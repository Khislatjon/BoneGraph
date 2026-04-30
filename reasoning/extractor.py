"""
reasoning/extractor.py
======================
Step 4.2 — LLM triple extraction from the BoneMind corpus chunks.

Overview
--------
Reads chunks from chunks.db in priority order (textbooks first, then
papers ranked by citation count descending), sends each to a local
Ollama model for structured triple extraction, canonicalises the output,
and writes new nodes and edges to ontology.db via OntologyStore.

Progress is tracked in the extraction_progress table inside ontology.db.
The pipeline is fully resumable — interrupting and restarting will skip
already-processed chunks.

Priority order
--------------
1. Textbook chunks (1,983 chunks — highest quality, shortest text)
2. Paper chunks from papers with ≥ N citations (default: 100)
3. Remaining paper chunks (when --source all is used)

Extraction model
----------------
Uses the Ollama HTTP API with the configured EXTRACTION_MODEL
(default: huatuogpt-bone:latest).  The model is prompted to return
a JSON array of (node_1, relation, node_2) triples using the
controlled vocabulary defined in reasoning/ontology.py.

Canonicalisation
----------------
Extracted node names are converted to snake_case and checked against
known seed node IDs and labels.  Matched nodes reuse the canonical
seed ID; unmatched nodes are added as type "concept" for later
classification.

Usage::

    # Textbooks only (recommended first run — ~30 min)
    python -m reasoning.extractor

    # Textbooks + papers with ≥ 100 citations (~4–8 hours)
    python -m reasoning.extractor --source papers --min-citations 100

    # Everything
    python -m reasoning.extractor --source all

    # Quick test on 20 chunks
    python -m reasoning.extractor --limit 20

    # Show extraction progress without running
    python -m reasoning.extractor --stats
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import requests

from config.settings import (
    CHUNKS_DB_PATH,
    EXTRACTION_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    ONTOLOGY_DB_PATH,
    PAPERS_DB_PATH,
)
from reasoning.graph_db import OntologyStore
from reasoning.ontology import RELATION_TYPES, Edge, Node
from reasoning.seed import SEED_NODES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Seed node lookup tables ───────────────────────────────────────────────────
# Built once at module load — used by canonicalise_node_id().

# Exact node_id → node_id (trivial, but unifies the lookup path)
_SEED_BY_ID: dict[str, str] = {n.node_id: n.node_id for n in SEED_NODES}

# Normalised label → canonical node_id
# e.g. "Fracture Toughness" → "fracture_toughness"
_SEED_BY_LABEL: dict[str, str] = {
    re.sub(r"[^a-z0-9]+", "_", n.label.lower()).strip("_"): n.node_id
    for n in SEED_NODES
}

# Common bone science abbreviations → canonical node_id
_ABBREV: dict[str, str] = {
    "bmd":          "bone_mineral_density",
    "bv_tv":        "trabecular_connectivity",
    "tb_th":        "trabecular_thickness",
    "tb_sp":        "trabecular_spacing",
    "tb_n":         "trabecular_connectivity",
    "ct_th":        "cortical_thickness",
    "kic":          "fracture_toughness",
    "k_ic":         "fracture_toughness",
    "e_modulus":    "elastic_modulus",
    "youngs_modulus":"elastic_modulus",
    "young_s_modulus":"elastic_modulus",
    "ult":          "ultimate_strength",
    "uts":          "tensile_strength",
    "ucs":          "compressive_strength",
    "oi":           "osteogenesis_imperfecta",
    "avnc":         "avascular_necrosis",
    "avascular_necrosis_of_bone": "avascular_necrosis",
    "ha":           "hydroxyapatite_scaffold",
    "ha_scaffold":  "hydroxyapatite_scaffold",
    "ha_crystal":   "hydroxyapatite_crystal",
    "bmp":          "bone_morphogenetic_protein",
    "bmp_2":        "BMP2",
    "bmp2":         "BMP2",
    "tgf":          "TGF_beta",
    "rankl":        "RANKL",
    "opg":          "OPG",
    "pth":          "PTH",
    "igf":          "IGF1",
    "igf_1":        "IGF1",
    "age":          "advanced_glycation_endproduct",
    "ages":         "advanced_glycation_endproduct",
    "bmu":          "bone_remodelling",   # basic multicellular unit drives remodelling
    "frax":         "FRAX_score",
    "dxa":          "DXA_measurement",
    "dexa":         "DXA_measurement",
    "qct":          "bone_mineral_density",
    "peek":         "PEEK_implant",
    "tcp":          "beta_tricalcium_phosphate",
    "b_tcp":        "beta_tricalcium_phosphate",
    "dbm":          "demineralized_bone_matrix",
}

# Simple type-inference heuristics for extracted nodes.
# Checked in order — first match wins.
_TYPE_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(osteoporosis|osteopenia|fracture|sarcoma|metastasis|disease|"
                r"arthritis|necrosis|malacia|rickets|imperfecta|syndrome|"
                r"porosis|penia)$"), "pathology"),
    (re.compile(r"(modulus|strength|toughness|density|porosity|stiffness|"
                r"hardness|ductility|anisotropy|viscosity|elasticity|"
                r"crystallinity|connectivity|thickness|spacing|ratio|rate|"
                r"count|content)$"), "property"),
    (re.compile(r"(remodelling|remodeling|formation|resorption|mineralisation|"
                r"mineralization|crosslinking|propagation|initiation|healing|"
                r"ossification|apoptosis|signaling|signalling|transduction|"
                r"accumulation|deflection|bridging|differentiation|"
                r"regeneration|resorption|calcification|osteogenesis|"
                r"angiogenesis|vascularization|vascularisation)$"), "process"),
    (re.compile(r"(law|mechanostat|theory|mechanism|effect|principle|model)$"),
                "mechanism"),
    (re.compile(r"(scaffold|implant|cement|graft|glass|membrane|coating|"
                r"composite|ceramic|polymer|biomaterial)$"), "material"),
    (re.compile(r"(osteoblast|osteoclast|osteocyte|cell|stem_cell|"
                r"progenitor)$"), "cell"),
    (re.compile(r"(collagen|fibril|osteon|lamella|lacuna|canal|bone|"
                r"trabecula|cortex|periosteum|endosteum|marrow|"
                r"crystal|platelet|matrix|network)"), "structure"),
]


def canonicalise_node_id(raw: str) -> str:
    """
    Convert a raw LLM-extracted node name to a canonical snake_case ID.

    Lookup order:
    1. Abbreviation map (_ABBREV)
    2. Exact match against seed node_ids
    3. Normalised-label match against seed nodes
    4. Return cleaned snake_case string as a new node id

    Parameters
    ----------
    raw : str
        The node name as returned by the LLM (any capitalisation, spacing).

    Returns
    -------
    str
        Canonical snake_case node_id.
    """
    if not raw or not raw.strip():
        return ""

    # Step 1 — normalise to snake_case
    cleaned = raw.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = cleaned.strip("_")

    if not cleaned:
        return ""

    # Step 2 — abbreviation lookup
    if cleaned in _ABBREV:
        return _ABBREV[cleaned]

    # Step 3 — exact seed node_id match
    if cleaned in _SEED_BY_ID:
        return cleaned

    # Step 4 — normalised seed label match
    if cleaned in _SEED_BY_LABEL:
        return _SEED_BY_LABEL[cleaned]

    # Step 5 — return as a new node id (will be added as type "concept")
    return cleaned


def infer_node_type(node_id: str) -> str:
    """
    Heuristically infer the node type from its id.

    Parameters
    ----------
    node_id : str
        Canonical snake_case node id.

    Returns
    -------
    str
        One of NODE_TYPES; defaults to "concept" if no rule matches.
    """
    for pattern, ntype in _TYPE_HINTS:
        if pattern.search(node_id):
            return ntype
    return "concept"


# ── Noise filter ─────────────────────────────────────────────────────────────
# Node IDs matching any of these patterns are methodological / statistical
# artefacts, not bone science concepts — skip triples containing them.
_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(statistical|kruskal|wallis|anova|t_test|chi_square|p_value|"
               r"spearman|pearson|regression|machine|microtome|microscope|"
               r"software|program|analysis_program|image_analysis|"
               r"grinding|polishing|sectioning_process|histometric|"
               r"scanning_electron|transmission_electron)"),
]


def _is_noise(node_id: str) -> bool:
    """Return True if node_id looks like a methodological / statistical artefact."""
    return any(p.search(node_id) for p in _NOISE_PATTERNS)


# ── Triple dataclass ──────────────────────────────────────────────────────────


@dataclass
class RawTriple:
    """A single (node_1, relation, node_2) triple as returned by the LLM."""

    node_1: str
    relation: str
    node_2: str


# ── Ollama client ─────────────────────────────────────────────────────────────

# Extraction prompt — shown to the LLM once per chunk.
_EXTRACTION_PROMPT = """\
You are a bone science expert. Extract causal and structural relationships \
from the text below as JSON triples.

Valid relations (use EXACTLY one of these):
determines, increases, decreases, activates, inhibits, leads_to, \
is_part_of, predicts, analogous_to, correlates_with, measures

Node naming rules:
- Use snake_case (e.g. fracture_toughness, cortical_bone)
- Keep names concise (2-4 words max)
- Only bone science concepts

Examples of good triples:
[
  {{"node_1": "porosity", "relation": "decreases", "node_2": "elastic_modulus"}},
  {{"node_1": "osteoclast_resorption", "relation": "leads_to", "node_2": "bone_resorption"}},
  {{"node_1": "bone_mineral_density", "relation": "predicts", "node_2": "fracture_risk"}},
  {{"node_1": "collagen_crosslink_density", "relation": "determines", "node_2": "fracture_toughness"}}
]

Rules:
- Extract 3–8 triples, or [] if no clear relationships exist
- Only extract relationships explicitly stated or strongly implied
- Return ONLY a valid JSON array — no explanation, no markdown

Text:
{text}

JSON:"""


def _call_ollama(text: str, model: str, base_url: str, timeout: int) -> str:
    """
    Send a chunk of text to Ollama and return the raw response string.

    Parameters
    ----------
    text : str
        The chunk text to extract triples from.
    model : str
        Ollama model name (e.g. "huatuogpt-bone:latest").
    base_url : str
        Ollama server URL (e.g. "http://localhost:11434").
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    str
        Raw text response from the model.

    Raises
    ------
    requests.RequestException
        If the HTTP request fails.
    """
    prompt = _EXTRACTION_PROMPT.format(text=text[:2000])  # cap at 2000 chars
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,   # low temperature for consistent structured output
            "num_predict": 512,   # triples are short — cap token usage
        },
    }
    resp = requests.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _parse_triples(raw_response: str) -> list[RawTriple]:
    """
    Parse the LLM response into a list of RawTriple objects.

    Handles common failure modes:
    - Response wrapped in markdown code fences
    - Extra text before/after the JSON array
    - Individual malformed entries (skipped, rest kept)

    Parameters
    ----------
    raw_response : str
        The raw string returned by the LLM.

    Returns
    -------
    list[RawTriple]
        Parsed triples (may be empty if parsing fully fails).
    """
    if not raw_response:
        return []

    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", raw_response).strip()

    # Find the first JSON array in the response
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []

    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    if not isinstance(items, list):
        return []

    triples: list[RawTriple] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        n1 = item.get("node_1", "").strip()
        rel = item.get("relation", "").strip().lower()
        n2 = item.get("node_2", "").strip()
        if n1 and rel and n2 and rel in RELATION_TYPES:
            triples.append(RawTriple(node_1=n1, relation=rel, node_2=n2))

    return triples


# ── Extraction pipeline ───────────────────────────────────────────────────────


class ExtractionPipeline:
    """
    Orchestrates triple extraction across the corpus.

    Reads chunks from chunks.db, calls Ollama for each, and writes
    results to ontology.db.  Progress is tracked so the pipeline
    can be safely interrupted and resumed.

    Parameters
    ----------
    model : str
        Ollama model name.
    base_url : str
        Ollama server base URL.
    timeout : int
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        model: str = EXTRACTION_MODEL,
        base_url: str = OLLAMA_URL,
        timeout: int = OLLAMA_TIMEOUT,
    ) -> None:
        self.model    = model
        self.base_url = base_url
        self.timeout  = timeout

    # ── Chunk loading ─────────────────────────────────────────────────────────

    def _load_citation_map(self) -> dict[str, int]:
        """Return mapping paper_id → citation_count from papers.db."""
        conn = sqlite3.connect(PAPERS_DB_PATH)
        rows = conn.execute(
            "SELECT paper_id, COALESCE(citation_count, 0) FROM papers"
        ).fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}

    def _iter_chunks(
        self,
        source: str,
        min_citations: int,
        limit: int | None,
        processed: set[str],
    ) -> Iterator[tuple[str, str, str, int]]:
        """
        Yield (chunk_id, text, source_type, citation_count) in priority order.

        Priority: textbooks first, then papers ordered by citation_count DESC.

        Parameters
        ----------
        source : str
            "textbooks" | "papers" | "all"
        min_citations : int
            Only include paper chunks from papers with >= this many citations.
            Ignored when source == "textbooks".
        limit : int | None
            Maximum number of chunks to yield (None = no limit).
        processed : set[str]
            Chunk IDs already processed — these are skipped.
        """
        cite_map = self._load_citation_map()

        conn = sqlite3.connect(CHUNKS_DB_PATH)
        conn.row_factory = sqlite3.Row

        # Build WHERE clause
        if source == "textbooks":
            where = "WHERE source_type = 'textbook'"
        elif source == "papers":
            where = "WHERE source_type = 'paper'"
        else:  # all
            where = ""

        rows = conn.execute(
            f"SELECT id, text, source_type, source_id FROM chunks {where}"
        ).fetchall()
        conn.close()

        # Annotate with citation count and sort
        annotated: list[tuple[str, str, str, int]] = []
        for row in rows:
            chunk_id    = str(row["id"])   # cast to str: chunks.db id is INTEGER
            source_type = row["source_type"]
            source_id   = row["source_id"]
            citations   = cite_map.get(source_id, 0) if source_type == "paper" else 999_999

            # Apply citation filter for paper chunks
            if source_type == "paper" and source != "textbooks" and citations < min_citations:
                continue

            annotated.append((chunk_id, row["text"], source_type, citations))

        # Sort: textbooks first (citation=999_999 sorts highest), then by citation DESC
        annotated.sort(key=lambda x: x[3], reverse=True)

        yielded = 0
        for chunk_id, text, source_type, citations in annotated:
            if chunk_id in processed:
                continue
            if not text or len(text.strip()) < 50:
                continue
            if limit is not None and yielded >= limit:
                break
            yield chunk_id, text, source_type, citations
            yielded += 1

    # ── Core run loop ─────────────────────────────────────────────────────────

    def run(
        self,
        source: str = "textbooks",
        min_citations: int = 100,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> None:
        """
        Run the extraction pipeline.

        Parameters
        ----------
        source : str
            "textbooks" — textbook chunks only (default, recommended first run)
            "papers"    — paper chunks filtered by min_citations
            "all"       — everything (textbooks + all papers)
        min_citations : int
            Citation threshold for paper chunks (default 100).
        limit : int | None
            Cap on number of chunks to process (useful for testing).
        dry_run : bool
            If True, print what would be processed but do not call Ollama.
        """
        with OntologyStore() as store:
            processed = store.get_processed_chunk_ids()
            logger.info(
                "Resuming: %d chunks already processed", len(processed)
            )

            # Count eligible chunks
            total = sum(
                1 for _ in self._iter_chunks(source, min_citations, limit, processed)
            )
            if total == 0:
                logger.info("No new chunks to process. Run --stats to check progress.")
                return

            logger.info(
                "Source: %s | min_citations: %d | eligible: %d chunks%s",
                source, min_citations, total,
                f" (capped at {limit})" if limit else "",
            )
            if dry_run:
                logger.info("DRY RUN — no Ollama calls will be made.")

            # ── Main loop ──────────────────────────────────────────────────
            n_done = n_empty = n_failed = n_triples_total = 0
            t_start = time.time()

            for i, (chunk_id, text, source_type, citations) in enumerate(
                self._iter_chunks(source, min_citations, limit, processed), start=1
            ):
                if dry_run:
                    print(
                        f"  [{i:>5}/{total}] {source_type:>8} "
                        f"cites={citations:>6}  chunk={chunk_id[:20]}"
                    )
                    continue

                # ── Call Ollama ────────────────────────────────────────────
                try:
                    raw = _call_ollama(
                        text, self.model, self.base_url, self.timeout
                    )
                    triples = _parse_triples(raw)
                except requests.RequestException as exc:
                    logger.warning("Ollama error on chunk %s: %s", chunk_id, exc)
                    store.record_progress(chunk_id, "failed")
                    n_failed += 1
                    continue
                except Exception as exc:
                    logger.warning("Parse error on chunk %s: %s", chunk_id, exc)
                    store.record_progress(chunk_id, "failed")
                    n_failed += 1
                    continue

                if not triples:
                    store.record_progress(chunk_id, "empty")
                    n_empty += 1
                else:
                    n_written = self._write_triples(store, triples, chunk_id)
                    store.record_progress(chunk_id, "done", n_written)
                    n_triples_total += n_written
                    n_done += 1

                # ── Progress log every 50 chunks ───────────────────────────
                if i % 200 == 0 or i == total:
                    elapsed = time.time() - t_start
                    rate = i / elapsed if elapsed > 0 else 0
                    eta  = (total - i) / rate if rate > 0 else 0
                    logger.info(
                        "[%d/%d]  done=%d  empty=%d  failed=%d  "
                        "triples=%d  %.1f chunk/s  ETA %.0f min",
                        i, total, n_done, n_empty, n_failed,
                        n_triples_total, rate, eta / 60,
                    )

            # ── Final summary ──────────────────────────────────────────────
            if not dry_run:
                elapsed = time.time() - t_start
                db_stats = store.stats()
                logger.info("=" * 60)
                logger.info("Extraction complete in %.1f min", elapsed / 60)
                logger.info(
                    "  Chunks processed : %d done · %d empty · %d failed",
                    n_done, n_empty, n_failed,
                )
                logger.info("  Triples written  : %d", n_triples_total)
                logger.info(
                    "  Graph now        : %d nodes · %d edges",
                    db_stats["total_nodes"], db_stats["total_edges"],
                )
                logger.info("=" * 60)

    # ── Triple writing ────────────────────────────────────────────────────────

    def _write_triples(
        self,
        store: OntologyStore,
        triples: list[RawTriple],
        chunk_id: str,
    ) -> int:
        """
        Canonicalise and persist triples from one chunk.

        New nodes are created on the fly with inferred types.
        New edges accumulate weight if they already exist.

        Parameters
        ----------
        store : OntologyStore
            Open database connection.
        triples : list[RawTriple]
            Raw triples from the LLM.
        chunk_id : str
            Source chunk ID stored as evidence on each edge.

        Returns
        -------
        int
            Number of triples successfully written.
        """
        written = 0
        for triple in triples:
            n1 = canonicalise_node_id(triple.node_1)
            n2 = canonicalise_node_id(triple.node_2)

            if not n1 or not n2 or n1 == n2:
                continue  # skip degenerate triples

            # Skip methodological / statistical noise
            if _is_noise(n1) or _is_noise(n2):
                continue

            # Ensure both nodes exist in the DB
            for nid in (n1, n2):
                if not store.get_node(nid):
                    ntype = infer_node_type(nid)
                    label = nid.replace("_", " ")
                    store.upsert_node(
                        Node(
                            node_id=nid,
                            label=label,
                            node_type=ntype,
                            description="",
                            source="extracted",
                        )
                    )

            # Write the edge (upsert accumulates weight + evidence)
            store.upsert_edge(
                Edge(
                    source=n1,
                    relation=triple.relation,
                    target=n2,
                    weight=1.0,
                    evidence=[chunk_id],
                    edge_source="extracted",
                )
            )
            written += 1

        return written


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4.2 — LLM triple extraction from corpus chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m reasoning.extractor                           # textbooks only\n"
            "  python -m reasoning.extractor --source papers --min-citations 100\n"
            "  python -m reasoning.extractor --source all\n"
            "  python -m reasoning.extractor --limit 20               # quick test\n"
            "  python -m reasoning.extractor --dry-run                # preview only\n"
            "  python -m reasoning.extractor --stats                  # progress report\n"
        ),
    )
    parser.add_argument(
        "--source",
        choices=["textbooks", "papers", "all"],
        default="textbooks",
        help="Which chunks to process (default: textbooks).",
    )
    parser.add_argument(
        "--min-citations",
        type=int,
        default=100,
        metavar="N",
        help="Only process paper chunks from papers with >= N citations (default: 100).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Stop after processing N chunks (default: no limit).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=EXTRACTION_MODEL,
        help=f"Ollama model name (default: {EXTRACTION_MODEL}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print eligible chunks without calling Ollama.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print extraction progress and graph stats, then exit.",
    )
    return parser.parse_args()


def _print_stats() -> None:
    """Print extraction progress and current graph stats."""
    with OntologyStore() as store:
        ex  = store.extraction_stats()
        db  = store.stats()

    total_chunks = 248_629  # known corpus size
    processed    = ex["total_processed"]
    pct          = 100 * processed / total_chunks if total_chunks else 0

    print()
    print("─" * 58)
    print("  BoneMind — Phase 4 extraction progress")
    print("─" * 58)
    print(f"  Chunks processed : {processed:>7,} / {total_chunks:,}  ({pct:.1f}%)")
    print(f"    done           : {ex['done']:>7,}")
    print(f"    empty          : {ex['empty']:>7,}  (no triples found)")
    print(f"    failed         : {ex['failed']:>7,}  (Ollama / parse error)")
    print(f"  Triples written  : {ex['total_triples']:>7,}")
    print()
    print(f"  Graph now")
    print(f"    Total nodes    : {db['total_nodes']:>7,}  "
          f"(seed: {db['seed_nodes']} · extracted: {db['extracted_nodes']})")
    print(f"    Total edges    : {db['total_edges']:>7,}  "
          f"(seed: {db['seed_edges']} · extracted: {db['extracted_edges']})")
    print("─" * 58)
    print()


def main() -> None:
    args = _parse_args()

    if args.stats:
        _print_stats()
        sys.exit(0)

    # Verify Ollama is reachable before starting
    if not args.dry_run:
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if args.model not in models:
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s",
                    args.model, models,
                )
                logger.warning("Pull it with: ollama pull %s", args.model)
                sys.exit(1)
            logger.info("Ollama OK — using model: %s", args.model)
        except requests.RequestException as exc:
            logger.error("Cannot reach Ollama at %s: %s", OLLAMA_URL, exc)
            logger.error("Start Ollama with: ollama serve")
            sys.exit(1)

    pipeline = ExtractionPipeline(
        model=args.model,
        base_url=OLLAMA_URL,
        timeout=OLLAMA_TIMEOUT,
    )
    pipeline.run(
        source=args.source,
        min_citations=args.min_citations,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
