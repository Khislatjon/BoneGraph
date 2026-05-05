"""
scripts/clean_graph.py
──────────────────────
Six-stage graph cleanup for ontology.db.

Reads  data/db/ontology.db   (the noisy LLM-extracted graph)
Writes data/db/ontology_clean.db   (cleaned graph; original is left intact)

Stages
------
1. Drop cross-domain nodes        — tumor, cancer, covid, breast, silk, …
2. Drop sentence-fragment nodes   — label > 5 words or > 50 chars
3. Canonicalise nodes             — strip direction prefixes/suffixes
                                    (increased_X → X, X_decrease → X, …)
                                    and merge edges accordingly
4. Resolve same-relation A↔B      — for (A,r,B) and (B,r,A) keep higher weight
5. Drop low-weight extracted edges — keep all seed edges; drop extracted
                                     edges with weight < min_weight (default 2)
6. Drop orphan nodes              — nodes with degree 0 after edge cleanup

Each stage logs how many nodes/edges it removed.  The original DB is not
touched; the cleaned DB is written separately so you can A/B test.

Usage::

    python scripts/clean_graph.py
    python scripts/clean_graph.py --min-weight 3
    python scripts/clean_graph.py --dry-run

After review, swap in the cleaned DB::

    mv data/db/ontology.db data/db/ontology_raw.db
    mv data/db/ontology_clean.db data/db/ontology.db
"""

from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# ── locate DB relative to project root ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_INPUT  = PROJECT_ROOT / "data" / "db" / "ontology.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "db" / "ontology_clean.db"


# ── Stage 1 — cross-domain keyword filter ─────────────────────────────────────
#
# Any node whose node_id CONTAINS one of these substrings is dropped.
# Curated to catch obvious off-topic contamination (cancer biology, viral
# infection, non-bone tissues, food/silk corpus leakage) while preserving
# bone-relevant biomimetic terms (mussel adhesive, nacre, hydroxyapatite).
#
CROSS_DOMAIN_KEYWORDS = [
    # cancer / oncology (osteosarcoma is preserved by the explicit allow-list)
    "tumor", "tumour", "cancer", "carcinoma", "oncolog",
    "malignan", "metasta", "leukemia", "lymphoma", "melanoma",
    # viral / pathogen
    "covid", "sars_cov", "ebola", "hiv_", "viral_infection", "influenza",
    # non-bone tissues
    "breast_", "_breast", "prostate", "ovarian", "uterine", "cervical",
    "pancreatic", "hepatic_cancer",
    # food / silk / unrelated materials in the corpus
    "silk_", "_silk", "cheese", "dough", "hake_", "whiting_",
    "pacific_whiting", "beethoven", "symphony", "fishmeal",
]

# Substrings that LOOK cross-domain but are bone-relevant — keep these.
ALLOW_LIST = [
    "osteosarcoma", "bone_cancer", "bone_metastasis", "skeletal_metastas",
    "spinal_metastas", "bone_tumor", "bone_tumour",
]


def is_cross_domain(node_id: str) -> bool:
    nid = node_id.lower()
    if any(allowed in nid for allowed in ALLOW_LIST):
        return False
    return any(kw in nid for kw in CROSS_DOMAIN_KEYWORDS)


# ── Stage 3 — canonicalisation rules ──────────────────────────────────────────
#
# Direction-encoding affixes that should be stripped to merge variants.
# Applied iteratively until the node_id stabilises.
#
DIRECTION_PREFIXES = [
    "increased_", "decreased_", "reduced_", "elevated_",
    "higher_", "lower_", "low_", "high_", "abnormal_",
    "new_", "future_", "subsequent_", "old_",
    "loss_of_", "lack_of_", "absence_of_", "presence_of_",
    "degree_of_", "level_of_", "accuracy_of_",
]

DIRECTION_SUFFIXES = [
    "_increase", "_decrease", "_reduction", "_decline",
    "_increases", "_decreases",
    "_levels", "_level",
]


def canonical_id(node_id: str) -> str:
    """
    Strip direction-encoding affixes iteratively.

    Examples
    --------
    increased_fracture_risk  → fracture_risk
    low_bmd_levels           → bmd
    fracture_risk_increase   → fracture_risk
    """
    nid = node_id
    changed = True
    while changed:
        changed = False
        for pfx in DIRECTION_PREFIXES:
            if nid.startswith(pfx) and len(nid) > len(pfx) + 1:
                nid = nid[len(pfx):]
                changed = True
                break
        for sfx in DIRECTION_SUFFIXES:
            if nid.endswith(sfx) and len(nid) > len(sfx) + 1:
                nid = nid[:-len(sfx)]
                changed = True
                break
    return nid


# ── stats helpers ─────────────────────────────────────────────────────────────


def graph_stats(con: sqlite3.Connection) -> tuple[int, int]:
    """Return (n_nodes, n_edges)."""
    n = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    e = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    return n, e


def log_stage(name: str, before: tuple[int, int], after: tuple[int, int]) -> None:
    n0, e0 = before
    n1, e1 = after
    print(
        f"  {name:<46} "
        f"nodes {n0:>6} → {n1:>6} ({n1 - n0:+d})  "
        f"edges {e0:>6} → {e1:>6} ({e1 - e0:+d})"
    )


# ── Stage 1 ───────────────────────────────────────────────────────────────────


def stage1_drop_cross_domain(con: sqlite3.Connection) -> int:
    """Delete nodes whose ID contains a cross-domain keyword. Cascade to edges."""
    cur = con.execute("SELECT node_id FROM nodes")
    bad = [row[0] for row in cur.fetchall() if is_cross_domain(row[0])]

    if not bad:
        return 0

    placeholders = ",".join("?" * len(bad))
    con.execute(
        f"DELETE FROM edges WHERE source_node IN ({placeholders}) "
        f"OR target_node IN ({placeholders})",
        bad + bad,
    )
    con.execute(f"DELETE FROM nodes WHERE node_id IN ({placeholders})", bad)
    con.commit()
    return len(bad)


# ── Stage 2 ───────────────────────────────────────────────────────────────────


def stage2_drop_sentence_fragments(
    con: sqlite3.Connection,
    max_words: int = 5,
    max_chars: int = 50,
) -> int:
    """
    Drop nodes whose label is a sentence fragment.

    A node label is a sentence fragment if it has > max_words words OR
    its node_id length > max_chars characters.
    """
    cur = con.execute("SELECT node_id, label FROM nodes")
    bad: list[str] = []
    for nid, label in cur.fetchall():
        n_words = len(label.split())
        if n_words > max_words or len(nid) > max_chars:
            bad.append(nid)

    if not bad:
        return 0

    # Delete in batches (SQLite parameter limit is ~999)
    BATCH = 500
    for i in range(0, len(bad), BATCH):
        chunk = bad[i:i + BATCH]
        ph = ",".join("?" * len(chunk))
        con.execute(
            f"DELETE FROM edges WHERE source_node IN ({ph}) OR target_node IN ({ph})",
            chunk + chunk,
        )
        con.execute(f"DELETE FROM nodes WHERE node_id IN ({ph})", chunk)
    con.commit()
    return len(bad)


# ── Stage 3 ───────────────────────────────────────────────────────────────────


def stage3_canonicalise(con: sqlite3.Connection) -> tuple[int, int]:
    """
    Merge direction-encoded variants into their canonical form.

    Returns (n_nodes_merged, n_edges_collapsed).

    Algorithm
    ---------
    1. Build map: old_node_id → canonical_node_id.
    2. For each old_id whose canonical differs:
       - Ensure the canonical node exists (insert if missing).
       - Rewrite all edges to use canonical endpoints.
    3. Deduplicate edges with identical (source, relation, target),
       summing their weights.  This collapses the redundant evidence
       that was scattered across variant nodes.
    4. Drop the now-orphaned variant nodes.
    """
    rows = con.execute("SELECT node_id, label, node_type FROM nodes").fetchall()
    canon_map: dict[str, str] = {}
    canonical_to_label: dict[str, tuple[str, str]] = {}

    for nid, label, node_type in rows:
        c = canonical_id(nid)
        canon_map[nid] = c
        if c not in canonical_to_label:
            canonical_to_label[c] = (label, node_type)

    # Step 1 — make sure every canonical id exists as a node.
    existing = {nid for nid, _, _ in rows}
    new_nodes = []
    for c, (label, node_type) in canonical_to_label.items():
        if c not in existing:
            # Synthesise a label from the canonical id.
            new_label = c.replace("_", " ")
            new_nodes.append((c, new_label, node_type, "", "extracted"))
    if new_nodes:
        con.executemany(
            "INSERT OR IGNORE INTO nodes (node_id, label, node_type, description, source) "
            "VALUES (?, ?, ?, ?, ?)",
            new_nodes,
        )

    # Step 2 — load all edges, rewrite endpoints, sum weights, build new table.
    edges = con.execute(
        "SELECT source_node, target_node, relation, weight, evidence, edge_source "
        "FROM edges"
    ).fetchall()

    merged: dict[tuple[str, str, str], dict] = {}
    for src, tgt, rel, weight, evidence, esource in edges:
        c_src = canon_map.get(src, src)
        c_tgt = canon_map.get(tgt, tgt)
        if c_src == c_tgt:
            # Self-loops created by canonicalisation are dropped.
            continue
        key = (c_src, rel, c_tgt)
        if key not in merged:
            merged[key] = {
                "weight": weight,
                "evidence": evidence,
                # If any merged edge was a seed, the result is a seed.
                "edge_source": esource,
            }
        else:
            merged[key]["weight"] += weight
            if esource == "seed":
                merged[key]["edge_source"] = "seed"

    # Step 3 — wipe edges and re-insert deduplicated set.
    n_before = len(edges)
    con.execute("DELETE FROM edges")
    con.executemany(
        "INSERT INTO edges (edge_id, source_node, target_node, relation, "
        "weight, evidence, edge_source) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                f"{src}|{rel}|{tgt}",
                src, tgt, rel,
                data["weight"],
                data["evidence"],
                data["edge_source"],
            )
            for (src, rel, tgt), data in merged.items()
        ],
    )

    # Step 4 — delete variant nodes whose canonical differs.
    variants = [old for old, c in canon_map.items() if old != c]
    BATCH = 500
    for i in range(0, len(variants), BATCH):
        chunk = variants[i:i + BATCH]
        ph = ",".join("?" * len(chunk))
        con.execute(f"DELETE FROM nodes WHERE node_id IN ({ph})", chunk)

    con.commit()
    return len(variants), n_before - len(merged)


# ── Stage 4 ───────────────────────────────────────────────────────────────────


def stage4_resolve_reverse_pairs(con: sqlite3.Connection) -> int:
    """
    For (A, r, B) and (B, r, A) on the same relation, keep the higher-weight
    side and delete the lower-weight one.  Drops only same-relation
    contradictions; opposite-direction relations (e.g. is_part_of vs leads_to)
    are preserved.
    """
    edges = con.execute(
        "SELECT edge_id, source_node, target_node, relation, weight FROM edges"
    ).fetchall()

    # Index by canonical pair + relation
    forward: dict[tuple[str, str, str], tuple[str, float]] = {}
    to_drop: list[str] = []

    for edge_id, src, tgt, rel, weight in edges:
        # Canonical pair (alphabetical) so both directions land on the same key.
        a, b = sorted((src, tgt))
        key = (a, b, rel)
        if key not in forward:
            forward[key] = (edge_id, weight)
        else:
            other_id, other_w = forward[key]
            if weight > other_w:
                to_drop.append(other_id)
                forward[key] = (edge_id, weight)
            else:
                to_drop.append(edge_id)

    if not to_drop:
        return 0

    BATCH = 500
    for i in range(0, len(to_drop), BATCH):
        chunk = to_drop[i:i + BATCH]
        ph = ",".join("?" * len(chunk))
        con.execute(f"DELETE FROM edges WHERE edge_id IN ({ph})", chunk)
    con.commit()
    return len(to_drop)


# ── Stage 5 ───────────────────────────────────────────────────────────────────


def stage5_drop_low_weight(con: sqlite3.Connection, min_weight: float) -> int:
    """
    Drop extracted edges with weight < min_weight.
    Seed edges are preserved regardless of weight.
    """
    cur = con.execute(
        "DELETE FROM edges WHERE edge_source != 'seed' AND weight < ?",
        (min_weight,),
    )
    con.commit()
    return cur.rowcount


# ── Stage 6 ───────────────────────────────────────────────────────────────────


def stage6_drop_orphans(con: sqlite3.Connection) -> int:
    """Remove nodes with no incoming or outgoing edges."""
    cur = con.execute(
        """
        DELETE FROM nodes
        WHERE node_id NOT IN (SELECT source_node FROM edges)
          AND node_id NOT IN (SELECT target_node FROM edges)
        """
    )
    con.commit()
    return cur.rowcount


# ── orchestration ─────────────────────────────────────────────────────────────


def run_cleanup(input_path: Path, output_path: Path, min_weight: float) -> None:
    print(f"Source     : {input_path}")
    print(f"Destination: {output_path}")
    print(f"min_weight : {min_weight} (seed edges always preserved)")
    print()

    if output_path.exists():
        output_path.unlink()
    shutil.copy(input_path, output_path)

    con = sqlite3.connect(output_path)
    con.execute("PRAGMA foreign_keys = OFF")  # bypass FK during deletes

    initial = graph_stats(con)
    print(f"Initial graph: {initial[0]} nodes, {initial[1]} edges")
    print()
    print("Stage results:")

    before = initial
    n_dropped = stage1_drop_cross_domain(con)
    after = graph_stats(con)
    log_stage(f"1. Drop cross-domain nodes ({n_dropped})", before, after)

    before = after
    n_dropped = stage2_drop_sentence_fragments(con)
    after = graph_stats(con)
    log_stage(f"2. Drop sentence fragments ({n_dropped})", before, after)

    before = after
    n_merged, n_collapsed = stage3_canonicalise(con)
    after = graph_stats(con)
    log_stage(
        f"3. Canonicalise variants ({n_merged} merged, {n_collapsed} edges collapsed)",
        before, after,
    )

    before = after
    n_dropped = stage4_resolve_reverse_pairs(con)
    after = graph_stats(con)
    log_stage(f"4. Resolve reverse-pair contradictions ({n_dropped})", before, after)

    before = after
    n_dropped = stage5_drop_low_weight(con, min_weight)
    after = graph_stats(con)
    log_stage(f"5. Drop low-weight extracted edges ({n_dropped})", before, after)

    before = after
    n_dropped = stage6_drop_orphans(con)
    after = graph_stats(con)
    log_stage(f"6. Drop orphan nodes ({n_dropped})", before, after)

    con.execute("VACUUM")
    con.close()

    final = after
    print()
    print(f"Done. Cleaned graph: {final[0]} nodes, {final[1]} edges")
    print(
        f"Reduction: nodes -{initial[0] - final[0]} "
        f"({(1 - final[0] / initial[0]) * 100:.1f}%), "
        f"edges -{initial[1] - final[1]} "
        f"({(1 - final[1] / initial[1]) * 100:.1f}%)"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    p.add_argument("--input",  type=Path, default=DEFAULT_INPUT,
                   help=f"input DB (default: {DEFAULT_INPUT})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"output DB (default: {DEFAULT_OUTPUT})")
    p.add_argument("--min-weight", type=float, default=2.0,
                   help="drop extracted edges with weight < this (default: 2.0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        print(f"Input DB not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    run_cleanup(args.input, args.output, args.min_weight)


if __name__ == "__main__":
    main()
