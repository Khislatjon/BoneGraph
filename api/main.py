"""
api/main.py
===========
FastAPI backend for the BoneMind React frontend.

Endpoints
---------
GET  /api/stats            — corpus statistics
POST /api/ask              — SSE stream: RAG retrieval + Ollama LLM
                              · accepts optional `history` (JSON array of {role,content})
                                for multi-turn follow-up questions
                              · topic guard: a general-purpose classifier model (llama3.2:3b)
                                decides if the question is bone-science-related; off-topic
                                questions are rejected before retrieval and before the main
                                LLM call. (llama3.2:1b was tried first but misclassified
                                clear bone questions like "trabecular vs cortical bone" as
                                off-topic — too small to follow the classification prompt.)
                              · results deduplicated by title before context is built
                                so the LLM never sees the same paper under two rank numbers
                              · answer generated at temperature 0 for full determinism
                              · References section is rebuilt server-side from the retrieved
                                results — the LLM's own References block is discarded because
                                huatuogpt-bone tends to renumber citations, breaking the link
                                between inline [N] and the cited paper
POST /api/search           — semantic search, JSON response
POST /api/reason           — LRM hypothesis generation, JSON response
POST /api/reason_v2        — v2 equation-graph reasoner (forward / abductive /
                              counterfactual), JSON response
POST /api/reason_v2/ask    — Phase 4 free-text entry: single LLM call routes
                              the query, SPECTER2 anchors the variables, the
                              deterministic reasoner runs.
GET  /api/reason_v2/presets — v2 preset queries + variable/relation metadata
POST /api/analyse          — VLM image analysis (multipart), JSON response
GET  /                     — serves frontend/index.html
"""

import base64
import io
import json
import sqlite3
import time
from pathlib import Path

import requests
from fastapi import FastAPI, Form, UploadFile, File, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from retrieval.retriever import BoneMindRetriever
from reasoning.bone_relations import build_bone_registry, COVARIATE_SCHEMA
from reasoning.semantic_anchor import SemanticVariableAnchor
from reasoning.query_router import QueryRouter
from reasoning.lrm import LRM
from reasoning.novelty import NoveltyClassifier, CORPUS_DISCLAIMER
from config.settings import PAPERS_DB_PATH, TEXTBOOKS_DB_PATH, CHUNKS_DB_PATH

# ── Ollama config ──────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "huatuogpt-bone"
GUARD_MODEL  = "llama3.2:3b"  # general-purpose classifier for topic guard
VLM_MODEL    = "llava:13b"

VLM_PROMPT = (
    "You are a radiologist specialised in musculoskeletal and bone imaging. "
    "Describe this image systematically:\n"
    "1. Modality and body part\n"
    "2. Cortical bone: thickness, continuity, any thinning or breaks\n"
    "3. Trabecular bone: density and pattern\n"
    "4. Joint spaces if visible\n"
    "5. Any fractures, lesions, or pathological changes\n"
    "6. Overall impression in one sentence\n"
    "Use precise radiological terminology.\n\n"
    "Format your response as JSON:\n"
    '{"modality":"...","quality":"...","density":"...","findings":["..."],'
    '"assessment":"...","recommendation":"...","confidence":"HIGH|MODERATE|LOW"}'
)

SYSTEM_PROMPT = """You are BoneMind, an expert AI assistant specialised in bone science.
You have access to a curated corpus of peer-reviewed bone science literature and textbooks.

MOST IMPORTANT RULE — citations are mandatory:
Every sentence in your answer that states a fact MUST end with a citation like [1] or [2][3]
before the full stop. Use the number shown at the start of each context passage. If you write
a sentence without a citation, that sentence will be rejected. No exceptions.

RULES — follow exactly:

1. CONTEXT ONLY. Answer exclusively from the numbered context passages provided. Do not use
   knowledge from your training that is not supported by the context.

2. CITE EVERY CLAIM. After each factual claim, add [N] matching the passage number.
   Do not cite a passage you did not actually use. Cite multiple passages when relevant: [2][5].

3. INSUFFICIENT CONTEXT. If the context does not contain enough information, say:
   "The corpus does not contain sufficient information to answer this fully."
   Then summarise the most relevant available context and suggest a more specific query.

4. OUT-OF-DOMAIN. If the question is not about bone science (morphology, structure-function
   relationships, mechanics, pathology, imaging, biomaterials, or simulation), respond:
   "This question is outside BoneMind's domain. I cover bone science only."

5. HYPOTHESES. If you extend beyond direct evidence, mark it explicitly:
   **Hypothesis:** [speculative claim]
   Use this sparingly — only for well-grounded extrapolations, not speculation.

6. EVIDENCE QUALITY. Distinguish study types where relevant: systematic reviews and RCTs
   carry more weight than single studies; in vitro and animal data should be flagged as such.

7. UNCERTAINTY. When evidence is limited, conflicting, or inconclusive, say so clearly.
   Never overstate confidence.

8. NO FABRICATION. Do not invent authors, titles, statistics, or any data not present in
   the context passages.

9. STRUCTURE. Use markdown headers and bullet points for multi-part answers. Be concise —
   synthesise the evidence; do not copy-paste large verbatim passages.

10. REFERENCES SECTION. End every answer with a ## References section listing each cited
    passage as: [N] Title — Authors (Year) · Venue

---

EXAMPLE OF A CORRECT RESPONSE:

Question: How does denosumab affect cortical porosity, and why does that matter mechanically?

Answer:

Denosumab has been shown to decrease cortical porosity by inhibiting osteoclast-mediated
remodelling [2]. This is clinically significant because cortical porosity is a major
determinant of whole-bone stiffness and fracture resistance [3].

## References
[2] The effect of 8 or 5 years of denosumab treatment — Papapoulos et al. (2015) · Osteoporosis International
[3] Establishing Biomechanical Mechanisms in Mouse Models — Jepsen et al. (2015) · JBMR

---

CITATION RULES — enforced strictly:
- Every sentence that states a fact MUST end with [N] before the full stop.
- Never cite a passage number that was not provided in the context.
- If two passages support the same claim, cite both: [1][3].
- The ## References section is mandatory, even for short answers."""


# ── Startup: load models once ──────────────────────────────────────────────────
print("Loading BoneMind retriever...")
retriever = BoneMindRetriever()
retriever.load()

print("Loading LRM (bone knowledge graph)...")
lrm = LRM()

print("Building v2 equation-graph registry...")
bone_registry = build_bone_registry()

print("Wiring v2 semantic anchor (SPECTER2)...")
semantic_anchor = SemanticVariableAnchor(bone_registry, retriever=retriever)

print("Wiring v2 query router (Ollama)...")
query_router = QueryRouter(ollama_url=OLLAMA_URL, model=GUARD_MODEL)

print("Loading novelty classifier...")
novelty_clf = NoveltyClassifier(
    use_semantic=True,
    model=retriever._model,
    tokenizer=retriever._tokenizer,
    device=retriever._device,
)

print("Loading corpus stats...")


def _load_stats() -> dict:
    stats = {}
    conn = sqlite3.connect(PAPERS_DB_PATH)
    stats["papers_total"]    = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    stats["pdfs_downloaded"] = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE pdf_local_path IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    conn = sqlite3.connect(TEXTBOOKS_DB_PATH)
    stats["textbooks"] = conn.execute("SELECT COUNT(*) FROM textbooks").fetchone()[0]
    conn.close()
    conn = sqlite3.connect(CHUNKS_DB_PATH)
    stats["chunks"] = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    return stats


STATS = _load_stats()
print(
    f"Ready — {STATS['pdfs_downloaded']:,} papers · "
    f"{STATS['textbooks']} textbooks · "
    f"{STATS['chunks']:,} chunks"
)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="BoneMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_context(results: list[dict]) -> str:
    parts = []
    for r in results:
        label = f"[{r['rank']}]"
        if r["source_type"] == "paper":
            header = f"{label} {r['title']} — {r['authors'] or 'Unknown'} ({r['year'] or ''}) · {r['venue'] or ''}"
        else:
            header = f"{label} {r['title']} [Textbook]"
        parts.append(f"{header}\n{r['text'].strip()}")
    return "\n\n".join(parts)


def _inject_ref_links(answer: str, results: list[dict]) -> str:
    """
    Authoritative references handling:
      1. Strip whatever the LLM wrote in `## References` (it tends to renumber).
      2. Parse inline [N] citations from the body — those numbers are trusted
         because they came directly from the numbered context passages.
      3. Rebuild the References section from `results` keyed by rank, in order
         of first appearance in the body.

    Citations the LLM invented (rank not present in results) are dropped from
    the rebuilt list so the user never sees a phantom reference.
    """
    import re

    # Drop the model's References section entirely
    body, _, _ = answer.partition("## References")
    body = body.rstrip()

    by_rank: dict[int, dict] = {r["rank"]: r for r in results}

    # Find inline citations [N] in order of first appearance, dedup, keep only
    # ranks that actually exist in the retrieved results.
    seen: list[int] = []
    for m in re.finditer(r"\[(\d+)\]", body):
        n = int(m.group(1))
        if n in by_rank and n not in seen:
            seen.append(n)

    if not seen:
        return body  # no valid citations → no References section

    lines = ["## References", ""]
    for n in sorted(seen):
        r = by_rank[n]
        title   = r["title"]
        authors = r["authors"] or "Unknown"
        year    = r["year"] or ""
        venue   = r["venue"] or ""
        url = None
        if r["source_type"] == "paper":
            if r.get("doi"):
                url = f"https://doi.org/{r['doi']}"
            elif r.get("openalex_id"):
                url = f"https://openalex.org/{r['openalex_id']}"
        meta_bits = [authors, f"({year})" if year else "", venue]
        meta = " · ".join(b for b in meta_bits if b).replace(" · (", " (")
        line = f"[{n}] {title} — {meta}"
        if url:
            line += f" · [Open paper]({url})"
        lines.append(line)
        lines.append("")

    return body + "\n\n" + "\n".join(lines).rstrip() + "\n"


def _dedupe_venue(venue: str) -> str:
    if "/" not in venue:
        return venue
    parts = [p.strip() for p in venue.split("/")]
    seen = [parts[0]]
    for p in parts[1:]:
        norm = p.lower().replace("the ", "").strip()
        if not any(
            norm in s.lower().replace("the ", "").strip() or
            s.lower().replace("the ", "").strip() in norm
            for s in seen
        ):
            seen.append(p)
    return " / ".join(seen)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/test", response_class=HTMLResponse)
def serve_test():
    return (FRONTEND_DIR / "test.html").read_text(encoding="utf-8")


@app.get("/api/stats")
def get_stats():
    graph = lrm.graph_stats()
    return {
        "papers_total":    STATS["papers_total"],
        "pdfs_downloaded": STATS["pdfs_downloaded"],
        "textbooks":       STATS["textbooks"],
        "chunks":          STATS["chunks"],
        "graph_nodes":     graph["n_nodes"],
        "graph_edges":     graph["n_edges"],
    }


@app.post("/api/ask")
async def ask(question: str = Form(...), top_k: int = Form(8), history: str = Form("[]"), all_questions: str = Form("[]")):
    """
    Server-Sent Events stream.
    Events:
      {"type": "sources", "results": [...]}   — after retrieval
      {"type": "token",   "content": "..."}   — each LLM token
      {"type": "done",    "answer": "..."}    — final answer with DOI links injected
      {"type": "error",   "message": "..."}   — on failure

    `history` is a JSON array of {role, content} objects for prior turns.
    Retrieval always uses only the latest question so sources stay relevant.
    """
    prior_turns: list[dict] = json.loads(history) if history else []
    prior_questions_all: list[str] = json.loads(all_questions) if all_questions else []

    GUARD_PROMPT = (
        "{prior_block}"
        "The user's current question is: {q}\n\n"
        "Is the current question (taking the previous questions as context "
        "to resolve pronouns) about bone, the skeleton, bones, bone cells "
        "(osteoblasts, osteoclasts, osteocytes), bone diseases (osteoporosis, "
        "fractures, osteoarthritis, bone tumours), bone mechanics, bone "
        "imaging, bone biomaterials, bone metabolism, or orthopaedics?\n\n"
        "Reply with exactly one word: YES or NO."
    )

    def _is_bone_science(q: str, prior_questions: list[str]) -> bool:
        # Include the last few user questions so the classifier can resolve
        # pronouns ("it", "this", "they"). Assistant replies are deliberately
        # excluded — their off-topic-adjacent vocabulary confuses small classifiers.
        recent = prior_questions[-8:]
        if recent:
            prior_block = (
                "The user's previous questions in this conversation were:\n"
                + "\n".join(f"- {p}" for p in recent)
                + "\n\n"
            )
        else:
            prior_block = ""
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": GUARD_MODEL,
                    "messages": [{"role": "user", "content": GUARD_PROMPT.format(q=q, prior_block=prior_block)}],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 5},
                },
                timeout=15,
            )
            resp.raise_for_status()
            answer = resp.json()["message"]["content"].strip().upper()
            return answer.startswith("YES")
        except Exception:
            return True  # fail open so a network hiccup doesn't block real questions

    def generate():
        q = question.strip()
        if not q:
            yield f"data: {json.dumps({'type':'error','message':'Empty question'})}\n\n"
            return

        # Topic guard: uses the full question history (no sliding window) so it can
        # resolve pronouns even when early turns fall outside the LLM context window.
        if not _is_bone_science(q, prior_questions_all):
            out = (
                "I'm BoneMind, a specialist assistant for bone science. "
                "Your question doesn't appear to be related to bone biology, skeletal mechanics, "
                "or a closely related biomedical topic. Please ask something within that domain "
                "and I'll do my best to answer from the literature.\n\n"
                "Note: BoneMind is a research tool and does not provide personal medical advice. "
                "For clinical decisions — including starting, adjusting, or stopping any medication — "
                "please consult a qualified healthcare professional."
            )
            yield f"data: {json.dumps({'type':'done','answer':out})}\n\n"
            return

        results = retriever.query(q, top_k=top_k)

        # Deduplicate by title so the LLM sees each paper only once
        seen_titles: set[str] = set()
        deduped = []
        for r in results:
            key = (r["title"] or "").strip().lower()
            if key not in seen_titles:
                seen_titles.add(key)
                deduped.append(r)
        # Re-number ranks to stay consecutive after dedup
        for i, r in enumerate(deduped, 1):
            r["rank"] = i
        results = deduped

        # Serialise results (exclude raw embedding bytes)
        results_payload = [
            {k: v for k, v in r.items() if k != "embedding"}
            for r in results
        ]
        yield f"data: {json.dumps({'type':'sources','results':results_payload})}\n\n"

        context  = _build_context(results)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *prior_turns,
            {"role": "user",   "content": f"Context passages:\n\n{context}\n\n---\n\nQuestion: {q}"},
        ]

        # Estimate the full assembled prompt size (Ollama's prompt_eval_count
        # only reports newly-evaluated tokens because it caches the prefix
        # between turns, which makes the badge stay flat across follow-ups).
        # ~3.5 chars/token is a good Llama-3 approximation for English/medical text.
        char_count = sum(len(m.get("content", "")) for m in messages)
        prompt_tokens = max(1, round(char_count / 3.5))
        completion_tokens = 0
        answer = ""
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": True, "options": {"temperature": 0, "num_ctx": 8192}},
                stream=True,
                timeout=180,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if data.get("done"):
                        completion_tokens = data.get("eval_count", 0)
                    else:
                        token = data["message"]["content"]
                        answer += token
                        yield f"data: {json.dumps({'type':'token','content':token})}\n\n"

            linked = _inject_ref_links(answer, results)
            yield f"data: {json.dumps({'type':'done','answer':linked,'prompt_tokens':prompt_tokens,'completion_tokens':completion_tokens,'context_window':8192})}\n\n"

        except requests.ConnectionError:
            yield f"data: {json.dumps({'type':'error','message':'Could not connect to Ollama. Run: ollama serve'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/search")
def search(query: str = Form(...), top_k: int = Form(10), source_filter: str = Form("All"), year_min: int = Form(1970), year_max: int = Form(2026)):
    q = query.strip()
    if not q:
        return {"query": q, "elapsed_ms": 0, "results": []}

    t0 = time.time()
    raw = retriever.query(q, top_k=top_k, year_min=year_min, year_max=year_max)
    elapsed_ms = (time.time() - t0) * 1000

    if source_filter == "Papers only":
        raw = [r for r in raw if r["source_type"] == "paper"]
    elif source_filter == "Textbooks only":
        raw = [r for r in raw if r["source_type"] == "textbook"]

    # Fetch total chunk counts per source so the UI can show "chunk X of Y".
    source_ids = list({r["source_id"] for r in raw})
    total_chunks_map: dict[str, int] = {}
    if source_ids:
        conn = sqlite3.connect(CHUNKS_DB_PATH)
        placeholders = ",".join("?" * len(source_ids))
        rows = conn.execute(
            f"SELECT source_id, COUNT(*) FROM chunks WHERE source_id IN ({placeholders}) GROUP BY source_id",
            source_ids,
        ).fetchall()
        conn.close()
        total_chunks_map = {row[0]: row[1] for row in rows}

    results = []
    for r in raw:
        venue = _dedupe_venue(r["venue"] or "")
        results.append({
            "rank":         r["rank"],
            "score":        round(r["score"], 4),
            "title":        r["title"],
            "authors":      r["authors"] or "",
            "year":         r["year"],
            "venue":        venue,
            "source_type":  r["source_type"],
            "page_number":  r["page_number"],
            "doi":          r["doi"],
            "chunk_index":  r["chunk_index"],
            "total_chunks": total_chunks_map.get(r["source_id"], 0),
            "excerpt":      r["text"].strip().replace("\n", " "),
        })

    return {"query": q, "elapsed_ms": round(elapsed_ms), "results": results}


@app.post("/api/reason")
def reason(
    query: str = Form(...),
    max_results: int = Form(8),
    physics_filter: bool = Form(True),
    keep_falsified: bool = Form(False),
):
    """
    Physics-driven hypothesis generation.

    Pipeline (Items 1, 2, 5):
      1. PhysicsGenerator runs each applicable physical law (currently
         Currey's law) over a small perturbation grid and emits
         candidate hypotheses with quantitative ΔY/Y predictions.
      2. PhysicsCritic runs a four-round adversarial falsification
         pass on each candidate.
      3. Novelty classifier checks corpus presence on survivors.

    The legacy graph-walk path (lrm.query) is no longer reached from
    the UI, but remains in the codebase for the eval suite.
    """
    q = query.strip()
    if not q:
        return {"concept": q, "chains": [], "elapsed": 0}

    t0 = time.perf_counter()

    hypotheses = lrm.query_physics(
        q,
        max_results=max_results,
        novelty_classifier=novelty_clf,
        keep_falsified=keep_falsified,
    )

    NOVELTY_DISPLAY = {
        "GROUNDED":    "Grounded",
        "SPECULATIVE": "Speculative",
        "NOVEL":       "Novel",
        "UNCERTAIN":   "Uncertain",
    }

    chains = []
    for h in hypotheses:
        # Look up labels from the graph for nicer display, falling back
        # to the snake_case → space form embedded on the hypothesis.
        nodes_display = []
        for nid, fallback in zip(h.chain, h.chain_labels):
            node = lrm._graph.get_node(nid)
            nodes_display.append(node.label if node else fallback)

        critique = h.critique
        validity = (
            "PLAUSIBLE" if (critique and critique.survived) else "IMPLAUSIBLE"
        )

        chains.append({
            # Chain — kept compatible with existing UI fields
            "nodes":         nodes_display,
            "relations":     h.relations,

            # Physics-driven additions (Items 1, 2)
            "law":           h.law,
            "law_form":      h.law_form,
            "scenario":      h.scenario,
            "perturbation":  h.perturbation,
            "prediction":    h.prediction,
            "input_var":     h.input_var,
            "output_var":    h.output_var,
            "delta_input":   h.delta_input,
            "delta_output":  h.delta_output,
            "assumed_inputs": h.assumed_inputs,

            # Critic results (Item 5)
            "validity":         validity,
            "critic_rounds_total":  critique.rounds_total if critique else 0,
            "critic_rounds_passed": critique.rounds_passed if critique else 0,
            "critic_failure":   critique.failure_reason if critique else "",
            "critic_checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in (critique.checks if critique else [])
            ],

            # Novelty / corpus grounding
            "novelty":         h.novelty or "UNCERTAIN",
            "novelty_display": NOVELTY_DISPLAY.get(h.novelty, h.novelty or "Uncertain"),
            "explanation":     h.novelty_reason,
            "disclaimer":      bool(h.corpus_disclaimer),
            "corpus_disclaimer": h.corpus_disclaimer,

            # Summary + score
            "summary":     h.summary,
            "score":       round(h.score, 3),
        })

    elapsed = round(time.perf_counter() - t0, 2)
    graph_stats = lrm.graph_stats()
    return {
        "concept":     q,
        "elapsed":     elapsed,
        "graph_stats": {
            "nodes":      graph_stats["n_nodes"],
            "edges":      graph_stats["n_edges"],
            "components": graph_stats["n_components"],
        },
        "chains": chains,
    }


# ── v2 equation-graph reasoner ────────────────────────────────────────────────
# Phase 2: forward / abductive / counterfactual inference over a typed equation
# graph (Currey, Paris–Vashishth, beam bending, Frost mechanostat, plus the
# density / inertia / strain bridges).  The chain of Relations is discovered
# by traversing shared variable symbols, not pre-encoded.

_V2_PRESETS: dict[str, dict] = {
    # ── Forward ────────────────────────────────────────────────────────────
    "fwd_porosity_to_modulus": {
        "mode": "forward",
        "label": "Forward · Porosity → elastic modulus",
        "target": "E",
        "given": {"phi": 0.10},
        "description": (
            "How a 10 % porosity sample maps to elastic modulus via "
            "the density bridge and Currey's law."
        ),
    },
    "fwd_porosity_to_crack_growth": {
        "mode": "forward",
        "label": "Forward · Porosity → crack growth (at ΔK = 1.0 MPa·√m)",
        "target": "da_dN",
        "given": {"phi": 0.10, "dK": 1.0},
        "description": (
            "How a 10 % porosity sample maps to fatigue crack growth via "
            "the density bridge and the Vashishth-modulated Paris law."
        ),
    },
    "fwd_geometry_to_remodeling": {
        "mode": "forward",
        "label": "Forward · Loading + geometry → BMD adaptation rate",
        "target": "dBMD_dt",
        "given": {"phi": 0.10, "R": 13.0, "t": 4.5, "M": 150_000.0},
        "description": (
            "Femoral midshaft under a 150 N·m bending moment: full chain "
            "through inertia → stress → strain → Frost mechanostat."
        ),
    },

    # ── Abductive ──────────────────────────────────────────────────────────
    "abd_low_modulus": {
        "mode": "abductive",
        "label": "Abductive · Patient with low elastic modulus (E = 12 GPa)",
        "target": "E",
        "observed": 12.0,
        "observed_std": 1.0,
        "infer": ["phi"],
        "description": (
            "A cortical sample measures E = 12 ± 1 GPa. Invert Currey + "
            "density bridge to recover the most likely porosity."
        ),
    },
    "abd_high_crack_growth": {
        "mode": "abductive",
        "label": "Abductive · Elevated fatigue crack growth (da/dN = 5e-9 m/cycle)",
        "target": "da_dN",
        "observed": 5.0e-9,
        "observed_std": 1.0e-9,
        "infer": ["phi"],
        "given": {"dK": 1.0},
        "description": (
            "Observed da/dN at ΔK = 1.0 MPa·√m is high. Invert "
            "Paris–Vashishth to recover the porosity that explains it."
        ),
    },
    "abd_moderate_remodeling": {
        "mode": "abductive",
        "label": "Abductive · Mid-zone adaptation response (dBMD/dt = +1.0 %/yr)",
        "target": "dBMD_dt",
        # +1.0 %/yr sits in the Frost transition window, so the chosen
        # cyclic moment is genuinely informative.  Inferring only M (with
        # cortex thickness held at the literature midpoint) keeps the
        # marginal interpretable; saturation values like +1.8 would flatten
        # the posterior.
        "observed": 1.0,
        "observed_std": 0.2,
        "infer": ["M"],
        "given": {"phi": 0.10, "R": 13.0, "t": 4.5},
        "description": (
            "A patient gains 1.0 ± 0.2 %/yr cortical BMD. With porosity, "
            "radius and cortex thickness pinned at literature midpoints, "
            "infer the cyclic bending moment that explains the response."
        ),
    },

    # ── Counterfactual ─────────────────────────────────────────────────────
    "cf_drop_porosity": {
        "mode": "counterfactual",
        "label": "Counterfactual · do(porosity = 0.05) on elastic modulus",
        "target": "E",
        "given": {"phi": 0.30},
        "intervention": {"phi": 0.05},
        "description": (
            "If we could reduce porosity from 30 % to 5 % (sealed cortex), "
            "how much would elastic modulus change?"
        ),
    },
    "cf_thinner_cortex": {
        "mode": "counterfactual",
        "label": "Counterfactual · do(cortical thickness = 2.5 mm) on remodeling",
        "target": "dBMD_dt",
        # Baseline picks a moderate moment so peak strain sits in the
        # Frost transition window — otherwise the system is saturated
        # and the intervention can't show a visible Δ.
        "given": {"phi": 0.10, "R": 13.0, "t": 5.5, "M": 40_000.0},
        "intervention": {"t": 2.5},
        "description": (
            "Thinning the cortex from 5.5 mm to 2.5 mm (osteoporotic "
            "progression) raises bending stress and strain — does the "
            "Frost response intensify?"
        ),
    },
    "cf_double_load": {
        "mode": "counterfactual",
        "label": "Counterfactual · do(moment ×2) on fatigue crack growth",
        "target": "da_dN",
        "given": {"phi": 0.10, "dK": 0.8},
        "intervention": {"dK": 1.6},
        "description": (
            "Doubling the cyclic stress-intensity range under the same "
            "porosity — how much does da/dN climb via the Paris exponent?"
        ),
    },

    # ── Phase 3: covariate-aware demos ────────────────────────────────────
    "cov_old_vertebra_modulus": {
        "mode": "forward",
        "label": "Phase 3 · Same porosity, two patients: E in 75 F vertebra",
        "target": "E",
        "given": {"phi": 0.15},
        "covariates": {
            "age": 75, "sex": "F", "site": "vertebra",
            "disease": ["osteoporosis"],
        },
        "description": (
            "Predict elastic modulus at φ = 0.15 for a 75-year-old female "
            "osteoporotic vertebra. Compare against the same φ on a young "
            "femur — v2 gives ~3× different answer, v0 cannot."
        ),
    },
    "cov_abductive_clinical_flip": {
        "mode": "abductive",
        "label": "Phase 3 · E = 12 GPa observation, infer φ in 75 F vertebra",
        "target": "E",
        "observed": 12.0,
        "observed_std": 1.0,
        "infer": ["phi"],
        "covariates": {
            "age": 75, "sex": "F", "site": "vertebra",
            "disease": ["osteoporosis"],
        },
        "description": (
            "Same observed E = 12 GPa, but in an elderly vertebra the "
            "literature-shifted Currey constants imply φ is near-zero — "
            "an entirely different clinical reading than the young-femur case."
        ),
    },
    "cov_glucocorticoid_remodeling": {
        "mode": "counterfactual",
        "label": "Phase 3 · do(thickness ↓) on remodeling — with vs without steroids",
        "target": "dBMD_dt",
        "given": {"phi": 0.10, "R": 13.0, "t": 5.5, "M": 40_000.0},
        "intervention": {"t": 2.5},
        "covariates": {
            "age": 65, "sex": "F",
            "disease": ["glucocorticoid"],
        },
        "description": (
            "Cortical thinning normally drives a strong anabolic response "
            "(Frost). On chronic glucocorticoids the anabolic rate is cut "
            "to ≈20 %, so the same intervention produces a much weaker Δ."
        ),
    },
}


# ── Result-shape helpers ─────────────────────────────────────────────────────


def _v2_var(reg, symbol: str) -> dict:
    v = reg.variable(symbol)
    if v is None:
        return {"symbol": symbol, "name": symbol, "unit": ""}
    return {
        "symbol":      v.symbol,
        "name":        v.name,
        "unit":        v.unit,
        "lo":          v.lo,
        "hi":          v.hi,
        "description": v.description,
    }


def _v2_render_forward(
    reg, target: str, given: dict[str, float],
    covariates: dict | None = None,
) -> dict:
    fr = reg.forward(
        target, given=given, n_samples=4000, seed=12345, covariates=covariates,
    )
    rel_by_name = {r.name: r for r in reg.relations()}
    steps = []
    for s in fr.steps:
        rel = rel_by_name.get(s.relation_name)
        steps.append({
            "relation_name": s.relation_name,
            "description":   rel.description if rel else "",
            "latex":         rel.latex if rel else "",
            "citation":      s.citation,
            "inputs":        s.inputs,
            "output_var":    s.output_var,
            "output_unit":   (reg.variable(s.output_var).unit
                              if reg.variable(s.output_var) else ""),
            "output_mean":   s.output_mean,
            "output_p5":     s.output_p5,
            "output_p95":    s.output_p95,
        })
    target_var = reg.variable(target)
    chain = reg._find_chain(target, set(given.keys())) or []
    return {
        "mode":          "forward",
        "target":        target,
        "target_info":   _v2_var(reg, target),
        "given":         given,
        "given_info":    {k: _v2_var(reg, k) for k in given},
        "covariates":    covariates or {},
        "applied_shifts": reg.applied_shifts(chain, covariates),
        "chain_vars":    [_v2_var(reg, v) for v in fr.chain_vars],
        "steps":         steps,
        "result": {
            "variable":             target,
            "unit":                 target_var.unit if target_var else "",
            "mean":                 fr.mean,
            "median":               fr.median,
            "p5":                   fr.p5,
            "p95":                  fr.p95,
            "relative_uncertainty": fr.relative_uncertainty,
            "n_samples":            int(fr.samples.size),
        },
        "citations":     fr.citations,
    }


def _v2_render_abductive(
    reg,
    target: str,
    observed: float,
    observed_std: float | None,
    infer: list[str] | None,
    given: dict[str, float] | None,
    covariates: dict | None = None,
) -> dict:
    ar = reg.abductive(
        target,
        observed=observed,
        observed_std=observed_std,
        infer=infer,
        given=given,
        n_samples=8000,
        seed=12345,
        covariates=covariates,
    )
    inferred = [
        {
            "variable":       _v2_var(reg, p.variable),
            "prior_mean":     p.prior_mean,
            "prior_p5":       p.prior_p5,
            "prior_p95":      p.prior_p95,
            "posterior_mean": p.posterior_mean,
            "posterior_p5":   p.posterior_p5,
            "posterior_p95":  p.posterior_p95,
            "shift_score":    p.shift_score,
        }
        for p in ar.inferred
    ]
    # Re-resolve chain so we can attach the applied-shifts trace.
    chain_given = {**(given or {}), **{v: 0.0 for v in (infer or [])}}
    chain = reg._find_chain(target, set(chain_given.keys())) or []
    return {
        "mode":         "abductive",
        "target":       target,
        "target_info":  _v2_var(reg, target),
        "observed":     ar.observed,
        "observed_std": ar.observed_std,
        "given":        given or {},
        "given_info":   {k: _v2_var(reg, k) for k in (given or {})},
        "covariates":   covariates or {},
        "applied_shifts": reg.applied_shifts(chain, covariates),
        "chain_vars":   [_v2_var(reg, v) for v in ar.chain_vars],
        "inferred":     inferred,
        "result": {
            "effective_sample_size": ar.effective_sample_size,
            "n_samples":             ar.n_samples,
        },
        "citations":    ar.citations,
    }


def _v2_render_counterfactual(
    reg,
    target: str,
    given: dict[str, float],
    intervention: dict[str, float],
    covariates: dict | None = None,
) -> dict:
    cf = reg.counterfactual(
        target, given=given, intervention=intervention,
        n_samples=4000, seed=12345, covariates=covariates,
    )
    target_var = reg.variable(target)
    chain = reg._find_chain(target, set(given.keys())) or []
    return {
        "mode":             "counterfactual",
        "target":           target,
        "target_info":      _v2_var(reg, target),
        "given":            given,
        "given_info":       {k: _v2_var(reg, k) for k in given},
        "intervention":     intervention,
        "intervention_info": {k: _v2_var(reg, k) for k in intervention},
        "covariates":       covariates or {},
        "applied_shifts":   reg.applied_shifts(chain, covariates),
        "chain_vars":       [_v2_var(reg, v) for v in cf.chain_vars],
        "result": {
            "variable":        target,
            "unit":            target_var.unit if target_var else "",
            "baseline_mean":   cf.baseline_mean,
            "baseline_p5":     cf.baseline_p5,
            "baseline_p95":    cf.baseline_p95,
            "intervened_mean": cf.intervened_mean,
            "intervened_p5":   cf.intervened_p5,
            "intervened_p95":  cf.intervened_p95,
            "delta_mean":      cf.delta_mean,
            "delta_p5":        cf.delta_p5,
            "delta_p95":       cf.delta_p95,
            "relative_delta":  cf.relative_delta,
            "n_samples":       cf.n_samples,
        },
        "citations":        cf.citations,
    }


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/api/reason_v2/presets")
def reason_v2_presets():
    """Return the preset demo queries available to the v2 UI."""
    return {
        "presets": [
            {"id": pid, **p}
            for pid, p in _V2_PRESETS.items()
        ],
        "covariates": COVARIATE_SCHEMA,
        "variables": [
            {
                "symbol":      v.symbol,
                "name":        v.name,
                "unit":        v.unit,
                "lo":          v.lo,
                "hi":          v.hi,
                "description": v.description,
            }
            for v in (
                bone_registry.variable(s)
                for s in sorted(bone_registry.variables_in_graph())
            ) if v is not None
        ],
        "relations": [
            {
                "name":        r.name,
                "description": r.description,
                "latex":       r.latex,
                "citation":    r.citation,
                "inputs":      list(r.inputs),
                "output":      r.output,
            }
            for r in bone_registry.relations()
        ],
    }


def _coerce_float_dict(d: dict, field_name: str) -> dict[str, float]:
    if not isinstance(d, dict):
        raise ValueError(f"Field '{field_name}' must be an object.")
    try:
        return {str(k): float(v) for k, v in d.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Bad '{field_name}' values: {exc}") from exc


def _normalise_covariates(raw) -> dict | None:
    """
    Accept the UI's loose covariate dict (age, sex, site, disease list)
    and return a clean version, dropping unknown keys and coercing types.
    Returns ``None`` if nothing meaningful was supplied so the registry
    can short-circuit to the no-shift fast path.
    """
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError("Field 'covariates' must be an object.")
    out: dict = {}
    if "age" in raw and raw["age"] is not None:
        try:
            out["age"] = float(raw["age"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Bad 'covariates.age': {exc}") from exc
    if raw.get("sex") in {"M", "F"}:
        out["sex"] = raw["sex"]
    if raw.get("site"):
        out["site"] = str(raw["site"])
    diseases = raw.get("disease")
    if diseases:
        if isinstance(diseases, str):
            diseases = [diseases]
        out["disease"] = [str(d) for d in diseases]
    return out or None


@app.post("/api/reason_v2")
def reason_v2(payload: dict):
    """
    Forward / abductive / counterfactual inference over the equation graph.

    Common fields::

        {"preset": "<preset_id>"}                  # any mode, fully specified
        {"mode": "forward",        "target": "E", "given": {"phi": 0.10}}
        {"mode": "abductive",      "target": "E", "observed": 12.0,
                                   "observed_std": 1.0, "infer": ["phi"]}
        {"mode": "counterfactual", "target": "E", "given": {"phi": 0.30},
                                   "intervention": {"phi": 0.05}}

    The response shape varies with ``mode`` — see the renderers above.
    """
    # Preset shortcut: copy fields into payload and continue down the
    # normal dispatch path.
    preset_id = payload.get("preset")
    if preset_id is not None:
        preset = _V2_PRESETS.get(preset_id)
        if preset is None:
            return {"error": f"Unknown preset: {preset_id!r}"}
        payload = {**preset, **{k: v for k, v in payload.items() if k != "preset"}}

    mode = (payload.get("mode") or "forward").lower()
    target = payload.get("target")
    if not target:
        return {"error": "Field 'target' is required."}

    t0 = time.perf_counter()
    try:
        covariates = _normalise_covariates(payload.get("covariates"))
        if mode == "forward":
            given = _coerce_float_dict(payload.get("given") or {}, "given")
            out = _v2_render_forward(
                bone_registry, target, given, covariates=covariates,
            )
        elif mode == "abductive":
            observed = payload.get("observed")
            if observed is None:
                return {"error": "Field 'observed' is required for abductive mode."}
            observed = float(observed)
            observed_std = payload.get("observed_std")
            if observed_std is not None:
                observed_std = float(observed_std)
            infer = payload.get("infer")
            if infer is not None and not isinstance(infer, list):
                return {"error": "Field 'infer' must be a list of variable names."}
            given = _coerce_float_dict(payload.get("given") or {}, "given")
            out = _v2_render_abductive(
                bone_registry, target,
                observed=observed,
                observed_std=observed_std,
                infer=[str(v) for v in infer] if infer else None,
                given=given,
                covariates=covariates,
            )
        elif mode == "counterfactual":
            given = _coerce_float_dict(payload.get("given") or {}, "given")
            intervention = _coerce_float_dict(
                payload.get("intervention") or {}, "intervention",
            )
            if not intervention:
                return {"error": "Field 'intervention' is required for counterfactual mode."}
            out = _v2_render_counterfactual(
                bone_registry, target, given, intervention,
                covariates=covariates,
            )
        else:
            return {"error": f"Unknown mode: {mode!r}. Use forward, abductive, or counterfactual."}
    except ValueError as exc:
        return {"error": str(exc)}
    out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


# ── Phase 4: free-text routing → semantic anchor → deterministic reasoner ────


# Literature midpoints used when the router/anchor doesn't supply a value.
# Chain roots only — derived variables are never given directly.
_V2_DEFAULT_VALUES: dict[str, float] = {
    "phi": 0.10,
    "dK":  1.0,
    "R":   13.0,
    "t":   4.5,
    "M":   40_000.0,
}

# Minimum cosine similarity for a SPECTER2 match to be trusted as an
# anchor.  Below this we fall back to the top-1 result regardless —
# the threshold guards against the router emitting an empty hint.
_V2_ANCHOR_MIN_SCORE = 0.30


def _v2_anchor_symbol(
    text: str | None, fallback_symbol: str | None = None,
) -> tuple[str | None, float]:
    """Map a free-text hint to a variable symbol via SPECTER2 cosine."""
    if not text:
        return fallback_symbol, 0.0
    match = semantic_anchor.best_match(text, min_score=_V2_ANCHOR_MIN_SCORE)
    if match is None:
        return fallback_symbol, 0.0
    return match.symbol, match.score


def _v2_pick_target(
    target_hint: str | None,
    candidate_matches: list,
) -> tuple[str, float]:
    """
    Choose a forward-inference target that the registry can actually solve for.

    The semantic anchor sometimes ranks a chain root highest (e.g.
    ``phi`` for "how does porosity affect modulus?").  Chain roots can
    never be a forward target because no Relation produces them, so we
    walk down the top-k list until we hit a variable that has at least
    one producer.  Falls back to ``"E"`` if nothing qualifies.
    """
    producers = bone_registry._producers
    # First try the explicit hint.
    if target_hint:
        candidates = semantic_anchor.top_k(target_hint, k=5)
        for m in candidates:
            if m.symbol in producers and m.score >= _V2_ANCHOR_MIN_SCORE:
                return m.symbol, m.score
    # Then fall back to the query-wide top-k that the router already attached.
    for m in candidate_matches or []:
        if m.symbol in producers:
            return m.symbol, m.score
    return "E", 0.0


def _v2_chain_roots(target: str) -> list[str]:
    """Chain roots required to reach ``target``; literature defaults fill them."""
    given_set = set(_V2_DEFAULT_VALUES.keys())
    chain = bone_registry._find_chain(target, given_set)
    if chain is None:
        return []
    return bone_registry.chain_root_inputs(chain)


def _v2_build_ask_payload(
    decision, query: str,
) -> tuple[dict, dict]:
    """
    Translate a :class:`RouterDecision` into a /api/reason_v2 payload.

    Returns ``(payload, routing_block)``.  ``payload`` is suitable for
    direct dispatch through :func:`reason_v2`; ``routing_block`` is
    surfaced in the response so the UI can show *why* a particular
    mode + target was chosen.
    """
    mode = decision.mode if decision.mode in {"forward", "abductive", "counterfactual"} \
        else "forward"

    # 1) Target: trust the router's text hint, fall back to the top
    #    semantic match against the raw query.  Constrained to variables
    #    that have a producer so the chain finder can actually solve it.
    target, target_score = _v2_pick_target(
        decision.target_hint, decision.matched_variables,
    )

    payload: dict = {"mode": mode, "target": target}

    # 2) Mode-specific extraction.
    given_overrides: dict[str, float] = {}
    for phrase, value in (decision.given_hints or {}).items():
        sym, _ = _v2_anchor_symbol(phrase)
        if sym in _V2_DEFAULT_VALUES:
            given_overrides[sym] = float(value)

    if mode == "forward":
        roots = _v2_chain_roots(target)
        given = {r: _V2_DEFAULT_VALUES.get(r, 0.0) for r in roots}
        given.update(given_overrides)
        payload["given"] = given

    elif mode == "abductive":
        if decision.observed_value is None:
            # Without a number we cannot do abduction; fall back to forward.
            roots = _v2_chain_roots(target)
            given = {r: _V2_DEFAULT_VALUES.get(r, 0.0) for r in roots}
            given.update(given_overrides)
            payload = {"mode": "forward", "target": target, "given": given}
        else:
            payload["observed"] = float(decision.observed_value)
            if decision.observed_units:
                payload["observed_units"] = decision.observed_units
            # Infer porosity by default — the canonical clinical upstream.
            # Pin every other chain root at literature midpoints.
            roots = _v2_chain_roots(target)
            infer = ["phi"] if "phi" in roots else (roots[:1] if roots else [])
            payload["infer"] = infer
            given = {
                r: _V2_DEFAULT_VALUES.get(r, 0.0)
                for r in roots if r not in infer
            }
            given.update({k: v for k, v in given_overrides.items() if k not in infer})
            payload["given"] = given

    elif mode == "counterfactual":
        interv_sym, _ = _v2_anchor_symbol(decision.intervention_hint)
        if interv_sym is None or decision.intervention_value is None or \
                interv_sym not in _V2_DEFAULT_VALUES:
            # Missing intervention — degrade to forward.
            roots = _v2_chain_roots(target)
            given = {r: _V2_DEFAULT_VALUES.get(r, 0.0) for r in roots}
            given.update(given_overrides)
            payload = {"mode": "forward", "target": target, "given": given}
        else:
            roots = _v2_chain_roots(target)
            given = {r: _V2_DEFAULT_VALUES.get(r, 0.0) for r in roots}
            given.update(given_overrides)
            # Ensure the intervention key has a meaningful baseline.
            if interv_sym not in given:
                given[interv_sym] = _V2_DEFAULT_VALUES.get(interv_sym, 0.0)
            payload["given"] = given
            payload["intervention"] = {interv_sym: float(decision.intervention_value)}

    routing_block = {
        "query":      query,
        "mode":       payload["mode"],
        "target":     target,
        "target_info": _v2_var(bone_registry, target),
        "target_score": round(target_score, 3),
        "rationale":  decision.rationale,
        "confidence": round(decision.confidence, 3),
        "source":     decision.source,
        "matched_variables": [
            {
                "symbol":      m.symbol,
                "name":        m.name,
                "unit":        m.unit,
                "score":       round(m.score, 3),
            }
            for m in decision.matched_variables
        ],
    }
    return payload, routing_block


@app.post("/api/reason_v2/ask")
def reason_v2_ask(payload: dict):
    """
    Free-text entry point into the v2 reasoner.

    Request shape::

        {"query": "bone stiffness under cyclic load",
         "covariates": {"age": 60, "sex": "F", "site": "femur_cortical", "disease": []}}

    Pipeline:
      1. One Ollama call classifies mode and extracts numeric anchors.
      2. SPECTER2 cosine similarity maps text hints (and the raw query)
         to Variable symbols.
      3. Any chain roots not supplied by the user are filled from
         literature midpoints so the deterministic reasoner has a
         complete payload.
      4. The existing forward / abductive / counterfactual renderer
         runs unchanged.

    The response is the standard v2 result plus a ``routing`` block
    describing which mode / target the router chose, the matched
    variables (with cosine scores) and the rationale string.
    """
    query = (payload.get("query") or "").strip()
    if not query:
        return {"error": "Field 'query' is required."}

    t0 = time.perf_counter()
    try:
        decision = query_router.route(query, anchor=semantic_anchor)
    except Exception as exc:
        return {"error": f"Query router failed: {exc}"}

    inner_payload, routing_block = _v2_build_ask_payload(decision, query)
    # Forward covariates through verbatim — the router does not touch them.
    if payload.get("covariates"):
        inner_payload["covariates"] = payload["covariates"]

    result = reason_v2(inner_payload)
    if "error" in result:
        result["routing"] = routing_block
        return result
    result["routing"] = routing_block
    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


@app.post("/api/analyse")
async def analyse(image: UploadFile = File(...)):
    from PIL import Image as PILImage
    import numpy as np

    contents = await image.read()
    try:
        pil_img = PILImage.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return {"error": "Could not read image file."}

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": VLM_MODEL,
                "messages": [{"role": "user", "content": VLM_PROMPT, "images": [img_b64]}],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw_text = resp.json()["message"]["content"]
    except requests.ConnectionError:
        return {"error": "Could not connect to Ollama. Run: ollama serve && ollama pull llava:13b"}
    except Exception as e:
        return {"error": str(e)}

    # Parse JSON from the VLM response
    import re
    clean = re.sub(r"```json|```", "", raw_text).strip()
    # Extract first JSON object
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Fallback: return raw text as assessment
    return {
        "modality": "Bone scan",
        "quality": "N/A",
        "density": "N/A",
        "findings": [],
        "assessment": raw_text[:800],
        "recommendation": "Manual review recommended.",
        "confidence": "LOW",
    }
