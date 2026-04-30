"""
api/main.py
===========
FastAPI backend for the BoneLogic React frontend.

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
GET  /api/gaps             — research gap detection, JSON response
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

from retrieval.retriever import BoneLogicRetriever
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

SYSTEM_PROMPT = """You are BoneLogic, an expert AI assistant specialised in bone science.
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
   "This question is outside BoneLogic's domain. I cover bone science only."

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
print("Loading BoneLogic retriever...")
retriever = BoneLogicRetriever()
retriever.load()

print("Loading LRM (bone knowledge graph)...")
lrm = LRM()

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
app = FastAPI(title="BoneLogic API")

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
    for n in seen:
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
async def ask(question: str = Form(...), top_k: int = Form(8), history: str = Form("[]")):
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

        prior_user_questions = [
            t["content"] for t in prior_turns
            if t.get("role") == "user" and isinstance(t.get("content"), str)
        ]

        # Topic guard: dedicated small classifier model decides if question is on-topic.
        # Runs before retrieval so off-topic questions cost nothing beyond the guard call.
        if not _is_bone_science(q, prior_user_questions):
            out = (
                "I'm BoneLogic, a specialist assistant for bone science. "
                "Your question doesn't appear to be related to bone biology, skeletal mechanics, "
                "or a closely related biomedical topic. Please ask something within that domain "
                "and I'll do my best to answer from the literature."
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
def search(query: str = Form(...), top_k: int = Form(10), source_filter: str = Form("All")):
    q = query.strip()
    if not q:
        return {"query": q, "elapsed_ms": 0, "results": []}

    t0 = time.time()
    raw = retriever.query(q, top_k=top_k)
    elapsed_ms = (time.time() - t0) * 1000

    if source_filter == "Papers only":
        raw = [r for r in raw if r["source_type"] == "paper"]
    elif source_filter == "Textbooks only":
        raw = [r for r in raw if r["source_type"] == "textbook"]

    results = []
    for r in raw:
        venue = _dedupe_venue(r["venue"] or "")
        results.append({
            "rank":        r["rank"],
            "score":       round(r["score"], 4),
            "title":       r["title"],
            "authors":     r["authors"] or "",
            "year":        r["year"],
            "venue":       venue,
            "source_type": r["source_type"],
            "page_number": r["page_number"],
            "doi":         r["doi"],
            "excerpt":     r["text"].strip().replace("\n", " ")[:400],
        })

    return {"query": q, "elapsed_ms": round(elapsed_ms), "results": results}


@app.post("/api/reason")
def reason(query: str = Form(...), max_results: int = Form(8), physics_filter: bool = Form(True)):
    q = query.strip()
    if not q:
        return {"concept": q, "chains": [], "elapsed": 0}

    t0 = time.perf_counter()
    lrm.physics_filter = physics_filter
    hypotheses = lrm.query(q, max_results=max_results)

    chains = []
    for h in hypotheses:
        nr = novelty_clf.classify(h)

        # Map physics result to validity label
        validity = "IMPLAUSIBLE" if h.physics.is_implausible else "PLAUSIBLE"

        nodes = []
        for nid in h.chain:
            node = lrm._graph.get_node(nid)
            nodes.append(node.label if node else nid.replace("_", " "))

        relations = [e.relation for e in h.edges]

        novelty_display = {
            "GROUNDED":    "Grounded",
            "SPECULATIVE": "Speculative",
            "NOVEL":       "Novel",
            "UNCERTAIN":   "Uncertain",
        }.get(nr.label, nr.label)

        chains.append({
            "nodes":       nodes,
            "relations":   relations,
            "validity":    validity,
            "novelty":     nr.label,
            "novelty_display": novelty_display,
            "score":       round(h.score, 3),
            "summary":     h.summary,
            "explanation": nr.explanation,
            "disclaimer":  nr.show_disclaimer,
            "corpus_disclaimer": CORPUS_DISCLAIMER if nr.show_disclaimer else None,
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


@app.get("/api/gaps")
def gaps(top_n: int = Query(10)):
    try:
        gap_list = lrm.find_gaps(top_n=top_n)
    except Exception as e:
        return {"error": str(e), "gaps": []}

    return {
        "gaps": [
            {
                "rank":         i + 1,
                "label":        g.label,
                "node_type":    g.node_type,
                "betweenness":  round(g.betweenness, 5),
                "n_edges":      g.n_edges,
                "gap_score":    round(g.gap_score, 5),
            }
            for i, g in enumerate(gap_list)
        ]
    }


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
