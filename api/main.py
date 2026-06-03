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
POST /api/reason           — equation-graph reasoner (forward / abductive /
                              counterfactual) over a typed variable graph.
POST /api/reason/ask       — free-text entry: single LLM call routes the
                              query, SPECTER2 anchors the variables, the
                              deterministic reasoner runs.
GET  /api/reason/presets   — preset demo queries + variable/relation metadata
GET  /api/reason/explore   — active exploration: walk the variable graph
                              without a user query, score candidates against
                              the corpus, rank by surprise.
POST /api/reason/agents    — Proposer/Critic agent loop: Proposer suggests a
                              hypothesis, the Explorer evaluates it
                              deterministically, the Critic decides whether
                              it is interesting.
POST /api/analyse          — VLM image analysis (multipart), JSON response
GET  /                     — serves frontend/index.html
"""

import base64
import io
import json
import math
import re
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
from reasoning.explorer import Explorer, ExplorationCandidate
from reasoning.semantic_anchor import SemanticVariableAnchor
from reasoning.query_router import QueryRouter
from reasoning.graph_db import OntologyStore
from reasoning.novelty import NoveltyClassifier, CORPUS_DISCLAIMER
from reasoning.agent_tools import ToolDispatcher
from reasoning.proposer_agent import ProposerAgent
from reasoning.critic_agent import CriticAgent
from config.settings import PAPERS_DB_PATH, TEXTBOOKS_DB_PATH, CHUNKS_DB_PATH

# ── Ollama config ──────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "huatuogpt-bone"
GUARD_MODEL  = "llama3.2:3b"  # general-purpose classifier for topic guard


# ── Shared topic guard (Ask + Reasoning tabs) ────────────────────────────────
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


_BONE_VOCAB = {
    # general
    "bone", "bones", "skeleton", "skeletal", "osseous",
    # tissue types
    "cortical", "trabecular", "cancellous", "compact bone", "spongy bone",
    "lamellar", "woven",
    # cells / biology
    "osteoblast", "osteoclast", "osteocyte", "osteoid", "osteogen",
    # diseases
    "osteoporosis", "osteopenia", "osteomalacia", "osteoarthritis",
    "osteosarcoma", "paget", "fragility", "fracture",
    # mechanics & physics
    "wolff", "mechanostat", "remodel", "bmd", "bv/tv", "porosity",
    "modulus of bone", "bone modulus", "bone stiffness", "bone density",
    "bone strength", "bone toughness",
    # imaging & diagnostics
    "dxa", "dexa", "t-score", "z-score", "qct", "micro-ct", "ucbt",
    # anatomy
    "femur", "femoral", "tibia", "vertebr", "humer", "radius (bone)",
    "metaphys", "diaphys", "epiphys", "calcaneus",
    # therapy / drugs commonly associated
    "bisphosphonate", "denosumab", "teriparatide", "alendronate",
    # tumours
    "lytic", "metastatic bone", "bone metastas",
    # field
    "orthopaed", "orthoped", "biomaterial", "biomechan",
}


def _looks_bone_related(text: str) -> bool:
    low = (text or "").lower()
    return any(term in low for term in _BONE_VOCAB)


def _is_bone_science(q: str, prior_questions: list[str]) -> bool:
    """Lightweight bone-science classifier used by Ask and Reasoning tabs.

    Two-stage:
      1. Fast lexical pass — if the question (or a recent question, for pronoun
         carry-over) contains explicit bone vocabulary, accept immediately. This
         is the common path and avoids the small classifier's misfires.
      2. LLM fallback — for ambiguous wording, ask llama3.2:3b. Fails open on
         network errors so a hiccup doesn't block real questions.
    """
    if _looks_bone_related(q):
        return True
    # Pronoun-carry: if the user just said "and trabecular?" after a bone question
    for prev in prior_questions[-3:]:
        if _looks_bone_related(prev):
            return True

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
        return True
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

print("Loading bone knowledge graph (for /api/stats)...")
with OntologyStore() as _store:
    _graph_stats = _store.stats()

print("Building equation-graph registry...")
bone_registry = build_bone_registry()

print("Wiring semantic anchor (SPECTER2)...")
semantic_anchor = SemanticVariableAnchor(bone_registry, retriever=retriever)

print("Wiring query router (Ollama)...")
query_router = QueryRouter(ollama_url=OLLAMA_URL, model=GUARD_MODEL)

print("Loading novelty classifier...")
novelty_clf = NoveltyClassifier(
    use_semantic=True,
    model=retriever._model,
    tokenizer=retriever._tokenizer,
    device=retriever._device,
)

print("Wiring explorer...")
explorer = Explorer(bone_registry, novelty_classifier=novelty_clf)
# Lazily computed on first request and cached in process memory.
_exploration_cache: list[ExplorationCandidate] | None = None

print("Wiring proposer + critic agents...")
_agent_dispatcher = ToolDispatcher(registry=bone_registry, retriever=retriever)
_proposer = ProposerAgent(
    registry=bone_registry,
    ollama_url=OLLAMA_URL,
    model=OLLAMA_MODEL,
)
_critic = CriticAgent(
    dispatcher=_agent_dispatcher,
    ollama_url=OLLAMA_URL,
    model=OLLAMA_MODEL,
)
# In-memory scratchpad: append-only list of evaluated hypothesis dicts.
# Cleared on server restart.  Agents read this to avoid repeating sweeps.
# Capped so a long-running server can't grow unbounded — the proposer
# prompt only reads the last 8 entries anyway.
_agent_scratchpad: list[dict] = []
_AGENT_SCRATCHPAD_MAX = 64

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
    return {
        "papers_total":    STATS["papers_total"],
        "pdfs_downloaded": STATS["pdfs_downloaded"],
        "textbooks":       STATS["textbooks"],
        "chunks":          STATS["chunks"],
        "graph_nodes":     _graph_stats["total_nodes"],
        "graph_edges":     _graph_stats["total_edges"],
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


# ── Reasoning tab — reasoning agent (Phase 2) ─────────────────────────────────
#
# Pure LLM, no retrieval. Produces a numbered chain of reasoning steps about
# bone mechanics / fracture / fragility. Later phases add (3) physical-grounding
# filter, (4) user-feedback memory, (5) critic agent.

REASONING_SYSTEM_PROMPT = """You are BoneMind's reasoning agent. Given a question about bone
mechanics, fracture, fragility, remodelling, or related pathology, work through it as a chain
of reasoning points.

Output format (markdown):

**Point 1.** One-sentence claim or mechanism.
*Basis.* 1–2 sentences explaining the biological or mechanical grounding.

Use as few points as the question actually needs — anywhere from 1 to 4.
- A direct factual question often needs only one point.
- A "why" question with a single causal chain may need 2.
- Only use 3–4 when the answer genuinely depends on multiple mechanisms that build on each other.
Do not pad. If one point is enough, stop there. Never repeat the same idea across points.

Rules:
- Reason about mechanisms (cause → effect), not just list facts.
- Stay within bone science (mechanics, biology, pathology, imaging, biomaterials). If the question
  is out of scope, reply with exactly: "Out of scope for BoneMind's reasoning tab."
- Be honest about uncertainty. If a point is speculative, say so in the *Basis* line.
- No preamble, no closing summary. Start directly with "**Point 1.**"."""


CRITIC_SYSTEM_PROMPT = """You are BoneMind's critic agent. You review another agent's bone-science reasoning for correctness, internal consistency, completeness, and appropriately calibrated uncertainty.

You will receive:
  - the original question
  - the reasoning agent's answer (in Point / Basis format)
  - a list of physical-grounding rule violations (may be empty)
  - a short summary of the user's own learned rules (may be empty)
  - retrieved literature passages from the bone-science corpus (may be empty), each tagged [L1], [L2], …
  - knowledge-graph facts: individual curated relationships (may be empty), each one independent

Reply with ONE JSON object on a single line. No prose. No markdown fences. Shape:

{"verdict":"accept" | "dispute" | "conflicting_evidence",
 "notes":["short bullet 1","short bullet 2"],
 "suggested_revision":"<short instruction to the reasoning agent>",
 "majority":"<position supported by most evidence, citing [L#]>",
 "minority":"<the competing position, citing [L#]>"}

Rules:
- VERDICT must be "accept", "dispute", or "conflicting_evidence".
- GROUND YOUR REVIEW IN THE LITERATURE. When a passage supports or contradicts the answer, cite it by tag in your notes, e.g. "Contradicts [L2]: cortical modulus is ~18 GPa".
- Do NOT treat the literature as infallible: passages may be off-topic or only tangentially related. Use judgement; if no passage is relevant, review on correctness alone and say so.
- KNOWLEDGE-GRAPH FACTS are individual curated relationships. Each line is ONE fact. Use them to check the answer's direction/sign (e.g. "porosity [decreases] mechanical strength"). Treat each edge independently — do NOT chain several edges into a multi-step causal argument, because chaining curated edges can imply relationships the graph never asserted.
- ACCEPT if the answer is broadly correct and well-reasoned. A physical-grounding violation flagged on an edge case the agent already qualified is still acceptable — say so in notes.
- DISPUTE only if there is a factual error, internal contradiction, missing key mechanism, a genuine physical-grounding violation that the agent did not address, or a claim the retrieved literature clearly contradicts.
- CONFLICTING_EVIDENCE only when the retrieved literature genuinely disagrees — one body of passages supports the answer and another contradicts it, and the question has no single settled answer. You MUST fill both "majority" and "minority", and EACH must cite at least one [L#] tag. If you cannot cite both sides from the passages provided, do NOT use this verdict — use accept or dispute instead. Do not invent a conflict to hedge.
- NOTES: 1–3 short bullets, each ≤ 15 words. Be specific. Cite [L#] where relevant. No generic praise.
- SUGGESTED_REVISION: present for "dispute" (what to fix) and "conflicting_evidence" (how to present both positions). Omit for "accept". Do NOT rewrite the answer yourself.
- "majority" / "minority": fill ONLY for "conflicting_evidence"; leave empty otherwise.
- Do not dispute or flag conflict just to look thorough. If the answer is good, accept it.
"""


REVISION_PROMPT_TEMPLATE = """Your earlier answer was reviewed by a critic. Critic verdict: DISPUTE.

Critic notes:
{notes}

Suggested revision: {suggestion}

Please revise your answer in the same Point / Basis format. Address the critic's points concretely. If you believe the critic is wrong about a specific point, you may keep your original claim — but explain why in that Point's *Basis* line.

Original question: {question}
"""


CONFLICT_REVISION_PROMPT_TEMPLATE = """Your earlier answer was reviewed by a critic. Critic verdict: CONFLICTING EVIDENCE — the literature genuinely disagrees on this question.

Majority position (bulk of evidence): {majority}
Minority position (competing evidence): {minority}
Critic notes:
{notes}

Please revise your answer in the same Point / Basis format. Do NOT pick one side as definitive. Instead:
- Lead with the majority position as the most likely answer given the bulk of literature.
- Then explicitly note the minority position as a competing view that some evidence supports.
- Make clear the question is not fully settled.

Original question: {question}
"""


CRITIC_MAX_ROUNDS = 2  # max revision iterations (= up to 4 LLM calls total)


def _llm_stream(messages, result_out: dict, temperature=0.3):
    """Generator yielding token strings as Ollama emits them. After the
    iterator is exhausted, result_out['full'] and result_out['eval_count']
    are populated."""
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": True,
              "options": {"temperature": temperature, "num_ctx": 8192}},
        stream=True,
        timeout=240,
    )
    resp.raise_for_status()
    chunks: list[str] = []
    evc = 0
    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if data.get("done"):
            evc = data.get("eval_count", 0)
            continue
        t = data["message"]["content"]
        chunks.append(t)
        yield t
    result_out["full"] = "".join(chunks)
    result_out["eval_count"] = evc


# A1 — literature evidence for the critic.
# The agent stays pure-LLM; only the critic sees retrieved passages so it can
# dispute with citations. Budget is capped so the critic prompt stays bounded
# (~1200 tokens ≈ ~4200 chars across all passages).
LIT_CHAR_BUDGET = 4200
LIT_TOP_K = 3


def _fetch_literature(question: str) -> list[dict]:
    """Top-K corpus passages for the critic. Deduped by title, snippet
    trimmed so the whole block stays within LIT_CHAR_BUDGET. Returns
    [{rank, title, year, snippet, score}]. Never raises."""
    try:
        results = retriever.query(question, top_k=LIT_TOP_K * 2)
    except Exception:
        return []
    seen: set[str] = set()
    out: list[dict] = []
    budget = LIT_CHAR_BUDGET
    per = max(400, LIT_CHAR_BUDGET // max(1, LIT_TOP_K))
    for r in results:
        title = (r.get("title") or "").strip()
        key = title.lower()
        if key and key in seen:
            continue
        seen.add(key)
        snippet = (r.get("text") or "").strip().replace("\n", " ")
        snippet = snippet[:per]
        if budget - len(snippet) < 0:
            break
        budget -= len(snippet)
        out.append({
            "rank": len(out) + 1,
            "title": title or "(untitled)",
            "year": r.get("year"),
            "snippet": snippet,
            "score": r.get("score"),
        })
        if len(out) >= LIT_TOP_K:
            break
    return out


def _format_literature(passages: list[dict]) -> str:
    if not passages:
        return "(no literature retrieved)"
    lines = []
    for p in passages:
        yr = f" ({p['year']})" if p.get("year") else ""
        lines.append(f"[L{p['rank']}] {p['title']}{yr}\n    {p['snippet']}")
    return "\n".join(lines)


def _call_critic(question: str, agent_answer: str, violations: list,
                 user_rule_summary: str, literature: list[dict] | None = None,
                 kg: dict | None = None) -> dict:
    """Single non-streaming call to the critic. Returns parsed dict with keys
    verdict, notes, suggested_revision. Falls back to verdict='accept' on
    parse failure so a flaky critic never blocks the user."""
    from reasoning.kg_context import format_facts
    viol_str = "\n".join(f"- [{v['name']}] {v['detail']}" for v in violations) or "(none)"
    lit_str = _format_literature(literature or [])
    kg_str = format_facts(kg or {})
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"AGENT ANSWER:\n{agent_answer}\n\n"
        f"PHYSICAL-GROUNDING VIOLATIONS:\n{viol_str}\n\n"
        f"USER RULES IN EFFECT:\n{user_rule_summary or '(none)'}\n\n"
        f"RETRIEVED LITERATURE:\n{lit_str}\n\n"
        f"KNOWLEDGE-GRAPH FACTS (independent edges — do NOT chain them into a causal path):\n{kg_str}\n"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2, "num_predict": 300, "num_ctx": 8192},
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
    except Exception as e:
        return {"verdict": "accept", "notes": [f"(critic unavailable: {e})"],
                "suggested_revision": "", "majority": "", "minority": ""}

    parsed = _safe_json(raw)
    if not parsed:
        return {"verdict": "accept", "notes": ["(critic returned unparseable output)"],
                "suggested_revision": "", "majority": "", "minority": ""}
    verdict = str(parsed.get("verdict", "accept")).strip().lower()
    if verdict not in ("accept", "dispute", "conflicting_evidence"):
        verdict = "accept"
    notes = parsed.get("notes") or []
    if isinstance(notes, str):
        notes = [notes]
    notes = [str(n).strip() for n in notes if str(n).strip()][:3]
    suggestion = str(parsed.get("suggested_revision") or "").strip()
    majority = str(parsed.get("majority") or "").strip()
    minority = str(parsed.get("minority") or "").strip()

    # Gate: conflicting_evidence is only valid if BOTH sides are present and
    # EACH cites at least one [L#] passage. Otherwise the critic is hedging
    # without grounds — downgrade to a normal review.
    if verdict == "conflicting_evidence":
        cite = lambda s: bool(re.search(r"\[L\d+\]", s))
        if not (majority and minority and cite(majority) and cite(minority)):
            verdict = "dispute" if suggestion else "accept"
            notes = (notes + ["(conflict claim not substantiated by citations from both sides — downgraded)"])[:3]
            majority = minority = ""

    return {"verdict": verdict, "notes": notes, "suggested_revision": suggestion,
            "majority": majority, "minority": minority}


def _safe_json(raw: str):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except Exception:
        mm = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not mm:
            return None
        try:
            return json.loads(mm.group(0))
        except Exception:
            return None


def _strip_thinking(text: str) -> str:
    m = re.search(r"final response\s*", text, flags=re.IGNORECASE)
    return text[m.end():].lstrip() if m else text


def _user_rules_summary(user_rules: list[dict]) -> str:
    if not user_rules:
        return ""
    lines = []
    for r in user_rules[:10]:
        if r["kind"] == "range":
            p = r["params"]
            lines.append(f"- {r['name']}: {p.get('lo')}–{p.get('hi')} {p.get('unit','')} "
                         f"(context: {', '.join(p.get('context_terms', []))})")
        elif r["kind"] == "forbid_pattern":
            p = r["params"]
            lines.append(f"- {r['name']}: avoid {p.get('forbidden_terms', [])} near "
                         f"{p.get('context_terms', [])} unless {p.get('exception_terms', [])}")
    return "\n".join(lines)


@app.post("/api/reason/chat")
async def reason_chat(
    question: str = Form(...),
    history: str = Form("[]"),
    all_questions: str = Form("[]"),
):
    """SSE stream — reasoning agent + critic loop with conditional revision.

    Pipeline per request:
      Round 1: reasoning agent answers (streamed)
               → physical-grounding check
      Round 2: critic reviews (non-streaming, JSON verdict)
               → if ACCEPT, done.
               → if DISPUTE, agent revises (streamed)
                          → physical-grounding check
                          → critic re-reviews
                          → done (even if still disputed; surface unresolved)

    Hard cap: CRITIC_MAX_ROUNDS iterations (≤4 LLM calls total).

    SSE events:
      {"type": "round_start",    "phase": "agent"|"critic"|"agent_revise", "round": N}
      {"type": "literature",     "passages": [{rank,title,year,snippet,score}]}  (A1; critic-only evidence)
      {"type": "kg_context",     "facts": [{subject,relation,object,weight}], "anchors": [...]}  (A4; critic-only)
      {"type": "token",          "content": "...", "phase": "agent"|"agent_revise"}
      {"type": "physical_check", "round": N, "passed": bool, "violations": [...]}
      {"type": "critic_review",  "round": N, "verdict": "accept"|"dispute",
                                 "notes": [...], "suggested_revision": "..."}
      {"type": "done",           "answer": "...", "rounds": [...],
                                 "prompt_tokens": N, "completion_tokens": N,
                                 "context_window": N, "out_of_scope": bool,
                                 "critic_resolved": bool}
      {"type": "error",          "message": "..."}
    """
    from reasoning.physical_grounding import check as physical_check
    from reasoning.feedback_store import list_user_rules
    prior_turns: list[dict] = json.loads(history) if history else []
    prior_questions_all: list[str] = json.loads(all_questions) if all_questions else []
    user_rules = list_user_rules()
    user_rule_summary = _user_rules_summary(user_rules)

    def generate():
        q = question.strip()
        if not q:
            yield f"data: {json.dumps({'type':'error','message':'Empty question'})}\n\n"
            return

        if not _is_bone_science(q, prior_questions_all):
            out = (
                "This question is outside BoneMind's reasoning scope. "
                "I cover bone mechanics, fracture and fragility, remodelling, "
                "imaging, biomaterials, and related pathology. "
                "Please ask within that domain."
            )
            yield f"data: {json.dumps({'type':'done','answer':out,'prompt_tokens':0,'completion_tokens':0,'context_window':8192,'out_of_scope':True,'rounds':[],'critic_resolved':True})}\n\n"
            return

        base_messages = [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            *prior_turns,
            {"role": "user",   "content": q},
        ]

        char_count = sum(len(m.get("content", "")) for m in base_messages)
        prompt_tokens = max(1, round(char_count / 3.5))
        completion_tokens = 0

        rounds: list[dict] = []

        # A1 + A4 — evidence for the critic (fetched once, reused across both
        # critic rounds). The reasoning agent never sees either channel.
        literature = _fetch_literature(q)
        from reasoning.kg_context import kg_facts as _kg_facts
        try:
            kg = _kg_facts(q)
        except Exception:
            kg = {"facts": [], "anchors": []}

        try:
            if literature:
                yield f"data: {json.dumps({'type':'literature','passages':literature})}\n\n"
            if kg.get("facts"):
                yield f"data: {json.dumps({'type':'kg_context', **kg})}\n\n"
            # ── Round 1: Reasoning agent ───────────────────────────────
            yield f"data: {json.dumps({'type':'round_start','phase':'agent','round':1})}\n\n"
            agent_result: dict = {}
            for t in _llm_stream(base_messages, agent_result):
                yield f"data: {json.dumps({'type':'token','content':t,'phase':'agent'})}\n\n"
            answer = agent_result.get("full", "")
            completion_tokens += agent_result.get("eval_count", 0)
            answer_visible = _strip_thinking(answer)

            pc = physical_check(answer_visible, user_rules=user_rules)
            yield f"data: {json.dumps({'type':'physical_check','round':1, **pc})}\n\n"
            rounds.append({"role": "agent", "round": 1, "content": answer, "physical_check": pc})

            # ── Round 2: Critic review ─────────────────────────────────
            yield f"data: {json.dumps({'type':'round_start','phase':'critic','round':2})}\n\n"
            critic1 = _call_critic(q, answer_visible, pc["violations"], user_rule_summary, literature, kg)
            yield f"data: {json.dumps({'type':'critic_review','round':2, **critic1})}\n\n"
            rounds.append({"role": "critic", "round": 2, **critic1})

            critic_resolved = critic1["verdict"] == "accept"
            final_answer = answer

            # ── Optional Round 3: Agent revises ────────────────────────
            # Both "dispute" and "conflicting_evidence" trigger a revision, with
            # different instructions: dispute → fix the error; conflicting →
            # present majority + minority positions without picking a side.
            if critic1["verdict"] in ("dispute", "conflicting_evidence"):
                notes_block = "\n".join(f"- {n}" for n in critic1["notes"]) or "- (no specifics provided)"
                if critic1["verdict"] == "conflicting_evidence":
                    revision_user = CONFLICT_REVISION_PROMPT_TEMPLATE.format(
                        majority=critic1.get("majority") or "(not specified)",
                        minority=critic1.get("minority") or "(not specified)",
                        notes=notes_block,
                        question=q,
                    )
                else:
                    revision_user = REVISION_PROMPT_TEMPLATE.format(
                        notes=notes_block,
                        suggestion=critic1["suggested_revision"] or "(no concrete suggestion provided)",
                        question=q,
                    )
                revision_messages = base_messages + [
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": revision_user},
                ]
                yield f"data: {json.dumps({'type':'round_start','phase':'agent_revise','round':3})}\n\n"
                rev_result: dict = {}
                for t in _llm_stream(revision_messages, rev_result):
                    yield f"data: {json.dumps({'type':'token','content':t,'phase':'agent_revise'})}\n\n"
                revised = rev_result.get("full", "")
                completion_tokens += rev_result.get("eval_count", 0)
                revised_visible = _strip_thinking(revised)
                pc2 = physical_check(revised_visible, user_rules=user_rules)
                yield f"data: {json.dumps({'type':'physical_check','round':3, **pc2})}\n\n"
                rounds.append({"role": "agent", "round": 3, "content": revised, "physical_check": pc2})

                # ── Round 4: Critic re-reviews ─────────────────────────
                yield f"data: {json.dumps({'type':'round_start','phase':'critic','round':4})}\n\n"
                critic2 = _call_critic(q, revised_visible, pc2["violations"], user_rule_summary, literature, kg)
                yield f"data: {json.dumps({'type':'critic_review','round':4, **critic2})}\n\n"
                rounds.append({"role": "critic", "round": 4, **critic2})

                # accept OR a correctly-presented conflict are both resolved
                # terminal states; only a remaining "dispute" is unresolved.
                critic_resolved = critic2["verdict"] in ("accept", "conflicting_evidence")
                final_answer = revised
                # latest physical check applies to the displayed answer
                pc = pc2

            yield f"data: {json.dumps({'type':'done', 'answer': final_answer, 'rounds': rounds, 'literature': literature, 'kg': kg, 'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'context_window': 8192, 'critic_resolved': critic_resolved, 'final_physical_check': pc})}\n\n"

        except requests.ConnectionError:
            yield f"data: {json.dumps({'type':'error','message':'Could not connect to Ollama. Run: ollama serve'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Reasoning tab — feedback + user rules (Phase 4) ───────────────────────────

@app.post("/api/reason/feedback")
async def reason_feedback(
    turn_id: str = Form(...),
    polarity: int = Form(...),
    question: str = Form(""),
    answer: str = Form(""),
    feedback_text: str = Form(""),
):
    """Record a thumbs event and (on thumbs-down with text) propose a rule.

    Response shapes:
      thumbs-up:
        {"ok": true, "stats": {...}}
      thumbs-down without text:
        {"ok": true, "correction_id": N, "proposed_rule": null, "stats": {...}}
      thumbs-down with text:
        {"ok": true, "correction_id": N,
         "proposed_rule": {"kind":"range"|"forbid_pattern", "name":..., "params":...}
                       OR  {"kind":"none", "reason":...},
         "stats": {...}}
    """
    from reasoning.feedback_store import save_event, save_correction, stats
    from reasoning.rule_extractor import extract

    if polarity not in (-1, 1):
        return {"ok": False, "error": "polarity must be -1 or +1"}

    save_event(turn_id=turn_id, polarity=polarity)

    if polarity == 1:
        return {"ok": True, "stats": stats()}

    # thumbs-down — persist the correction payload (even if empty text)
    correction_id = save_correction(
        turn_id=turn_id,
        question=question or "",
        answer=answer or "",
        feedback_text=feedback_text or "",
    )

    proposed = None
    if feedback_text.strip():
        proposed = extract(question=question, answer=answer, feedback_text=feedback_text)
    return {
        "ok": True,
        "correction_id": correction_id,
        "proposed_rule": proposed,
        "stats": stats(),
    }


@app.post("/api/reason/rules/confirm")
async def reason_rule_confirm(
    name: str = Form(...),
    kind: str = Form(...),
    params: str = Form(...),                          # JSON-encoded
    source_correction_id: int = Form(None),
):
    """Persist a confirmed (possibly user-edited) rule."""
    from reasoning.feedback_store import save_user_rule, stats
    try:
        params_obj = json.loads(params)
    except Exception as e:
        return {"ok": False, "error": f"params JSON parse error: {e}"}
    if kind not in ("range", "forbid_pattern"):
        return {"ok": False, "error": f"unknown kind: {kind}"}
    rule = save_user_rule(name=name, kind=kind, params=params_obj,
                          source_correction_id=source_correction_id)
    return {"ok": True, "rule": rule, "stats": stats()}


@app.get("/api/reason/rules")
async def reason_rules_list():
    from reasoning.feedback_store import list_user_rules, stats
    return {"rules": list_user_rules(), "stats": stats()}


@app.delete("/api/reason/rules/{rule_db_id}")
async def reason_rule_delete(rule_db_id: int):
    from reasoning.feedback_store import delete_user_rule, stats
    deleted = delete_user_rule(rule_db_id)
    return {"ok": deleted, "stats": stats()}


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


# ── Reasoner — equation graph ────────────────────────────────────────────────
# Forward / abductive / counterfactual inference over a typed equation graph
# (Currey, Paris–Vashishth, beam bending, Frost mechanostat, plus the density
# / inertia / strain bridges).  The chain of Relations is discovered by
# traversing shared variable symbols, not pre-encoded.

_REASON_PRESETS: dict[str, dict] = {
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
            "femur — composition through cortical_inertia and Hooke chain."
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


def _var_info(reg, symbol: str) -> dict:
    v = reg.variable(symbol)
    if v is None:
        return {"symbol": symbol, "display_symbol": symbol, "name": symbol, "unit": ""}
    return {
        "symbol":         v.symbol,
        "display_symbol": v.render(),
        "name":           v.name,
        "unit":           v.unit,
        "lo":             v.lo,
        "hi":             v.hi,
        "description":    v.description,
    }


def _display_map(reg) -> dict[str, str]:
    """ASCII symbol → Unicode display, for the front-end to look up by key."""
    out: dict[str, str] = {}
    for sym in reg.variables_in_graph():
        v = reg.variable(sym)
        if v is not None:
            out[sym] = v.render()
    return out


def _render_forward(
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
        "target_info":   _var_info(reg, target),
        "given":         given,
        "given_info":    {k: _var_info(reg, k) for k in given},
        "covariates":    covariates or {},
        "applied_shifts": reg.applied_shifts(chain, covariates),
        "chain_vars":    [_var_info(reg, v) for v in fr.chain_vars],
        "display_map":   _display_map(reg),
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


def _render_abductive(
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
            "variable":       _var_info(reg, p.variable),
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
        "target_info":  _var_info(reg, target),
        "observed":     ar.observed,
        "observed_std": ar.observed_std,
        "given":        given or {},
        "given_info":   {k: _var_info(reg, k) for k in (given or {})},
        "covariates":   covariates or {},
        "applied_shifts": reg.applied_shifts(chain, covariates),
        "chain_vars":   [_var_info(reg, v) for v in ar.chain_vars],
        "display_map":  _display_map(reg),
        "inferred":     inferred,
        "result": {
            "effective_sample_size": ar.effective_sample_size,
            "n_samples":             ar.n_samples,
        },
        "citations":    ar.citations,
    }


def _render_counterfactual(
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
        "target_info":      _var_info(reg, target),
        "given":            given,
        "given_info":       {k: _var_info(reg, k) for k in given},
        "intervention":     intervention,
        "intervention_info": {k: _var_info(reg, k) for k in intervention},
        "covariates":       covariates or {},
        "applied_shifts":   reg.applied_shifts(chain, covariates),
        "chain_vars":       [_var_info(reg, v) for v in cf.chain_vars],
        "display_map":      _display_map(reg),
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


@app.get("/api/reason/presets")
def reason_presets():
    """Return the preset demo queries available to the reasoning UI."""
    return {
        "presets": [
            {"id": pid, **p}
            for pid, p in _REASON_PRESETS.items()
        ],
        "covariates": COVARIATE_SCHEMA,
        "variables": [
            {
                "symbol":         v.symbol,
                "display_symbol": v.render(),
                "name":           v.name,
                "unit":           v.unit,
                "lo":             v.lo,
                "hi":             v.hi,
                "description":    v.description,
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


@app.post("/api/reason")
def reason(payload: dict):
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
        preset = _REASON_PRESETS.get(preset_id)
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
            out = _render_forward(
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
            out = _render_abductive(
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
            out = _render_counterfactual(
                bone_registry, target, given, intervention,
                covariates=covariates,
            )
        elif mode == "explore":
            return {
                "error": (
                    "Mode 'explore' does not take a target/given payload — "
                    "call GET /api/reason/explore for active exploration "
                    "or POST /api/reason/ask with a free-text query."
                ),
            }
        else:
            return {"error": f"Unknown mode: {mode!r}. Use forward, abductive, counterfactual, or explore."}
    except ValueError as exc:
        return {"error": str(exc)}
    out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


# ── Phase 4: free-text routing → semantic anchor → deterministic reasoner ────


# Literature midpoints used when the router/anchor doesn't supply a value.
# Chain roots only — derived variables are never given directly.
_REASON_DEFAULT_VALUES: dict[str, float] = {
    "phi": 0.10,
    "dK":  1.0,
    "R":   13.0,
    "t":   4.5,
    "M":   40_000.0,
}

# Minimum cosine similarity for a SPECTER2 match to be trusted as an
# anchor.  Below this we fall back to the top-1 result regardless —
# the threshold guards against the router emitting an empty hint.
_REASON_ANCHOR_MIN_SCORE = 0.30


def _anchor_symbol(
    text: str | None, fallback_symbol: str | None = None,
) -> tuple[str | None, float]:
    """Map a free-text hint to a variable symbol via SPECTER2 cosine."""
    if not text:
        return fallback_symbol, 0.0
    match = semantic_anchor.best_match(text, min_score=_REASON_ANCHOR_MIN_SCORE)
    if match is None:
        return fallback_symbol, 0.0
    return match.symbol, match.score


def _pick_target(
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
            if m.symbol in producers and m.score >= _REASON_ANCHOR_MIN_SCORE:
                return m.symbol, m.score
    # Then fall back to the query-wide top-k that the router already attached.
    for m in candidate_matches or []:
        if m.symbol in producers:
            return m.symbol, m.score
    return "E", 0.0


def _chain_roots(target: str) -> list[str]:
    """Chain roots required to reach ``target``; literature defaults fill them."""
    given_set = set(_REASON_DEFAULT_VALUES.keys())
    chain = bone_registry._find_chain(target, given_set)
    if chain is None:
        return []
    return bone_registry.chain_root_inputs(chain)


def _build_ask_payload(
    decision, query: str,
) -> tuple[dict, dict]:
    """
    Translate a :class:`RouterDecision` into a /api/reason payload.

    Returns ``(payload, routing_block)``.  ``payload`` is suitable for
    direct dispatch through :func:`reason`; ``routing_block`` is
    surfaced in the response so the UI can show *why* a particular
    mode + target was chosen.
    """
    mode = decision.mode if decision.mode in {"forward", "abductive", "counterfactual"} \
        else "forward"

    # 1) Target: trust the router's text hint, fall back to the top
    #    semantic match against the raw query.  Constrained to variables
    #    that have a producer so the chain finder can actually solve it.
    target, target_score = _pick_target(
        decision.target_hint, decision.matched_variables,
    )

    payload: dict = {"mode": mode, "target": target}

    # 2) Mode-specific extraction.
    given_overrides: dict[str, float] = {}
    for phrase, value in (decision.given_hints or {}).items():
        sym, _ = _anchor_symbol(phrase)
        if sym in _REASON_DEFAULT_VALUES:
            given_overrides[sym] = float(value)

    if mode == "forward":
        roots = _chain_roots(target)
        given = {r: _REASON_DEFAULT_VALUES.get(r, 0.0) for r in roots}
        given.update(given_overrides)
        payload["given"] = given

    elif mode == "abductive":
        if decision.observed_value is None:
            # Without a number we cannot do abduction; fall back to forward.
            roots = _chain_roots(target)
            given = {r: _REASON_DEFAULT_VALUES.get(r, 0.0) for r in roots}
            given.update(given_overrides)
            payload = {"mode": "forward", "target": target, "given": given}
        else:
            payload["observed"] = float(decision.observed_value)
            if decision.observed_units:
                payload["observed_units"] = decision.observed_units
            # Infer porosity by default — the canonical clinical upstream.
            # Pin every other chain root at literature midpoints.
            roots = _chain_roots(target)
            infer = ["phi"] if "phi" in roots else (roots[:1] if roots else [])
            payload["infer"] = infer
            given = {
                r: _REASON_DEFAULT_VALUES.get(r, 0.0)
                for r in roots if r not in infer
            }
            given.update({k: v for k, v in given_overrides.items() if k not in infer})
            payload["given"] = given

    elif mode == "counterfactual":
        interv_sym, _ = _anchor_symbol(decision.intervention_hint)
        if interv_sym is None or decision.intervention_value is None or \
                interv_sym not in _REASON_DEFAULT_VALUES:
            # Missing intervention — degrade to forward.
            roots = _chain_roots(target)
            given = {r: _REASON_DEFAULT_VALUES.get(r, 0.0) for r in roots}
            given.update(given_overrides)
            payload = {"mode": "forward", "target": target, "given": given}
        else:
            roots = _chain_roots(target)
            given = {r: _REASON_DEFAULT_VALUES.get(r, 0.0) for r in roots}
            given.update(given_overrides)
            # Ensure the intervention key has a meaningful baseline.
            if interv_sym not in given:
                given[interv_sym] = _REASON_DEFAULT_VALUES.get(interv_sym, 0.0)
            payload["given"] = given
            payload["intervention"] = {interv_sym: float(decision.intervention_value)}

    routing_block = {
        "query":      query,
        "mode":       payload["mode"],
        "target":     target,
        "target_info": _var_info(bone_registry, target),
        "target_score": round(target_score, 3),
        "rationale":  decision.rationale,
        "confidence": round(decision.confidence, 3),
        "source":     decision.source,
        "matched_variables": [
            {
                "symbol":         m.symbol,
                "display_symbol": m.display_symbol or m.symbol,
                "name":           m.name,
                "unit":           m.unit,
                "score":          round(m.score, 3),
            }
            for m in decision.matched_variables
        ],
    }
    return payload, routing_block


@app.post("/api/reason/ask")
def reason_ask(payload: dict):
    """
    Free-text entry point into the reasoner.

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

    The response is the standard reasoner result plus a ``routing`` block
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

    # Phase 5: explore mode short-circuits into the exploration engine
    # rather than the deterministic reasoner.  The routing block still
    # describes what the LLM understood from the query.
    if decision.mode == "explore":
        candidates = _run_exploration()
        rendered = [_explore_render_candidate(c) for c in candidates]
        routing_block = {
            "query":      query,
            "mode":       "explore",
            "target":     None,
            "rationale":  decision.rationale,
            "confidence": round(decision.confidence, 3),
            "source":     decision.source,
            "matched_variables": [
                {
                    "symbol": m.symbol,
                    "display_symbol": m.display_symbol or m.symbol,
                    "name": m.name,
                    "unit":   m.unit,   "score": round(m.score, 3),
                }
                for m in decision.matched_variables
            ],
        }
        return {
            "mode":       "explore",
            "routing":    routing_block,
            "candidates": rendered,
            "corpus_disclaimer": CORPUS_DISCLAIMER,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    inner_payload, routing_block = _build_ask_payload(decision, query)
    # Forward covariates through verbatim — the router does not touch them.
    if payload.get("covariates"):
        inner_payload["covariates"] = payload["covariates"]

    result = reason(inner_payload)
    if "error" in result:
        result["routing"] = routing_block
        return result
    result["routing"] = routing_block
    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


# ── Active exploration ──────────────────────────────────────────


def _explore_render_candidate(c: ExplorationCandidate) -> dict:
    """Shape one :class:`ExplorationCandidate` for the front-end."""
    sweep_var = bone_registry.variable(c.sweep_var)
    target_v  = bone_registry.variable(c.target)
    chain_var_info = [
        {
            "symbol":         v.symbol if (v := bone_registry.variable(s)) else s,
            "display_symbol": v.render() if v else s,
            "name":           v.name if v else s,
            "unit":           v.unit if v else "",
        }
        for s in c.chain_vars
    ]
    sweep_points = [
        {
            "value":      val,
            "mean":       mean,
            "p5":         p5,
            "p95":        p95,
        }
        for val, mean, p5, p95 in zip(
            c.sweep_values, c.sweep_means, c.sweep_p5, c.sweep_p95,
        )
    ]
    rel_change = c.relative_change
    # Float('inf') / nan would serialise to null in some clients; clip them.
    import math as _math
    rel_change_safe = rel_change if _math.isfinite(rel_change) else None

    return {
        "title":          c.title,
        "target":         c.target,
        "target_info":    {
            "symbol":         target_v.symbol if target_v else c.target,
            "display_symbol": target_v.render() if target_v else c.target,
            "name":           target_v.name   if target_v else c.target,
            "unit":           target_v.unit   if target_v else "",
        },
        "sweep_var":      c.sweep_var,
        "sweep_var_info": {
            "symbol":         sweep_var.symbol if sweep_var else c.sweep_var,
            "display_symbol": sweep_var.render() if sweep_var else c.sweep_var,
            "name":           sweep_var.name   if sweep_var else c.sweep_var,
            "unit":           sweep_var.unit   if sweep_var else "",
        },
        "held":           c.held,
        "chain_vars":     chain_var_info,
        "citations":      c.citations,
        "sweep_points":   sweep_points,
        "direction":      c.direction,
        "relative_change": rel_change_safe,
        "physics_confidence": round(c.physics_confidence, 3),
        "magnitude":          round(c.magnitude, 3),
        "corpus": {
            "label":         c.corpus_label,
            "score":         round(c.corpus_score, 3),
            "similarity":    round(c.corpus_similarity, 3),
            "keyword_hits":  c.corpus_keyword_hits,
            "query":         c.corpus_query,
        },
        "surprise_score": round(c.surprise_score, 3),
        "summary":        c.summary,
        "error":          c.error,
    }


def _run_exploration(refresh: bool = False) -> list[ExplorationCandidate]:
    """Lazy + cached exploration run."""
    global _exploration_cache
    if refresh or _exploration_cache is None:
        _exploration_cache = explorer.run()
    return _exploration_cache


@app.get("/api/reason/explore")
def reason_explore(refresh: bool = Query(False)):
    """
    Active exploration over the equation graph.

    Walks a hand-picked but principled set of variable-graph sweeps,
    runs forward inference at each sweep point, asks the corpus how
    attested the rendered claim is, and ranks the candidates by

        surprise = magnitude × physics_confidence × corpus_weight

    The first call computes ~7 sweeps with corpus contact (≈10s on a
    warm SPECTER2 index); subsequent calls return the cached result
    in <1ms.  Pass ``?refresh=true`` to force re-evaluation.
    """
    t0 = time.perf_counter()
    candidates = _run_exploration(refresh=refresh)
    rendered = [_explore_render_candidate(c) for c in candidates]
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "n_candidates":      len(rendered),
        "candidates":        rendered,
        "corpus_disclaimer": CORPUS_DISCLAIMER,
        "cached":            (not refresh) and _exploration_cache is not None
                              and elapsed_ms < 50,
        "elapsed_ms":        elapsed_ms,
    }


# ── Phase 7/8 — agent-driven exploration ──────────────────────────────────────

@app.post("/api/reason/agents")
def reason_agents(payload: dict):
    """
    Run the Proposer → Physics → Critic agent loop.

    The Proposer (Phase 7) uses an LLM to walk the variable graph and
    propose (target, sweep_var, given, sweep_values) tuples not in the
    hand-picked sweep table.  Each proposal is evaluated by the
    deterministic Explorer (same pipeline as /explore).  The Critic
    (Phase 8) then calls corpus_search and decides whether the physics
    prediction is interesting, trivial, out-of-domain, or needs more data.

    Body (all optional):
      n_proposals : int  — number of hypotheses to generate (default 2, max 4)
      bone_type   : str  — "cortical" | "transitional" | "trabecular" — narrows
                          Variable sweep ranges to the active tissue regime
      covariates  : dict — patient context passed to forward()
    """
    n = min(int(payload.get("n_proposals") or 2), 4)
    bone_type = str(payload.get("bone_type") or "cortical").strip().lower()
    if bone_type not in {"cortical", "transitional", "trabecular"}:
        bone_type = "cortical"
    t0 = time.perf_counter()

    proposals_out = []
    for _ in range(n):
        # ── Step 1: Proposer ─────────────────────────────────────────────────
        proposal = _proposer.propose(
            scratchpad=_agent_scratchpad,
            bone_type=bone_type,
        )
        if proposal is None:
            proposals_out.append({
                "error": "Proposer agent unavailable (Ollama unreachable or parsing failed).",
                "source": "fallback",
            })
            continue

        # Validate proposed symbols exist in the registry.
        if not bone_registry.variable(proposal.target):
            proposals_out.append({
                "error": f"Proposer proposed unknown target '{proposal.target}'.",
                "proposal": {"target": proposal.target, "sweep_var": proposal.sweep_var},
                "source": "agent",
            })
            continue
        if not bone_registry.variable(proposal.sweep_var):
            proposals_out.append({
                "error": f"Proposer proposed unknown sweep_var '{proposal.sweep_var}'.",
                "proposal": {"target": proposal.target, "sweep_var": proposal.sweep_var},
                "source": "agent",
            })
            continue

        # ── Step 2: Physics evaluation (deterministic) ───────────────────────
        # Filter `given` to known symbols so a hallucinated key (e.g. "density"
        # instead of "rho") can't crash the Explorer downstream.
        held_raw = {
            k: v for k, v in proposal.given.items()
            if bone_registry.variable(k) is not None
        }
        # Auto-repair: drop the sweep_var from held and fill any missing
        # chain-root inputs with midpoint values, so the engine can resolve
        # the derivation even when the Proposer forgot a required pin.
        held = bone_registry.complete_given(
            target=proposal.target,
            sweep_var=proposal.sweep_var,
            given=held_raw,
            bone_type=bone_type,
        )
        if held is None:
            proposals_out.append({
                "error": (
                    f"No derivation chain from sweep_var '{proposal.sweep_var}' "
                    f"to target '{proposal.target}' exists in the registry."
                ),
                "proposal": {"target": proposal.target, "sweep_var": proposal.sweep_var,
                             "rationale": proposal.rationale},
                "source": "agent",
            })
            continue

        # Clamp sweep values to the swept Variable's registered range so the
        # Proposer can't push a sweep beyond physical validity (e.g. R=30 mm
        # when the range is [5, 25]).
        sweep_var_def = bone_registry.variable(proposal.sweep_var)
        if sweep_var_def is not None:
            lo, hi = sweep_var_def.range_for(bone_type)
            clipped = [max(lo, min(hi, v)) for v in proposal.sweep_values]
            if len(set(clipped)) < 2:
                clipped = [lo, 0.5 * (lo + hi), hi]
            sweep_values = clipped
        else:
            sweep_values = proposal.sweep_values

        cand = explorer.evaluate_proposal(
            target=proposal.target,
            sweep_var=proposal.sweep_var,
            sweep_values=sweep_values,
            held=held,
            title=f"[agent] {proposal.sweep_var} → {proposal.target}",
            skip_novelty=True,   # Critic agent runs its own corpus search.
        )

        if cand is None:
            proposals_out.append({
                "error": "Explorer could not evaluate the proposed sweep.",
                "proposal": {"target": proposal.target, "sweep_var": proposal.sweep_var,
                             "rationale": proposal.rationale},
                "source": "agent",
            })
            continue

        # Compare the Proposer's declared direction against what the engine
        # actually computed. A mismatch means the Proposer's rationale and
        # the engine result disagree — the Critic must be told so it can
        # address the disagreement.
        engine_dir = cand.direction
        pred_dir   = proposal.predicted_direction
        dir_map = {"increases": "up", "decreases": "down", "non-monotonic": "mixed"}
        engine_dir_norm = dir_map.get(engine_dir, "unknown")
        proposer_mismatch = (
            pred_dir in {"up", "down", "mixed"}
            and engine_dir_norm in {"up", "down", "mixed"}
            and pred_dir != engine_dir_norm
        )

        physics_summary = {
            "direction":           cand.direction,
            "relative_change":     cand.relative_change if not _isnan(cand.relative_change) else None,
            "physics_confidence":  cand.physics_confidence,
            "magnitude":           cand.magnitude,
            "summary":             cand.summary,
            "citations":           cand.citations,
            "corpus_label":        cand.corpus_label,
            "surprise_score":      cand.surprise_score,
            "predicted_direction": pred_dir,
            "proposer_mismatch":   proposer_mismatch,
        }

        # ── Step 3: Critic ───────────────────────────────────────────────────
        critique = _critic.critique(
            hypothesis={
                "target":              proposal.target,
                "sweep_var":           proposal.sweep_var,
                "given":               held,
                "sweep_values":        sweep_values,
                "rationale":           proposal.rationale,
                "predicted_direction": pred_dir,
            },
            physics=physics_summary,
        )

        # Append to scratchpad so later proposals avoid repeating.
        _agent_scratchpad.append({
            "target":    proposal.target,
            "sweep_var": proposal.sweep_var,
        })
        if len(_agent_scratchpad) > _AGENT_SCRATCHPAD_MAX:
            del _agent_scratchpad[: len(_agent_scratchpad) - _AGENT_SCRATCHPAD_MAX]

        proposals_out.append({
            "hypothesis": {
                "target":       proposal.target,
                "sweep_var":    proposal.sweep_var,
                "given":        held,
                "sweep_values": sweep_values,
                "rationale":    proposal.rationale,
                "title":        cand.title,
            },
            "physics":    {**physics_summary, **_explore_render_candidate(cand)},
            "critique":   {
                "verdict":          critique.verdict,
                "reason":           critique.reason,
                "confidence":       critique.confidence,
                "corpus_passages":  critique.corpus_passages,
                "source":           critique.source,
            },
            "proposer_source": proposal.source,
        })

    elapsed_s = round(time.perf_counter() - t0, 2)
    return {
        "proposals":         proposals_out,
        "n_proposals":       len(proposals_out),
        "scratchpad_size":   len(_agent_scratchpad),
        "corpus_disclaimer": CORPUS_DISCLAIMER,
        "elapsed_s":         elapsed_s,
    }


def _isnan(x) -> bool:
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return False


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
