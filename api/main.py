"""
api/main.py
===========
FastAPI backend for the BoneGraph React frontend.

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
POST /api/reason/chat       — SSE stream: reasoning agent → physical grounding →
                              critic loop with conditional revision (the live
                              Reasoning tab). See docs/reasoning_tab.md.
POST /api/reason/feedback   — record 👍/👎; on 👎 with text, extract a proposed rule
POST /api/reason/rules/*    — confirm / list / enable / delete / import user rules
POST /api/analyse          — VLM image analysis (multipart), JSON response
GET  /                     — serves frontend/index.html
"""

import base64
import io
import json
import math
import os
import re
import sqlite3
import time
from pathlib import Path

import requests
from fastapi import FastAPI, Form, UploadFile, File, Query, Request, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from retrieval.retriever import BoneGraphRetriever
from reasoning.graph_db import OntologyStore
from config.settings import PAPERS_DB_PATH, TEXTBOOKS_DB_PATH, CHUNKS_DB_PATH, D2IM_WEIGHTS_PATH

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

SYSTEM_PROMPT = """You are BoneGraph, an expert AI assistant specialised in bone science.
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
   "This question is outside BoneGraph's domain. I cover bone science only."

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
print("Loading BoneGraph retriever...")
retriever = BoneGraphRetriever()
retriever.load()

print("Loading bone knowledge graph (for /api/stats)...")
with OntologyStore() as _store:
    _graph_stats = _store.stats()

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
app = FastAPI(title="BoneGraph API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


# ── Auth ───────────────────────────────────────────────────────────────────────
# Email/password accounts (api/auth_store.py). The email becomes the
# `user_id` threaded through reasoning/feedback_store.py and
# vision/correction_store.py so Tier-2 learned rules and Vision corrections are
# scoped per account instead of all landing in the shared "local" bucket. The
# 8 built-in Tier-1 physical-grounding rules stay hardcoded and global — they
# are not touched by this.

def get_current_user(authorization: str = Header(None)) -> str:
    from api.auth_store import get_user_by_token
    token = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else ""
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user["email"]


@app.post("/api/auth/register")
async def auth_register(full_name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    from api.auth_store import create_user, create_session
    try:
        user = create_user(full_name, email, password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"token": create_session(user["id"]), "full_name": user["full_name"], "email": user["email"]}


@app.post("/api/auth/login")
async def auth_login(email: str = Form(...), password: str = Form(...)):
    from api.auth_store import verify_user, create_session
    user = verify_user(email, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"token": create_session(user["id"]), "full_name": user["full_name"], "email": user["email"]}


@app.post("/api/auth/logout")
async def auth_logout(authorization: str = Header(None)):
    from api.auth_store import delete_session
    if authorization and authorization.lower().startswith("bearer "):
        delete_session(authorization[7:].strip())
    return {"ok": True}


@app.get("/api/auth/me")
async def auth_me(user_id: str = Depends(get_current_user)):
    return {"email": user_id}


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
def get_stats(user_id: str = Depends(get_current_user)):
    return {
        "papers_total":    STATS["papers_total"],
        "pdfs_downloaded": STATS["pdfs_downloaded"],
        "textbooks":       STATS["textbooks"],
        "chunks":          STATS["chunks"],
        "graph_nodes":     _graph_stats["total_nodes"],
        "graph_edges":     _graph_stats["total_edges"],
    }


@app.post("/api/ask")
async def ask(question: str = Form(...), top_k: int = Form(8), history: str = Form("[]"), all_questions: str = Form("[]"),
              user_id: str = Depends(get_current_user)):
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
                "I'm BoneGraph, a specialist assistant for bone science. "
                "Your question doesn't appear to be related to bone biology, skeletal mechanics, "
                "or a closely related biomedical topic. Please ask something within that domain "
                "and I'll do my best to answer from the literature.\n\n"
                "Note: BoneGraph is a research tool and does not provide personal medical advice. "
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

REASONING_SYSTEM_PROMPT = """You are BoneGraph's reasoning agent. Given a question about bone
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
  is out of scope, reply with exactly: "Out of scope for BoneGraph's reasoning tab."
- Be honest about uncertainty. If a point is speculative, say so in the *Basis* line.
- No preamble, no closing summary. Start directly with "**Point 1.**"."""


CRITIC_SYSTEM_PROMPT = """You are BoneGraph's critic agent. You review another agent's bone-science reasoning for correctness, internal consistency, completeness, and appropriately calibrated uncertainty.

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
 "minority":"<the competing position, citing [L#]>",
 "majority_support":["L1","L2"],
 "minority_support":["L3"]}

Rules:
- VERDICT must be "accept", "dispute", or "conflicting_evidence".
- GROUND YOUR REVIEW IN THE LITERATURE. When a passage supports or contradicts the answer, cite it by tag in your notes, e.g. "Contradicts [L2]: cortical modulus is ~18 GPa".
- Do NOT treat the literature as infallible: passages may be off-topic or only tangentially related. Use judgement; if no passage is relevant, review on correctness alone and say so.
- KNOWLEDGE-GRAPH FACTS are individual curated relationships. Each line is ONE fact. Use them to check the answer's direction/sign (e.g. "porosity [decreases] mechanical strength"). Treat each edge independently — do NOT chain several edges into a multi-step causal argument, because chaining curated edges can imply relationships the graph never asserted.
- ACCEPT if the answer is broadly correct and well-reasoned. A physical-grounding violation flagged on an edge case the agent already qualified is still acceptable — say so in notes.
- DISPUTE only if there is a factual error, internal contradiction, missing key mechanism, a genuine physical-grounding violation that the agent did not address, or a claim the retrieved literature clearly contradicts.
- USER RULES ARE NON-NEGOTIABLE. If any violation is tagged "USER RULE — must dispute", you MUST return "dispute". The user explicitly taught that constraint through feedback, and it overrides your own judgement and any literature. Your suggested_revision must tell the agent to comply with that rule.
- CONFLICTING_EVIDENCE only when the retrieved literature genuinely disagrees — one body of passages supports the answer and another contradicts it, and the question has no single settled answer. You MUST fill both "majority" and "minority", and EACH must cite at least one [L#] tag. If you cannot cite both sides from the passages provided, do NOT use this verdict — use accept or dispute instead. Do not invent a conflict to hedge.
- For CONFLICTING_EVIDENCE you MUST also fill "majority_support" and "minority_support": the exact list of passage tags ("L1","L2",…) that back each side. Assign every passage that takes a position to exactly one side based on what it actually says — do NOT split them evenly to look balanced. A passage that is off-topic or takes no side is simply omitted from both lists. These lists drive a confidence score shown to the user, so they must reflect the real weight of evidence.
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
# (~1200 tokens ≈ ~4200 chars across all passages, split across LIT_TOP_K).
#
# LIT_TOP_K is the configurable retrieval count (A5 / Task 4). It also sets the
# denominator for the conflict confidence score — more passages → a more
# credible split. Default 5; raise via env (e.g. BONEGRAPH_LIT_TOP_K=10) on a
# box with a larger context window (16k). The char budget caps total tokens
# regardless, so a bigger K just yields more, shorter snippets.
LIT_CHAR_BUDGET = 4200
LIT_TOP_K = int(os.environ.get("BONEGRAPH_LIT_TOP_K", "5"))


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
    # Tag user-defined rule violations so the critic knows they are
    # non-negotiable (they're also enforced deterministically downstream).
    def _vline(v):
        tag = "USER RULE — must dispute" if v.get("source") == "user" else v["name"]
        return f"- [{tag}] {v['detail']}"
    viol_str = "\n".join(_vline(v) for v in violations) or "(none)"
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
                "options": {"temperature": 0.2, "num_predict": 380, "num_ctx": 8192},
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
                "suggested_revision": "", "majority": "", "minority": "", "score": None}
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
    score = None

    # Gate: conflicting_evidence is only valid if BOTH sides are present and
    # EACH is backed by at least one real retrieved [L#] passage. Otherwise the
    # critic is hedging without grounds — downgrade to a normal review.
    if verdict == "conflicting_evidence":
        available = {f"L{p['rank']}" for p in (literature or [])}
        # Prefer the explicit support arrays; fall back to the [L#] cited inside
        # the majority/minority prose if the arrays are missing/malformed.
        maj_sup = _support_tags(parsed.get("majority_support"), majority, available)
        min_sup = _support_tags(parsed.get("minority_support"), minority, available)
        # A tag cited on both sides discriminates nothing — drop it from both.
        overlap = set(maj_sup) & set(min_sup)
        maj_sup = [t for t in maj_sup if t not in overlap]
        min_sup = [t for t in min_sup if t not in overlap]

        if not (majority and minority and maj_sup and min_sup):
            verdict = "dispute" if suggestion else "accept"
            notes = (notes + ["(conflict claim not substantiated by citations from both sides — downgraded)"])[:3]
            majority = minority = ""
        else:
            n_maj, n_min = len(maj_sup), len(min_sup)
            total = n_maj + n_min
            maj_pct = round(100 * n_maj / total)
            score = {
                "majority_pct": maj_pct,
                "minority_pct": 100 - maj_pct,
                "n_supporting": total,
                "n_passages": len(literature or []),
                "majority_support": maj_sup,
                "minority_support": min_sup,
            }

    return {"verdict": verdict, "notes": notes, "suggested_revision": suggestion,
            "majority": majority, "minority": minority, "score": score}


def _enforce_user_rules(critic: dict, pc: dict) -> dict:
    """Deterministic guarantee: if the round flagged any user-defined rule
    violation, the verdict MUST be 'dispute' regardless of what the critic
    model decided. The user explicitly taught the rule; the model cannot
    override it. (The critic prompt is also told this, but we do not rely on
    a small model honouring it.)"""
    user_viol = [v for v in pc.get("violations", []) if v.get("source") == "user"]
    if not user_viol or critic.get("verdict") == "dispute":
        return critic
    details = "; ".join(v["detail"] for v in user_viol[:2])
    note = f"User-defined rule violated — must be corrected: {details}"
    return {
        **critic,
        "verdict": "dispute",
        "notes": ([note] + (critic.get("notes") or []))[:3],
        "suggested_revision": critic.get("suggested_revision")
            or "Revise so the answer complies with the user's rule(s); do not restate the violating value.",
        "user_override": True,
    }


def _support_tags(array_val, prose: str, available: set) -> list:
    """Resolve the passages backing one side of a conflict into canonical
    "L#" tags. Prefers the critic's explicit support array; if that is empty or
    malformed, falls back to the [L#] citations inside the side's prose. Keeps
    only tags that match a real retrieved passage, de-duplicated in order."""
    raw_tags: list[str] = []
    if isinstance(array_val, str):
        array_val = [array_val]
    if isinstance(array_val, list):
        for t in array_val:
            m = re.search(r"(\d+)", str(t))
            if m:
                raw_tags.append(f"L{m.group(1)}")
    if not raw_tags:
        raw_tags = [f"L{n}" for n in re.findall(r"\[L(\d+)\]", prose or "")]
    out: list[str] = []
    for t in raw_tags:
        if t in available and t not in out:
            out.append(t)
    return out


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
        elif r["kind"] == "comparative":
            p = r["params"]
            a = "/".join(p.get("first_terms", []))
            b = "/".join(p.get("second_terms", []))
            axis = ", ".join(p.get("comparator_terms", []))
            lines.append(f"- {r['name']}: {a} comes before / outranks {b} "
                         f"(axis: {axis})"
                         + (f" — {p['explanation']}" if p.get('explanation') else ""))
    return "\n".join(lines)


@app.post("/api/reason/chat")
async def reason_chat(
    question: str = Form(...),
    history: str = Form("[]"),
    all_questions: str = Form("[]"),
    mode: str = Form("deep"),
    user_id: str = Depends(get_current_user),
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
    user_rules = list_user_rules(user_id=user_id)
    user_rule_summary = _user_rules_summary(user_rules)
    # B3 — "quick" mode skips the critic loop and the (critic-only) evidence
    # fetch for fast answers; physical grounding + user rules still run.
    quick = (mode or "deep").strip().lower() == "quick"

    def generate():
        q = question.strip()
        if not q:
            yield f"data: {json.dumps({'type':'error','message':'Empty question'})}\n\n"
            return

        if not _is_bone_science(q, prior_questions_all):
            out = (
                "This question is outside BoneGraph's reasoning scope. "
                "I cover bone mechanics, fracture and fragility, remodelling, "
                "imaging, biomaterials, and related pathology. "
                "Please ask within that domain."
            )
            yield f"data: {json.dumps({'type':'done','answer':out,'prompt_tokens':0,'completion_tokens':0,'context_window':8192,'out_of_scope':True,'rounds':[],'critic_resolved':True})}\n\n"
            return

        # Prime the agent with the user's learned rules so it is correct on the
        # FIRST pass, rather than relying on the critic loop to repair a wrong
        # draft after the fact. These are authoritative: they override the
        # agent's priors AND any premise baked into the question (e.g. a leading
        # "why does A fail before B?" when the user taught B-before-A). The
        # deterministic physical-grounding check still runs as a backstop in case
        # the agent ignores the prime.
        agent_system = REASONING_SYSTEM_PROMPT
        if user_rule_summary:
            agent_system += (
                "\n\nLEARNED CONSTRAINTS — the user has explicitly taught you the "
                "following. Treat them as authoritative: they override your own "
                "priors and any assumption implied by the question. If the question "
                "presupposes something a constraint contradicts, correct the premise "
                "instead of going along with it.\n" + user_rule_summary
            )
        base_messages = [
            {"role": "system", "content": agent_system},
            *prior_turns,
            {"role": "user",   "content": q},
        ]

        char_count = sum(len(m.get("content", "")) for m in base_messages)
        prompt_tokens = max(1, round(char_count / 3.5))
        completion_tokens = 0

        rounds: list[dict] = []

        # A1 + A4 — evidence for the critic (fetched once, reused across both
        # critic rounds). The reasoning agent never sees either channel.
        # Skipped entirely in quick mode (it's only used by the critic).
        if quick:
            literature, kg = [], {"facts": [], "anchors": []}
        else:
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

            # ── Quick mode: stop after grounding, skip the critic ──────
            if quick:
                yield f"data: {json.dumps({'type':'done', 'answer': answer, 'rounds': rounds, 'literature': [], 'kg': kg, 'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'context_window': 8192, 'critic_resolved': True, 'final_physical_check': pc, 'mode': 'quick'})}\n\n"
                return

            # ── Round 2: Critic review ─────────────────────────────────
            yield f"data: {json.dumps({'type':'round_start','phase':'critic','round':2})}\n\n"
            critic1 = _call_critic(q, answer_visible, pc["violations"], user_rule_summary, literature, kg)
            critic1 = _enforce_user_rules(critic1, pc)
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
                critic2 = _enforce_user_rules(critic2, pc2)
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
    user_id: str = Depends(get_current_user),
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

    save_event(turn_id=turn_id, polarity=polarity, user_id=user_id)

    if polarity == 1:
        return {"ok": True, "stats": stats(user_id=user_id)}

    # thumbs-down — persist the correction payload (even if empty text)
    correction_id = save_correction(
        turn_id=turn_id,
        question=question or "",
        answer=answer or "",
        feedback_text=feedback_text or "",
        user_id=user_id,
    )

    proposed = None
    if feedback_text.strip():
        proposed = extract(question=question, answer=answer, feedback_text=feedback_text)
    return {
        "ok": True,
        "correction_id": correction_id,
        "proposed_rule": proposed,
        "stats": stats(user_id=user_id),
    }


@app.post("/api/reason/rules/confirm")
async def reason_rule_confirm(
    name: str = Form(...),
    kind: str = Form(...),
    params: str = Form(...),                          # JSON-encoded
    source_correction_id: int = Form(None),
    user_id: str = Depends(get_current_user),
):
    """Persist a confirmed (possibly user-edited) rule."""
    from reasoning.feedback_store import save_user_rule, stats
    try:
        params_obj = json.loads(params)
    except Exception as e:
        return {"ok": False, "error": f"params JSON parse error: {e}"}
    if kind not in ("range", "forbid_pattern", "comparative"):
        return {"ok": False, "error": f"unknown kind: {kind}"}
    rule = save_user_rule(name=name, kind=kind, params=params_obj,
                          source_correction_id=source_correction_id, user_id=user_id)
    return {"ok": True, "rule": rule, "stats": stats(user_id=user_id)}


@app.get("/api/reason/rules")
async def reason_rules_list(user_id: str = Depends(get_current_user)):
    # The management view needs ALL rules (incl. disabled). The grounding check
    # itself uses list_user_rules(enabled_only=True) — unchanged.
    from reasoning.feedback_store import list_user_rules, stats
    return {"rules": list_user_rules(user_id=user_id, enabled_only=False), "stats": stats(user_id=user_id)}


@app.post("/api/reason/rules/{rule_db_id}/enabled")
async def reason_rule_set_enabled(rule_db_id: int, enabled: bool = Form(...), user_id: str = Depends(get_current_user)):
    from reasoning.feedback_store import set_rule_enabled, stats
    ok = set_rule_enabled(rule_db_id, enabled, user_id=user_id)
    return {"ok": ok, "stats": stats(user_id=user_id)}


@app.delete("/api/reason/rules/{rule_db_id}")
async def reason_rule_delete(rule_db_id: int, user_id: str = Depends(get_current_user)):
    from reasoning.feedback_store import delete_user_rule, stats
    deleted = delete_user_rule(rule_db_id, user_id=user_id)
    return {"ok": deleted, "stats": stats(user_id=user_id)}


@app.get("/api/reason/rules/template")
async def reason_rules_template():
    """Downloadable CSV template (header + examples) for bulk import."""
    from reasoning.rule_import import TEMPLATE_CSV
    return PlainTextResponse(TEMPLATE_CSV, headers={
        "Content-Disposition": 'attachment; filename="bonegraph_rules_template.csv"'
    })


@app.post("/api/reason/rules/import")
async def reason_rules_import(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    """Bulk-import rules from a CSV or XLSX file (A5).

    Validates every row, dedupes against the file and the existing store, and
    enforces a practical cap so the violation badge stays meaningful. Bad rows
    are reported, not silently dropped.

    Response:
      {"imported": N, "skipped": [{"row": i, "name": ..., "reason": ...}],
       "total_now": M, "stats": {...}}
    """
    from reasoning.rule_import import parse_file, validate_row, rule_signature, MAX_IMPORT_RULES
    from reasoning.feedback_store import list_user_rules, save_user_rule, stats

    data = await file.read()
    try:
        rows = parse_file(file.filename or "", data)
    except Exception as e:
        return {"imported": 0, "skipped": [], "total_now": None, "error": str(e)}

    existing = list_user_rules(user_id=user_id, enabled_only=False)
    existing_sigs = set()
    for r in existing:
        try:
            existing_sigs.add(rule_signature(r))
        except Exception:
            pass

    current_count = len(existing)
    imported = 0
    skipped: list[dict] = []
    seen_in_file: set = set()

    for i, row in enumerate(rows, 1):
        rule, err = validate_row(row)
        if err:
            skipped.append({"row": i, "name": str(row.get("name", "")), "reason": err})
            continue
        sig = rule_signature(rule)
        if sig in existing_sigs or sig in seen_in_file:
            skipped.append({"row": i, "name": rule["name"], "reason": "duplicate of an existing rule"})
            continue
        if current_count + imported >= MAX_IMPORT_RULES:
            skipped.append({"row": i, "name": rule["name"],
                            "reason": f"rule cap reached ({MAX_IMPORT_RULES}) — not imported"})
            continue
        save_user_rule(rule["name"], rule["kind"], rule["params"],
                       source_correction_id=None, origin="imported", user_id=user_id)
        seen_in_file.add(sig)
        imported += 1

    return {
        "imported": imported,
        "skipped": skipped,
        "total_now": current_count + imported,
        "stats": stats(user_id=user_id),
    }


@app.post("/api/search")
def search(query: str = Form(...), top_k: int = Form(10), source_filter: str = Form("All"), year_min: int = Form(1970), year_max: int = Form(2026),
           user_id: str = Depends(get_current_user)):
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


# ── Vision tab — Phase 1: chat shell + structured VLM identification ───────────
#
# Mirrors the Reasoning tab's /api/reason/chat pipeline, with the agent swapped
# for a vision-language model answering "what am I looking at?". Phase 1 ships
# ONLY round 1 (the VLM agent) — grounding, critic, evidence and correction
# memory are Phases 2–3 (see docs/vision/architecture.md).
#
# The VLM is a deliberately ISOLATED, swappable adapter: everything downstream
# depends on the structured identification schema below, never on llava-specific
# behaviour. Swapping llava:13b → a medical VLM → a fine-tuned model is a change
# to VLM_MODEL + this one prompt, with zero changes to the loop.

VISION_AGENT_PROMPT = (
    "You are a bone-imaging identification assistant. You are shown ONE image "
    "and (optionally) a user prompt. Identify what the image shows.\n\n"
    "Scope: musculoskeletal / bone imaging only — X-ray, MRI, micro-CT, "
    "histology. If the image is clearly NOT a bone image, set \"identification\" "
    "to \"not a bone image\" and \"confidence\" to \"LOW\".\n\n"
    "BE AS SPECIFIC AS THE IMAGE ALLOWS. Name the actual anatomical structure, "
    "not just the tissue class:\n"
    "  - For radiographs / MRI: name the specific bone(s) and region, and the "
    "view if discernible — e.g. \"radius and ulna (forearm), AP view\", "
    "\"proximal femur\", \"lumbar vertebra L4\". Do NOT answer with a broad tissue "
    "term like \"cortical bone\" when the bone itself is identifiable.\n"
    "  - For micro-CT / histology, where gross anatomy is not visible: identify "
    "the tissue type and structure — e.g. \"trabecular bone (cancellous network)\", "
    "\"cortical bone (osteonal)\".\n"
    "Only fall back to a broad term if the specific structure genuinely cannot "
    "be determined, and lower the confidence accordingly. Put the tissue class "
    "(cortical / trabecular) in \"tissue_type\".\n\n"
    "Return ONLY a JSON object with these exact fields (no prose, no markdown):\n"
    "{\n"
    '  "identification": "<most specific anatomical structure, e.g. radius and ulna (forearm), proximal femur, lumbar vertebra L4>",\n'
    '  "tissue_type": "<cortical | trabecular | mixed | not applicable>",\n'
    '  "modality": "<X-ray | MRI | micro-CT | histology | unknown>",\n'
    '  "morphology": "<1-2 sentence description of the visible morphology>",\n'
    '  "estimated_scale": "<rough field of view, e.g. whole bone (cm), trabecular network (mm), unknown>",\n'
    '  "features": ["<short salient feature>", "..."],\n'
    '  "confidence": "HIGH | MODERATE | LOW",\n'
    '  "runner_up": "<second most likely identification, or empty string>"\n'
    "}"
)


# Follow-up turns answer conversationally (prose), not as the rigid schema —
# the first turn anchors the identification, then the user can chat about the
# image like with an LLM ("is there anything unusual here?", "what about the
# top edge?"). History gives the model the context for "this"/"it" references.
VISION_CHAT_PROMPT = (
    "You are a bone-imaging assistant discussing ONE image with the user. You "
    "already identified this image earlier in the conversation (see history). "
    "Answer the user's question conversationally and concisely, grounded in what "
    "is actually visible in the image and consistent with your earlier "
    "identification. Write normal prose — do NOT return JSON. When the user says "
    "\"this\", \"it\", or \"the image\", they mean the uploaded image. Scope: bone "
    "imaging only; if asked something outside that, say so briefly. For research "
    "and education only — not a clinical diagnosis."
)

VLM_NUM_CTX = 4096  # llava:13b context window — also reported to the UI's CtxRing


def _vlm_stream(img_b64: str, instruction: str, prompt: str, history: list[dict],
                result_out: dict, temperature: float = 0.2):
    """Generator yielding raw token strings from the VLM adapter. `instruction`
    frames the task (structured identification vs conversational answer);
    `prompt` is the user's text. After the iterator is exhausted, result_out
    holds 'full', 'eval_count', and 'prompt_chars'.

    The VLM call is the ONLY model-specific code in the vision loop — keep it
    confined here so the model stays swappable (see the block comment above)."""
    user_content = instruction
    if prompt:
        user_content += f"\n\nUser question: {prompt}"
    messages: list[dict] = []
    # Light continuity: replay prior TEXT turns only (re-sending images each turn
    # is unreliable across VLMs). The current image is attached to this turn.
    for t in history:
        role = t.get("role")
        content = t.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": user_content, "images": [img_b64]})
    result_out["prompt_chars"] = sum(len(m.get("content", "")) for m in messages)

    resp = requests.post(
        OLLAMA_URL,
        json={"model": VLM_MODEL, "messages": messages, "stream": True,
              "options": {"temperature": temperature, "num_ctx": VLM_NUM_CTX}},
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


def _parse_identification(raw: str) -> dict:
    """Parse the VLM's raw output into the structured identification schema.
    Never raises — falls back to a low-confidence 'unclear' identification so a
    malformed VLM response never breaks the stream."""
    parsed = _safe_json(raw) or {}
    feats = parsed.get("features")
    if isinstance(feats, str):
        feats = [feats]
    if not isinstance(feats, list):
        feats = []
    conf = str(parsed.get("confidence", "")).strip().upper()
    if conf not in ("HIGH", "MODERATE", "LOW"):
        conf = "LOW"
    return {
        "identification":  str(parsed.get("identification") or "").strip() or "unclear",
        "tissue_type":     str(parsed.get("tissue_type") or "").strip(),
        "modality":        str(parsed.get("modality") or "").strip() or "unknown",
        "morphology":      str(parsed.get("morphology") or "").strip(),
        "estimated_scale": str(parsed.get("estimated_scale") or "").strip() or "unknown",
        "features":        [str(f).strip() for f in feats if str(f).strip()][:8],
        "confidence":      conf,
        "runner_up":       str(parsed.get("runner_up") or "").strip(),
    }


# Cosine ≥ this means "essentially the same scan" — surface the prior
# correction. Tuned conservatively; see vision/correction_store.py.
VISION_RECALL_THRESHOLD = 0.9


def _vision_recall(pil_img, user_id: str) -> list[dict]:
    """Recall prior corrections for a similar image. Embeds the image with
    BiomedCLIP and cosine-matches against the correction store. Cheap-exits when
    nothing is stored (avoids loading the encoder on a fresh install), and never
    raises — a recall failure must not break the chat stream."""
    try:
        from vision.correction_store import stats, recall
        if stats(user_id=user_id).get("corrections", 0) == 0:
            return []
        from vision import encoder
        q = encoder.embed_image(pil_img)
        return recall(q, threshold=VISION_RECALL_THRESHOLD, top_k=3, user_id=user_id)
    except Exception:
        return []


def _vision_classify(pil_img) -> dict | None:
    """Predict the bone region with the trained BiomedCLIP head — the hybrid's
    grounding signal (docs/vision/training.md). Cheap-exits when the head isn't
    installed (so a fresh install still works), and never raises — a classifier
    failure must not break the chat stream."""
    try:
        from vision import classifier
        if not classifier.is_available():
            return None
        return classifier.predict(pil_img)
    except Exception:
        return None


@app.post("/api/vision/chat")
async def vision_chat(
    image: UploadFile = File(...),
    prompt: str = Form(""),
    history: str = Form("[]"),
    mode: str = Form("deep"),
    user_id: str = Depends(get_current_user),
):
    """SSE stream — vision agent (VLM) over an uploaded image.

    First turn (no prior assistant reply in history) → a structured
    identification card. Follow-up turns → a conversational prose answer, like
    an LLM, with the image + history for context. `mode` is accepted for forward
    compatibility (the critic loop arrives in Phase 2).

    SSE events:
      {"type": "round_start",    "phase": "vision", "round": 1}
      {"type": "token",          "content": "...", "phase": "vision"}  (prose, follow-up turns)
      {"type": "identification", "schema": {...}}                      (first turn only)
      {"type": "done",           "kind": "identify"|"chat",
                                 "identification": {...} | "answer": "...",
                                 "prompt_tokens": N, "completion_tokens": N,
                                 "context_window": N, "mode": "..."}
      {"type": "error",          "message": "..."}
    """
    from PIL import Image as PILImage

    contents = await image.read()
    try:
        pil_img = PILImage.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        def err_gen():
            yield f"data: {json.dumps({'type':'error','message':'Could not read image file.'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    prior_turns: list[dict] = json.loads(history) if history else []
    user_prompt = (prompt or "").strip()
    # A follow-up = the conversation already has an assistant reply. First turn
    # identifies (structured); follow-ups chat (prose).
    is_followup = any(t.get("role") == "assistant" and t.get("content") for t in prior_turns)

    def generate():
        try:
            yield f"data: {json.dumps({'type':'round_start','phase':'vision','round':1})}\n\n"
            res: dict = {}
            instruction = VISION_CHAT_PROMPT if is_followup else VISION_AGENT_PROMPT

            # Correction memory: recall prior 👎 corrections for a similar image
            # and surface them to the VLM as context — the one extra input the
            # agent is allowed (docs/vision/architecture.md, principle 1).
            recalled = _vision_recall(pil_img, user_id)
            if recalled:
                yield f"data: {json.dumps({'type':'recalled','corrections':recalled})}\n\n"
                notes = "\n".join(f"- {h['feedback_text']}" for h in recalled if h.get("feedback_text"))
                if notes:
                    instruction += (
                        "\n\n[Correction memory] A user previously corrected your reading of a "
                        "very similar image:\n" + notes +
                        "\nTreat these corrections as authoritative and do not repeat the mistake."
                    )

            # Trained-classifier grounding: a MURA-trained head predicts the bone
            # region and the prediction is fed to the VLM as a hint (the hybrid,
            # docs/vision/training.md). First turn only — it anchors the initial
            # identification; follow-up turns already have that context.
            if not is_followup:
                prediction = _vision_classify(pil_img)
                if prediction:
                    yield f"data: {json.dumps({'type':'classifier','prediction':prediction})}\n\n"
                    if prediction.get("in_scope", True):
                        conf = round(prediction["confidence"] * 100)
                        topk = prediction.get("topk") or []
                        runner = topk[1] if len(topk) > 1 else None
                        runner_txt = f" Runner-up: {runner[0]} ({round(runner[1]*100)}%)." if runner else ""
                        instruction += (
                            f"\n\n[Region classifier] A classifier trained on MURA upper-limb "
                            f"radiographs predicts this image is: {prediction['label']} "
                            f"({conf}% confidence).{runner_txt} It was trained ONLY on upper-limb "
                            f"X-rays (elbow, finger, forearm, hand, humerus, shoulder, wrist); use "
                            f"it to inform your identification when consistent with the image, but "
                            f"rely on what you actually see — if this is not an upper-limb "
                            f"radiograph, disregard it."
                        )
                    else:
                        # Out-of-distribution: the image is unlike the head's training
                        # data, so its label is withheld (not injected). Keep this note
                        # SCOPE-NEUTRAL — the old "do NOT assume upper-limb radiograph"
                        # wording pushed the VLM toward dismissing bone micro-CT as "not
                        # a bone image". We only need to say the classifier doesn't apply;
                        # the confident-wrong failure came from the injected label, which
                        # is already withheld here.
                        instruction += (
                            "\n\n[Region classifier] No region prediction applies to this "
                            "image — it is outside the classifier's scope (upper-limb "
                            "X-rays), so it was not used. Assess the image on its own merits "
                            "and describe what you actually see."
                        )

            for t in _vlm_stream(img_b64, instruction, user_prompt, prior_turns, res):
                yield f"data: {json.dumps({'type':'token','content':t,'phase':'vision'})}\n\n"
            prompt_tokens = max(1, round(res.get("prompt_chars", 0) / 3.5))
            usage = {"prompt_tokens": prompt_tokens,
                     "completion_tokens": res.get("eval_count", 0),
                     "context_window": VLM_NUM_CTX}
            if is_followup:
                answer = res.get("full", "").strip()
                yield f"data: {json.dumps({'type':'done','kind':'chat','answer':answer, **usage, 'mode':(mode or 'deep')})}\n\n"
            else:
                ident = _parse_identification(res.get("full", ""))
                yield f"data: {json.dumps({'type':'identification','schema':ident})}\n\n"
                rounds = [{"role": "vision", "round": 1, "identification": ident}]
                yield f"data: {json.dumps({'type':'done','kind':'identify','identification':ident,'rounds':rounds, **usage, 'mode':(mode or 'deep')})}\n\n"
        except requests.ConnectionError:
            yield f"data: {json.dumps({'type':'error','message':'Could not connect to Ollama. Run: ollama serve && ollama pull llava:13b'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/vision/feedback")
async def vision_feedback(
    turn_id: str = Form(...),
    polarity: int = Form(...),
    prompt: str = Form(""),
    identification: str = Form(""),
    feedback_text: str = Form(""),
    image: UploadFile = File(None),
    user_id: str = Depends(get_current_user),
):
    """Record a 👍/👎 and, on 👎 with text, store an image-keyed correction.

    The Vision tab has no critic and no rules — correction memory is the only
    feedback surface. A 👎 with a free-text note embeds the image (original +
    augmented copies, via BiomedCLIP) so the correction is recalled the next
    time a similar image shows up. A 👎 with no text records the event only —
    there is nothing to recall.

    Response:
      thumbs-up / empty thumbs-down:
        {"ok": true, "correction_id": null, "stats": {...}}
      thumbs-down with text:
        {"ok": true, "correction_id": N, "n_embeddings": K, "stats": {...}}
    """
    from vision.correction_store import save_event, save_correction, stats

    if polarity not in (-1, 1):
        return {"ok": False, "error": "polarity must be -1 or +1"}

    save_event(turn_id=turn_id, polarity=polarity, user_id=user_id)

    if polarity == 1 or not feedback_text.strip():
        return {"ok": True, "correction_id": None, "stats": stats(user_id=user_id)}

    if image is None:
        return {"ok": False, "error": "image is required to store a correction"}

    from PIL import Image as PILImage
    from vision import encoder

    contents = await image.read()
    try:
        pil_img = PILImage.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return {"ok": False, "error": "could not read image file"}

    vecs, variants = encoder.embed_with_augments(pil_img)
    res = save_correction(
        turn_id=turn_id,
        feedback_text=feedback_text.strip(),
        embeddings=vecs,
        prompt=prompt or "",
        identification=identification or "",
        variants=variants,
        user_id=user_id,
    )
    return {"ok": True, "correction_id": res["id"],
            "n_embeddings": res["n_embeddings"], "stats": stats(user_id=user_id)}


@app.get("/api/vision/corrections")
async def vision_corrections_list(user_id: str = Depends(get_current_user)):
    """List stored corrections (for a management view)."""
    from vision.correction_store import list_corrections, stats
    return {"corrections": list_corrections(user_id=user_id), "stats": stats(user_id=user_id)}


@app.delete("/api/vision/corrections/{correction_id}")
async def vision_correction_delete(correction_id: int, user_id: str = Depends(get_current_user)):
    """Delete a correction and its embeddings (cascade)."""
    from vision.correction_store import delete_correction, stats
    deleted = delete_correction(correction_id, user_id=user_id)
    return {"ok": deleted, "stats": stats(user_id=user_id)}


# ── Mechanics tab — D2IM displacement & strain prediction ─────────────────────
#
# The quantitative counterpart to Vision: where Vision says *what* a slice is,
# Mechanics predicts *how it deforms*. A single forward pass through the D2IM CNN
# (mechanics/d2im.py) turns one undeformed micro-CT slice into displacement and
# axial-strain fields. Unlike the LLM tabs there is nothing to stream — it is one
# compute — so this returns plain JSON.
#
# D2IM is an isolated, swappable adapter: this endpoint depends only on the
# predict/render contract, never on TensorFlow. If TF or the weights are absent,
# /status reports it and /predict returns a clear, actionable error rather than
# crashing — the other three tabs are unaffected.

MECHANICS_DISCLAIMER = (
    "D2IM predictions are a research model's estimate from image content alone, "
    "not a measurement. The bundled model was trained on our own vertebrae dataset, "
    "so it generalises best to similar micro-CT slices. For research and educational "
    "use only — not a clinical tool."
)


@app.get("/api/mechanics/status")
async def mechanics_status(user_id: str = Depends(get_current_user)):
    """Report whether the Mechanics tab can run (TF installed + weights present),
    so the UI can enable the tab or show setup instructions."""
    from mechanics import d2im
    return d2im.status()


@app.post("/api/mechanics/predict")
async def mechanics_predict(
    image: UploadFile = File(...),
    mask: UploadFile = File(None),
    user_id: str = Depends(get_current_user),
):
    """Predict displacement (u, v, w) and axial strain ε_zz for one bone slice.

    `image` is an undeformed micro-CT slice (TIFF/PNG/JPG). `mask` is an optional
    bone-region mask; when omitted, an approximate mask is derived from the scan
    (Otsu) and `mask_source` is reported as 'auto'. Returns a labelled figure
    (base64 PNG) plus summary statistics in physical units.
    """
    from mechanics import d2im

    if not d2im.available():
        st = d2im.status()
        return {"ok": False, "error": st["reason"], "status": st}

    img_bytes = await image.read()
    mask_bytes = await mask.read() if mask is not None else None
    try:
        result = d2im.predict(
            img_bytes, scan_name=image.filename or "",
            mask_bytes=mask_bytes, mask_name=(mask.filename if mask else "") or "",
        )
        figure = d2im.render_figure(result)
    except FileNotFoundError as e:
        return {"ok": False, "error": str(e), "status": d2im.status()}
    except Exception as e:
        return {"ok": False, "error": f"Prediction failed: {e}"}

    return {
        "ok": True,
        "figure": figure,
        "stats": result["stats"],
        "params": result["params"],
        "mask_source": result["mask_source"],
        "model": Path(D2IM_WEIGHTS_PATH).name,
        "disclaimer": MECHANICS_DISCLAIMER,
    }


# ── Public-beta feedback ──────────────────────────────────────────────────────

@app.post("/api/feedback")
async def submit_feedback(
    request: Request,
    message: str = Form(...),
    name: str = Form(...),
    category: str = Form("other"),
    email: str = Form(""),
    tab: str = Form(""),
    page: str = Form(""),
):
    """Store one free-text feedback submission from the beta UI."""
    from api.beta_feedback import save_feedback
    msg = (message or "").strip()
    if not msg:
        raise HTTPException(status_code=422, detail="Feedback message is required.")
    if not (name or "").strip():
        raise HTTPException(status_code=422, detail="Full name is required.")
    fid = save_feedback(
        msg,
        name=name,
        category=category,
        email=email,
        tab=tab,
        page=page,
        user_agent=request.headers.get("user-agent", ""),
    )
    return {"ok": True, "id": fid}


@app.get("/api/feedback/list")
async def list_feedback_admin(token: str = Query("")):
    """Read submitted feedback. Gated by FEEDBACK_ADMIN_TOKEN when that env var
    is set (recommended in production so submitted emails aren't public); open
    locally when it's unset."""
    from api.beta_feedback import list_feedback, count
    admin_token = os.getenv("FEEDBACK_ADMIN_TOKEN", "")
    if admin_token and token != admin_token:
        raise HTTPException(status_code=403, detail="Forbidden")
    return {"count": count(), "feedback": list_feedback()}


@app.get("/admin", response_class=HTMLResponse)
def serve_admin():
    """Server-rendered admin dashboard (signups + beta feedback)."""
    return (FRONTEND_DIR / "admin.html").read_text(encoding="utf-8")


@app.get("/api/admin/overview")
async def admin_overview(token: str = Query("")):
    """Users + feedback for the /admin dashboard. Gated by FEEDBACK_ADMIN_TOKEN
    when that env var is set (recommended in production); open locally when unset."""
    from api.beta_feedback import list_feedback, count as count_feedback
    from api.auth_store import list_users, count_users
    admin_token = os.getenv("FEEDBACK_ADMIN_TOKEN", "")
    if admin_token and token != admin_token:
        raise HTTPException(status_code=403, detail="Forbidden")
    return {
        "users": {"count": count_users(), "items": list_users()},
        "feedback": {"count": count_feedback(), "items": list_feedback()},
    }


@app.delete("/api/admin/feedback/{feedback_id}")
async def admin_delete_feedback(feedback_id: int, token: str = Query("")):
    """Delete one feedback item from the /admin dashboard. Same gate as the overview."""
    from api.beta_feedback import delete_feedback
    admin_token = os.getenv("FEEDBACK_ADMIN_TOKEN", "")
    if admin_token and token != admin_token:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not delete_feedback(feedback_id):
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True, "deleted": feedback_id}
