"""
app.py
=======
BoneLogic — Gradio web UI.

Tab 1  Ask BoneLogic   Query → retrieve top-k chunks → stream HuatuoGPT-o1-8B answer
Tab 2  Analyse Image   Upload X-ray/MRI → LLaVA report → cross-modal retrieval → LLM answer
Tab 3  Search Corpus   Raw semantic search (chunks ranked by cosine similarity)
Tab 4  Reason          Graph traversal → physics-validated causal hypotheses + gap detection

Run with:
    python app.py

Opens automatically at http://localhost:7860
Requires Ollama running:
    ollama serve   (in a separate terminal, if not already running as a service)
Models needed:
    huatuogpt-bone   (Tab 1 + Tab 2 LLM answers)
    llava:13b        (Tab 2 image description)
"""

import json
import sqlite3
import time
import requests
import gradio as gr

from retrieval.retriever import BoneLogicRetriever
from reasoning.lrm import LRM
from reasoning.novelty import NoveltyClassifier, CORPUS_DISCLAIMER
from config.settings import PAPERS_DB_PATH, TEXTBOOKS_DB_PATH, CHUNKS_DB_PATH

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "huatuogpt-bone"
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
    "Use precise radiological terminology."
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
determinant of whole-bone stiffness and fracture resistance — greater porosity reduces
the effective cross-sectional area available to resist bending loads [3].

Biomechanical testing in mouse models confirms that cortical geometry and porosity
together govern torsional and bending stiffness; torsion tests are particularly sensitive
to changes in cortical organisation [3].

## References
[2] The effect of 8 or 5 years of denosumab treatment in postmenopausal women with osteoporosis — Papapoulos et al. (2015) · Osteoporosis International
[3] Establishing Biomechanical Mechanisms in Mouse Models — Jepsen et al. (2015) · Journal of Bone and Mineral Research

---

CITATION RULES — enforced strictly:
- Every sentence that states a fact MUST end with [N] before the full stop.
- Use the passage number exactly as given in the context (the number in square brackets at the start of each passage).
- Never cite a passage number that was not provided in the context.
- If two passages support the same claim, cite both: [1][3].
- The ## References section is mandatory, even for short answers."""


# ── Startup: load retriever, LRM, and novelty classifier ─────────────────────
print("Loading BoneLogic retriever...")
retriever = BoneLogicRetriever()
retriever.load()

print("Loading LRM (bone knowledge graph)...")
lrm = LRM()

# Reuse the retriever's already-loaded SPECTER2 model for novelty scoring
# so we don't load the 1.6 GB model a second time.
print("Loading novelty classifier...")
novelty_clf = NoveltyClassifier(
    use_semantic=True,
    model=retriever._model,
    tokenizer=retriever._tokenizer,
    device=retriever._device,
)


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


# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_context(results: list[dict]) -> str:
    """Format retrieved chunks as a numbered context block for the LLM."""
    parts = []
    for r in results:
        label = f"[{r['rank']}]"
        if r["source_type"] == "paper":
            authors = r["authors"] or "Unknown"
            year    = r["year"] or ""
            venue   = r["venue"] or ""
            header  = f"{label} {r['title']} — {authors} ({year}) · {venue}"
        else:
            header = f"{label} {r['title']} [Textbook]"
        parts.append(f"{header}\n{r['text'].strip()}")
    return "\n\n".join(parts)


def _build_sources_html(results: list[dict], collapsible: bool = False) -> str:
    """
    Render retrieved chunks as a styled HTML panel.

    collapsible=True  →  show first 5 lines of text with a Read more / Show less toggle.
                         Uses inline JS so it works inside Gradio's gr.HTML component.
    collapsible=False →  show full text (used by Tab 2 / Search Corpus).
    """
    if not results:
        return "<p style='color:#888'>No relevant passages found.</p>"

    # Characters that roughly fill 3 lines inside the panel (~80 chars/line)
    PREVIEW_CHARS = 240

    html = ""
    for r in results:
        is_tb      = r["source_type"] == "textbook"
        border     = "#2ca02c" if is_tb else "#1f77b4"
        badge_bg   = border
        icon       = "📚" if is_tb else "📄"
        source_lbl = "Textbook" if is_tb else "Paper"

        meta_parts = []
        if r["authors"]:     meta_parts.append(r["authors"])
        if r["year"]:        meta_parts.append(str(r["year"]))
        if r["venue"]:
            venue = r["venue"]
            # Deduplicate "Name/Name" patterns (exact or near-duplicate parts)
            if "/" in venue:
                parts = [p.strip() for p in venue.split("/")]
                seen = [parts[0]]
                for p in parts[1:]:
                    # Skip if this part is contained in or contains an already-seen part
                    norm = p.lower().replace("the ", "").strip()
                    already = any(
                        norm in s.lower().replace("the ", "").strip() or
                        s.lower().replace("the ", "").strip() in norm
                        for s in seen
                    )
                    if not already:
                        seen.append(p)
                venue = " / ".join(seen)
            meta_parts.append(f"<em style='color:green'>{venue}</em>")
        if r["page_number"]: meta_parts.append(f"p.&nbsp;{r['page_number']}")
        meta_line = " &nbsp;·&nbsp; ".join(meta_parts)

        text = r["text"].strip().replace("\n", " ")

        if collapsible and len(text) > PREVIEW_CHARS:
            uid      = f"bl_src_{r['rank']}"
            preview  = text[:PREVIEW_CHARS].rsplit(" ", 1)[0] + "…"
            text_block = f"""
            <div style="background:white; border:1px solid #e0e0e0; border-radius:4px;
                        padding:10px 14px; font-size:0.87em; line-height:1.6; color:#000000 !important;">
                <span id="{uid}_short" style="color:#000000 !important;">{preview}</span>
                <span id="{uid}_full" style="display:none; color:#000000 !important;">{text}</span>
                <br>
                <button id="{uid}_btn"
                    onclick="
                        var s=document.getElementById('{uid}_short');
                        var f=document.getElementById('{uid}_full');
                        var b=document.getElementById('{uid}_btn');
                        if(f.style.display==='none'){{
                            s.style.display='none'; f.style.display='inline'; b.textContent='Show less';
                        }} else {{
                            f.style.display='none'; s.style.display='inline'; b.textContent='Read more';
                        }}"
                    style="margin-top:6px; background:none; border:none; color:#1f77b4;
                           font-size:0.85em; cursor:pointer; padding:0; font-weight:600;">
                    Read more
                </button>
            </div>"""
        else:
            text_block = f"""
            <div style="background:white; border:1px solid #e0e0e0; border-radius:4px;
                        padding:10px 14px; font-size:0.87em; line-height:1.6;
                        color:#000000 !important;">{text}</div>"""

        html += f"""
        <div style="border-left:4px solid {border}; background:#f8f9fa;
                    border-radius:6px; padding:14px 18px; margin-bottom:12px;">
            <div style="margin-bottom:4px">
                <span style="background:{badge_bg}; color:white; border-radius:10px;
                             padding:2px 8px; font-size:0.82em; font-weight:bold;
                             margin-right:8px;">[{r['rank']}] {r['score']:.3f}</span>
                <strong style="color:#000000;">{icon} {r['title']}</strong>
                <span style="color:#666; font-size:0.8em; margin-left:8px">{source_lbl}</span>
            </div>
            <div style="color:green; font-size:0.85em; margin-bottom:8px">{meta_line}</div>
            {text_block}
        </div>
        """
    return html


def _stream_ollama(messages: list[dict]):
    """
    Generator that streams tokens from Ollama.
    Yields str tokens. Raises requests.ConnectionError if Ollama is not running.
    """
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": True},
        stream=True,
        timeout=180,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            data = json.loads(line)
            if not data.get("done"):
                yield data["message"]["content"]


def _inject_ref_links(answer: str, results: list[dict]) -> str:
    """
    Post-process the completed LLM answer to append DOI/OpenAlex links
    to each reference line in the ## References section.

    Matches lines like:  [1] Title — Authors (Year) · Venue
    and appends:         · [Open paper](https://doi.org/...)
    """
    import re
    # Build rank → URL mapping (prefer DOI, fall back to OpenAlex)
    ref_urls: dict[int, str] = {}
    for r in results:
        if r["source_type"] != "paper":
            continue
        if r["doi"]:
            ref_urls[r["rank"]] = f"https://doi.org/{r['doi']}"
        elif r["openalex_id"]:
            ref_urls[r["rank"]] = f"https://openalex.org/{r['openalex_id']}"

    if not ref_urls:
        return answer

    def _replace(m: re.Match) -> str:
        n = int(m.group(1))
        rest = m.group(2)
        url = ref_urls.get(n)
        if url:
            return f"[{n}]{rest} · [Open paper]({url})"
        return m.group(0)

    # Only modify lines inside the References section
    if "## References" not in answer:
        return answer

    body, _, refs = answer.partition("## References")
    refs = re.sub(r"^\[(\d+)\](.*)", _replace, refs, flags=re.MULTILINE)

    # Ensure each [N] reference starts on its own line with a blank line before it
    # so markdown renders them as separate paragraphs, not one collapsed block
    refs = re.sub(r"\n(\[\d+\])", r"\n\n\1", refs)
    refs = refs.lstrip("\n")

    return body + "## References\n\n" + refs


# ── Tab 1: Ask BoneLogic (RAG + LLM) ─────────────────────────────────────────
def ask(question: str, top_k: int):
    """
    Generator for Gradio streaming.
    Yields (answer_so_far: str, sources_html: str) tuples.
    Sources are emitted immediately after retrieval; answer streams token by token.
    """
    question = question.strip()
    if not question:
        yield "", "<p style='color:#888'>Enter a question above and press Ask.</p>"
        return

    # Retrieve relevant chunks
    results = retriever.query(question, top_k=int(top_k))
    sources_html = _build_sources_html(results, collapsible=True)

    # Show sources immediately, blank answer while LLM warms up
    yield "_Thinking…_", sources_html

    # Build LLM messages
    context = _build_context(results)
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": f"Context passages:\n\n{context}\n\n---\n\nQuestion: {question}"},
    ]

    # Stream answer
    answer = ""
    try:
        for token in _stream_ollama(messages):
            answer += token
            yield answer, sources_html
        # Inject DOI links into references once streaming is complete
        linked = _inject_ref_links(answer, results)
        if linked != answer:
            yield linked, sources_html
    except requests.ConnectionError:
        yield (
            "⚠️ **Could not connect to Ollama.**\n\n"
            "Make sure the model is running:\n"
            "```\nollama serve\n```\n"
            "And the model is loaded:\n"
            "```\nollama run huatuogpt-bone\n```",
            sources_html,
        )
    except requests.HTTPError as e:
        yield f"⚠️ **Ollama error:** {e}", sources_html
    except Exception as e:
        yield f"⚠️ **Unexpected error:** {e}", sources_html


# ── Tab 2: Analyse Image (VLM only) ──────────────────────────────────────────
def _describe_image(image_array) -> str:
    """
    Send an image (numpy H×W×3 array from gr.Image) to LLaVA via Ollama
    and return the radiological report. Streams internally, returns full text.
    """
    import base64
    import io
    from PIL import Image as PILImage
    import numpy as np

    # Convert numpy array → PNG bytes → base64
    pil_img = PILImage.fromarray(image_array.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": VLM_MODEL,
            "messages": [{"role": "user", "content": VLM_PROMPT, "images": [img_b64]}],
            "stream": True,
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    report = ""
    for line in resp.iter_lines():
        if line:
            data = json.loads(line)
            if not data.get("done"):
                report += data["message"]["content"]
    return report


def analyse_image(image_array):
    """Generator for Tab 2 — yields VLM report text as it streams."""
    if image_array is None:
        yield "_Upload an image first._"
        return

    yield "_Analysing image…_"
    try:
        report = _describe_image(image_array)
    except requests.ConnectionError:
        yield "⚠️ **Could not connect to Ollama.**\n\nRun `ollama serve` and make sure `llava:13b` is pulled."
        return
    except Exception as e:
        yield f"⚠️ **VLM error:** {e}"
        return

    yield report


# ── Tab 3: Search Corpus (raw retrieval) ──────────────────────────────────────
def search(query: str, top_k: int, source_filter: str) -> str:
    if not query.strip():
        return "<p style='color:#888'>Enter a query above and press Search.</p>"

    t0      = time.time()
    results = retriever.query(query.strip(), top_k=int(top_k))
    elapsed = (time.time() - t0) * 1000

    if source_filter == "Papers only":
        results = [r for r in results if r["source_type"] == "paper"]
    elif source_filter == "Textbooks only":
        results = [r for r in results if r["source_type"] == "textbook"]

    if not results:
        return "<p style='color:#888'>No results. Try changing the source filter.</p>"

    html = (
        f"<p style='color:#555; margin-bottom:16px'>"
        f"<strong>{len(results)} results</strong> &nbsp;·&nbsp; {elapsed:.0f}ms</p>"
    )
    html += _build_sources_html(results)
    return html


# ── Tab 4: Reason — hypothesis rendering helpers ─────────────────────────────

_PHYSICS_COLORS = {
    "PLAUSIBLE":   ("#059669", "#D1FAE5"),   # green  (text, bg)
    "IMPLAUSIBLE": ("#DC2626", "#FEE2E2"),   # red
    "UNCERTAIN":   ("#6B7280", "#F3F4F6"),   # grey
}
_NOVELTY_COLORS = {
    "GROUNDED":    ("#1D4ED8", "#DBEAFE"),   # blue
    "SPECULATIVE": ("#D97706", "#FEF3C7"),   # amber
    "NOVEL":       ("#7C3AED", "#EDE9FE"),   # purple
    "UNCERTAIN":   ("#6B7280", "#F3F4F6"),   # grey
}


def _badge(label: str, color_map: dict) -> str:
    """Render a small colored pill badge."""
    text_col, bg_col = color_map.get(label, ("#374151", "#F9FAFB"))
    return (
        f"<span style='background:{bg_col}; color:{text_col}; "
        f"border:1px solid {text_col}33; border-radius:12px; "
        f"padding:2px 10px; font-size:0.78em; font-weight:700; "
        f"letter-spacing:0.04em;'>{label}</span>"
    )


def _chain_html(nodes: list[str], graph) -> str:
    """Render a causal chain as styled node pills with arrows between them."""
    parts = []
    for nid in nodes:
        node = graph.get_node(nid)
        label = node.label if node else nid.replace("_", " ")
        ntype = node.node_type if node else "concept"
        # Soft background per node type
        _TYPE_BG = {
            "structure": "#EFF6FF", "property": "#F0FDF4",
            "process":   "#FFF7ED", "pathology": "#FFF1F2",
            "factor":    "#FAF5FF", "cell":      "#F0FDFA",
            "mechanism": "#FFFBEB", "clinical":  "#F8FAFC",
            "material":  "#F0FDF4", "scale":     "#F8FAFC",
            "concept":   "#F9FAFB",
        }
        bg = _TYPE_BG.get(ntype, "#F9FAFB")
        parts.append(
            f"<span style='background:{bg}; border:1px solid #D1D5DB; "
            f"border-radius:6px; padding:3px 10px; font-size:0.88em; "
            f"font-weight:600; color:#111827;'>{label}</span>"
        )
    arrow = "<span style='color:#9CA3AF; font-size:1em; margin:0 4px;'>→</span>"
    return arrow.join(parts)


def _render_hypotheses(hypotheses, graph) -> str:
    """Render a list of HypothesisResult objects as styled HTML cards."""
    if not hypotheses:
        return (
            "<div style='color:#6B7280; padding:24px; text-align:center;'>"
            "No hypotheses found. Try a different query or broaden your terms."
            "</div>"
        )

    html = ""
    for i, h in enumerate(hypotheses, 1):
        nr = novelty_clf.classify(h)

        p_badge = _badge(h.physics.status, _PHYSICS_COLORS)
        n_badge = _badge(nr.label, _NOVELTY_COLORS)
        chain   = _chain_html(h.chain, graph)
        rels    = " &nbsp;|&nbsp; ".join(
            f"<em style='color:#6B7280'>{e.relation}</em>" for e in h.edges
        )

        disclaimer = ""
        if nr.show_disclaimer:
            disclaimer = (
                f"<div style='background:#FFFBEB; border:1px solid #FCD34D; "
                f"border-radius:6px; padding:8px 12px; margin-top:10px; "
                f"font-size:0.82em; color:#92400E;'>"
                f"⚠️ {CORPUS_DISCLAIMER}</div>"
            )

        physics_note = ""
        if h.physics.law:
            physics_note = (
                f"<span style='color:#6B7280; font-size:0.82em;'>"
                f"&nbsp;({h.physics.law})</span>"
            )

        novelty_note = (
            f"<span style='color:#6B7280; font-size:0.82em;'>"
            f"&nbsp;{nr.explanation[:90]}{'…' if len(nr.explanation) > 90 else ''}"
            f"</span>"
        )

        html += f"""
        <div style='border:1px solid #E5E7EB; border-radius:10px;
                    padding:18px 22px; margin-bottom:14px;
                    background:#FAFAFA; box-shadow:0 1px 3px rgba(0,0,0,0.06);'>

            <div style='font-size:0.78em; color:#9CA3AF; margin-bottom:8px;
                        font-weight:600; letter-spacing:0.05em;'>
                HYPOTHESIS {i} &nbsp;·&nbsp; score {h.score:.3f}
            </div>

            <div style='margin-bottom:10px; line-height:2;'>
                {chain}
            </div>

            <div style='font-size:0.83em; color:#6B7280; margin-bottom:12px;'>
                {rels}
            </div>

            <div style='display:flex; gap:8px; align-items:center;
                        flex-wrap:wrap; margin-bottom:10px;'>
                {p_badge}{physics_note}
                &nbsp;&nbsp;
                {n_badge}{novelty_note}
            </div>

            <div style='font-size:0.9em; color:#374151; line-height:1.6;
                        border-top:1px solid #F3F4F6; padding-top:10px;'>
                {h.summary}
            </div>

            {disclaimer}
        </div>
        """
    return html


def _render_gaps(gaps) -> str:
    """Render research gap results as a ranked HTML table."""
    if not gaps:
        return "<p style='color:#6B7280'>No gaps found.</p>"

    rows = ""
    for i, g in enumerate(gaps, 1):
        rows += (
            f"<tr style='border-bottom:1px solid #F3F4F6;'>"
            f"<td style='padding:8px 12px; color:#6B7280; font-size:0.85em;'>{i}</td>"
            f"<td style='padding:8px 12px; font-weight:600; color:#111827;'>{g.label}</td>"
            f"<td style='padding:8px 12px; color:#6B7280; font-size:0.85em;'>{g.node_type}</td>"
            f"<td style='padding:8px 12px; font-size:0.85em;'>"
            f"<span style='background:#EDE9FE; color:#7C3AED; border-radius:8px; "
            f"padding:2px 8px;'>{g.betweenness:.4f}</span></td>"
            f"<td style='padding:8px 12px; color:#6B7280; font-size:0.85em;'>{g.n_edges}</td>"
            f"<td style='padding:8px 12px; font-size:0.85em;'>"
            f"<strong style='color:#7C3AED;'>{g.gap_score:.4f}</strong></td>"
            f"</tr>"
        )
    return (
        f"<table style='width:100%; border-collapse:collapse; "
        f"font-size:0.9em; background:white;'>"
        f"<thead><tr style='background:#F9FAFB; border-bottom:2px solid #E5E7EB;'>"
        f"<th style='padding:8px 12px; text-align:left; color:#6B7280;'>#</th>"
        f"<th style='padding:8px 12px; text-align:left; color:#6B7280;'>Concept</th>"
        f"<th style='padding:8px 12px; text-align:left; color:#6B7280;'>Type</th>"
        f"<th style='padding:8px 12px; text-align:left; color:#6B7280;'>Betweenness</th>"
        f"<th style='padding:8px 12px; text-align:left; color:#6B7280;'>Edges</th>"
        f"<th style='padding:8px 12px; text-align:left; color:#6B7280;'>Gap score</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def reason(query: str, max_results: int, physics_filter: bool):
    """
    Generator for the Reason tab.
    Yields (hypotheses_html, gaps_html, graph_stats_md) tuples.
    """
    query = query.strip()
    if not query:
        yield (
            "<p style='color:#6B7280; padding:20px;'>Enter a concept or question above.</p>",
            "",
            "",
        )
        return

    yield (
        "<p style='color:#6B7280; padding:20px;'>🔬 Traversing knowledge graph…</p>",
        "",
        "",
    )

    try:
        lrm.physics_filter = physics_filter
        hypotheses = lrm.query(query, max_results=int(max_results))
        hyp_html   = _render_hypotheses(hypotheses, lrm._graph)

        s = lrm.graph_stats()
        stats_md = (
            f"**Graph:** {s['n_nodes']:,} nodes · {s['n_edges']:,} edges · "
            f"{s['n_components']} components · "
            f"largest component: {s['giant_component']} nodes"
        )

        yield hyp_html, "", stats_md

    except Exception as e:
        yield f"<p style='color:#DC2626'>⚠️ Error: {e}</p>", "", ""


def find_gaps(top_n: int):
    """Return rendered gap detection results."""
    try:
        gaps     = lrm.find_gaps(top_n=int(top_n))
        gaps_html = _render_gaps(gaps)
        return gaps_html
    except Exception as e:
        return f"<p style='color:#DC2626'>⚠️ Error: {e}</p>"


def reload_graph():
    """Reload the knowledge graph from ontology.db (picks up new extractions)."""
    lrm.reload()
    s = lrm.graph_stats()
    return (
        f"✅ Graph reloaded: **{s['n_nodes']:,} nodes · {s['n_edges']:,} edges** "
        f"({s['n_components']} components)"
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────
_CSS = """
.answer-markdown { padding-top: 18px !important; }
"""

with gr.Blocks(title="BoneLogic", theme=gr.themes.Soft(), css=_CSS) as demo:

    gr.Markdown(
        f"# 🦴 BoneLogic\n"
        f"Intelligent reasoning system for bone science · "
        f"{STATS['pdfs_downloaded']:,} papers · "
        f"{STATS['textbooks']} textbooks · "
        f"{STATS['chunks']:,} chunks · SPECTER2 + HuatuoGPT-o1-8B"
    )

    # ── Tab 1 ──────────────────────────────────────────────────────────────────
    with gr.Tab("Ask BoneLogic"):
        gr.Markdown(
            "Ask a question about bone science. "
            "The system retrieves the most relevant passages from the corpus and "
            "streams a reasoned answer grounded in that evidence."
        )

        with gr.Row():
            with gr.Column(scale=5):
                q_box = gr.Textbox(
                    placeholder="e.g. How does cortical porosity affect fracture toughness in osteoporotic bone?",
                    label="Question",
                    lines=2,
                )
            with gr.Column(scale=1, min_width=120):
                ask_btn = gr.Button("🧠 Ask", variant="primary", scale=1)

        with gr.Row():
            ask_top_k = gr.Slider(
                minimum=3, maximum=20, value=8, step=1,
                label="Passages retrieved (top-k)",
            )

        with gr.Row():
            with gr.Column(scale=3):
                answer_md = gr.Markdown(
                    value="_Enter a question and press Ask._",
                    label="Answer",
                    elem_classes=["answer-markdown"],
                )
            with gr.Column(scale=2):
                sources_html = gr.HTML(
                    value="<p style='color:#888'>Sources will appear here after retrieval.</p>",
                    label="Retrieved passages",
                )

        # Example questions
        gr.Markdown("#### Example questions:")
        with gr.Row():
            ex_questions = [
                "How does trabecular architecture change in osteoporosis?",
                "What determines fracture toughness in cortical bone?",
                "How does hydroxyapatite crystallinity affect scaffold performance?",
                "What is the role of osteocytes in mechanosensing?",
                "How does bone remodelling respond to fatigue loading?",
            ]
            for eq in ex_questions:
                gr.Button(eq, size="sm").click(
                    fn=lambda q=eq: q,
                    outputs=q_box,
                )

        # Wire streaming
        ask_btn.click(
            fn=ask,
            inputs=[q_box, ask_top_k],
            outputs=[answer_md, sources_html],
        )
        q_box.submit(
            fn=ask,
            inputs=[q_box, ask_top_k],
            outputs=[answer_md, sources_html],
        )

    # ── Tab 2 ──────────────────────────────────────────────────────────────────
    with gr.Tab("Analyse Image"):
        gr.Markdown(
            "Upload a bone X-ray or MRI. "
            "LLaVA 13b describes the image findings."
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_upload = gr.Image(
                    type="numpy",
                    label="Upload X-ray / MRI",
                )
            with gr.Column(scale=1):
                img_btn = gr.Button("🔬 Analyse", variant="primary")

        vlm_report_md = gr.Markdown(
            value="_Upload an image and press Analyse._",
            label="VLM Report",
            elem_classes=["answer-markdown"],
        )

        img_btn.click(
            fn=analyse_image,
            inputs=[img_upload],
            outputs=[vlm_report_md],
        )

    # ── Tab 3 ──────────────────────────────────────────────────────────────────
    with gr.Tab("Search Corpus"):
        gr.Markdown(
            "Raw semantic search — returns the top-k passages ranked by cosine similarity "
            "to your query (SPECTER2 embeddings). No LLM involved."
        )

        with gr.Row():
            with gr.Column(scale=4):
                s_box = gr.Textbox(
                    placeholder="e.g. cortical bone fracture toughness, osteoblast differentiation...",
                    label="Query",
                    lines=1,
                )
            with gr.Column(scale=1):
                search_btn = gr.Button("🔍 Search", variant="primary")

        with gr.Row():
            s_top_k = gr.Slider(minimum=3, maximum=30, value=10, step=1,
                                label="Number of results")
            s_filter = gr.Radio(
                choices=["All", "Papers only", "Textbooks only"],
                value="All", label="Source filter",
            )

        s_results = gr.HTML(
            value="<p style='color:#888'>Enter a query above and press Search.</p>"
        )

        with gr.Row():
            s_examples = [
                "cortical bone fracture toughness",
                "osteoporosis trabecular microstructure",
                "bone scaffold hydroxyapatite tissue engineering",
                "finite element model femur stress",
                "osteoblast osteoclast bone remodelling",
                "vertebral biomechanics spine loading",
            ]
            for ex in s_examples:
                gr.Button(f"🔎 {ex}", size="sm").click(
                    fn=lambda q=ex: (q, search(q, 10, "All")),
                    outputs=[s_box, s_results],
                )

        search_btn.click(fn=search, inputs=[s_box, s_top_k, s_filter], outputs=s_results)
        s_box.submit(fn=search, inputs=[s_box, s_top_k, s_filter], outputs=s_results)

    # ── Tab 4 ──────────────────────────────────────────────────────────────────
    with gr.Tab("🔬 Reason"):
        gr.Markdown(
            "Traverse the **bone knowledge graph** to generate causal hypothesis chains. "
            "Each hypothesis is validated by the physics engine and scored for novelty "
            "against the full corpus."
        )

        with gr.Row():
            with gr.Column(scale=5):
                r_query = gr.Textbox(
                    placeholder="e.g. aging and fracture risk, collagen toughness, RANKL osteoclast",
                    label="Concept or question",
                    lines=1,
                )
            with gr.Column(scale=1, min_width=130):
                r_btn = gr.Button("🔬 Reason", variant="primary")

        with gr.Row():
            r_max = gr.Slider(
                minimum=1, maximum=15, value=8, step=1,
                label="Max hypotheses",
            )
            r_physics = gr.Checkbox(
                value=True,
                label="Filter out IMPLAUSIBLE chains",
            )
            r_reload = gr.Button("↺ Reload graph", size="sm")

        r_stats_md = gr.Markdown(value="", label="")

        r_results = gr.HTML(
            value="<p style='color:#6B7280; padding:20px;'>"
                  "Enter a concept above and press Reason.</p>"
        )

        # ── Example queries ───────────────────────────────────────────────────
        gr.Markdown("#### Example queries:")
        with gr.Row():
            _r_examples = [
                "aging fracture risk",
                "collagen bone toughness",
                "RANKL osteoclast resorption",
                "porosity elastic modulus",
                "osteoporosis bone mineral density",
            ]
            for ex in _r_examples:
                gr.Button(ex, size="sm").click(fn=lambda q=ex: q, outputs=r_query)

        # ── Research gap detection ─────────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown(
            "#### 🕳️ Research Gap Detection\n"
            "Identifies concepts that sit on many causal paths in the graph "
            "but have few direct connections — candidate knowledge gaps."
        )
        with gr.Row():
            gap_top_n = gr.Slider(minimum=5, maximum=20, value=10, step=1,
                                  label="Number of gaps to show")
            gap_btn   = gr.Button("Find Gaps", variant="secondary")

        gap_results = gr.HTML(
            value="<p style='color:#6B7280'>Press Find Gaps to detect research gaps.</p>"
        )

        # ── Wire ───────────────────────────────────────────────────────────────
        r_btn.click(
            fn=reason,
            inputs=[r_query, r_max, r_physics],
            outputs=[r_results, gap_results, r_stats_md],
        )
        r_query.submit(
            fn=reason,
            inputs=[r_query, r_max, r_physics],
            outputs=[r_results, gap_results, r_stats_md],
        )
        r_reload.click(
            fn=reload_graph,
            outputs=r_stats_md,
        )
        gap_btn.click(
            fn=find_gaps,
            inputs=[gap_top_n],
            outputs=gap_results,
        )

    gr.Markdown(
        f"---\n"
        f"**Corpus:** {STATS['pdfs_downloaded']:,} papers (full text) · "
        f"{STATS['papers_total']:,} papers (metadata) · "
        f"{STATS['textbooks']} textbooks · "
        f"{STATS['chunks']:,} chunks · "
        f"SPECTER2 768-dim embeddings · HuatuoGPT-o1-8B via Ollama"
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
