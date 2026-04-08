"""
app.py
=======
BoneLogic — Gradio web UI for querying the bone science corpus.

Run with:
    python app.py

Opens automatically in your browser at http://localhost:7860
"""

import sqlite3
import time
import gradio as gr
from retrieval.retriever import BoneLogicRetriever
from config.settings import PAPERS_DB_PATH, TEXTBOOKS_DB_PATH, CHUNKS_DB_PATH


def _load_stats() -> dict:
    """Read live corpus stats from the databases."""
    stats = {}

    conn = sqlite3.connect(PAPERS_DB_PATH)
    stats["pdfs_downloaded"] = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE pdf_local_path IS NOT NULL"
    ).fetchone()[0]
    stats["papers_total"] = conn.execute(
        "SELECT COUNT(*) FROM papers"
    ).fetchone()[0]
    conn.close()

    conn = sqlite3.connect(TEXTBOOKS_DB_PATH)
    stats["textbooks"] = conn.execute(
        "SELECT COUNT(*) FROM textbooks"
    ).fetchone()[0]
    conn.close()

    conn = sqlite3.connect(CHUNKS_DB_PATH)
    stats["chunks"] = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    conn.close()

    return stats


# ── Load retriever and stats once at startup ──────────────────────────────────
print("Loading BoneLogic retriever...")
retriever = BoneLogicRetriever()
retriever.load()
STATS = _load_stats()
print(f"Retriever ready. {STATS['pdfs_downloaded']:,} papers | {STATS['textbooks']} textbooks | {STATS['chunks']:,} chunks")


# ── Query function ────────────────────────────────────────────────────────────
def search(query: str, top_k: int, source_filter: str) -> str:
    """
    Execute a query and return formatted HTML results.
    Called by Gradio on every search button click or Enter press.
    """
    if not query.strip():
        return "<p style='color:#888'>Enter a query above and press Search.</p>"

    t0 = time.time()
    results = retriever.query(query.strip(), top_k=int(top_k))
    elapsed_ms = (time.time() - t0) * 1000

    # Apply source filter
    if source_filter == "Papers only":
        results = [r for r in results if r["source_type"] == "paper"]
    elif source_filter == "Textbooks only":
        results = [r for r in results if r["source_type"] == "textbook"]

    if not results:
        return "<p style='color:#888'>No results found. Try changing the source filter.</p>"

    # Build HTML output
    html = f"<p style='color:#555; margin-bottom:16px'><strong>{len(results)} results</strong> &nbsp;·&nbsp; {elapsed_ms:.0f}ms</p>"

    for r in results:
        is_textbook = r["source_type"] == "textbook"
        border_color = "#2ca02c" if is_textbook else "#1f77b4"
        badge_color  = "#2ca02c" if is_textbook else "#1f77b4"
        icon         = "📚" if is_textbook else "📄"
        source_label = "Textbook" if is_textbook else "Paper"

        # Metadata line
        meta_parts = []
        if r["authors"]:
            meta_parts.append(r["authors"])
        if r["year"]:
            meta_parts.append(str(r["year"]))
        if r["venue"]:
            meta_parts.append(f"<em style='color:#000000'>{r['venue']}</em>")
        if r["page_number"]:
            meta_parts.append(f"p.&nbsp;{r['page_number']}")
        meta_line = " &nbsp;·&nbsp; ".join(meta_parts)

        # Full chunk text — no truncation
        full_text = r["text"].strip().replace("\n", " ")

        html += f"""
        <div style="
            border-left: 4px solid {border_color};
            background: #f8f9fa;
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 14px;
        ">
            <div style="margin-bottom:4px">
                <span style="
                    background:{badge_color}; color:white;
                    border-radius:10px; padding:2px 10px;
                    font-size:0.85em; font-weight:bold;
                    margin-right:8px;
                ">{r['score']:.3f}</span>
                <strong style="color:#000000;">{icon} {r['title']}</strong>
                <span style="color:#555; font-size:0.8em; margin-left:8px">{source_label}</span>
            </div>
            <div style="color:#000000; font-size:0.87em; margin-bottom:8px">{meta_line}</div>
            <div style="
                background:white; border:1px solid #e0e0e0;
                border-radius:4px; padding:10px 14px;
                font-size:0.88em; line-height:1.6; color:#000000;
            ">{full_text}</div>
        </div>
        """

    return html


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="BoneLogic", theme=gr.themes.Soft()) as demo:

    gr.Markdown(f"""
    # 🦴 BoneLogic — Bone Science Knowledge Retrieval
    Search across **{STATS['pdfs_downloaded']:,} downloaded papers** and **{STATS['textbooks']} textbooks** using semantic similarity (SPECTER2).
    Results are ranked by relevance to your query.
    """)

    with gr.Row():
        with gr.Column(scale=4):
            query_box = gr.Textbox(
                placeholder="e.g. cortical bone fracture toughness, osteoporosis trabecular density...",
                label="Query",
                lines=1,
            )
        with gr.Column(scale=1):
            search_btn = gr.Button("🔍 Search", variant="primary")

    with gr.Row():
        top_k_slider = gr.Slider(
            minimum=3, maximum=30, value=10, step=1, label="Number of results"
        )
        source_filter = gr.Radio(
            choices=["All", "Papers only", "Textbooks only"],
            value="All",
            label="Source filter",
        )

    results_box = gr.HTML(
        value="<p style='color:#888'>Enter a query above and press Search.</p>"
    )

    # Example queries
    gr.Markdown("#### Try an example:")
    with gr.Row():
        examples = [
            "cortical bone fracture toughness",
            "osteoporosis trabecular microstructure",
            "bone scaffold hydroxyapatite tissue engineering",
            "finite element model femur stress",
            "osteoblast osteoclast bone remodelling",
            "vertebral bone biomechanics spine",
        ]
        for example in examples:
            gr.Button(f"🔎 {example}", size="sm").click(
                fn=lambda q=example: (q, search(q, 10, "All")),
                outputs=[query_box, results_box],
            )

    # Wire up search button and Enter key
    search_btn.click(
        fn=search,
        inputs=[query_box, top_k_slider, source_filter],
        outputs=results_box,
    )
    query_box.submit(
        fn=search,
        inputs=[query_box, top_k_slider, source_filter],
        outputs=results_box,
    )

    gr.Markdown(f"""
    ---
    **Corpus:** {STATS['pdfs_downloaded']:,} papers (full text) · {STATS['papers_total']:,} papers (metadata) · {STATS['textbooks']} textbooks · {STATS['chunks']:,} chunks · SPECTER2 768-dim embeddings
    """)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
