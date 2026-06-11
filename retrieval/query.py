"""
retrieval/query.py
===================
Interactive CLI for querying the BoneGraph corpus.

Usage
-----
    # Single query
    python -m retrieval.query "cortical bone fracture toughness"

    # Control number of results
    python -m retrieval.query "osteoporosis trabecular microstructure" --top-k 5

    # Interactive mode — keep asking questions without reloading embeddings
    python -m retrieval.query --interactive

The first run takes ~10–15 seconds to load the embeddings into RAM.
Subsequent queries in interactive mode are nearly instant (~50ms each).
"""

from __future__ import annotations

import argparse
import logging
import textwrap

from retrieval.retriever import BoneGraphRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Maximum characters of chunk text to display per result.
PREVIEW_CHARS = 400


def _format_result(r: dict, show_text: bool = True) -> str:
    """Format a single retrieval result for console display."""
    lines = []

    # Header line
    source_label = "📄 Paper" if r["source_type"] == "paper" else "📚 Textbook"
    lines.append(f"\n{'─' * 70}")
    lines.append(f"  #{r['rank']}  Score: {r['score']:.4f}  {source_label}")
    lines.append(f"{'─' * 70}")

    # Document info
    lines.append(f"  Title   : {r['title']}")
    if r["authors"]:
        lines.append(f"  Authors : {r['authors']}")
    if r["year"]:
        lines.append(f"  Year    : {r['year']}")
    if r["venue"]:
        label = "Journal" if r["source_type"] == "paper" else "Source"
        lines.append(f"  {label:<7} : {r['venue']}")
    if r["page_number"]:
        lines.append(f"  Page    : {r['page_number']}  (chunk {r['chunk_index']})")

    # Text preview
    if show_text:
        preview = r["text"][:PREVIEW_CHARS].strip()
        if len(r["text"]) > PREVIEW_CHARS:
            preview += "..."
        lines.append(f"\n  {textwrap.fill(preview, width=68, subsequent_indent='  ')}")

    return "\n".join(lines)


def run_query(retriever: BoneGraphRetriever, query: str, top_k: int) -> None:
    """Execute a query and print formatted results."""
    import time
    t0 = time.time()
    results = retriever.query(query, top_k=top_k)
    elapsed_ms = (time.time() - t0) * 1000

    print(f"\n{'═' * 70}")
    print(f"  Query  : \"{query}\"")
    print(f"  Results: {len(results)}   ({elapsed_ms:.0f}ms)")
    print(f"{'═' * 70}")

    for r in results:
        print(_format_result(r))

    print(f"\n{'─' * 70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the BoneGraph corpus")
    parser.add_argument("query", nargs="?", help="Query string")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results (default 10)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode — ask multiple questions")
    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.print_help()
        return

    # Load retriever once.
    retriever = BoneGraphRetriever()
    retriever.load()

    if args.interactive:
        print("\n" + "═" * 70)
        print("  BoneGraph RAG — Interactive Query Mode")
        print("  Type your question and press Enter. Type 'quit' to exit.")
        print("═" * 70)
        while True:
            try:
                query = input("\n  Query > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Exiting.")
                break
            run_query(retriever, query, top_k=args.top_k)
    else:
        run_query(retriever, args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
