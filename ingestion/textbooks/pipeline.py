"""
ingestion/textbooks/pipeline.py
================================
CLI entrypoint for the textbook ingestion pipeline.

Usage
-----
    # Scan the textbooks folder and register all PDFs in the database
    python -m ingestion.textbooks.pipeline

    # After running, inspect results with:
    python scripts/inspect_textbooks.py

What this script does
---------------------
1. Opens (or creates) the textbooks SQLite database.
2. Scans data/raw/textbooks/ for PDF files.
3. For each PDF: reads page count, file size, derives title and source.
4. Upserts all metadata into the database.
5. Prints a summary table to the console.
"""

from __future__ import annotations

import logging

from ingestion.textbooks.scanner import scan_textbooks
from ingestion.textbooks.storage import TextbookStore

# Configure logging so progress is visible in the terminal.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting textbook ingestion pipeline")

    with TextbookStore() as store:
        counts = scan_textbooks(store)

        # ── Print summary ──────────────────────────────────────────────────────
        stats = store.stats()
        print("\n" + "=" * 60)
        print("  TEXTBOOK INGESTION COMPLETE")
        print("=" * 60)
        print(f"  Registered : {counts['registered']} books")
        print(f"  Errors     : {counts['errors']} books")
        print(f"  Sources    : {stats['sources']}")
        print(f"  Total pages: {stats['total_pages'] or 0:,}")
        print(f"  Total size : {stats['total_size_mb'] or 0:.1f} MB")
        print("=" * 60)

        # ── Print per-book table ───────────────────────────────────────────────
        print(f"\n{'Source':<25} {'Title':<50} {'Pages':>6} {'MB':>6}")
        print("-" * 92)
        for row in store.all_textbooks():
            title_short = row['title'][:48] + '..' if len(row['title']) > 50 else row['title']
            pages = row['page_count'] or '-'
            print(
                f"{row['source']:<25} {title_short:<50} "
                f"{str(pages):>6} {row['file_size_mb']:>6.1f}"
            )


if __name__ == "__main__":
    main()
