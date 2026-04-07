"""
processing/extract_textbooks.py
================================
Extracts text from all textbook PDFs and tracks extraction status
in the textbooks SQLite database.

Usage
-----
    # Extract all textbooks
    python -m processing.extract_textbooks

    # Re-extract everything
    python -m processing.extract_textbooks --force

Output files are written to:
    data/processed/text/textbooks/<source>/<filename>.txt

The subfolder per source (e.g. "MDPI Books", "OpenStax") mirrors the raw
textbooks folder structure, making it easy to browse by source.
"""

from __future__ import annotations

import argparse
import logging

from config.settings import RAW_TEXTBOOKS_DIR, TEXTBOOKS_DB_PATH, PROCESSED_DIR
from ingestion.textbooks.storage import TextbookStore
from processing.extractor import extract_text_from_pdf
from pathlib import Path

logger = logging.getLogger(__name__)

TEXTBOOKS_TEXT_DIR = PROCESSED_DIR / "text" / "textbooks"


def _ensure_columns(store: TextbookStore) -> None:
    """Add text_path and extraction_status columns if they don't exist."""
    cur = store._conn.execute("PRAGMA table_info(textbooks)")
    existing = {row[1] for row in cur.fetchall()}

    if "text_path" not in existing:
        store._conn.execute("ALTER TABLE textbooks ADD COLUMN text_path TEXT")
        logger.info("Added text_path column to textbooks table")

    if "extraction_status" not in existing:
        store._conn.execute(
            "ALTER TABLE textbooks ADD COLUMN extraction_status TEXT"
        )
        logger.info("Added extraction_status column to textbooks table")

    store._conn.commit()


def run(force: bool = False) -> None:
    """Extract text from all registered textbook PDFs."""
    TEXTBOOKS_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    with TextbookStore() as store:
        _ensure_columns(store)

        if force:
            cur = store._conn.execute("SELECT * FROM textbooks")
        else:
            cur = store._conn.execute(
                """SELECT * FROM textbooks
                   WHERE text_path IS NULL OR extraction_status = 'failed'"""
            )

        books = cur.fetchall()
        total = len(books)
        logger.info("Textbooks to extract: %d", total)

        counts = {"extracted": 0, "scanned": 0, "failed": 0}

        for idx, row in enumerate(books, start=1):
            pdf_path = RAW_TEXTBOOKS_DIR / row["file_path"]
            source = row["source"]
            title = row["title"]

            logger.info("[%d/%d] Extracting: [%s] %s", idx, total, source, title[:50])

            # Mirror source subfolder in output directory.
            safe_name = Path(row["file_path"]).stem
            out_path = TEXTBOOKS_TEXT_DIR / source / f"{safe_name}.txt"

            result = extract_text_from_pdf(pdf_path, out_path)
            status = result["status"]
            counts[status] = counts.get(status, 0) + 1

            store._conn.execute(
                """UPDATE textbooks
                   SET text_path = ?, extraction_status = ?, updated_at = datetime('now')
                   WHERE file_path = ?""",
                (
                    str(out_path) if status == "extracted" else None,
                    status,
                    row["file_path"],
                ),
            )
            store._conn.commit()

            logger.info(
                "  -> %s | %d pages | %s chars",
                status, result["page_count"], f"{result['char_count']:,}"
            )

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  TEXTBOOK TEXT EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"  Extracted : {counts.get('extracted', 0)}")
    print(f"  Scanned   : {counts.get('scanned', 0)}  (no text layer)")
    print(f"  Failed    : {counts.get('failed', 0)}")
    print("=" * 50)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Extract text from textbook PDFs")
    parser.add_argument("--force", action="store_true", help="Re-extract all textbooks")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
