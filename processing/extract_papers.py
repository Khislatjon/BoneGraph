"""
processing/extract_papers.py
==============================
Extracts text from all downloaded paper PDFs and tracks extraction status
in the papers SQLite database.

Usage
-----
    # Extract all papers that haven't been extracted yet
    python -m processing.extract_papers

    # Re-extract everything (including previously extracted)
    python -m processing.extract_papers --force

What this script does
---------------------
1. Queries the papers DB for all rows where pdf_local_path IS NOT NULL
   (i.e. PDFs that have been downloaded).
2. Skips papers already extracted (text_path IS NOT NULL) unless --force.
3. For each paper: calls extract_text_from_pdf() from extractor.py.
4. Writes the .txt file to data/processed/text/papers/<paper_id>.txt.
5. Updates the papers DB with the text_path and extraction_status.

Output file naming
------------------
Files are named <paper_id>.txt with "/" replaced by "_" to be filesystem-safe.
They are stored flat in data/processed/text/papers/ (no year subfolders) since
text files are small and fast to scan.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3

from config.settings import PAPERS_DB_PATH, PROCESSED_DIR
from processing.extractor import extract_text_from_pdf
from pathlib import Path

logger = logging.getLogger(__name__)

# Where extracted paper text files are written.
PAPERS_TEXT_DIR = PROCESSED_DIR / "text" / "papers"


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """
    Add text_path and extraction_status columns to the papers table if they
    don't exist yet.  This is safe to run on an existing database — SQLite
    ignores "ALTER TABLE ADD COLUMN" if the column already exists... actually
    it raises an error, so we check first.
    """
    cur = conn.execute("PRAGMA table_info(papers)")
    existing_columns = {row[1] for row in cur.fetchall()}

    if "text_path" not in existing_columns:
        conn.execute("ALTER TABLE papers ADD COLUMN text_path TEXT")
        logger.info("Added text_path column to papers table")

    if "extraction_status" not in existing_columns:
        conn.execute("ALTER TABLE papers ADD COLUMN extraction_status TEXT")
        logger.info("Added extraction_status column to papers table")

    conn.commit()


def run(force: bool = False) -> None:
    """
    Extract text from all downloaded paper PDFs.

    Parameters
    ----------
    force : bool
        If True, re-extract papers even if already extracted.
        If False (default), skip papers that already have a text_path.
    """
    PAPERS_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(PAPERS_DB_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_columns(conn)

    # Fetch papers that have a downloaded PDF.
    if force:
        cur = conn.execute(
            "SELECT paper_id, pdf_local_path FROM papers WHERE pdf_local_path IS NOT NULL"
        )
    else:
        # Skip already-extracted papers to allow resuming interrupted runs.
        cur = conn.execute(
            """SELECT paper_id, pdf_local_path FROM papers
               WHERE pdf_local_path IS NOT NULL
               AND (text_path IS NULL OR extraction_status = 'failed')"""
        )

    papers = cur.fetchall()
    total = len(papers)
    logger.info("Papers to extract: %d", total)

    counts = {"extracted": 0, "scanned": 0, "failed": 0, "skipped": 0}

    for idx, row in enumerate(papers, start=1):
        paper_id = row["paper_id"]
        pdf_path = Path(row["pdf_local_path"])

        if not pdf_path.exists():
            logger.warning("[%d/%d] PDF not found on disk: %s", idx, total, pdf_path)
            counts["skipped"] += 1
            continue

        # Output path: flat directory, paper_id as filename.
        safe_id = paper_id.replace("/", "_")
        out_path = PAPERS_TEXT_DIR / f"{safe_id}.txt"

        if idx % 100 == 0 or idx == total:
            logger.info(
                "[%d/%d] Extracting %s...", idx, total, paper_id[:20]
            )

        result = extract_text_from_pdf(pdf_path, out_path)
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1

        # Write results back to the database.
        conn.execute(
            """UPDATE papers
               SET text_path = ?, extraction_status = ?, updated_at = datetime('now')
               WHERE paper_id = ?""",
            (
                str(out_path) if status == "extracted" else None,
                status,
                paper_id,
            ),
        )

        # Commit every 100 rows to avoid losing progress on interruption.
        if idx % 100 == 0:
            conn.commit()

    conn.commit()
    conn.close()

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  PAPER TEXT EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"  Extracted : {counts.get('extracted', 0):,}")
    print(f"  Scanned   : {counts.get('scanned', 0):,}  (no text layer)")
    print(f"  Failed    : {counts.get('failed', 0):,}")
    print(f"  Skipped   : {counts.get('skipped', 0):,}  (PDF missing)")
    print("=" * 50)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Extract text from paper PDFs")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract papers that have already been processed",
    )
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
