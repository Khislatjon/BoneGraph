"""
scripts/filter_english.py
==========================
Detects the language of each paper using its abstract (falling back to
the title if no abstract is available), marks non-English papers in
papers.db, and removes their chunks from chunks.db.

Why abstract-based detection?
------------------------------
The abstract is already stored in papers.db — no need to read PDF files.
A 100–300 word abstract gives langdetect enough signal to identify the
language accurately. The title alone can be ambiguous (e.g. single-word
titles or proper nouns), so we fall back to it only when the abstract is
missing.

What gets changed
-----------------
1. A `language` column is added to papers.db (e.g. "en", "de", "fr").
2. A `language_filtered` column is set to 1 for non-English papers.
3. Chunks for non-English papers are deleted from chunks.db.
4. Text files for non-English papers are NOT deleted — only chunks are
   removed so the filtering is reversible.

Usage
-----
    # Dry run — show what would be filtered without changing anything
    python scripts/filter_english.py --dry-run

    # Apply the filter
    python scripts/filter_english.py
"""

from __future__ import annotations

import argparse
import logging
import sqlite3

from langdetect import detect, LangDetectException

from config.settings import PAPERS_DB_PATH, CHUNKS_DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Minimum characters of text needed to attempt language detection.
# Very short strings give unreliable results.
MIN_TEXT_LENGTH = 50


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Add language and language_filtered columns if they don't exist."""
    cur = conn.execute("PRAGMA table_info(papers)")
    existing = {row[1] for row in cur.fetchall()}

    if "language" not in existing:
        conn.execute("ALTER TABLE papers ADD COLUMN language TEXT")
        logger.info("Added 'language' column to papers table")

    if "language_filtered" not in existing:
        conn.execute("ALTER TABLE papers ADD COLUMN language_filtered INTEGER DEFAULT 0")
        logger.info("Added 'language_filtered' column to papers table")

    conn.commit()


def detect_language(text: str) -> str | None:
    """
    Detect the language of a text string.

    Returns a 2-letter ISO language code (e.g. 'en', 'de', 'fr')
    or None if detection fails or text is too short.
    """
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return None
    try:
        return detect(text.strip())
    except LangDetectException:
        return None


def run(dry_run: bool = False) -> None:
    conn = sqlite3.connect(PAPERS_DB_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_columns(conn)

    # Fetch all papers that have not been language-checked yet.
    rows = conn.execute(
        """SELECT paper_id, title, abstract
           FROM papers
           WHERE language IS NULL
           AND (abstract IS NOT NULL OR title IS NOT NULL)"""
    ).fetchall()

    total = len(rows)
    logger.info("Papers to check: %d", total)

    counts = {"en": 0, "non_en": 0, "unknown": 0}
    non_english_ids = []

    for idx, row in enumerate(rows, start=1):
        paper_id = row["paper_id"]

        # Use abstract if available, otherwise title.
        text = row["abstract"] or row["title"] or ""
        lang = detect_language(text)

        if lang is None:
            counts["unknown"] += 1
            lang_value = "unknown"
            filtered = 0
        elif lang == "en":
            counts["en"] += 1
            lang_value = "en"
            filtered = 0
        else:
            counts["non_en"] += 1
            lang_value = lang
            filtered = 1
            non_english_ids.append(paper_id)

        if not dry_run:
            conn.execute(
                "UPDATE papers SET language = ?, language_filtered = ? WHERE paper_id = ?",
                (lang_value, filtered, paper_id),
            )

        if idx % 1000 == 0 or idx == total:
            logger.info(
                "[%d/%d]  en: %d  non-en: %d  unknown: %d",
                idx, total, counts["en"], counts["non_en"], counts["unknown"],
            )

    if not dry_run:
        conn.commit()

    conn.close()

    # ── Remove chunks for non-English papers ──────────────────────────────────
    if non_english_ids and not dry_run:
        logger.info("Removing chunks for %d non-English papers...", len(non_english_ids))
        chunk_conn = sqlite3.connect(CHUNKS_DB_PATH)
        chunk_conn.executemany(
            "DELETE FROM chunks WHERE source_type = 'paper' AND source_id = ?",
            [(pid,) for pid in non_english_ids],
        )
        chunk_conn.commit()
        deleted_chunks = chunk_conn.execute(
            "SELECT changes()"
        ).fetchone()[0]
        chunk_conn.close()
        logger.info("Deleted chunks for non-English papers.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  LANGUAGE FILTER" + ("  [DRY RUN]" if dry_run else "  COMPLETE"))
    print("=" * 55)
    print(f"  English         : {counts['en']:,}")
    print(f"  Non-English     : {counts['non_en']:,}")
    print(f"  Unknown         : {counts['unknown']:,}  (kept — too short to detect)")
    print(f"  Non-English IDs : {len(non_english_ids):,} papers")
    if not dry_run and non_english_ids:
        print(f"  Chunks removed  : from {len(non_english_ids):,} papers")
    print("=" * 55)

    if dry_run and non_english_ids:
        print("\nSample non-English papers that would be filtered:")
        conn2 = sqlite3.connect(PAPERS_DB_PATH)
        conn2.row_factory = sqlite3.Row
        placeholders = ",".join("?" * min(10, len(non_english_ids)))
        samples = conn2.execute(
            f"SELECT title, language FROM papers WHERE paper_id IN ({placeholders})",
            non_english_ids[:10],
        ).fetchall()
        conn2.close()
        for s in samples:
            print(f"  [{s['language']}] {s['title']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter non-English papers from the corpus")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be filtered without making any changes"
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
