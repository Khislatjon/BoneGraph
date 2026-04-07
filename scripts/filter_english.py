"""
scripts/filter_english.py
==========================
Detects the language of each paper using its extracted text file (when
available), falling back to the abstract, then the title.

Why text-file-based detection?
--------------------------------
Some papers have an English abstract but a non-English body (e.g. a paper
translated its abstract to English for indexing while the full text is in
German or French).  Detecting language on only the abstract misses these.
Reading the first ~2,000 characters of the extracted .txt file gives a much
more reliable signal about the actual body language.

Detection priority
------------------
1. extracted text file (text_path) — first TEXT_CHARS_TO_READ characters
2. abstract (fallback when no text file exists)
3. title (final fallback)

What gets changed
-----------------
1. A `language` column is added to papers.db (e.g. "en", "de", "fr").
2. A `language_source` column records how detection was done:
   "text_file" | "abstract" | "title" | "unknown"
3. A `language_filtered` column is set to 1 for non-English papers.
4. Chunks for non-English papers are deleted from chunks.db.
5. Extracted .txt files are NOT deleted — filtering is reversible.

Passes
------
Pass A — NEW papers (language IS NULL): detect using best available signal.
Pass B — RE-CHECK papers previously marked 'en' via abstract/title but now
          having a text file available: re-detect from text file.

Usage
-----
    # Dry run — show what would be filtered without changing anything
    python -m scripts.filter_english --dry-run

    # Full run (Pass A + Pass B)
    python -m scripts.filter_english

    # Re-check only (skip Pass A, only re-examine abstract-detected papers)
    python -m scripts.filter_english --recheck-only
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

from langdetect import detect, LangDetectException

from config.settings import PAPERS_DB_PATH, CHUNKS_DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# How many characters to read from the extracted text file for detection.
# ~2,000 chars is enough to get past any translated title/abstract preamble
# and into the actual body language.
TEXT_CHARS_TO_READ = 2000

# Minimum characters needed for a reliable langdetect result.
MIN_TEXT_LENGTH = 50


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Add language, language_source and language_filtered columns if missing."""
    cur = conn.execute("PRAGMA table_info(papers)")
    existing = {row[1] for row in cur.fetchall()}

    if "language" not in existing:
        conn.execute("ALTER TABLE papers ADD COLUMN language TEXT")
        logger.info("Added 'language' column to papers table")

    if "language_source" not in existing:
        conn.execute("ALTER TABLE papers ADD COLUMN language_source TEXT")
        logger.info("Added 'language_source' column to papers table")

    if "language_filtered" not in existing:
        conn.execute(
            "ALTER TABLE papers ADD COLUMN language_filtered INTEGER DEFAULT 0"
        )
        logger.info("Added 'language_filtered' column to papers table")

    conn.commit()


def detect_language(text: str) -> str | None:
    """
    Detect the ISO 639-1 language code of *text*.
    Returns None if text is too short or detection fails.
    """
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return None
    try:
        return detect(text.strip())
    except LangDetectException:
        return None


def _read_text_file(text_path: str | None) -> str:
    """
    Read the first TEXT_CHARS_TO_READ characters from a .txt file.
    Returns an empty string if path is None, missing, or unreadable.
    """
    if not text_path:
        return ""
    try:
        p = Path(text_path)
        if not p.exists():
            return ""
        with p.open("r", encoding="utf-8", errors="ignore") as fh:
            return fh.read(TEXT_CHARS_TO_READ)
    except Exception:
        return ""


def _detect_best(row: sqlite3.Row) -> tuple[str, str]:
    """
    Return (lang_code, source_label) using the best available text signal.
    source_label is one of: "text_file", "abstract", "title", "unknown"
    """
    # 1. Try extracted text file
    text_file_content = _read_text_file(row["text_path"])
    lang = detect_language(text_file_content)
    if lang is not None:
        return lang, "text_file"

    # 2. Try abstract
    lang = detect_language(row["abstract"] or "")
    if lang is not None:
        return lang, "abstract"

    # 3. Try title
    lang = detect_language(row["title"] or "")
    if lang is not None:
        return lang, "title"

    return "unknown", "unknown"


def _delete_chunks(non_english_ids: list, dry_run: bool) -> int:
    """Delete chunks for non-English papers. Returns count of affected papers."""
    if not non_english_ids or dry_run:
        return 0
    chunk_conn = sqlite3.connect(CHUNKS_DB_PATH)
    chunk_conn.executemany(
        "DELETE FROM chunks WHERE source_type = 'paper' AND source_id = ?",
        [(pid,) for pid in non_english_ids],
    )
    chunk_conn.commit()
    chunk_conn.close()
    return len(non_english_ids)


def run_pass_a(conn: sqlite3.Connection, dry_run: bool) -> dict:
    """
    Pass A: Process papers with language IS NULL.
    Uses text_file > abstract > title for detection.
    """
    rows = conn.execute(
        """SELECT paper_id, title, abstract, text_path
           FROM papers
           WHERE language IS NULL
           AND (abstract IS NOT NULL OR title IS NOT NULL OR text_path IS NOT NULL)"""
    ).fetchall()

    total = len(rows)
    logger.info("Pass A — new papers to check: %d", total)

    counts = {"en": 0, "non_en": 0, "unknown": 0}
    non_english_ids = []

    for idx, row in enumerate(rows, start=1):
        paper_id = row["paper_id"]
        lang, source = _detect_best(row)

        if lang == "unknown":
            counts["unknown"] += 1
            filtered = 0
        elif lang == "en":
            counts["en"] += 1
            filtered = 0
        else:
            counts["non_en"] += 1
            filtered = 1
            non_english_ids.append(paper_id)

        if not dry_run:
            conn.execute(
                """UPDATE papers
                   SET language = ?, language_source = ?, language_filtered = ?
                   WHERE paper_id = ?""",
                (lang, source, filtered, paper_id),
            )

        if idx % 1000 == 0 or idx == total:
            logger.info(
                "Pass A [%d/%d]  en: %d  non-en: %d  unknown: %d",
                idx, total, counts["en"], counts["non_en"], counts["unknown"],
            )

    if not dry_run:
        conn.commit()

    return {"counts": counts, "non_english_ids": non_english_ids}


def run_pass_b(conn: sqlite3.Connection, dry_run: bool) -> dict:
    """
    Pass B: Re-check papers that were previously marked 'en' using only the
    abstract/title but now have an extracted text file available.
    These are the papers with English abstracts but non-English body text.
    """
    rows = conn.execute(
        """SELECT paper_id, title, abstract, text_path
           FROM papers
           WHERE language = 'en'
           AND (language_source IN ('abstract', 'title') OR language_source IS NULL)
           AND text_path IS NOT NULL"""
    ).fetchall()

    total = len(rows)
    logger.info("Pass B — re-checking %d previously abstract-detected 'en' papers with text files...", total)

    counts = {"confirmed_en": 0, "reclassified": 0}
    non_english_ids = []

    for idx, row in enumerate(rows, start=1):
        paper_id = row["paper_id"]

        # Re-detect using text file only (we already know abstract was 'en')
        text_file_content = _read_text_file(row["text_path"])
        lang = detect_language(text_file_content)

        if lang is None or lang == "en":
            counts["confirmed_en"] += 1
            if not dry_run:
                conn.execute(
                    "UPDATE papers SET language_source = 'text_file' WHERE paper_id = ?",
                    (paper_id,),
                )
        else:
            counts["reclassified"] += 1
            non_english_ids.append(paper_id)
            if not dry_run:
                conn.execute(
                    """UPDATE papers
                       SET language = ?, language_source = 'text_file', language_filtered = 1
                       WHERE paper_id = ?""",
                    (lang, paper_id),
                )

        if idx % 500 == 0 or idx == total:
            logger.info(
                "Pass B [%d/%d]  confirmed en: %d  reclassified non-en: %d",
                idx, total, counts["confirmed_en"], counts["reclassified"],
            )

    if not dry_run:
        conn.commit()

    return {"counts": counts, "non_english_ids": non_english_ids}


def run(dry_run: bool = False, recheck_only: bool = False) -> None:
    conn = sqlite3.connect(PAPERS_DB_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_columns(conn)

    all_non_english_ids = []

    # ── Pass A: new papers ────────────────────────────────────────────────────
    if not recheck_only:
        result_a = run_pass_a(conn, dry_run)
        all_non_english_ids.extend(result_a["non_english_ids"])
    else:
        result_a = {"counts": {"en": 0, "non_en": 0, "unknown": 0}, "non_english_ids": []}
        logger.info("Pass A skipped (--recheck-only)")

    # ── Pass B: re-check abstract-detected English papers ────────────────────
    result_b = run_pass_b(conn, dry_run)
    all_non_english_ids.extend(result_b["non_english_ids"])

    conn.close()

    # ── Delete chunks for all newly found non-English papers ─────────────────
    if all_non_english_ids:
        logger.info(
            "Removing chunks for %d non-English papers...", len(all_non_english_ids)
        )
        _delete_chunks(all_non_english_ids, dry_run)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  LANGUAGE FILTER" + ("  [DRY RUN]" if dry_run else "  COMPLETE"))
    print("=" * 60)

    if not recheck_only:
        ca = result_a["counts"]
        print(f"\n  Pass A — new papers:")
        print(f"    English         : {ca['en']:,}")
        print(f"    Non-English     : {ca['non_en']:,}")
        print(f"    Unknown         : {ca['unknown']:,}  (kept — too short)")
        print(f"    Filtered out    : {len(result_a['non_english_ids']):,} papers")

    cb = result_b["counts"]
    print(f"\n  Pass B — re-checked abstract-only 'en' papers:")
    print(f"    Confirmed English    : {cb['confirmed_en']:,}")
    print(f"    Reclassified non-en  : {cb['reclassified']:,}  ← papers with English abstract but non-English body")
    print(f"    Filtered out         : {len(result_b['non_english_ids']):,} papers")

    print(f"\n  Total newly filtered : {len(all_non_english_ids):,} papers")
    if not dry_run and all_non_english_ids:
        print(f"  Chunks deleted from  : {len(all_non_english_ids):,} papers")
    print("=" * 60)

    # Sample non-English papers found in dry-run
    if dry_run and all_non_english_ids:
        print("\nSample non-English papers that would be filtered (Pass B):")
        conn2 = sqlite3.connect(PAPERS_DB_PATH)
        conn2.row_factory = sqlite3.Row
        sample_ids = result_b["non_english_ids"][:10]
        if sample_ids:
            placeholders = ",".join("?" * len(sample_ids))
            samples = conn2.execute(
                f"SELECT title, language, language_source FROM papers WHERE paper_id IN ({placeholders})",
                sample_ids,
            ).fetchall()
            for s in samples:
                print(f"  [{s['language']} via {s['language_source']}] {s['title']}")
        conn2.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter non-English papers using extracted text files"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be filtered without making any changes",
    )
    parser.add_argument(
        "--recheck-only", action="store_true",
        help="Skip Pass A (new papers) and only re-check previously abstract-detected English papers",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run, recheck_only=args.recheck_only)


if __name__ == "__main__":
    main()
