"""
ingestion/papers/pipeline.py
==============================
Top-level orchestrator for the paper ingestion pipeline.

What this script does
---------------------
1. Takes a list of keyword search queries (from keywords.py or the CLI).
2. For each keyword, calls the Semantic Scholar API to fetch up to max_papers papers.
3. Stores the paper metadata (title, abstract, authors, PDF URL…) in SQLite.
4. Optionally downloads the open-access PDFs to disk.

This is the main entry point for Phase 1 data collection.  Run it from the
command line or from PyCharm's run configuration.

CLI usage examples
------------------
# Run all 133 keywords (default) — takes ~20-25 minutes with API key
python -m ingestion.papers.pipeline

# Run only mechanics and pathology groups
python -m ingestion.papers.pipeline --groups mechanics pathology

# Run a single custom query, restrict to 2015-2024, download PDFs
python -m ingestion.papers.pipeline --keywords "femur fracture osteoporosis" --year 2015-2024 --download

# Limit to 50 papers per keyword (fast test run)
python -m ingestion.papers.pipeline --groups imaging --max 50
"""

from __future__ import annotations  # enables X | Y union syntax on Python 3.9

import argparse
import logging
import sys
from pathlib import Path

# Insert repo root into sys.path so this module can be run directly as a script
# (python ingestion/papers/pipeline.py) as well as as a module
# (python -m ingestion.papers.pipeline).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ingestion.papers.downloader import download_all_open_access
from ingestion.papers.keywords import ALL_KEYWORDS, KEYWORD_GROUPS
from ingestion.papers.openalex import OpenAlexClient
from ingestion.papers.storage import PaperStore
from config.settings import DEFAULT_YEAR_RANGE, DEFAULT_LANGUAGE

# Configure logging to print timestamped, levelled messages to the console.
# Format: "09:14:22  INFO      ingestion.papers.pipeline  Starting 133 keyword queries."
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(
    keywords: list[str],
    max_per_keyword: int,
    year_range: str | None,
    language: str | None,
    download_pdfs: bool,
) -> None:
    """
    Execute the full ingestion pipeline for the given list of keywords.

    Parameters
    ----------
    keywords : list[str]
        The search queries to run, e.g. ["bone fracture", "cortical bone"].
    max_per_keyword : int
        Maximum number of papers to fetch per keyword query.
    year_range : str | None
        Optional year filter passed to the S2 API, e.g. "2015-2024".
    download_pdfs : bool
        If True, attempt to download open-access PDFs after metadata ingestion.
    """
    # Instantiate the OpenAlex client — no API key needed.
    client = OpenAlexClient()

    # Open the database connection for the entire run.
    # The `with` statement guarantees the connection closes cleanly even on error.
    with PaperStore() as store:
        total_new = 0      # papers we've never seen before across all keywords
        total_updated = 0  # papers already in the DB that were refreshed

        for i, keyword in enumerate(keywords, 1):
            logger.info("[%d/%d] Querying: '%s'", i, len(keywords), keyword)

            # search_papers() is a generator — calling list() materialises all
            # pages into memory before we start writing to the DB.
            papers = list(
                client.search_papers(
                    keyword,
                    max_papers=max_per_keyword,
                    year_range=year_range,
                    language=language,
                )
            )

            # Upsert all papers into SQLite.  Duplicates (same paper found by
            # multiple keywords) are merged, not duplicated.
            new, updated = store.upsert_papers(papers, keyword)

            # Record this keyword query in the audit log table.
            store.record_search_run(keyword, len(papers), len(papers))

            total_new += new
            total_updated += updated
            logger.info(
                "  → %d papers fetched | %d new, %d already known",
                len(papers), new, updated,
            )

        # Print the final database totals after all keywords are done.
        stats = store.stats()
        logger.info(
            "Ingestion complete. DB totals: %d unique papers, %d with open-access PDF.",
            stats["total"], stats["with_open_pdf"],
        )

        # Phase 1b (optional): download the open-access PDFs.
        if download_pdfs:
            logger.info("Starting PDF downloads…")
            counts = download_all_open_access(store)
            logger.info("PDF download summary: %s", counts)


def main() -> None:
    """
    Parse command-line arguments and launch the pipeline.

    This function is called when you run:
        python -m ingestion.papers.pipeline [options]
    """
    parser = argparse.ArgumentParser(
        description="BoneMind Phase 1 — paper ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ingestion.papers.pipeline
  python -m ingestion.papers.pipeline --groups mechanics pathology
  python -m ingestion.papers.pipeline --keywords "femur fracture" --year 2015-2024 --download
  python -m ingestion.papers.pipeline --groups imaging --max 50
        """,
    )

    # --groups: run only selected keyword groups (e.g. mechanics, pathology).
    # choices= is populated dynamically from KEYWORD_GROUPS so it always
    # matches the current keyword file without hardcoding group names here.
    parser.add_argument(
        "--groups",
        nargs="*",                          # zero or more group names
        choices=list(KEYWORD_GROUPS.keys()),
        default=None,
        help="Keyword groups to run (default: all groups).",
    )

    # --keywords: provide custom search strings directly on the command line.
    # When specified, this takes priority over --groups.
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="Custom keyword strings (overrides --groups).",
    )

    # --year: restrict results to a publication year range.
    # OpenAlex format: "YYYY-YYYY" (inclusive on both ends).
    parser.add_argument(
        "--year",
        default=DEFAULT_YEAR_RANGE,
        help=f"Year range filter (default: {DEFAULT_YEAR_RANGE}).",
    )

    # --language: ISO 639-1 language code to filter by.
    # OpenAlex supports native language filtering — non-English papers are
    # excluded at query time before they touch the database.
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language filter, ISO 639-1 code (default: {DEFAULT_LANGUAGE}). Pass '' to disable.",
    )

    # --max: cap the number of papers retrieved per keyword.
    # Lower values are useful for quick test runs.
    parser.add_argument(
        "--max",
        type=int,
        default=500,
        help="Max papers per keyword query (default: 500).",
    )

    # --download: after fetching metadata, also download open-access PDFs.
    # Off by default because PDF downloads take much longer and use disk space.
    parser.add_argument(
        "--download",
        action="store_true",            # flag: present = True, absent = False
        help="Download open-access PDFs after metadata ingestion.",
    )

    args = parser.parse_args()

    # Resolve which keyword list to use, in priority order:
    # 1. Explicit --keywords strings from the CLI.
    # 2. Keywords from the selected --groups.
    # 3. ALL_KEYWORDS (everything in keywords.py) as the default.
    if args.keywords:
        keywords = args.keywords
    elif args.groups:
        # Flatten: for each selected group name, extend the list with its keywords.
        keywords = [kw for g in args.groups for kw in KEYWORD_GROUPS[g]]
    else:
        keywords = ALL_KEYWORDS

    lang = args.language or None  # convert empty string to None (disables filter)
    logger.info("Running %d keyword queries (max %d papers each, lang=%s, years=%s).",
                len(keywords), args.max, lang, args.year)
    run(keywords, args.max, args.year, lang, args.download)


# This block runs only when the script is executed directly, not when imported.
if __name__ == "__main__":
    main()
