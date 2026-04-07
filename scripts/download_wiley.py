"""
scripts/download_wiley.py
==========================
Downloads Wiley PDFs via Unpaywall instead of hitting Wiley directly.

Why not download from Wiley directly?
--------------------------------------
Wiley protects all URLs with Cloudflare bot detection (TLS fingerprinting,
navigator.webdriver checks, CAPTCHA challenges) that cannot be bypassed
programmatically without constant manual intervention.

Instead, we look up each paper's DOI on Unpaywall, which finds legal
open-access copies hosted elsewhere — PubMed Central, institutional
repositories, arXiv, author pages — that have no bot protection.

All 2,850 Wiley papers have DOIs, so this covers all of them.

Usage
-----
    python -m scripts.download_wiley

    # Dry run
    python -m scripts.download_wiley --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from pathlib import Path

import requests

from config.settings import PAPERS_DB_PATH, RAW_PAPERS_DIR
from ingestion.papers.downloader import download_pdf
from ingestion.papers.resolvers import resolve_via_unpaywall

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_WILEY_PREFIXES = [
    "https://onlinelibrary.wiley.com",
    "https://anatomypubs.onlinelibrary.wiley.com",
]

_DELAY = 0.15  # Unpaywall allows generous rate limits


def _pdf_path(paper_id: str, year: int | None) -> Path:
    subdir = RAW_PAPERS_DIR / str(year or "unknown")
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{paper_id.replace('/', '_')}.pdf"


def run(dry_run: bool = False) -> None:
    placeholders = " OR ".join(f"pdf_url LIKE '{p}%'" for p in _WILEY_PREFIXES)
    conn = sqlite3.connect(PAPERS_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(f"""
        SELECT paper_id, pdf_url, year, external_ids_json
        FROM papers
        WHERE pdf_local_path IS NULL
        AND ({placeholders})
    """).fetchall()

    logger.info("Wiley papers to attempt via Unpaywall: %d", len(rows))

    if dry_run:
        print(f"\n  Wiley papers : {len(rows):,}")
        conn.close()
        return

    session = requests.Session()
    counts = {"downloaded": 0, "failed": 0, "no_unpaywall": 0, "skipped": 0}
    total = len(rows)

    for idx, row in enumerate(rows, start=1):
        paper_id = row["paper_id"]
        year     = row["year"]
        dest     = _pdf_path(paper_id, year)

        if dest.exists():
            conn.execute("UPDATE papers SET pdf_local_path = ? WHERE paper_id = ?",
                         (str(dest), paper_id))
            conn.commit()
            counts["skipped"] += 1
            continue

        try:
            ext_ids = json.loads(row["external_ids_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            ext_ids = {}
        doi = ext_ids.get("DOI")

        if not doi:
            counts["no_unpaywall"] += 1
            continue

        unpaywall_url = resolve_via_unpaywall(doi, session=session)

        # Skip if Unpaywall just returns a Wiley URL — same 403 problem.
        if unpaywall_url and "wiley.com" in unpaywall_url:
            unpaywall_url = None

        if not unpaywall_url:
            counts["no_unpaywall"] += 1
            if idx % 100 == 0:
                logger.info("[%d/%d] progress — downloaded: %d  no_unpaywall: %d  failed: %d",
                            idx, total, counts["downloaded"], counts["no_unpaywall"], counts["failed"])
            time.sleep(_DELAY)
            continue

        logger.info("[%d/%d] %s → %s", idx, total, doi, unpaywall_url[:70])
        success = download_pdf(unpaywall_url, dest)

        if success:
            conn.execute("UPDATE papers SET pdf_local_path = ? WHERE paper_id = ?",
                         (str(dest), paper_id))
            conn.commit()
            counts["downloaded"] += 1
        else:
            counts["failed"] += 1

        time.sleep(_DELAY)

    session.close()
    conn.close()

    print("\n" + "=" * 50)
    print("  Wiley via Unpaywall — Complete")
    print("=" * 50)
    print(f"  Downloaded      : {counts['downloaded']:,}")
    print(f"  Failed          : {counts['failed']:,}")
    print(f"  No Unpaywall    : {counts['no_unpaywall']:,}  (no free copy found)")
    print(f"  Skipped         : {counts['skipped']:,}  (already on disk)")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Wiley PDFs via Unpaywall")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
