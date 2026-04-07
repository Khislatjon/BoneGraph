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

from config.settings import PAPERS_DB_PATH, RAW_PAPERS_DIR, WILEY_TDM_TOKEN

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

_WILEY_TDM_BASE = "https://api.wiley.com/onlinelibrary/tdm/v1/articles"

_HEADERS = {
    "Wiley-TDM-Client-Token": WILEY_TDM_TOKEN,
    "Accept": "application/pdf,*/*",
}

_DELAY = 2.0


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

    logger.info("Wiley papers to download via TDM token: %d", len(rows))

    if dry_run:
        print(f"\n  Wiley papers : {len(rows):,}")
        conn.close()
        return

    session = requests.Session()
    session.headers.update(_HEADERS)

    counts = {"downloaded": 0, "failed": 0, "skipped": 0}
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
            counts["failed"] += 1
            continue

        # Use the Wiley TDM API — no Cloudflare, token-authenticated
        tdm_url = f"{_WILEY_TDM_BASE}/{doi}"
        logger.info("[%d/%d] %s", idx, total, tdm_url[:80])

        try:
            response = session.get(tdm_url, stream=True, timeout=60)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                logger.warning("Got HTML instead of PDF: %s", tdm_url[:70])
                counts["failed"] += 1
                time.sleep(_DELAY)
                continue

            with open(dest, "wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    fh.write(chunk)

            with open(dest, "rb") as fh:
                magic = fh.read(4)

            if magic != b"%PDF":
                logger.warning("Not a valid PDF: %s", tdm_url[:70])
                dest.unlink()
                counts["failed"] += 1
            else:
                conn.execute("UPDATE papers SET pdf_local_path = ? WHERE paper_id = ?",
                             (str(dest), paper_id))
                conn.commit()
                counts["downloaded"] += 1

        except requests.RequestException as exc:
            logger.error("Failed %s: %s", tdm_url[:70], exc)
            if dest.exists():
                dest.unlink()
            counts["failed"] += 1

        time.sleep(_DELAY)

    session.close()
    conn.close()

    print("\n" + "=" * 50)
    print("  Wiley TDM Download — Complete")
    print("=" * 50)
    print(f"  Downloaded : {counts['downloaded']:,}")
    print(f"  Failed     : {counts['failed']:,}")
    print(f"  Skipped    : {counts['skipped']:,}  (already on disk)")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Wiley PDFs via Unpaywall")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
