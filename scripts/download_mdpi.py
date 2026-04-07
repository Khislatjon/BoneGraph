"""
scripts/download_mdpi.py
=========================
Downloads MDPI PDFs using a browser User-Agent.

MDPI URLs from OpenAlex already point directly to PDF files (/pdf path).
The only reason they fail in the main downloader is the bot-like User-Agent.
A browser User-Agent is sufficient — MDPI has no Cloudflare protection.

Usage
-----
    python -m scripts.download_mdpi

    # Dry run
    python -m scripts.download_mdpi --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from pathlib import Path

import requests

from config.settings import PAPERS_DB_PATH, RAW_PAPERS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_MDPI_PREFIX = "https://www.mdpi.com"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
    "Accept-Encoding": "gzip,deflate,br",
}

_DELAY = 0.5


def _pdf_path(paper_id: str, year: int | None) -> Path:
    subdir = RAW_PAPERS_DIR / str(year or "unknown")
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{paper_id.replace('/', '_')}.pdf"


def run(dry_run: bool = False) -> None:
    conn = sqlite3.connect(PAPERS_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT paper_id, pdf_url, year
        FROM papers
        WHERE pdf_local_path IS NULL
        AND pdf_url LIKE 'https://www.mdpi.com%'
    """).fetchall()

    logger.info("MDPI papers to download: %d", len(rows))

    if dry_run:
        print(f"\n  MDPI papers : {len(rows):,}")
        conn.close()
        return

    session = requests.Session()
    session.headers.update(_HEADERS)

    counts = {"downloaded": 0, "failed": 0, "skipped": 0}
    total = len(rows)

    for idx, row in enumerate(rows, start=1):
        paper_id = row["paper_id"]
        pdf_url  = row["pdf_url"]
        year     = row["year"]
        dest     = _pdf_path(paper_id, year)

        if dest.exists():
            conn.execute("UPDATE papers SET pdf_local_path = ? WHERE paper_id = ?",
                         (str(dest), paper_id))
            conn.commit()
            counts["skipped"] += 1
            continue

        logger.info("[%d/%d] %s", idx, total, pdf_url[:80])

        try:
            response = session.get(pdf_url, stream=True, timeout=60)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                logger.warning("Got HTML instead of PDF: %s", pdf_url[:70])
                counts["failed"] += 1
                time.sleep(_DELAY)
                continue

            with open(dest, "wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    fh.write(chunk)

            with open(dest, "rb") as fh:
                magic = fh.read(4)

            if magic != b"%PDF":
                logger.warning("Not a valid PDF: %s", pdf_url[:70])
                dest.unlink()
                counts["failed"] += 1
            else:
                conn.execute("UPDATE papers SET pdf_local_path = ? WHERE paper_id = ?",
                             (str(dest), paper_id))
                conn.commit()
                counts["downloaded"] += 1

        except requests.RequestException as exc:
            logger.error("Failed %s: %s", pdf_url[:70], exc)
            if dest.exists():
                dest.unlink()
            counts["failed"] += 1

        time.sleep(_DELAY)

    session.close()
    conn.close()

    print("\n" + "=" * 50)
    print("  MDPI Download — Complete")
    print("=" * 50)
    print(f"  Downloaded : {counts['downloaded']:,}")
    print(f"  Failed     : {counts['failed']:,}")
    print(f"  Skipped    : {counts['skipped']:,}  (already on disk)")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MDPI PDFs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
