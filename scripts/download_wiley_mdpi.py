"""
scripts/download_wiley_mdpi.py
===============================
Downloads PDFs from publishers that block bot User-Agents, using a realistic
Chrome browser User-Agent to bypass 403 responses.

Publishers covered
------------------
- Wiley Online Library     (onlinelibrary.wiley.com)
- Wiley Anatomy sub-domain (anatomypubs.onlinelibrary.wiley.com)
- MDPI                     (www.mdpi.com)
- Cell Press               (www.cell.com)
- AJNR                     (www.ajnr.org)
- Brieflands               (brieflands.com)

Safe to run in parallel with the main downloader — this script only touches
papers whose pdf_url matches the above prefixes. Both scripts write to
different paper_id rows so there is no SQLite write conflict.

Usage
-----
    # In a new PyCharm terminal (while main downloader is still running):
    python -m scripts.download_wiley_mdpi

    # Dry run — show how many papers would be attempted without downloading
    python -m scripts.download_wiley_mdpi --dry-run
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

# Browser User-Agent — passes publisher bot-detection checks.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# URL prefixes that identify publishers handled by this script.
# All require a browser User-Agent to avoid 403 responses.
_PUBLISHERS = {
    "Wiley":           "https://onlinelibrary.wiley.com/doi/pdfdirect",
    "Wiley Anatomy":   "https://anatomypubs.onlinelibrary.wiley.com",
    "MDPI":            "https://www.mdpi.com",
    "Cell":            "http://www.cell.com",
    "AJNR":            "http://www.ajnr.org/",
    "Brieflands":      "https://brieflands.com",
}

# Seconds between downloads — be polite to publisher servers.
_DELAY = 1.0


def _pdf_path(paper_id: str, year: int | None) -> Path:
    """Return the local path where a paper's PDF should be saved."""
    subdir = RAW_PAPERS_DIR / str(year or "unknown")
    subdir.mkdir(parents=True, exist_ok=True)
    safe_id = paper_id.replace("/", "_")
    return subdir / f"{safe_id}.pdf"


def _download_pdf(url: str, dest: Path, session: requests.Session) -> bool:
    """
    Download a PDF using a browser User-Agent.
    Returns True if a valid PDF was saved, False otherwise.
    """
    try:
        response = session.get(url, stream=True, timeout=60)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            logger.warning("Got HTML instead of PDF: %s", url[:70])
            return False

        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)

        # Verify PDF magic number
        with open(dest, "rb") as fh:
            if fh.read(4) != b"%PDF":
                logger.warning("Not a valid PDF: %s", url[:70])
                dest.unlink()
                return False

        return True

    except requests.RequestException as exc:
        logger.error("Download failed for %s: %s", url[:70], exc)
        if dest.exists():
            dest.unlink()
        return False


def run(dry_run: bool = False) -> None:
    # Build the WHERE clause to match Wiley and MDPI urls
    placeholders = " OR ".join(
        f"pdf_url LIKE '{prefix}%'" for prefix in _PUBLISHERS.values()
    )
    query = f"""
        SELECT paper_id, pdf_url, year
        FROM papers
        WHERE pdf_local_path IS NULL
        AND ({placeholders})
    """

    conn = sqlite3.connect(PAPERS_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(query).fetchall()

    # Count by publisher
    pub_counts = {}
    for name, prefix in _PUBLISHERS.items():
        pub_counts[name] = sum(1 for r in rows if r["pdf_url"].startswith(prefix))

    logger.info("Papers to download — %s  |  Total: %d",
                "  |  ".join(f"{k}: {v}" for k, v in pub_counts.items()), len(rows))

    if dry_run:
        print()
        for name, count in pub_counts.items():
            print(f"  {name:<20}: {count:,}")
        print(f"  {'Total':<20}: {len(rows):,}")
        conn.close()
        return

    session = requests.Session()
    session.headers.update(_BROWSER_HEADERS)

    counts = {"downloaded": 0, "failed": 0, "skipped": 0}
    total = len(rows)

    for idx, row in enumerate(rows, start=1):
        paper_id = row["paper_id"]
        pdf_url  = row["pdf_url"]
        year     = row["year"]

        dest = _pdf_path(paper_id, year)

        # Skip if already on disk (e.g. main downloader got it first)
        if dest.exists():
            conn.execute(
                "UPDATE papers SET pdf_local_path = ? WHERE paper_id = ?",
                (str(dest), paper_id),
            )
            conn.commit()
            counts["skipped"] += 1
            continue

        publisher = next((name for name, prefix in _PUBLISHERS.items() if pdf_url.startswith(prefix)), "Unknown")
        logger.info("[%d/%d] [%s] %s", idx, total, publisher, pdf_url[:70])

        success = _download_pdf(pdf_url, dest, session)

        if success:
            conn.execute(
                "UPDATE papers SET pdf_local_path = ? WHERE paper_id = ?",
                (str(dest), paper_id),
            )
            conn.commit()
            counts["downloaded"] += 1
        else:
            counts["failed"] += 1

        time.sleep(_DELAY)

    session.close()
    conn.close()

    print("\n" + "=" * 50)
    print("  Publisher Download Complete")
    print("=" * 50)
    print(f"  Downloaded : {counts['downloaded']:,}")
    print(f"  Failed     : {counts['failed']:,}")
    print(f"  Skipped    : {counts['skipped']:,}  (already on disk)")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Wiley and MDPI papers using a browser User-Agent"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show counts without downloading anything",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
