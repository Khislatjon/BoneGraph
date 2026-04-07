"""
scripts/download_pdfs.py
=========================
Standalone PDF downloader — runs Pass 1 and Pass 2 without re-fetching
metadata from OpenAlex.

Pass 1 — papers with an open-access PDF URL from OpenAlex (has_pdf=1)
          Uses 3-tier resolver: direct link → publisher transform → Unpaywall
          Wiley URLs are automatically rewritten to the TDM API endpoint.
Pass 2 — papers with no PDF URL but a DOI, tried via Unpaywall

Usage
-----
    python -m scripts.download_pdfs
"""

from __future__ import annotations

import logging

from ingestion.papers.downloader import download_all_open_access
from ingestion.papers.storage import PaperStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting PDF download (Pass 1 + Pass 2)...")
    with PaperStore() as store:
        counts = download_all_open_access(store)
    logger.info("Done. %s", counts)


if __name__ == "__main__":
    main()
