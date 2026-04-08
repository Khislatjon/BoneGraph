"""
scripts/download_pass1.py
==========================
Pass 1 — Download PDFs for papers with an open-access URL from OpenAlex.

Uses 3-tier resolver: direct link → publisher transform → CrossRef fallback.
Wiley URLs are automatically rewritten to the TDM API endpoint.

Run Pass 2 separately with:
    python -m scripts.download_pass2

Usage
-----
    python -m scripts.download_pass1
"""

from __future__ import annotations

import logging

from ingestion.papers.downloader import download_pass1
from ingestion.papers.storage import PaperStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting PDF download — Pass 1...")
    with PaperStore() as store:
        counts = download_pass1(store)
    logger.info("Pass 1 done. %s", counts)


if __name__ == "__main__":
    main()
