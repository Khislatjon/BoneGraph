"""
scripts/download_pass2.py
==========================
Pass 2 — Try Unpaywall for papers that have a DOI but no open-access PDF
URL from OpenAlex.

Unpaywall often surfaces freely available copies that OpenAlex didn't flag —
preprints, institutional repositories, PubMed Central, etc.

Can be run in parallel with Pass 1 in a separate terminal tab since they
operate on different sets of papers (Pass 1: has_pdf=1, Pass 2: has_pdf=0).

Usage
-----
    python -m scripts.download_pass2
"""

from __future__ import annotations

import logging

from ingestion.papers.downloader import download_pass2
from ingestion.papers.storage import PaperStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting PDF download — Pass 2 (DOI → Unpaywall)...")
    with PaperStore() as store:
        counts = download_pass2(store)
    logger.info("Pass 2 done. %s", counts)


if __name__ == "__main__":
    main()
