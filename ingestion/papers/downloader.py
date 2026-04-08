"""
ingestion/papers/downloader.py
================================
Downloads open-access PDFs for papers that have a free legal PDF URL.

How it decides what to download
--------------------------------
Semantic Scholar's `openAccessPdf` field contains a URL when a legal free
copy of the paper is available (e.g. on PubMed Central, arXiv, or the
publisher's website).  We only download papers where this field is populated
and the PDF has not already been saved to disk.

URL Resolution
--------------
Not all URLs from Semantic Scholar are direct PDF links. Some are doi.org
links, publisher landing pages, or viewer pages. Before downloading, each
URL is passed through a 3-tier resolver (see ingestion/papers/resolvers.py):

  Tier 1 — Direct PDF check      : URL already points to a .pdf → use as-is
  Tier 2 — Publisher transforms  : rewrite known publisher URLs to PDF paths
  Tier 3 — Unpaywall API         : look up DOI to find a direct PDF URL

File organisation
-----------------
PDFs are saved to:  data/raw/papers/<year>/<paper_id>.pdf

Grouping by year makes the folder browsable and avoids very large flat directories.
The paper_id is used as the filename (with "/" replaced by "_" for filesystem safety).

What happens after download
---------------------------
The local file path is written back to the `pdf_local_path` column in the
database.  The processing pipeline later reads this column to find PDFs to parse.
"""

from __future__ import annotations  # enables X | Y union syntax on Python 3.9

import logging
import time
from pathlib import Path

import requests

from config.settings import RAW_PAPERS_DIR, REQUEST_DELAY_SECONDS, WILEY_TDM_TOKEN
from ingestion.papers.resolvers import resolve_pdf_url, resolve_via_crossref
from ingestion.papers.storage import PaperStore

logger = logging.getLogger(__name__)

# HTTP headers sent with every download request.
# A realistic browser User-Agent is required — many academic publishers
# (Wiley, MDPI, Cell, AJNR, etc.) return 403 Forbidden for bot-like strings.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
    "Accept-Encoding": "gzip,deflate,br",
}

# Wiley TDM API base — bypasses Cloudflare with token authentication.
_WILEY_TDM_BASE = "https://api.wiley.com/onlinelibrary/tdm/v1/articles"
_WILEY_PREFIXES = (
    "https://onlinelibrary.wiley.com",
    "https://anatomypubs.onlinelibrary.wiley.com",
)


def _is_wiley(url: str) -> bool:
    return any(url.startswith(p) for p in _WILEY_PREFIXES)


def _wiley_tdm_headers() -> dict:
    return {
        "Wiley-TDM-Client-Token": WILEY_TDM_TOKEN,
        "Accept": "application/pdf,*/*",
    }


def _pdf_path(paper_id: str, year: int | None) -> Path:
    """
    Construct the local filesystem path where a paper's PDF should be saved.

    Parameters
    ----------
    paper_id : str
        The Semantic Scholar paper ID.  Used as the filename.
    year : int | None
        Publication year, used as the sub-folder name.
        If unknown, falls back to the folder "unknown".

    Returns
    -------
    Path
        A Path like:  data/raw/papers/2022/abc123def456.pdf
    """
    # Group PDFs by year to keep the directory manageable.
    subdir = RAW_PAPERS_DIR / str(year or "unknown")
    # Create the year sub-folder if it doesn't exist.
    subdir.mkdir(parents=True, exist_ok=True)
    # Replace "/" in paper IDs with "_" — slashes are not valid in filenames.
    safe_id = paper_id.replace("/", "_")
    return subdir / f"{safe_id}.pdf"


def download_pdf(url: str, dest: Path, timeout: int = 60) -> bool:
    """
    Stream-download a PDF from *url* and save it to *dest*.

    We use streaming (stream=True) so the file is written in chunks rather
    than loading the entire PDF into memory first.  This is important for
    large PDFs (some papers are 50+ MB).

    Content-type guard
    ------------------
    Some URLs redirect to HTML error pages or login pages instead of PDFs.
    We check the Content-Type header and the first 4 bytes of the response
    for the PDF magic number (%PDF).  If neither confirms a PDF, we abort
    and delete the partial file rather than storing useless HTML.

    Parameters
    ----------
    url : str
        The direct URL to the PDF file.
    dest : Path
        Where to save the file on disk.
    timeout : int
        How many seconds to wait for the server to respond before giving up.
        60 s is generous but needed for slow academic servers.

    Returns
    -------
    bool
        True if the download succeeded and a real PDF was written.
        False if any error occurred or the response was not a PDF.
    """
    try:
        # Wiley TDM API URLs need the token header; all others use browser headers.
        actual_headers = _wiley_tdm_headers() if "api.wiley.com" in url else _HEADERS

        # stream=True — download the body in chunks, not all at once.
        response = requests.get(url, headers=actual_headers, stream=True, timeout=timeout)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        # Hard reject: if the server explicitly says HTML, this is a landing
        # page redirect, not a PDF.  Abort before writing anything to disk.
        if "text/html" in content_type:
            logger.warning(
                "Server returned HTML instead of PDF — likely a landing page: %s", url
            )
            return False

        # Write the response body to disk in 8 KB chunks.
        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)

        # Verify the file starts with the PDF magic number: "%PDF"
        # This catches cases where Content-Type was missing or wrong but
        # the server still returned HTML (some journals do this).
        with open(dest, "rb") as fh:
            magic = fh.read(4)

        if magic != b"%PDF":
            logger.warning(
                "File does not start with %%PDF magic number — discarding: %s", url
            )
            dest.unlink()   # Delete the invalid file
            return False

        logger.debug("Downloaded %s → %s", url, dest)
        return True

    except requests.RequestException as exc:
        logger.error("Download failed for %s: %s", url, exc)
        # Delete any partial file so we don't store corrupted data.
        if dest.exists():
            dest.unlink()
        return False


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(_HEADERS)
    return session


def _log_counts(label: str, counts: dict) -> None:
    logger.info(
        "%s — downloaded: %d  |  failed: %d  |  unresolvable: %d  |  skipped: %d",
        label,
        counts["downloaded"], counts["failed"],
        counts["unresolvable"], counts["skipped"],
    )


def download_pass1(
    store: PaperStore,
    delay: float = REQUEST_DELAY_SECONDS,
) -> dict[str, int]:
    """
    Pass 1 — papers with an open-access PDF URL from OpenAlex (has_pdf=1).

    For each paper:
      1. Run the stored URL through the 3-tier resolver to get a direct PDF URL.
      2. Download and verify the PDF (%PDF magic number).
      3. On failure, try Unpaywall as a last-resort fallback (if paper has a DOI).
      4. Record the local path in the DB.
    """
    import json

    papers = store.get_papers_with_pdf()
    counts = {"downloaded": 0, "failed": 0, "unresolvable": 0, "skipped": 0}

    logger.info("Pass 1: %d papers with open-access PDF URLs.", len(papers))

    session = _make_session()
    total = len(papers)

    for idx, row in enumerate(papers, start=1):
        paper_id = row["paper_id"]
        pdf_url  = row["pdf_url"]
        year     = row["year"]

        try:
            ext_ids = json.loads(row["external_ids_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            ext_ids = {}
        doi = ext_ids.get("DOI")

        dest = _pdf_path(paper_id, year)
        if dest.exists():
            store.set_local_pdf_path(paper_id, str(dest))
            counts["skipped"] += 1
            continue

        resolved_url = resolve_pdf_url(
            paper_id=paper_id,
            stored_url=pdf_url,
            doi=doi,
            session=session,
        )

        if not resolved_url:
            logger.info(
                "[%d/%d] No resolvable PDF for paper %s (stored_url=%s)",
                idx, total, paper_id, pdf_url,
            )
            counts["unresolvable"] += 1
            continue

        logger.info("[%d/%d] Downloading %s → %s", idx, total, resolved_url[:70], dest.name)
        success = download_pdf(resolved_url, dest)

        # CrossRef fallback on failure (only if not already resolved via CrossRef)
        if not success and doi:
            crossref_url = resolve_via_crossref(doi, session=session)
            if crossref_url and crossref_url != resolved_url:
                logger.info("[%d/%d] Tier 3 fallback → %s", idx, total, crossref_url[:70])
                success = download_pdf(crossref_url, dest)

        if success:
            store.set_local_pdf_path(paper_id, str(dest))
            counts["downloaded"] += 1
        else:
            counts["failed"] += 1

        time.sleep(delay)

    session.close()
    _log_counts("Pass 1 complete", counts)
    return counts


def download_pass2(
    store: PaperStore,
    delay: float = REQUEST_DELAY_SECONDS,
) -> dict[str, int]:
    """
    Pass 2 — papers with a DOI but no open-access PDF URL from OpenAlex.

    Tries Unpaywall for each paper to find a freely available PDF that
    OpenAlex didn't flag (e.g. preprints, institutional repositories).
    """
    import json

    doi_papers = store.get_papers_doi_only()
    counts = {"downloaded": 0, "failed": 0, "unresolvable": 0, "skipped": 0}

    logger.info("Pass 2: %d papers with DOI but no PDF URL — trying Unpaywall.", len(doi_papers))

    session = _make_session()
    total_doi = len(doi_papers)

    for idx, row in enumerate(doi_papers, start=1):
        paper_id = row["paper_id"]
        year     = row["year"]

        try:
            ext_ids = json.loads(row["external_ids_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            ext_ids = {}
        doi = ext_ids.get("DOI")

        if not doi:
            continue

        dest = _pdf_path(paper_id, year)
        if dest.exists():
            store.set_local_pdf_path(paper_id, str(dest))
            counts["skipped"] += 1
            continue

        resolved_url = resolve_pdf_url(
            paper_id=paper_id,
            stored_url=None,
            doi=doi,
            session=session,
        )

        if not resolved_url:
            counts["unresolvable"] += 1
        else:
            logger.info("[P2 %d/%d] Downloading %s → %s", idx, total_doi, resolved_url[:70], dest.name)
            success = download_pdf(resolved_url, dest)

            if success:
                store.set_local_pdf_path(paper_id, str(dest))
                counts["downloaded"] += 1
            else:
                counts["failed"] += 1

            time.sleep(delay)

        if idx % 500 == 0:
            logger.info(
                "Pass 2 progress [%d/%d] — downloaded: %d  failed: %d  unresolvable: %d  skipped: %d",
                idx, total_doi,
                counts["downloaded"], counts["failed"],
                counts["unresolvable"], counts["skipped"],
            )

    session.close()
    _log_counts("Pass 2 complete", counts)
    return counts


def download_all_open_access(
    store: PaperStore,
    delay: float = REQUEST_DELAY_SECONDS,
    try_doi_fallback: bool = True,
) -> dict[str, int]:
    """Run Pass 1 then optionally Pass 2. Returns combined counts."""
    counts = download_pass1(store, delay=delay)
    if try_doi_fallback:
        p2 = download_pass2(store, delay=delay)
        for k in counts:
            counts[k] += p2[k]
    return counts
