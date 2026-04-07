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

from config.settings import RAW_PAPERS_DIR, REQUEST_DELAY_SECONDS
from ingestion.papers.resolvers import resolve_pdf_url
from ingestion.papers.storage import PaperStore

logger = logging.getLogger(__name__)

# HTTP headers sent with every download request.
# A descriptive User-Agent is good practice — it identifies the bot to server
# admins and signals legitimate academic use.
_HEADERS = {
    "User-Agent": "BoneLogic-Research-Bot/1.0 (academic; contact: bonelogic@research.org)"
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
        # stream=True — download the body in chunks, not all at once.
        response = requests.get(url, headers=_HEADERS, stream=True, timeout=timeout)
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


def download_all_open_access(
    store: PaperStore,
    delay: float = REQUEST_DELAY_SECONDS,
    try_doi_fallback: bool = True,
) -> dict[str, int]:
    """
    Download PDFs for every paper in the store that has an open-access URL
    but no local file recorded yet.

    Steps for each paper
    --------------------
    1. Query the DB for papers with has_pdf=1 AND pdf_local_path IS NULL.
    2. Run the stored URL through the 3-tier resolver (resolvers.py) to get
       a direct, downloadable PDF URL:
         Tier 1 — URL is already a direct PDF link → use as-is
         Tier 2 — Rewrite using publisher-specific URL patterns
         Tier 3 — Query Unpaywall API with the paper's DOI
    3. Call download_pdf() to stream the file to disk.
    4. Verify the downloaded file starts with the PDF magic number (%PDF).
    5. Write the local path back to the DB with set_local_pdf_path().
    6. Wait `delay` seconds between downloads to be polite to servers.

    Parameters
    ----------
    store : PaperStore
        An open PaperStore connection (must be inside a with-block or
        after connect()).
    delay : float
        Seconds to sleep between consecutive downloads.  Defaults to the
        same delay used for API calls (1.1 s).

    Returns
    -------
    dict with keys:
        "downloaded"   — PDFs successfully fetched and verified
        "failed"       — URLs that returned errors or non-PDF content
        "unresolvable" — papers where no direct PDF URL could be found
        "skipped"      — papers already downloaded in a previous run
    """
    import json

    papers = store.get_papers_with_pdf()
    counts = {"downloaded": 0, "failed": 0, "unresolvable": 0, "skipped": 0}

    logger.info("Pass 1: %d papers with open-access PDF URLs.", len(papers))

    logger.info("Found %d papers with open-access PDF URLs to download.", len(papers))

    # Reuse a single HTTP session for all Unpaywall requests —
    # this is more efficient than creating a new connection per request.
    session = requests.Session()
    session.headers.update(_HEADERS)

    total   = len(papers)
    for idx, row in enumerate(papers, start=1):
        paper_id  = row["paper_id"]
        pdf_url   = row["pdf_url"]   # may be a doi.org link or landing page
        year      = row["year"]

        # Extract the DOI from the stored external_ids_json column.
        # The DOI is needed for Tier 3 (Unpaywall) resolution.
        try:
            ext_ids = json.loads(row["external_ids_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            ext_ids = {}
        doi = ext_ids.get("DOI")

        # ── Skip if already downloaded ────────────────────────────────────────
        dest = _pdf_path(paper_id, year)
        if dest.exists():
            store.set_local_pdf_path(paper_id, str(dest))
            counts["skipped"] += 1
            continue

        # ── Resolve to a direct PDF URL (3-tier chain) ────────────────────────
        resolved_url = resolve_pdf_url(
            paper_id   = paper_id,
            stored_url = pdf_url,
            doi        = doi,
            session    = session,
        )

        if not resolved_url:
            # All resolution tiers failed — no downloadable PDF found.
            logger.info(
                "[%d/%d] No resolvable PDF for paper %s (stored_url=%s)",
                idx, total, paper_id, pdf_url
            )
            counts["unresolvable"] += 1
            continue

        # ── Download the PDF ──────────────────────────────────────────────────
        logger.info(
            "[%d/%d] Downloading %s → %s",
            idx, total, resolved_url[:70], dest.name
        )
        success = download_pdf(resolved_url, dest)

        if success:
            store.set_local_pdf_path(paper_id, str(dest))
            counts["downloaded"] += 1
        else:
            counts["failed"] += 1

        # Pause between downloads — polite to servers and avoids bans.
        time.sleep(delay)

    # ── Pass 2: DOI-only papers (no OpenAlex PDF URL) ────────────────────────
    # Many papers have a DOI but were not flagged as open-access by OpenAlex.
    # Unpaywall often finds freely available PDFs for these — e.g. preprints,
    # institutional repositories, or publisher open-access pages OpenAlex missed.
    if try_doi_fallback:
        doi_papers = store.get_papers_doi_only()
        logger.info(
            "Pass 2: %d papers with DOI but no PDF URL — trying Unpaywall.",
            len(doi_papers),
        )

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

            # No stored_url → resolver skips Tiers 1 & 2 and goes straight to Unpaywall.
            resolved_url = resolve_pdf_url(
                paper_id   = paper_id,
                stored_url = None,
                doi        = doi,
                session    = session,
            )

            if not resolved_url:
                counts["unresolvable"] += 1
                continue

            logger.info(
                "[P2 %d/%d] Downloading %s → %s",
                idx, total_doi, resolved_url[:70], dest.name,
            )
            success = download_pdf(resolved_url, dest)

            if success:
                store.set_local_pdf_path(paper_id, str(dest))
                counts["downloaded"] += 1
            else:
                counts["failed"] += 1

            time.sleep(delay)

    session.close()

    logger.info(
        "Download complete — downloaded: %d  |  failed: %d  "
        "|  unresolvable: %d  |  skipped: %d",
        counts["downloaded"], counts["failed"],
        counts["unresolvable"], counts["skipped"],
    )
    return counts
