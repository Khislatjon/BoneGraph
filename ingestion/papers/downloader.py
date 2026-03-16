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

import logging
import time
from pathlib import Path

import requests

from config.settings import RAW_PAPERS_DIR, REQUEST_DELAY_SECONDS
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
        True if the download succeeded and the file was written.
        False if any error occurred (the partial file is deleted on failure).
    """
    try:
        # stream=True tells requests not to download the body immediately —
        # we'll read it in chunks with iter_content() below.
        response = requests.get(url, headers=_HEADERS, stream=True, timeout=timeout)
        response.raise_for_status()

        # Sanity-check the Content-Type header.  Some URLs redirect to HTML
        # error pages instead of PDFs — we warn but still save the file so
        # the user can inspect it.
        content_type = response.headers.get("Content-Type", "")
        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            logger.warning("Unexpected content type '%s' for %s", content_type, url)

        # Write the response body to disk in 8 KB chunks.
        # chunk_size=8192 is a good balance between memory use and I/O calls.
        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)

        logger.debug("Downloaded %s → %s", url, dest)
        return True

    except requests.RequestException as exc:
        # Any network error (timeout, connection refused, bad status code).
        logger.error("Download failed for %s: %s", url, exc)
        # Delete the partial file so we don't store corrupted data.
        if dest.exists():
            dest.unlink()
        return False


def download_all_open_access(
    store: PaperStore,
    delay: float = REQUEST_DELAY_SECONDS,
) -> dict[str, int]:
    """
    Download PDFs for every paper in the store that has an open-access URL
    but no local file recorded yet.

    The function:
    1. Queries the DB for papers with has_pdf=1 AND pdf_local_path IS NULL.
    2. For each such paper, calls download_pdf() to fetch the file.
    3. On success, writes the local path back to the DB with set_local_pdf_path().
    4. Waits `delay` seconds between downloads to avoid hammering servers.

    Parameters
    ----------
    store : PaperStore
        An open PaperStore connection (must be inside a with-block or after connect()).
    delay : float
        Seconds to sleep between consecutive downloads.  Defaults to the
        same delay used for API calls (1.1 s).

    Returns
    -------
    dict with keys:
        "downloaded" — PDFs successfully fetched and saved
        "failed"     — URLs that returned errors
        "skipped"    — papers with no URL, or already downloaded
    """
    # Get all papers that still need their PDFs downloaded.
    papers = store.get_papers_with_pdf()
    counts = {"downloaded": 0, "failed": 0, "skipped": 0}

    logger.info("Found %d papers with open-access PDFs to download.", len(papers))

    for row in papers:
        paper_id = row["paper_id"]
        pdf_url  = row["pdf_url"]
        year     = row["year"]

        # Skip rows where the URL is somehow NULL (shouldn't happen, but defensive).
        if not pdf_url:
            counts["skipped"] += 1
            continue

        # Compute the destination path for this paper's PDF.
        dest = _pdf_path(paper_id, year)

        # If the file already exists on disk (e.g. from a previous interrupted
        # run), just update the DB record and skip the download.
        if dest.exists():
            store.set_local_pdf_path(paper_id, str(dest))
            counts["skipped"] += 1
            continue

        # Attempt the download.
        success = download_pdf(pdf_url, dest)
        if success:
            # Record the local path so the processing pipeline can find this file.
            store.set_local_pdf_path(paper_id, str(dest))
            counts["downloaded"] += 1
        else:
            counts["failed"] += 1

        # Pause between downloads to be a polite HTTP client.
        time.sleep(delay)

    logger.info(
        "Download complete: %d downloaded, %d failed, %d skipped.",
        counts["downloaded"], counts["failed"], counts["skipped"],
    )
    return counts
