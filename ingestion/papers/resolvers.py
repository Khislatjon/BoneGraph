"""
ingestion/papers/resolvers.py
==============================
Resolves stored PDF URLs to actual downloadable PDF file URLs.

The Problem
-----------
Semantic Scholar's openAccessPdf field sometimes contains:
  1. Direct PDF links        → work as-is   (e.g. .../article.pdf)
  2. doi.org links           → redirect to publisher landing page
  3. EuropePMC viewer pages  → show a PDF viewer, not the raw file
  4. PubMed Central pages    → HTML page that needs /pdf/ appended
  5. Publisher landing pages → need publisher-specific URL patterns

This module applies a 3-tier resolution chain to find a direct PDF URL
for each paper regardless of what S2 gave us.

Resolution Strategy (tried in order)
-------------------------------------
Tier 1 — Direct-PDF check
    If the URL already points directly to a PDF (ends in .pdf or matches
    known direct-PDF path patterns), return it immediately. No network call.

Tier 2 — Publisher URL transforms
    Pattern-match the stored URL against known publishers and rewrite it
    to a direct PDF URL. Covers: EuropePMC, PubMed Central, PLOS, Frontiers,
    MDPI, Wiley, Springer. No network call.

Tier 3 — Unpaywall API
    If Tiers 1-2 fail (e.g. doi.org links, unrecognised publishers), call
    the Unpaywall API with the paper's DOI. Unpaywall is a free, legal
    database of open-access papers — it returns a direct PDF URL where one
    exists. Requires UNPAYWALL_EMAIL in your .env file (no API key needed).

Usage
-----
    from ingestion.papers.resolvers import resolve_pdf_url

    url = resolve_pdf_url(
        paper_id   = "abc123",
        stored_url = "https://doi.org/10.1007/s10704-024-00836-w",
        doi        = "10.1007/s10704-024-00836-w",
        session    = my_requests_session,
    )
    # url is now a direct PDF link, or None if unresolvable
"""

from __future__ import annotations

import logging
import re
import time
from urllib.parse import quote

import requests

from config.settings import REQUEST_DELAY_SECONDS, CROSSREF_EMAIL, WILEY_TDM_TOKEN

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# CrossRef REST API base URL.
# Documentation: https://api.crossref.org
_CROSSREF_BASE = "https://api.crossref.org/works"

# Headers for CrossRef API requests.
# Explicitly set Accept: application/json to override any session-level
# Accept header (downloader sessions use application/pdf,*/* which breaks
# JSON APIs).
_HEADERS = {
    "User-Agent": (
        "BoneLogic-Research-Bot/1.0 "
        f"(PhD research project; academic use only; mailto:{CROSSREF_EMAIL})"
    ),
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",  # no brotli — requests doesn't support it by default
}

# Small courtesy delay between CrossRef requests.
_CROSSREF_DELAY = 0.05


# ── Tier 1: Direct PDF detection ──────────────────────────────────────────────

# URL substrings that are strongly associated with direct PDF responses
# (not landing pages or HTML viewers).
_DIRECT_PDF_PATTERNS = [
    "/article/file?",           # PLOS journals — parameter-based PDF link
    "blobtype=pdf",             # EuropePMC backend PDF endpoint
    "/epdf/",                   # Wiley direct PDF path
    "/pdfdirect",               # Wiley direct PDF variant path
    "/content/pdf/",            # Springer PDF content path
    "/article/am/pii/",         # ScienceDirect accepted manuscript (served as PDF)
    "type=printable",           # PLOS printable PDF
    "renderformats=PDF",        # Some biomedical publishers
]


def is_direct_pdf(url: str) -> bool:
    """
    Return True if the URL is very likely to deliver a raw PDF file
    rather than an HTML landing page or viewer.

    Checks two things:
      1. The URL path ends with ".pdf" (explicit file extension).
      2. The URL contains a known direct-PDF path substring used by
         major academic publishers.

    Parameters
    ----------
    url : str
        The URL to inspect. Case-insensitive.

    Returns
    -------
    bool
        True if the URL appears to point directly to a PDF file.

    Examples
    --------
    >>> is_direct_pdf("https://www.nature.com/articles/s41598-018-36742-0.pdf")
    True
    >>> is_direct_pdf("https://doi.org/10.1007/s10704-024-00836-w")
    False
    """
    if not url:
        return False

    lower = url.lower()

    # Explicit .pdf extension is the strongest signal
    if lower.endswith(".pdf"):
        return True

    # Known direct-PDF path fragments from common academic publishers
    return any(pattern in lower for pattern in _DIRECT_PDF_PATTERNS)


# ── Tier 2: Publisher-specific URL transforms ──────────────────────────────────

def _transform_europepmc(url: str) -> str | None:
    """
    Convert a EuropePMC article viewer URL to the backend direct PDF URL.

    EuropePMC hosts open-access biomedical papers and exposes a backend
    endpoint that serves raw PDFs given a PMC accession number.

    Transform examples:
        https://europepmc.org/articles/pmc4955555?pdf=render
        → https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC4955555&blobtype=pdf

        https://europepmc.org/articles/PMC6263169
        → https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6263169&blobtype=pdf

    Parameters
    ----------
    url : str
        Any URL. Returns None if it is not a EuropePMC article URL.

    Returns
    -------
    str | None
        Direct PDF download URL, or None if the pattern did not match.
    """
    if "europepmc.org/articles" not in url.lower():
        return None

    # Extract the numeric PMC ID from the URL path (case-insensitive)
    match = re.search(r"/articles/[Pp][Mm][Cc](\d+)", url)
    if not match:
        return None

    pmc_id = f"PMC{match.group(1)}"
    return (
        f"https://europepmc.org/backend/ptpmcrender.fcgi"
        f"?accid={pmc_id}&blobtype=pdf"
    )


def _transform_pubmed_central(url: str) -> str | None:
    """
    Convert a PubMed Central article page URL to a direct PDF URL.

    PubMed Central (NCBI) hosts open-access papers. Their article HTML
    pages have a /pdf/ sub-path that serves the PDF directly.

    Transform example:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6263169
        → https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6263169/pdf/

    Parameters
    ----------
    url : str
        Any URL. Returns None if it is not an NCBI PMC URL.

    Returns
    -------
    str | None
        Direct PDF URL, or None if the pattern did not match.
    """
    if "ncbi.nlm.nih.gov/pmc" not in url:
        return None

    # Already pointing to PDF sub-path
    if "/pdf/" in url:
        return url

    match = re.search(r"pmc/articles/(PMC\d+)", url)
    if not match:
        return None

    pmc_id = match.group(1)
    return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"


def _transform_plos(url: str) -> str | None:
    """
    Convert a PLOS journal article page to a direct printable PDF URL.

    PLOS (Public Library of Science) publishes fully open-access journals
    (PLOS ONE, PLOS Medicine, PLOS Biology, etc.). Their article PDF is
    served at the same path but with /article/file?... and type=printable.

    Transform example:
        https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0149604
        → https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0149604&type=printable

    Parameters
    ----------
    url : str
        Any URL. Returns None if it is not a PLOS article URL.

    Returns
    -------
    str | None
        Direct PDF URL, or None if the pattern did not match.
    """
    if "journals.plos.org" not in url:
        return None

    # Match the /article?id= pattern and capture journal base + article ID
    match = re.match(r"(https://journals\.plos\.org/[^/]+)/article\?id=(.+)", url)
    if not match:
        return None

    journal_base = match.group(1)   # e.g. https://journals.plos.org/plosone
    article_id   = match.group(2)   # e.g. 10.1371/journal.pone.0149604

    return f"{journal_base}/article/file?id={article_id}&type=printable"


def _transform_frontiers(url: str) -> str | None:
    """
    Convert a Frontiers journal article /full page to a /pdf URL.

    Frontiers in [Medicine / Bioengineering / Physiology / etc.] is
    open-access. The full-text HTML page ends in /full; replacing it
    with /pdf gives a direct PDF.

    Transform example:
        https://www.frontiersin.org/articles/10.3389/fbioe.2021.123456/full
        → https://www.frontiersin.org/articles/10.3389/fbioe.2021.123456/pdf

    Parameters
    ----------
    url : str
        Any URL. Returns None if it is not a Frontiers /full URL.

    Returns
    -------
    str | None
        Direct PDF URL, or None if the pattern did not match.
    """
    if "frontiersin.org" not in url:
        return None

    if url.rstrip("/").endswith("/full"):
        # Replace trailing /full with /pdf
        return url.rstrip("/")[:-5] + "/pdf"

    return None


def _transform_mdpi(url: str) -> str | None:
    """
    Convert an MDPI article page URL to a direct PDF URL.

    MDPI (Multidisciplinary Digital Publishing Institute) publishes many
    open-access journals (e.g. Materials, IJMS, Biomolecules). Their
    article PDFs are served by appending /pdf to the article URL.

    Transform example:
        https://www.mdpi.com/1422-0067/21/5/1640
        → https://www.mdpi.com/1422-0067/21/5/1640/pdf

    Parameters
    ----------
    url : str
        Any URL. Returns None if it is not an MDPI article URL.

    Returns
    -------
    str | None
        Direct PDF URL, or None if the pattern did not match.
    """
    if "mdpi.com" not in url:
        return None

    # Already a PDF URL
    if url.rstrip("/").endswith("/pdf") or url.lower().endswith(".pdf"):
        return url

    # Avoid double-appending PDF paths
    if "pdf" in url.lower():
        return None

    return url.rstrip("/") + "/pdf"


def _transform_wiley(url: str) -> str | None:
    """
    Convert a Wiley Online Library article page to a direct PDF URL.

    Wiley open-access papers can be downloaded at the /pdfdirect sub-path.
    This covers both onlinelibrary.wiley.com and the ASBMR Wiley sub-domain.

    Transform example:
        https://onlinelibrary.wiley.com/doi/10.1002/jso.21005
        → https://onlinelibrary.wiley.com/doi/10.1002/jso.21005/pdfdirect

    Parameters
    ----------
    url : str
        Any URL. Returns None if it is not a Wiley Online Library URL.

    Returns
    -------
    str | None
        Direct PDF URL, or None if the pattern did not match.
    """
    if "wiley.com" not in url:
        return None

    # Already a direct PDF path — no transform needed
    if "/pdfdirect" in url or "/epdf" in url:
        return url

    # Strip query string before appending PDF path
    base = url.split("?")[0].rstrip("/")
    return base + "/pdfdirect"


def _transform_springer(url: str) -> str | None:
    """
    Convert a Springer article page URL to a direct PDF URL.

    Springer open-access papers are served at /content/pdf/{doi}.pdf.

    Transform example:
        https://link.springer.com/article/10.1007/s10704-024-00836-w
        → https://link.springer.com/content/pdf/10.1007/s10704-024-00836-w.pdf

    Parameters
    ----------
    url : str
        Any URL. Returns None if it is not a link.springer.com article URL.

    Returns
    -------
    str | None
        Direct PDF URL, or None if the pattern did not match.
    """
    if "link.springer.com" not in url:
        return None

    # Already a PDF
    if url.lower().endswith(".pdf") or "/content/pdf/" in url:
        return url

    # Match Springer article path: /article/{doi}
    match = re.match(r"https://link\.springer\.com/article/(.+?)(\?.*)?$", url)
    if not match:
        return None

    doi_path = match.group(1)
    return f"https://link.springer.com/content/pdf/{doi_path}.pdf"


# Registry: all Tier 2 transforms, applied in order.
# Add new publisher transforms here to extend coverage automatically.
_PUBLISHER_TRANSFORMS = [
    _transform_europepmc,
    _transform_pubmed_central,
    _transform_plos,
    _transform_frontiers,
    _transform_mdpi,
    _transform_wiley,
    _transform_springer,
]


def apply_publisher_transforms(url: str) -> str | None:
    """
    Try every registered publisher transform and return the first match.

    Iterates through _PUBLISHER_TRANSFORMS. Each function returns a
    rewritten URL if it recognises the host, or None otherwise.

    Parameters
    ----------
    url : str
        The original URL from the database.

    Returns
    -------
    str | None
        A (hopefully) direct PDF URL if any transform matched, else None.
    """
    for transform_fn in _PUBLISHER_TRANSFORMS:
        result = transform_fn(url)
        if result is not None:
            logger.debug("Publisher transform [%s]: %s → %s",
                         transform_fn.__name__, url[:60], result[:60])
            return result
    return None


# ── Tier 3: CrossRef API ───────────────────────────────────────────────────────

def resolve_via_crossref(
    doi: str,
    session: requests.Session | None = None,
) -> str | None:
    """
    Ask the CrossRef API for a direct PDF URL for a given DOI.

    CrossRef (https://api.crossref.org) is a free DOI registration agency API.
    The `message.link` array in the response contains PDF URLs provided by
    publishers. We pick the first entry with content-type "application/pdf".

    No API key required. Adding your email to the User-Agent header opts into
    the polite pool with higher rate limits.

    API documentation: https://api.crossref.org

    Response structure we care about:
    {
        "message": {
            "link": [
                {
                    "URL": "https://example.com/article.pdf",
                    "content-type": "application/pdf",
                    ...
                }
            ]
        }
    }

    Parameters
    ----------
    doi : str
        The paper's DOI, e.g. "10.1242/dev.117.2.409"
    session : requests.Session | None
        Optional persistent HTTP session. If None, a one-shot request is made.

    Returns
    -------
    str | None
        Direct PDF URL if found in the CrossRef link array, else None.
    """
    if not doi:
        return None

    encoded_doi = quote(doi, safe="")
    endpoint = f"{_CROSSREF_BASE}/{encoded_doi}"

    try:
        requester = session if session is not None else requests
        response  = requester.get(endpoint, headers=_HEADERS, timeout=15)

        # 404 means DOI not registered with CrossRef
        if response.status_code == 404:
            logger.debug("DOI not found in CrossRef: %s", doi)
            return None

        response.raise_for_status()

        if not response.content:
            logger.warning("CrossRef returned empty body for DOI %s", doi)
            return None

        data = response.json()
        links = data.get("message", {}).get("link", [])

        # Pick the first link explicitly marked as application/pdf
        for link in links:
            if link.get("content-type") == "application/pdf":
                pdf_url = link.get("URL")
                if pdf_url:
                    logger.debug("CrossRef resolved %s → %s", doi, pdf_url[:60])
                    return pdf_url

        logger.debug("No PDF link in CrossRef response for DOI %s", doi)
        return None

    except requests.RequestException as exc:
        logger.warning("CrossRef API error for DOI %s: %s", doi, exc)
        return None


# ── Main entry point ───────────────────────────────────────────────────────────

def resolve_pdf_url(
    paper_id:   str,
    stored_url: str | None,
    doi:        str | None,
    session:    requests.Session | None = None,
) -> str | None:
    """
    Find the best directly-downloadable PDF URL for a paper.

    Applies a 3-tier resolution chain, stopping at the first success:

    Tier 1 — Direct PDF check (no network call):
        If stored_url already points to a raw PDF file, return it as-is.

    Tier 2 — Publisher URL transforms (no network call):
        Rewrite stored_url using publisher-specific patterns for:
        EuropePMC, PubMed Central, PLOS, Frontiers, MDPI, Wiley, Springer.

    Tier 3 — Unpaywall API (1 network call):
        If Tiers 1-2 fail, call the Unpaywall API with the paper's DOI.
        Requires UNPAYWALL_EMAIL to be set in .env.

    Parameters
    ----------
    paper_id : str
        Semantic Scholar paper ID (used only for log messages).
    stored_url : str | None
        The URL stored in the papers table from Semantic Scholar.
        May be None if S2 had no open-access URL for this paper.
    doi : str | None
        The paper's DOI if available. Used as the Unpaywall lookup key.
    session : requests.Session | None
        Optional HTTP session for Tier 3 Unpaywall requests.
        Reusing a session is more efficient for large batches.

    Returns
    -------
    str | None
        The best direct PDF URL found, or None if all tiers exhausted.
    """
    # ── Wiley TDM: intercept Wiley URLs and rewrite to TDM API endpoint ───────
    # Wiley's regular URLs are blocked by Cloudflare. The TDM API endpoint
    # at api.wiley.com bypasses this with a token — no Cloudflare involved.
    _wiley_prefixes = (
        "https://onlinelibrary.wiley.com",
        "https://anatomypubs.onlinelibrary.wiley.com",
    )
    if stored_url and any(stored_url.startswith(p) for p in _wiley_prefixes):
        if doi and WILEY_TDM_TOKEN:
            tdm_url = f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}"
            logger.debug("[%s] Wiley TDM: rewriting to %s", paper_id, tdm_url)
            return tdm_url

    # ── Tier 1: check if what we already have is a direct PDF link ────────────
    if stored_url and is_direct_pdf(stored_url):
        logger.debug("[%s] Tier 1: direct PDF link confirmed", paper_id)
        return stored_url

    # ── Tier 2: try publisher-specific URL rewrites ───────────────────────────
    if stored_url:
        transformed = apply_publisher_transforms(stored_url)
        if transformed:
            logger.debug("[%s] Tier 2: publisher transform applied", paper_id)
            return transformed

    # ── Tier 3: CrossRef API lookup ───────────────────────────────────────────
    if doi:
        time.sleep(_CROSSREF_DELAY)   # polite delay before API call
        crossref_url = resolve_via_crossref(doi, session=session)
        if crossref_url:
            # If CrossRef returned any Wiley URL (regular or TDM), rewrite it
            # to the TDM endpoint using the clean unencoded DOI. The TDM API
            # rejects URL-encoded DOIs (%2F instead of /).
            _wiley_domains = ("wiley.com",)
            if any(d in crossref_url for d in _wiley_domains) and WILEY_TDM_TOKEN:
                crossref_url = f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}"
            logger.debug("[%s] Tier 3: CrossRef resolved PDF", paper_id)
            return crossref_url

    # All tiers exhausted — no downloadable PDF found
    logger.debug(
        "[%s] No PDF resolved (stored_url=%s, doi=%s)",
        paper_id, stored_url, doi
    )
    return None
