"""
ingestion/papers/openalex.py
=============================
Client for the OpenAlex REST API (https://api.openalex.org).

Why OpenAlex?
-------------
- Completely free and open — no API key required.
- Natively filters by language (filter=language:en), so non-English papers
  are excluded at query time before touching the database.
- 200-result pages (vs Semantic Scholar's 100) and 10 req/s polite-pool
  rate limit — roughly 10× faster ingestion than S2 without a key.
- Covers ~250M works including all major bone science journals.

Polite pool
-----------
OpenAlex provides a "polite pool" with better reliability when an email
address is included in the User-Agent header (no registration needed).
Set UNPAYWALL_EMAIL in .env — the same address is reused here.

Abstract format
---------------
OpenAlex stores abstracts as an *inverted index*: a dict mapping each word
to the list of positions it appears at, e.g. {"Bone": [0], "fracture": [1]}.
reconstruct_abstract() reverses this into a plain text string.

Output format
-------------
_normalize() converts each OpenAlex Work dict into a plain dict that matches
exactly the keys expected by ingestion/papers/storage.py — the same keys
that Semantic Scholar used — so storage.py, downloader.py, and the rest of
the pipeline need zero changes.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Generator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import (
    DEFAULT_LANGUAGE,
    DEFAULT_YEAR_RANGE,
    MAX_PAPERS_PER_QUERY,
    MAX_RETRIES,
    OPENALEX_BASE_URL,
    OPENALEX_SELECT_FIELDS,
    REQUEST_DELAY_SECONDS,
    RETRY_BACKOFF_SECONDS,
    SEARCH_BATCH_SIZE,
    UNPAYWALL_EMAIL,
)

logger = logging.getLogger(__name__)


# ── Abstract reconstruction ───────────────────────────────────────────────────

def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """
    Rebuild a plain-text abstract from OpenAlex's inverted-index format.

    OpenAlex stores abstracts as {word: [position, ...]} to save space and
    avoid copyright issues with verbatim text storage.  We reverse the mapping
    to reconstruct the original word order.

    Parameters
    ----------
    inverted_index : dict or None
        The abstract_inverted_index field from an OpenAlex Work object.
        Example: {"Cortical": [0], "bone": [1, 5], "fracture": [2], ...}

    Returns
    -------
    str or None
        Reconstructed abstract text, or None if the index is absent/empty.
    """
    if not inverted_index:
        return None
    position_word: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word[pos] = word
    if not position_word:
        return None
    return " ".join(position_word[i] for i in sorted(position_word))


# ── Client ────────────────────────────────────────────────────────────────────

class OpenAlexClient:
    """
    Thin wrapper around the OpenAlex Works search endpoint.

    No API key required.  Providing an email in the User-Agent header opts
    into the polite pool (better rate-limit headroom).

    Usage
    -----
        client = OpenAlexClient()
        for paper in client.search_papers("cortical bone fracture"):
            print(paper["title"], paper["year"])
    """

    def __init__(self, email: str = UNPAYWALL_EMAIL):
        self.base_url = OPENALEX_BASE_URL
        self.email = email
        self.session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()

        # Polite pool: email in User-Agent gives more reliable rate limits.
        ua = "BoneLogic-Research-Bot/1.0"
        if self.email:
            ua += f" (mailto:{self.email})"
        session.headers.update({"User-Agent": ua})

        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        return session

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a GET request and return the parsed JSON response."""
        url = f"{self.base_url}/{endpoint}"
        response = self.session.get(url, params=params, timeout=30)

        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", RETRY_BACKOFF_SECONDS))
            logger.warning("Rate limited by OpenAlex. Waiting %ds before retry.", wait)
            time.sleep(wait)
            response = self.session.get(url, params=params, timeout=30)

        response.raise_for_status()
        return response.json()

    def search_papers(
        self,
        query: str,
        max_papers: int = MAX_PAPERS_PER_QUERY,
        year_range: str | None = DEFAULT_YEAR_RANGE,
        language: str | None = DEFAULT_LANGUAGE,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Yield normalised paper dicts from OpenAlex matching *query*.

        Uses cursor-based pagination — OpenAlex's recommended approach for
        deep result sets.  Each page returns a `next_cursor` token used to
        fetch the following page.

        Parameters
        ----------
        query : str
            Free-text search string (searched against title + abstract).
        max_papers : int
            Maximum papers to yield (default MAX_PAPERS_PER_QUERY).
        year_range : str or None
            Year filter in OpenAlex format: "1970-2026" (inclusive).
        language : str or None
            ISO 639-1 language code to filter by (default "en").
            Pass None to disable language filtering.

        Yields
        ------
        dict
            Normalised paper dict with the same keys as the old S2 format.
        """
        # Build the filter string — OpenAlex combines filters with commas.
        filters: list[str] = []
        if language:
            filters.append(f"language:{language}")
        if year_range:
            filters.append(f"publication_year:{year_range}")
        filter_str = ",".join(filters) if filters else None

        total_yielded = 0
        cursor = "*"  # OpenAlex cursor pagination starts with "*"

        logger.info("Searching OpenAlex: '%s' (max=%d, lang=%s, years=%s)",
                    query, max_papers, language, year_range)

        while total_yielded < max_papers:
            batch_size = min(SEARCH_BATCH_SIZE, max_papers - total_yielded)

            params: dict[str, Any] = {
                "search":    query,
                "select":    ",".join(OPENALEX_SELECT_FIELDS),
                "per-page":  batch_size,
                "cursor":    cursor,
            }
            if filter_str:
                params["filter"] = filter_str

            try:
                data = self._get("works", params)
            except requests.HTTPError as exc:
                logger.error("HTTP error for query '%s': %s", query, exc)
                break
            except requests.RequestException as exc:
                logger.error("Request failed for query '%s': %s", query, exc)
                break

            results = data.get("results", [])
            if not results:
                logger.info("No more results for '%s' at cursor %s.", query, cursor)
                break

            for work in results:
                normalized = self._normalize(work)
                if normalized:
                    yield normalized
                    total_yielded += 1
                    if total_yielded >= max_papers:
                        break

            # Advance cursor for next page.
            meta = data.get("meta", {})
            next_cursor = meta.get("next_cursor")
            if not next_cursor:
                break
            cursor = next_cursor

            time.sleep(REQUEST_DELAY_SECONDS)

        logger.info("Finished query '%s': yielded %d papers.", query, total_yielded)

    def _normalize(self, work: dict[str, Any]) -> dict[str, Any] | None:
        """
        Convert an OpenAlex Work object to our internal paper dict format.

        The output keys are identical to what Semantic Scholar returned so
        that storage.py, downloader.py, and the rest of the pipeline work
        without any modification.
        """
        raw_id = work.get("id", "")
        if not raw_id:
            return None

        # OpenAlex ID is a full URL — extract just the short ID (e.g. "W2741809807").
        paper_id = raw_id.rstrip("/").split("/")[-1]

        # ── Abstract ────────────────────────────────────────────────────────
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

        # ── Authors (up to 10) ───────────────────────────────────────────────
        authors = []
        for authorship in (work.get("authorships") or [])[:10]:
            name = (authorship.get("author") or {}).get("display_name", "")
            if name:
                authors.append({"name": name})

        # ── Venue ────────────────────────────────────────────────────────────
        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}
        venue = source.get("display_name") or ""

        # ── Open-access PDF URL ───────────────────────────────────────────────
        # Prefer primary_location.pdf_url (direct link); fall back to oa_url.
        pdf_url = primary_location.get("pdf_url")
        if not pdf_url:
            open_access = work.get("open_access") or {}
            pdf_url = open_access.get("oa_url")

        # ── External IDs ─────────────────────────────────────────────────────
        doi_raw = work.get("doi") or ""
        external_ids: dict[str, str] = {}
        if doi_raw:
            # Strip the "https://doi.org/" prefix if present.
            external_ids["DOI"] = doi_raw.replace("https://doi.org/", "")

        # ── Publication type ─────────────────────────────────────────────────
        work_type = work.get("type") or ""
        pub_types = [work_type] if work_type else []

        return {
            # Core identity
            "paperId":          paper_id,
            "title":            work.get("title"),
            "abstract":         abstract,
            "year":             work.get("publication_year"),
            "venue":            venue,
            # Counts
            "citationCount":    work.get("cited_by_count", 0),
            "referenceCount":   0,   # not requested (would require large list)
            # Open access
            "openAccessPdf":    {"url": pdf_url} if pdf_url else None,
            # Metadata
            "authors":          authors,
            "externalIds":      external_ids,
            "publicationDate":  work.get("publication_date"),
            "publicationTypes": pub_types,
            "fieldsOfStudy":    [],
            "s2FieldsOfStudy":  [],
            "language":         work.get("language"),
        }
