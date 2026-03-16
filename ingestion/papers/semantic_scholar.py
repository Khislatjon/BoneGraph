"""
ingestion/papers/semantic_scholar.py
=====================================
Client for the Semantic Scholar Graph API (https://api.semanticscholar.org).

Responsibilities
----------------
- Build and send HTTP requests to the S2 API.
- Handle pagination so the caller gets a simple stream of paper dicts.
- Enforce rate limiting (1 request/second with an API key).
- Automatically retry on transient failures (network errors, 5xx, 429).

This module does NOT write anything to disk or to the database — it only
fetches data from the network.  Storage is handled by storage.py.

Usage example
-------------
    client = SemanticScholarClient()
    for paper in client.search_papers("cortical bone fracture toughness"):
        print(paper["title"], paper["year"])
"""

from __future__ import annotations  # enables X | Y union syntax on Python 3.9

import logging
import time
from typing import Any, Generator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import (
    MAX_PAPERS_PER_QUERY,
    MAX_RETRIES,
    PAPER_FIELDS,
    REQUEST_DELAY_SECONDS,
    RETRY_BACKOFF_SECONDS,
    SEARCH_BATCH_SIZE,
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_BASE_URL,
)

# Module-level logger — messages appear as "ingestion.papers.semantic_scholar"
# in log output, making it easy to trace which module produced a log line.
logger = logging.getLogger(__name__)


class SemanticScholarClient:
    """
    Thin wrapper around the Semantic Scholar Graph API v1.

    The class holds a persistent requests.Session so that the underlying TCP
    connection to the S2 server is reused across multiple requests, which is
    faster than opening a new connection each time.

    Parameters
    ----------
    api_key : str
        Your Semantic Scholar API key.  Read from SEMANTIC_SCHOLAR_API_KEY in
        .env by default.  Pass an empty string to use the unauthenticated tier
        (slower — 1 req/5s).
    """

    def __init__(self, api_key: str = SEMANTIC_SCHOLAR_API_KEY):
        # Store the base URL so it can be referenced in all request methods.
        self.base_url = SEMANTIC_SCHOLAR_BASE_URL

        # The API key is sent as a custom HTTP header on every request.
        # When api_key is empty we send no header and the unauthenticated
        # rate limit applies.
        self.headers = {"x-api-key": api_key} if api_key else {}

        # Build the session once at construction time.
        self.session = self._build_session()

    # ── Session setup ─────────────────────────────────────────────────────────

    def _build_session(self) -> requests.Session:
        """
        Create a requests.Session pre-configured with:
        - Our API-key header on every outgoing request.
        - Automatic retries on transient errors (network failures, 5xx, 429).

        A Session is more efficient than bare requests.get() because it
        maintains a connection pool (keep-alive) to the same host.
        """
        session = requests.Session()

        # Retry strategy:
        #   total=MAX_RETRIES      — retry up to this many times total
        #   backoff_factor         — wait backoff_factor * (2^retry) seconds between retries
        #   status_forcelist       — HTTP status codes that should trigger a retry:
        #                            429 = Too Many Requests (rate limited)
        #                            500/502/503/504 = server-side errors
        #   allowed_methods        — only retry idempotent GET requests automatically
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=RETRY_BACKOFF_SECONDS,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )

        # Mount the retry adapter for all HTTPS URLs.
        # HTTPAdapter wraps urllib3 and applies the retry strategy.
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)

        # Merge our API key header into every request this session makes.
        session.headers.update(self.headers)
        return session

    # ── Core request ──────────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Send a GET request to `<base_url>/<endpoint>` with the given query
        parameters and return the parsed JSON response body as a dict.

        If the server responds with 429, we read the Retry-After header (or
        fall back to 60 s) and wait before retrying once manually.  The
        requests Retry adapter handles most retries, but a manual check here
        gives cleaner log output.

        Parameters
        ----------
        endpoint : str
            The API path after the base URL, e.g. "paper/search".
        params : dict
            URL query parameters, e.g. {"query": "bone", "limit": 100}.

        Returns
        -------
        dict
            Parsed JSON response.

        Raises
        ------
        requests.HTTPError
            If the response status code indicates a non-retriable error.
        """
        url = f"{self.base_url}/{endpoint}"
        response = self.session.get(url, params=params, timeout=30)

        # Manual 429 handling: read the server's recommended wait time.
        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 60))
            logger.warning("Rate limited by S2. Waiting %ds before retry.", wait)
            time.sleep(wait)
            # Retry the same request once after waiting.
            response = self.session.get(url, params=params, timeout=30)

        # Raise an exception for any 4xx/5xx that wasn't handled above.
        response.raise_for_status()
        return response.json()

    # ── Search ────────────────────────────────────────────────────────────────

    def search_papers(
        self,
        query: str,
        max_papers: int = MAX_PAPERS_PER_QUERY,
        year_range: str | None = None,
        fields: list[str] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Yield papers from the S2 full-text search index matching *query*.

        This is a *generator* — it fetches pages lazily and yields one paper
        dict at a time.  The caller never has to think about pagination.

        How pagination works
        --------------------
        S2 returns at most SEARCH_BATCH_SIZE (100) papers per request.
        We track an `offset` counter and keep requesting the next page until
        either we've yielded `max_papers` or S2 has no more results.

        Parameters
        ----------
        query : str
            Free-text search string, e.g. "bone fracture toughness cortical".
            Tip: S2 searches title + abstract, so multi-word phrases work well.
        max_papers : int
            Hard cap on the number of papers to yield.  Defaults to the value
            in settings (500).
        year_range : str | None
            Optional publication-year filter in S2 format:
            "2015-2024"  → papers published 2015 to 2024 inclusive
            "2020-"      → papers published 2020 or later
            None         → no year filter (all years)
        fields : list[str] | None
            Which paper fields to retrieve.  Defaults to PAPER_FIELDS from
            settings.  Override only if you need a custom subset.

        Yields
        ------
        dict
            A single paper record as returned by the S2 API.  Always contains
            at least "paperId" and "title".  Other fields depend on availability.
        """
        # Use the default fields from settings if none are provided.
        fields = fields or PAPER_FIELDS

        # offset tracks how many results we've already fetched from S2.
        # We increment it by the number of results in each response page.
        offset = 0

        # total_yielded counts papers given to the caller so we can stop at max_papers.
        total_yielded = 0

        logger.info("Searching S2: '%s' (max=%d)", query, max_papers)

        while total_yielded < max_papers:
            # Request exactly as many results as we still need, but never more
            # than SEARCH_BATCH_SIZE (S2's hard per-request cap of 100).
            batch_size = min(SEARCH_BATCH_SIZE, max_papers - total_yielded)

            # Build the query parameters for this page request.
            params: dict[str, Any] = {
                "query": query,
                "fields": ",".join(fields),  # S2 expects a comma-separated string
                "limit": batch_size,
                "offset": offset,            # skip the first `offset` results
            }
            if year_range:
                params["year"] = year_range  # e.g. "2015-2024"

            try:
                data = self._get("paper/search", params)
            except requests.HTTPError as exc:
                # Log and stop this query; don't crash the entire pipeline.
                logger.error("HTTP error for query '%s': %s", query, exc)
                break
            except requests.RequestException as exc:
                # Covers connection errors, timeouts, etc.
                logger.error("Request failed for query '%s': %s", query, exc)
                break

            # "data" is the list of paper dicts in this page.
            papers = data.get("data", [])
            if not papers:
                # Empty page means S2 has no more results for this query.
                logger.info("No more results for '%s' at offset %d.", query, offset)
                break

            # "total" is the total number of matching papers in the S2 index.
            # We use it to detect when we've exhausted all available results.
            total_in_index = data.get("total", 0)
            logger.debug(
                "Query '%s': offset=%d, page_size=%d, total_in_index=%d",
                query, offset, len(papers), total_in_index,
            )

            # Yield each paper in this page to the caller.
            for paper in papers:
                yield paper
                total_yielded += 1
                # Stop early if we've reached the max_papers cap.
                if total_yielded >= max_papers:
                    break

            # Advance offset to the next page.
            offset += len(papers)

            # Stop if S2's index has no more results beyond this offset.
            if offset >= total_in_index:
                break

            # Respect the rate limit before fetching the next page.
            time.sleep(REQUEST_DELAY_SECONDS)

        logger.info("Finished query '%s': yielded %d papers.", query, total_yielded)

    def get_paper(
        self, paper_id: str, fields: list[str] | None = None
    ) -> dict[str, Any] | None:
        """
        Fetch the full record for a single paper by its identifier.

        Parameters
        ----------
        paper_id : str
            Either a Semantic Scholar paper ID (40-char hex string) or an
            external ID in prefixed form, e.g.:
            - "DOI:10.1016/j.bone.2022.01.001"
            - "ARXIV:2301.12345"
            - "PMID:12345678"
        fields : list[str] | None
            Fields to retrieve.  Defaults to PAPER_FIELDS.

        Returns
        -------
        dict or None
            Paper record dict, or None if the request failed.
        """
        fields = fields or PAPER_FIELDS
        try:
            return self._get(f"paper/{paper_id}", {"fields": ",".join(fields)})
        except requests.HTTPError as exc:
            logger.error("Failed to fetch paper %s: %s", paper_id, exc)
            return None

    def get_paper_references(
        self, paper_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Return the list of papers *cited by* the given paper (its reference list).

        Useful for citation-graph expansion: start from a seed paper and
        recursively collect the papers it cites to grow the corpus.

        Parameters
        ----------
        paper_id : str
            Semantic Scholar paper ID.
        limit : int
            Maximum number of references to return (max 1000 per S2 docs).

        Returns
        -------
        list[dict]
            Each element is a paper dict for one cited paper.
        """
        # We only need a lightweight field set for reference expansion.
        fields = "paperId,title,year,authors,venue,citationCount"
        try:
            data = self._get(
                f"paper/{paper_id}/references",
                {"fields": fields, "limit": limit},
            )
            # The "references" endpoint wraps each entry in a {"citedPaper": {...}} dict.
            return [entry.get("citedPaper", {}) for entry in data.get("data", [])]
        except requests.RequestException as exc:
            logger.error("Failed to fetch references for %s: %s", paper_id, exc)
            return []

    def get_paper_citations(
        self, paper_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Return the list of papers that *cite* the given paper.

        Useful for finding newer work that builds on an important paper.

        Parameters
        ----------
        paper_id : str
            Semantic Scholar paper ID.
        limit : int
            Maximum number of citing papers to return.

        Returns
        -------
        list[dict]
            Each element is a paper dict for one citing paper.
        """
        fields = "paperId,title,year,authors,venue,citationCount"
        try:
            data = self._get(
                f"paper/{paper_id}/citations",
                {"fields": fields, "limit": limit},
            )
            # The "citations" endpoint wraps each entry in a {"citingPaper": {...}} dict.
            return [entry.get("citingPaper", {}) for entry in data.get("data", [])]
        except requests.RequestException as exc:
            logger.error("Failed to fetch citations for %s: %s", paper_id, exc)
            return []
