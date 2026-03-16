"""
ingestion/papers/storage.py
============================
SQLite-backed store for paper metadata.

Why SQLite?
-----------
SQLite is a single-file, serverless relational database built into Python's
standard library.  No installation or running server is needed.  The entire
corpus metadata (titles, abstracts, authors, URLs…) for 25,000 papers fits in
~50 MB.  It is also fully queryable with SQL, which makes exploration easy.

Schema overview
---------------
papers       — one row per unique paper, keyed by S2 paperId.
search_runs  — audit log of every keyword query that was run.

Deduplication strategy
----------------------
The `paper_id` column is the PRIMARY KEY.  The INSERT … ON CONFLICT … DO UPDATE
pattern (also called "upsert") means that if we try to insert a paper that
already exists, SQLite silently updates it instead of raising an error.  This
is how we avoid duplicates when the same paper is found by multiple keywords.
The `keywords_matched` column accumulates all the query strings that led to
this paper, which is useful for understanding the coverage of your keyword set.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from config.settings import PAPERS_DB_PATH

logger = logging.getLogger(__name__)

# SQL executed once at startup to create tables if they don't already exist.
# "CREATE TABLE IF NOT EXISTS" is idempotent — safe to run on every connection.
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id          TEXT PRIMARY KEY,   -- Semantic Scholar's stable paper hash
    title             TEXT,               -- Full title of the paper
    abstract          TEXT,               -- Abstract — primary text for RAG
    year              INTEGER,            -- Publication year, e.g. 2022
    venue             TEXT,               -- Journal or conference name
    citation_count    INTEGER DEFAULT 0,  -- How many papers cite this one
    reference_count   INTEGER DEFAULT 0,  -- How many papers this one cites
    has_pdf           INTEGER DEFAULT 0,  -- 1 if open-access PDF URL is available
    pdf_url           TEXT,               -- URL of the open-access PDF (or NULL)
    pdf_local_path    TEXT,               -- Local filesystem path after download
    authors_json      TEXT,               -- JSON: [{authorId, name}, ...]
    external_ids_json TEXT,               -- JSON: {DOI: "...", ArXiv: "...", ...}
    fields_of_study   TEXT,               -- JSON: ["Medicine", "Biology", ...]
    pub_date          TEXT,               -- Full date string "YYYY-MM-DD"
    pub_types         TEXT,               -- JSON: ["JournalArticle", "Review", ...]
    keywords_matched  TEXT,               -- JSON: list of query strings that found this paper
    created_at        TEXT DEFAULT (datetime('now')),  -- First time we saw this paper
    updated_at        TEXT DEFAULT (datetime('now'))   -- Last time we touched this row
);

CREATE TABLE IF NOT EXISTS search_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword      TEXT NOT NULL,     -- The keyword string that was queried
    total_found  INTEGER,           -- How many results S2 reported in total
    fetched      INTEGER,           -- How many we actually retrieved (capped at max)
    run_at       TEXT DEFAULT (datetime('now'))
);
"""


class PaperStore:
    """
    Context-manager interface to the SQLite paper database.

    Typical usage::

        with PaperStore() as store:
            store.upsert_papers(papers, keyword="bone mechanics")
            print(store.stats())

    The context manager guarantees the connection is closed cleanly even if
    an exception occurs inside the `with` block.
    """

    def __init__(self, db_path: Path = PAPERS_DB_PATH):
        # db_path is where the SQLite file will be created (or opened if it
        # already exists).  Defaults to data/db/papers.db from settings.
        self.db_path = db_path

        # _conn holds the live database connection.  It starts as None and is
        # set by connect().  Using None as the sentinel lets us detect
        # accidental calls before connect().
        self._conn: sqlite3.Connection | None = None

    # ── Connection management ─────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Open the SQLite connection and create tables if they don't exist yet.

        row_factory = sqlite3.Row makes each returned row behave like an
        ordered dict — you can access columns by name (row["title"]) instead
        of by index (row[1]).
        """
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        # executescript runs the multi-statement CREATE TABLE SQL in one call.
        self._conn.executescript(CREATE_TABLE_SQL)
        self._conn.commit()
        logger.debug("Connected to paper store at %s", self.db_path)

    def close(self) -> None:
        """Flush any pending writes and close the connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        """Called at the start of a `with PaperStore() as store:` block."""
        self.connect()
        return self

    def __exit__(self, *_):
        """Called at the end of the `with` block (even if an exception occurred)."""
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        """
        Return the live connection, raising a clear error if connect() was
        never called.  Using a property hides the _conn implementation detail
        from callers.
        """
        if self._conn is None:
            raise RuntimeError("Call connect() or use as context manager first.")
        return self._conn

    # ── Write operations ──────────────────────────────────────────────────────

    def upsert_paper(self, paper: dict[str, Any], keyword: str) -> bool:
        """
        Insert a paper into the database, or update it if it already exists.

        "Upsert" = INSERT … ON CONFLICT … DO UPDATE.  SQLite detects a
        duplicate paper_id and updates the existing row instead of raising an
        error.  This is the core deduplication mechanism.

        When the same paper is found by multiple keywords, the `keywords_matched`
        column accumulates all of them so you can later see which queries found
        which papers.

        Parameters
        ----------
        paper : dict
            A paper dict as returned by SemanticScholarClient.search_papers().
        keyword : str
            The search keyword that produced this paper — recorded for auditing.

        Returns
        -------
        bool
            True if this was a brand-new paper (first time we've seen this ID).
            False if we updated an existing row.
        """
        pid = paper.get("paperId")
        if not pid:
            # S2 occasionally returns partial records without an ID — skip them.
            return False

        # Extract the open-access PDF URL if one exists.
        # The S2 field is either {"url": "https://..."} or None.
        open_pdf = paper.get("openAccessPdf") or {}
        pdf_url = open_pdf.get("url") if isinstance(open_pdf, dict) else None

        # Check if this paper is already in the database.
        existing = self.get_paper(pid)
        if existing:
            # Deserialise the current keywords list and append the new keyword
            # if it isn't already there.  Re-serialise to JSON for storage.
            matched: list[str] = json.loads(existing["keywords_matched"] or "[]")
            if keyword not in matched:
                matched.append(keyword)
            keywords_json = json.dumps(matched)
        else:
            # First time we see this paper — start the keyword list with one entry.
            keywords_json = json.dumps([keyword])

        # Execute the upsert.
        # Named placeholders (:pid, :title …) make the SQL readable and prevent
        # SQL injection even though all data here comes from the S2 API.
        self.conn.execute(
            """
            INSERT INTO papers (
                paper_id, title, abstract, year, venue,
                citation_count, reference_count,
                has_pdf, pdf_url,
                authors_json, external_ids_json,
                fields_of_study, pub_date, pub_types,
                keywords_matched, updated_at
            ) VALUES (
                :pid, :title, :abstract, :year, :venue,
                :cites, :refs,
                :has_pdf, :pdf_url,
                :authors, :ext_ids,
                :fos, :pub_date, :pub_types,
                :kw, datetime('now')
            )
            ON CONFLICT(paper_id) DO UPDATE SET
                citation_count   = excluded.citation_count,
                keywords_matched = excluded.keywords_matched,
                updated_at       = excluded.updated_at
            """,
            {
                "pid": pid,
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                # Default to 0 if the field is missing (some preprints omit counts).
                "cites": paper.get("citationCount", 0),
                "refs": paper.get("referenceCount", 0),
                # Store 1/0 integer because SQLite has no boolean type.
                "has_pdf": 1 if pdf_url else 0,
                "pdf_url": pdf_url,
                # Serialise nested structures to JSON strings for TEXT columns.
                "authors": json.dumps(paper.get("authors", [])),
                "ext_ids": json.dumps(paper.get("externalIds", {})),
                # s2FieldsOfStudy is a list of {category, source} dicts — extract just the labels.
                "fos": json.dumps(
                    [f.get("category") for f in (paper.get("s2FieldsOfStudy") or [])]
                ),
                "pub_date": paper.get("publicationDate"),
                "pub_types": json.dumps(paper.get("publicationTypes") or []),
                "kw": keywords_json,
            },
        )
        self.conn.commit()
        # Return True if this was a new insert (existing was None before).
        return existing is None

    def upsert_papers(self, papers: list[dict[str, Any]], keyword: str) -> tuple[int, int]:
        """
        Upsert a batch of papers and return counts of new vs updated rows.

        Parameters
        ----------
        papers : list[dict]
            List of paper dicts from the S2 API.
        keyword : str
            The search keyword that produced these papers.

        Returns
        -------
        tuple[int, int]
            (new_count, updated_count) — how many rows were inserted vs updated.
        """
        new, updated = 0, 0
        for p in papers:
            is_new = self.upsert_paper(p, keyword)
            if is_new:
                new += 1
            else:
                updated += 1
        return new, updated

    def record_search_run(self, keyword: str, total_found: int, fetched: int) -> None:
        """
        Insert a row into search_runs to record that a keyword was queried.

        This audit log lets you see which keywords have been run, when, and
        how many results each one returned.  Useful for resuming interrupted
        runs or checking coverage.

        Parameters
        ----------
        keyword : str
            The search string that was queried.
        total_found : int
            The total number of papers S2 reported matching this keyword
            (may be much larger than what we fetched, due to the max cap).
        fetched : int
            The number of paper records we actually retrieved and stored.
        """
        self.conn.execute(
            "INSERT INTO search_runs (keyword, total_found, fetched) VALUES (?, ?, ?)",
            (keyword, total_found, fetched),
        )
        self.conn.commit()

    def set_local_pdf_path(self, paper_id: str, local_path: str) -> None:
        """
        Record the local filesystem path of a downloaded PDF.

        Called by the downloader after a PDF is successfully saved to disk.
        This path is later used by the processing pipeline to open the file.

        Parameters
        ----------
        paper_id : str
            The S2 paper ID to update.
        local_path : str
            Absolute path to the saved PDF file on disk.
        """
        self.conn.execute(
            "UPDATE papers SET pdf_local_path = ?, updated_at = datetime('now') WHERE paper_id = ?",
            (local_path, paper_id),
        )
        self.conn.commit()

    # ── Read operations ────────────────────────────────────────────────────────

    def get_paper(self, paper_id: str) -> sqlite3.Row | None:
        """
        Return the database row for a single paper, or None if not found.

        sqlite3.Row objects support both index access (row[0]) and name access
        (row["title"]), and can be converted to a plain dict with dict(row).

        Parameters
        ----------
        paper_id : str
            The S2 paper ID to look up.
        """
        cur = self.conn.execute(
            "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
        )
        return cur.fetchone()

    def get_papers_with_pdf(self) -> list[sqlite3.Row]:
        """
        Return all papers that have an open-access PDF URL but haven't been
        downloaded to disk yet (pdf_local_path IS NULL).

        Used by the downloader to find papers that still need their PDFs fetched.
        """
        cur = self.conn.execute(
            "SELECT * FROM papers WHERE has_pdf = 1 AND pdf_local_path IS NULL"
        )
        return cur.fetchall()

    def total_papers(self) -> int:
        """Return the total number of papers currently in the database."""
        cur = self.conn.execute("SELECT COUNT(*) FROM papers")
        return cur.fetchone()[0]

    def stats(self) -> dict[str, int]:
        """
        Return a summary of database contents.

        Returns
        -------
        dict with keys:
            "total"          — total number of papers stored
            "with_open_pdf"  — how many have an open-access PDF URL available
        """
        cur = self.conn.execute(
            "SELECT COUNT(*) as total, SUM(has_pdf) as with_pdf FROM papers"
        )
        row = cur.fetchone()
        return {"total": row["total"] or 0, "with_open_pdf": row["with_pdf"] or 0}
