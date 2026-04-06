"""
ingestion/textbooks/storage.py
================================
SQLite-backed store for textbook metadata.

Schema overview
---------------
textbooks   — one row per textbook PDF, keyed by its file path relative to
              RAW_TEXTBOOKS_DIR.  Stores title, source, page count, file size,
              and processing status.

Deduplication
-------------
The `file_path` column is the PRIMARY KEY.  Re-running the scanner on an
already-registered book does an upsert (INSERT … ON CONFLICT … DO UPDATE),
so it is always safe to re-scan the folder.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from config.settings import TEXTBOOKS_DB_PATH

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS textbooks (
    file_path       TEXT PRIMARY KEY,   -- Path relative to RAW_TEXTBOOKS_DIR
    title           TEXT,               -- Human-readable title (derived from filename)
    source          TEXT,               -- Source name e.g. "MDPI Books", "OpenStax"
    file_size_mb    REAL,               -- File size in megabytes
    page_count      INTEGER,            -- Number of pages in the PDF
    status          TEXT DEFAULT 'registered',  -- registered | processed | error
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);
"""


class TextbookStore:
    """
    Manages the textbooks SQLite database.

    Usage
    -----
    Use as a context manager so the connection is always cleanly closed:

        with TextbookStore() as store:
            store.upsert_textbook(...)
    """

    def __init__(self, db_path: Path = TEXTBOOKS_DB_PATH):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> TextbookStore:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        if self._conn:
            self._conn.close()

    def connect(self) -> None:
        """Open the SQLite connection and create tables if they don't exist."""
        self._conn = sqlite3.connect(self.db_path)
        # Return rows as dict-like objects so columns can be accessed by name.
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(CREATE_TABLE_SQL)
        self._conn.commit()
        logger.info("Connected to textbooks DB at %s", self.db_path)

    def upsert_textbook(
        self,
        file_path: str,
        title: str,
        source: str,
        file_size_mb: float,
        page_count: int | None,
    ) -> None:
        """
        Insert a textbook record or update it if already present.

        Parameters
        ----------
        file_path    : Path relative to RAW_TEXTBOOKS_DIR — used as primary key.
        title        : Human-readable title derived from the filename.
        source       : Source folder name e.g. "MDPI Books".
        file_size_mb : File size in MB.
        page_count   : Number of pages (None if extraction failed).
        """
        self._conn.execute(
            """
            INSERT INTO textbooks (file_path, title, source, file_size_mb, page_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                title        = excluded.title,
                source       = excluded.source,
                file_size_mb = excluded.file_size_mb,
                page_count   = excluded.page_count,
                updated_at   = datetime('now')
            """,
            (file_path, title, source, file_size_mb, page_count),
        )
        self._conn.commit()

    def all_textbooks(self) -> list[sqlite3.Row]:
        """Return all registered textbooks ordered by source then title."""
        cur = self._conn.execute(
            "SELECT * FROM textbooks ORDER BY source, title"
        )
        return cur.fetchall()

    def stats(self) -> dict:
        """Return summary statistics for the textbook collection."""
        cur = self._conn.execute(
            """
            SELECT
                COUNT(*)                        AS total,
                SUM(file_size_mb)               AS total_size_mb,
                SUM(page_count)                 AS total_pages,
                COUNT(DISTINCT source)          AS sources
            FROM textbooks
            """
        )
        row = cur.fetchone()
        return dict(row)
