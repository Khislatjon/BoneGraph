"""
processing/chunk_store.py
==========================
SQLite-backed store for text chunks from both papers and textbooks.

Why a separate database?
------------------------
The chunks table will contain ~1–3 million rows (6,085 papers * average
~150 chunks each).  Keeping this in a separate chunks.db prevents it from
bloating papers.db and keeps both databases fast and independently browsable.

Schema
------
chunks
    id            — auto-increment primary key
    source_type   — "paper" or "textbook"
    source_id     — paper_id (S2 hash) or textbook file_path
    chunk_index   — sequential position within the document (0-based)
    page_number   — page this chunk starts on (1-indexed)
    text          — the chunk text
    char_count    — length of the text
    embedding     — NULL for now; will be filled in Phase 2 Step 3 (embedding)
    created_at    — timestamp
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from config.settings import CHUNKS_DB_PATH

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,   -- "paper" or "textbook"
    source_id   TEXT NOT NULL,   -- paper_id or textbook file_path
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    text        TEXT NOT NULL,
    char_count  INTEGER,
    embedding   BLOB,            -- NULL until embedding step
    created_at  TEXT DEFAULT (datetime('now'))
);

-- Index for fast lookup by source (used during embedding and retrieval).
CREATE INDEX IF NOT EXISTS idx_chunks_source
    ON chunks (source_type, source_id);
"""


class ChunkStore:
    """
    Manages the chunks SQLite database.

    Usage
    -----
        with ChunkStore() as store:
            store.insert_chunks(source_type, source_id, chunks)
    """

    def __init__(self, db_path: Path = CHUNKS_DB_PATH):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> ChunkStore:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        if self._conn:
            self._conn.close()

    def connect(self) -> None:
        """Open connection and create tables."""
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        # Execute both CREATE TABLE and CREATE INDEX statements.
        self._conn.executescript(CREATE_TABLE_SQL)
        self._conn.commit()
        logger.debug("Connected to chunks DB at %s", self.db_path)

    def source_already_chunked(self, source_id: str) -> bool:
        """Return True if this source already has chunks in the database."""
        cur = self._conn.execute(
            "SELECT 1 FROM chunks WHERE source_id = ? LIMIT 1", (source_id,)
        )
        return cur.fetchone() is not None

    def delete_chunks_for_source(self, source_id: str) -> None:
        """Remove all chunks for a given source (used with --force)."""
        self._conn.execute(
            "DELETE FROM chunks WHERE source_id = ?", (source_id,)
        )
        self._conn.commit()

    def insert_chunks(
        self,
        source_type: str,
        source_id: str,
        chunks: list[dict],
    ) -> None:
        """
        Bulk-insert all chunks for a single document.

        Parameters
        ----------
        source_type : "paper" or "textbook"
        source_id   : paper_id or textbook file_path
        chunks      : list of dicts from chunker.chunk_text()
        """
        rows = [
            (
                source_type,
                source_id,
                c["chunk_index"],
                c["page_number"],
                c["text"],
                c["char_count"],
            )
            for c in chunks
        ]
        self._conn.executemany(
            """INSERT INTO chunks
               (source_type, source_id, chunk_index, page_number, text, char_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def stats(self) -> dict:
        """Return summary statistics."""
        cur = self._conn.execute(
            """SELECT
                COUNT(*)                                    AS total_chunks,
                COUNT(DISTINCT source_id)                   AS total_sources,
                SUM(CASE WHEN source_type='paper'    THEN 1 ELSE 0 END) AS paper_chunks,
                SUM(CASE WHEN source_type='textbook' THEN 1 ELSE 0 END) AS textbook_chunks,
                SUM(char_count)                             AS total_chars
               FROM chunks"""
        )
        return dict(cur.fetchone())
