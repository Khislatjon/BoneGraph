"""
api/beta_feedback.py
====================

Lightweight persistence for **public-beta feedback** — the free-text notes,
bug reports, and feature requests submitted through the "Send feedback" button
in the UI.

This is deliberately separate from the per-tab learning stores
(``reasoning/feedback_store.py``, ``vision/correction_store.py``): those shape
model behaviour, whereas this is plain product feedback for the maintainer to
read. One table, no embeddings, no recall.

Feedback is the source of truth here in SQLite. Emailing it out (e.g. a nightly
digest to a university inbox) can be layered on top of ``list_feedback`` later
without changing the write path or the API.

Path: data/db/beta_feedback.db (alongside the other BoneGraph DBs).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "beta_feedback.db"

# Categories the UI offers. Anything else is coerced to "other" on the way in.
CATEGORIES = ("bug", "idea", "question", "praise", "other")

MAX_MESSAGE_CHARS = 4000
MAX_EMAIL_CHARS = 200
MAX_NAME_CHARS = 120


SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    category    TEXT    NOT NULL DEFAULT 'other',
    name        TEXT    NOT NULL DEFAULT '',   -- required: who sent it
    message     TEXT    NOT NULL,
    email       TEXT    NOT NULL DEFAULT '',   -- optional, only if user offers it
    tab         TEXT    NOT NULL DEFAULT '',   -- which tab they were on
    page        TEXT    NOT NULL DEFAULT '',   -- url / route
    user_agent  TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at);
"""


def _ensure_name_column(con: sqlite3.Connection) -> None:
    """Add the `name` column to pre-existing DBs created before it existed."""
    cols = {r["name"] for r in con.execute("PRAGMA table_info(feedback)")}
    if "name" not in cols:
        con.execute("ALTER TABLE feedback ADD COLUMN name TEXT NOT NULL DEFAULT ''")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        _ensure_name_column(con)
        yield con
        con.commit()
    finally:
        con.close()


def save_feedback(
    message: str,
    *,
    name: str,
    category: str = "other",
    email: str = "",
    tab: str = "",
    page: str = "",
    user_agent: str = "",
) -> int:
    """Persist one feedback submission. Returns the new row id.

    ``name`` and ``message`` are both required and non-empty; everything else is
    best-effort context. Values are trimmed and length-capped so a single
    submission can't blow up the table.
    """
    msg = (message or "").strip()
    if not msg:
        raise ValueError("feedback message must not be empty")
    nm = (name or "").strip()
    if not nm:
        raise ValueError("feedback name must not be empty")
    cat = category if category in CATEGORIES else "other"
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO feedback
               (created_at, category, name, message, email, tab, page, user_agent)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _now(),
                cat,
                nm[:MAX_NAME_CHARS],
                msg[:MAX_MESSAGE_CHARS],
                (email or "").strip()[:MAX_EMAIL_CHARS],
                (tab or "").strip()[:50],
                (page or "").strip()[:300],
                (user_agent or "").strip()[:300],
            ),
        )
        return int(cur.lastrowid)


def list_feedback(limit: int = 200) -> list[dict]:
    """Return the most recent feedback, newest first."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM feedback ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 1000)),),
        ).fetchall()
        return [dict(r) for r in rows]


def count() -> int:
    with _conn() as con:
        return int(con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])


def delete_feedback(feedback_id: int) -> bool:
    """Delete one feedback row by id. Returns True if a row was removed."""
    with _conn() as con:
        cur = con.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
        return cur.rowcount > 0
