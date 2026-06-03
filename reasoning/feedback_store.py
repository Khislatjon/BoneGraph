"""
reasoning/feedback_store.py
============================

Per-install (Tier-2) persistence for the Reasoning tab.

Three tables:

  feedback_events  — every thumbs-up / thumbs-down, for the scoreboard
  corrections      — the raw thumbs-down + free-text correction payload
  user_rules       — structured rules derived from corrections; merged with
                     the built-in Tier-1 rules in physical_grounding.check()

Path: data/db/reasoning_feedback.db (alongside the existing BoneMind DBs).

The store is intentionally single-user / single-machine. Multi-user shipping
is a future concern; the schema includes a `user_id` column so we don't have
to migrate later, but everything defaults to "local".
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "reasoning_feedback.db"
DEFAULT_USER = "local"


SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    NOT NULL DEFAULT 'local',
    turn_id     TEXT    NOT NULL,
    polarity    INTEGER NOT NULL,                  -- +1 or -1
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS corrections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT    NOT NULL DEFAULT 'local',
    turn_id       TEXT    NOT NULL,
    question      TEXT    NOT NULL,
    answer        TEXT    NOT NULL,
    feedback_text TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS user_rules (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              TEXT    NOT NULL DEFAULT 'local',
    rule_id              TEXT    NOT NULL,         -- e.g. user_cortical_modulus_1
    name                 TEXT    NOT NULL,
    kind                 TEXT    NOT NULL,         -- 'range' | 'forbid_pattern'
    params               TEXT    NOT NULL,         -- JSON
    source_correction_id INTEGER,
    origin               TEXT    NOT NULL DEFAULT 'feedback',  -- 'feedback' | 'imported'
    enabled              INTEGER NOT NULL DEFAULT 1,
    created_at           TEXT    NOT NULL,
    FOREIGN KEY(source_correction_id) REFERENCES corrections(id)
);

CREATE INDEX IF NOT EXISTS idx_user_rules_user ON user_rules(user_id, enabled);
CREATE INDEX IF NOT EXISTS idx_feedback_events_user ON feedback_events(user_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _migrate(con: sqlite3.Connection) -> None:
    """Lightweight, idempotent migrations for DBs created before a column existed."""
    cols = {r["name"] for r in con.execute("PRAGMA table_info(user_rules)")}
    if "origin" not in cols:
        con.execute("ALTER TABLE user_rules ADD COLUMN origin TEXT NOT NULL DEFAULT 'feedback'")


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        _migrate(con)
        yield con
        con.commit()
    finally:
        con.close()


# ── Events ────────────────────────────────────────────────────────────────────

def save_event(turn_id: str, polarity: int, user_id: str = DEFAULT_USER) -> int:
    if polarity not in (-1, 1):
        raise ValueError("polarity must be -1 or +1")
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO feedback_events (user_id, turn_id, polarity, created_at) VALUES (?, ?, ?, ?)",
            (user_id, turn_id, polarity, _now()),
        )
        return int(cur.lastrowid)


def stats(user_id: str = DEFAULT_USER) -> dict:
    with _conn() as con:
        up = con.execute(
            "SELECT COUNT(*) FROM feedback_events WHERE user_id = ? AND polarity = 1", (user_id,),
        ).fetchone()[0]
        down = con.execute(
            "SELECT COUNT(*) FROM feedback_events WHERE user_id = ? AND polarity = -1", (user_id,),
        ).fetchone()[0]
        rules = con.execute(
            "SELECT COUNT(*) FROM user_rules WHERE user_id = ? AND enabled = 1", (user_id,),
        ).fetchone()[0]
        return {"thumbs_up": up, "thumbs_down": down, "user_rules": rules}


# ── Corrections ───────────────────────────────────────────────────────────────

def save_correction(turn_id: str, question: str, answer: str, feedback_text: str,
                    user_id: str = DEFAULT_USER) -> int:
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO corrections
               (user_id, turn_id, question, answer, feedback_text, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, turn_id, question, answer, feedback_text, _now()),
        )
        return int(cur.lastrowid)


# ── User rules ────────────────────────────────────────────────────────────────

def save_user_rule(name: str, kind: str, params: dict, source_correction_id: int | None,
                   user_id: str = DEFAULT_USER, origin: str = "feedback") -> dict:
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO user_rules
               (user_id, rule_id, name, kind, params, source_correction_id, origin, enabled, created_at)
               VALUES (?, '', ?, ?, ?, ?, ?, 1, ?)""",
            (user_id, name, kind, json.dumps(params), source_correction_id, origin, _now()),
        )
        new_id = int(cur.lastrowid)
        rule_id = f"user_{kind}_{new_id}"
        con.execute("UPDATE user_rules SET rule_id = ? WHERE id = ?", (rule_id, new_id))
        return {"id": new_id, "rule_id": rule_id, "name": name, "kind": kind,
                "params": params, "origin": origin, "enabled": True}


def list_user_rules(user_id: str = DEFAULT_USER, enabled_only: bool = True) -> list[dict]:
    with _conn() as con:
        sql = "SELECT * FROM user_rules WHERE user_id = ?"
        args: tuple = (user_id,)
        if enabled_only:
            sql += " AND enabled = 1"
        sql += " ORDER BY id DESC"
        rows = con.execute(sql, args).fetchall()
        return [_row_to_rule(r) for r in rows]


def delete_user_rule(rule_db_id: int, user_id: str = DEFAULT_USER) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM user_rules WHERE id = ? AND user_id = ?",
                          (rule_db_id, user_id))
        return cur.rowcount > 0


def set_rule_enabled(rule_db_id: int, enabled: bool, user_id: str = DEFAULT_USER) -> bool:
    """Enable/disable a rule without deleting it. Disabled rules are kept in the
    store but excluded from the grounding check (list_user_rules(enabled_only=True))."""
    with _conn() as con:
        cur = con.execute(
            "UPDATE user_rules SET enabled = ? WHERE id = ? AND user_id = ?",
            (1 if enabled else 0, rule_db_id, user_id),
        )
        return cur.rowcount > 0


def _row_to_rule(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "rule_id": r["rule_id"],
        "name": r["name"],
        "kind": r["kind"],
        "params": json.loads(r["params"]),
        "source_correction_id": r["source_correction_id"],
        "origin": (r["origin"] if "origin" in r.keys() else "feedback"),
        "enabled": bool(r["enabled"]),
        "created_at": r["created_at"],
    }
