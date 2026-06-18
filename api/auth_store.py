"""
api/auth_store.py
==================
Username/password authentication for BoneGraph.

Single-user-per-account, multi-account: each registered user gets a unique
username that becomes the `user_id` threaded through reasoning/feedback_store.py
and vision/correction_store.py, so per-user data (Tier-2 learned rules, Vision
corrections) is scoped to the account that produced it instead of leaking into
a single shared "local" bucket.

No new dependencies: passwords are hashed with stdlib `hashlib.pbkdf2_hmac`
(per-user random salt), and session tokens are `secrets.token_urlsafe`. This
is a per-install tool, not a public-facing service — good enough for that
threat model without pulling in bcrypt/passlib/jose.

Path: data/db/auth.db (alongside the existing BoneGraph DBs).
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "auth.db"

PBKDF2_ITERATIONS = 200_000
MIN_PASSWORD_LENGTH = 8


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    password_salt TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    ).hex()


# ── Users ─────────────────────────────────────────────────────────────────

def create_user(username: str, password: str) -> dict:
    """Register a new account. Raises ValueError on invalid input or duplicate username."""
    username = username.strip()
    if not username:
        raise ValueError("Username is required")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    now = _now()
    with _conn() as con:
        try:
            cur = con.execute(
                """INSERT INTO users (username, password_hash, password_salt, created_at)
                   VALUES (?, ?, ?, ?)""",
                (username, password_hash, salt, now),
            )
        except sqlite3.IntegrityError:
            raise ValueError("Username is already taken")
        return {"id": int(cur.lastrowid), "username": username}


def verify_user(username: str, password: str) -> dict | None:
    """Return {"id", "username"} if the password matches, else None."""
    with _conn() as con:
        row = con.execute(
            "SELECT id, username, password_hash, password_salt FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row is None:
        return None
    candidate = _hash_password(password, row["password_salt"])
    if not secrets.compare_digest(candidate, row["password_hash"]):
        return None
    return {"id": row["id"], "username": row["username"]}


# ── Sessions ──────────────────────────────────────────────────────────────

def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with _conn() as con:
        con.execute(
            "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, _now()),
        )
    return token


def get_user_by_token(token: str) -> dict | None:
    if not token:
        return None
    with _conn() as con:
        row = con.execute(
            """SELECT u.id, u.username FROM sessions s
               JOIN users u ON u.id = s.user_id
               WHERE s.token = ?""",
            (token,),
        ).fetchone()
    return {"id": row["id"], "username": row["username"]} if row else None


def delete_session(token: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM sessions WHERE token = ?", (token,))
