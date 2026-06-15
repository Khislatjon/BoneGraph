"""
vision/correction_store.py
==========================

Per-install (Tier-2) persistence for the Vision tab.

Unlike the Reasoning tab there is **no critic and no rule layer** here — the
only learning surface is correction memory. Three concerns:

  feedback_events       — every 👍 / 👎, for the scoreboard
  corrections           — the raw 👎 + free-text correction payload
  correction_embeddings — one or more image-embedding vectors per correction,
                          so recall survives rotation / re-windowing of the
                          same scan (store the original + a few augments)

Recall is a cosine match: embed a new image, compare against every stored
vector, group by correction, keep each correction's best-matching variant, and
surface the ones above a threshold. Vectors are L2-normalised on the way in, so
cosine similarity is a plain dot product.

This module is **encoder-agnostic**: it stores and matches float vectors but
never loads a vision encoder. Whoever calls it (the API layer) is responsible
for turning an image into a vector with BiomedCLIP / CLIP / the VLM's own
tower. That keeps the store cheap to test and lets the encoder change without
touching persistence.

Memory cost is fixed-size: a 512-d float32 vector is ~2 KB, so a correction
with the original + ~4 augments is ~10 KB plus the text. 1,000 corrections is
~10 MB; storage is never the bottleneck (linear cosine scan time is, but only
past ~100k vectors — far beyond a single-user tool).

Path: data/db/vision_feedback.db (alongside the existing BoneGraph DBs).
Single-user / single-machine, with a `user_id` column reserved so we don't
have to migrate for multi-user later.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "vision_feedback.db"
DEFAULT_USER = "local"

# Cosine ≥ this means "essentially the same scan" (near-duplicate). Tuned
# conservatively: whole-image embeddings of one modality are globally similar,
# so a loose threshold retrieves the wrong correction. Validate against your
# own rotated/re-windowed copies before relaxing it.
DEFAULT_RECALL_THRESHOLD = 0.9


SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    NOT NULL DEFAULT 'local',
    turn_id     TEXT    NOT NULL,
    polarity    INTEGER NOT NULL,                  -- +1 or -1
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS corrections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        TEXT    NOT NULL DEFAULT 'local',
    turn_id        TEXT    NOT NULL,
    prompt         TEXT    NOT NULL DEFAULT '',     -- the user's "what is this?"
    identification TEXT    NOT NULL DEFAULT '',     -- what the VLM had said
    feedback_text  TEXT    NOT NULL,                -- the correction itself
    created_at     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS correction_embeddings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    correction_id INTEGER NOT NULL,
    user_id       TEXT    NOT NULL DEFAULT 'local',
    variant       TEXT    NOT NULL DEFAULT 'original',  -- 'original' | 'rot90' | ...
    dim           INTEGER NOT NULL,
    vector        BLOB    NOT NULL,                 -- L2-normalised float32
    created_at    TEXT    NOT NULL,
    FOREIGN KEY(correction_id) REFERENCES corrections(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_feedback_events_user ON feedback_events(user_id);
CREATE INDEX IF NOT EXISTS idx_corrections_user ON corrections(user_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_user ON correction_embeddings(user_id);
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


# ── Vector helpers ──────────────────────────────────────────────────────────

def _normalise(vec: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a 1-D float32 unit vector. Cosine then reduces to a dot product."""
    arr = np.asarray(vec, dtype=np.float32).ravel()
    if arr.size == 0:
        raise ValueError("embedding is empty")
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("embedding has zero norm; cannot normalise")
    return arr / norm


def _to_blob(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(vec, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


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
        corr = con.execute(
            "SELECT COUNT(*) FROM corrections WHERE user_id = ?", (user_id,),
        ).fetchone()[0]
        return {"thumbs_up": up, "thumbs_down": down, "corrections": corr}


# ── Corrections ───────────────────────────────────────────────────────────────

def save_correction(
    turn_id: str,
    feedback_text: str,
    embeddings: Sequence[Sequence[float]] | np.ndarray,
    *,
    prompt: str = "",
    identification: str = "",
    variants: Sequence[str] | None = None,
    user_id: str = DEFAULT_USER,
) -> dict:
    """Persist a 👎 correction together with its image embedding(s).

    Pass the original image's embedding plus a few augmented copies (rotations,
    flips, intensity re-windowings) so recall survives those transforms. A
    single vector is also fine. ``embeddings`` is a sequence of vectors (or a 2-D
    array, one row per variant); ``variants`` optionally names each row.

    Returns ``{"id", "n_embeddings"}``.
    """
    rows = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
    if rows.shape[0] == 0:
        raise ValueError("at least one embedding is required")
    if variants is not None and len(variants) != rows.shape[0]:
        raise ValueError("variants must match the number of embeddings")

    normed = [_normalise(r) for r in rows]
    dim = normed[0].size
    if any(v.size != dim for v in normed):
        raise ValueError("all embeddings must share the same dimensionality")

    names = list(variants) if variants is not None else (
        ["original"] + [f"aug{i}" for i in range(1, len(normed))]
    )
    now = _now()
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO corrections
               (user_id, turn_id, prompt, identification, feedback_text, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, turn_id, prompt, identification, feedback_text, now),
        )
        correction_id = int(cur.lastrowid)
        con.executemany(
            """INSERT INTO correction_embeddings
               (correction_id, user_id, variant, dim, vector, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [(correction_id, user_id, name, dim, _to_blob(vec), now)
             for name, vec in zip(names, normed)],
        )
    return {"id": correction_id, "n_embeddings": len(normed)}


def recall(
    query_embedding: Sequence[float] | np.ndarray,
    *,
    threshold: float = DEFAULT_RECALL_THRESHOLD,
    top_k: int = 3,
    user_id: str = DEFAULT_USER,
) -> list[dict]:
    """Return prior corrections whose best-matching variant is ≥ ``threshold``.

    Cosine similarity over every stored vector of matching dimensionality, kept
    as the best (max) score per correction, sorted high → low and capped at
    ``top_k``. Each hit is
    ``{"correction_id", "score", "variant", "feedback_text", "identification",
       "prompt", "created_at"}``.

    A brute-force NumPy scan — sub-millisecond for thousands of vectors. Swap in
    an ANN index only past ~100k, which a single-user tool will never reach.
    """
    q = _normalise(query_embedding)
    with _conn() as con:
        rows = con.execute(
            """SELECT e.correction_id, e.variant, e.dim, e.vector,
                      c.feedback_text, c.identification, c.prompt, c.created_at
               FROM correction_embeddings e
               JOIN corrections c ON c.id = e.correction_id
               WHERE e.user_id = ? AND e.dim = ?""",
            (user_id, q.size),
        ).fetchall()

    if not rows:
        return []

    matrix = np.vstack([_from_blob(r["vector"]) for r in rows])  # already unit vectors
    scores = matrix @ q  # cosine, since both sides are L2-normalised

    best: dict[int, dict] = {}
    for r, score in zip(rows, scores):
        score = float(score)
        if score < threshold:
            continue
        cid = int(r["correction_id"])
        if cid not in best or score > best[cid]["score"]:
            best[cid] = {
                "correction_id": cid,
                "score": score,
                "variant": r["variant"],
                "feedback_text": r["feedback_text"],
                "identification": r["identification"],
                "prompt": r["prompt"],
                "created_at": r["created_at"],
            }

    return sorted(best.values(), key=lambda h: h["score"], reverse=True)[:top_k]


def list_corrections(user_id: str = DEFAULT_USER) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            """SELECT c.*, COUNT(e.id) AS n_embeddings
               FROM corrections c
               LEFT JOIN correction_embeddings e ON e.correction_id = c.id
               WHERE c.user_id = ?
               GROUP BY c.id
               ORDER BY c.id DESC""",
            (user_id,),
        ).fetchall()
        return [_row_to_correction(r) for r in rows]


def delete_correction(correction_id: int, user_id: str = DEFAULT_USER) -> bool:
    """Delete a correction and its embeddings (ON DELETE CASCADE)."""
    with _conn() as con:
        cur = con.execute(
            "DELETE FROM corrections WHERE id = ? AND user_id = ?",
            (correction_id, user_id),
        )
        return cur.rowcount > 0


def _row_to_correction(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "turn_id": r["turn_id"],
        "prompt": r["prompt"],
        "identification": r["identification"],
        "feedback_text": r["feedback_text"],
        "n_embeddings": (r["n_embeddings"] if "n_embeddings" in r.keys() else None),
        "created_at": r["created_at"],
    }
