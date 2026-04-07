"""
processing/embed.py
====================
Embeds all text chunks using the SPECTER model and stores the resulting
vectors back into chunks.db.

Why SPECTER?
------------
SPECTER (Scientific Paper Embeddings using Citation-informed TransformERs) was
trained by Allen AI on 146 million citation relationships from Semantic Scholar.
It produces embeddings that capture scientific meaning — two papers on the same
topic will have similar vectors even if they use different terminology.  This
makes it ideal for our bone science corpus, which was collected from Semantic
Scholar itself.

How embeddings are stored
-------------------------
Each 768-dimensional embedding is serialised as a raw binary blob (numpy float32
array → bytes) and stored in the `embedding` column of the chunks table.  This
avoids any external dependency — everything stays in SQLite.

At query time, embeddings are deserialised and cosine similarity is computed
in Python/numpy.  For 200K chunks this is fast enough without a vector store.
If the corpus grows to millions of chunks, we can migrate to ChromaDB or FAISS.

Usage
-----
    # Embed all chunks that don't have embeddings yet
    python -m processing.embed

    # Re-embed everything from scratch
    python -m processing.embed --force

Estimated time
--------------
200,757 chunks / 64 per batch = ~3,137 batches.
CPU inference: ~1–2 seconds per batch → ~1–2 hours total.
Run overnight or in the background.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import CHUNKS_DB_PATH, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


def _serialize_embedding(vec: np.ndarray) -> bytes:
    """Convert a float32 numpy array to raw bytes for SQLite BLOB storage."""
    return vec.astype(np.float32).tobytes()


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    """Convert raw bytes from SQLite back to a float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


def run(force: bool = False) -> None:
    """
    Embed all un-embedded chunks in chunks.db.

    Parameters
    ----------
    force : bool
        If True, re-embed all chunks even if they already have embeddings.
    """
    logger.info("Loading SPECTER model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Model loaded. Embedding dimension: %d", model.get_sentence_embedding_dimension())

    conn = sqlite3.connect(CHUNKS_DB_PATH)
    conn.row_factory = sqlite3.Row

    # Fetch chunks that need embedding.
    if force:
        cur = conn.execute("SELECT id, text FROM chunks ORDER BY id")
    else:
        cur = conn.execute(
            "SELECT id, text FROM chunks WHERE embedding IS NULL ORDER BY id"
        )

    rows = cur.fetchall()
    total = len(rows)

    if total == 0:
        logger.info("All chunks already embedded. Use --force to re-embed.")
        conn.close()
        return

    logger.info("Chunks to embed: %d", total)

    embedded = 0
    start_time = time.time()

    # Process in batches for efficiency.
    for batch_start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = rows[batch_start: batch_start + EMBEDDING_BATCH_SIZE]
        ids = [r["id"] for r in batch]
        texts = [r["text"] for r in batch]

        # Encode the batch — returns a numpy array of shape (batch_size, 768).
        # show_progress_bar=False because we have our own progress logging.
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Serialize and write back to the database.
        conn.executemany(
            "UPDATE chunks SET embedding = ? WHERE id = ?",
            [(_serialize_embedding(emb), chunk_id)
             for emb, chunk_id in zip(embeddings, ids)],
        )
        conn.commit()

        embedded += len(batch)

        # Log progress every 1,000 chunks.
        if embedded % 1000 == 0 or embedded == total:
            elapsed = time.time() - start_time
            rate = embedded / elapsed if elapsed > 0 else 0
            remaining = (total - embedded) / rate if rate > 0 else 0
            logger.info(
                "Embedded %d/%d chunks  (%.0f chunks/sec  ~%.0f min remaining)",
                embedded, total, rate, remaining / 60,
            )

    conn.close()

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("  EMBEDDING COMPLETE")
    print("=" * 50)
    print(f"  Chunks embedded : {embedded:,}")
    print(f"  Model           : {EMBEDDING_MODEL}")
    print(f"  Dimensions      : 768")
    print(f"  Total time      : {elapsed / 60:.1f} minutes")
    print("=" * 50)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Embed text chunks with SPECTER")
    parser.add_argument("--force", action="store_true", help="Re-embed all chunks")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
