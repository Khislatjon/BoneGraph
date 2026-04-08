"""
processing/embed.py
====================
Embeds all text chunks using SPECTER2 and stores the resulting vectors
in chunks.db.

Why SPECTER2?
-------------
SPECTER2 (2023) is Allen AI's successor to the original SPECTER model.
It was trained on 164M citation relationships and uses task-specific adapters:
  - allenai/specter2        — document embedding (proximity), used here
  - allenai/specter2_adhoc_query — query embedding, used at retrieval time

This adapter split is important: at query time the retrieval layer must encode
the user's question with the adhoc_query adapter, not this one.

How it works
------------
SPECTER2 is loaded via the HuggingFace `adapters` library (not sentence-
transformers). The base model (allenai/specter2_base) is loaded once, then
the proximity adapter is attached. Each batch of chunk texts is tokenised,
passed through the model, and the CLS token (position 0) of the last hidden
state is taken as the 768-dimensional document embedding.

How embeddings are stored
-------------------------
Each 768-dim embedding is serialised as a raw float32 binary blob and written
to the `embedding` column of chunks.db. At retrieval time they are deserialised
and cosine similarity is computed in numpy. For ~250K chunks this is fast
enough without a dedicated vector store.

Usage
-----
    # Embed all chunks that don't have embeddings yet
    python -m processing.embed

    # Re-embed everything from scratch
    python -m processing.embed --force

Estimated time (CPU)
--------------------
~248K chunks / 64 per batch ≈ 3,900 batches.
CPU inference: ~1.5–2 seconds per batch → roughly 2–3 hours total.
Safe to run overnight.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time

import numpy as np
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

from config.settings import (
    CHUNKS_DB_PATH,
    SPECTER2_BASE_MODEL,
    SPECTER2_ADAPTER,
    EMBEDDING_BATCH_SIZE,
)

logger = logging.getLogger(__name__)

# SPECTER2 outputs 768-dimensional embeddings.
EMBEDDING_DIM = 768


def _load_model() -> tuple:
    """
    Load the SPECTER2 base model and attach the proximity adapter.

    Returns
    -------
    (tokenizer, model) — both ready for inference.
    """
    logger.info("Loading tokenizer: %s", SPECTER2_BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(SPECTER2_BASE_MODEL)

    logger.info("Loading base model: %s", SPECTER2_BASE_MODEL)
    model = AutoAdapterModel.from_pretrained(SPECTER2_BASE_MODEL)

    logger.info("Loading adapter: %s", SPECTER2_ADAPTER)
    model.load_adapter(SPECTER2_ADAPTER, source="hf", load_as="specter2", set_active=True)

    model.eval()
    logger.info("SPECTER2 ready — embedding dim: %d", EMBEDDING_DIM)
    return tokenizer, model


def _encode_batch(texts: list[str], tokenizer, model) -> np.ndarray:
    """
    Encode a batch of texts into 768-dim embeddings.

    Tokenises with max_length=512 (SPECTER2's limit), runs a forward pass,
    and extracts the CLS token from the last hidden state.

    Returns
    -------
    numpy array of shape (len(texts), 768), dtype float32.
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token (position 0) is the document-level embedding.
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy().astype(np.float32)


def _serialize(vec: np.ndarray) -> bytes:
    """Serialise a float32 numpy array to raw bytes for SQLite BLOB storage."""
    return vec.astype(np.float32).tobytes()


def _deserialize(blob: bytes) -> np.ndarray:
    """Deserialise raw bytes from SQLite back to a float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


def run(force: bool = False) -> None:
    """
    Embed all un-embedded chunks in chunks.db.

    Parameters
    ----------
    force : bool
        If True, re-embed chunks that already have embeddings.
    """
    tokenizer, model = _load_model()

    conn = sqlite3.connect(CHUNKS_DB_PATH)
    conn.row_factory = sqlite3.Row

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

    for batch_start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = rows[batch_start: batch_start + EMBEDDING_BATCH_SIZE]
        ids = [r["id"] for r in batch]
        texts = [r["text"] for r in batch]

        embeddings = _encode_batch(texts, tokenizer, model)

        conn.executemany(
            "UPDATE chunks SET embedding = ? WHERE id = ?",
            [(_serialize(emb), chunk_id) for emb, chunk_id in zip(embeddings, ids)],
        )
        conn.commit()

        embedded += len(batch)

        if embedded % 1000 == 0 or embedded == total:
            elapsed = time.time() - start_time
            rate = embedded / elapsed if elapsed > 0 else 0
            remaining = (total - embedded) / rate if rate > 0 else 0
            logger.info(
                "Embedded %d/%d  (%.0f chunks/sec  ~%.0f min remaining)",
                embedded, total, rate, remaining / 60,
            )

    conn.close()

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("  EMBEDDING COMPLETE")
    print("=" * 50)
    print(f"  Chunks embedded : {embedded:,}")
    print(f"  Model           : {SPECTER2_BASE_MODEL}")
    print(f"  Adapter         : {SPECTER2_ADAPTER}")
    print(f"  Dimensions      : {EMBEDDING_DIM}")
    print(f"  Total time      : {elapsed / 60:.1f} minutes")
    print("=" * 50)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Embed text chunks with SPECTER2")
    parser.add_argument("--force", action="store_true", help="Re-embed all chunks")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
