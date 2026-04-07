"""
processing/chunk_all.py
========================
Chunks all extracted text files (papers + textbooks) and stores the results
in data/db/chunks.db.

Usage
-----
    # Chunk everything not yet chunked
    python -m processing.chunk_all

    # Re-chunk everything from scratch
    python -m processing.chunk_all --force

What this script does
---------------------
1. Opens chunks.db (creates it if needed).
2. Queries papers.db for all papers with a text_path (extraction_status='extracted').
3. For each paper: reads the .txt file, calls chunk_text(), inserts into chunks.db.
4. Queries textbooks.db for all textbooks with a text_path.
5. For each textbook: same process.
6. Prints a summary at the end.

Resumability
------------
The script checks whether a source_id already has chunks before processing it.
This means interrupted runs can be safely resumed — already-chunked sources
are skipped automatically.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

from config.settings import PAPERS_DB_PATH, TEXTBOOKS_DB_PATH, PROCESSED_DIR
from processing.chunker import chunk_text
from processing.chunk_store import ChunkStore

logger = logging.getLogger(__name__)


def chunk_papers(store: ChunkStore, force: bool = False) -> dict:
    """Chunk all extracted paper text files."""
    conn = sqlite3.connect(PAPERS_DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT paper_id, text_path FROM papers WHERE extraction_status = 'extracted'"
    ).fetchall()
    conn.close()

    total = len(rows)
    counts = {"chunked": 0, "skipped": 0, "failed": 0}

    logger.info("Papers to chunk: %d", total)

    for idx, row in enumerate(rows, start=1):
        paper_id = row["paper_id"]
        text_path = Path(row["text_path"])

        # Skip if already chunked (unless --force).
        if not force and store.source_already_chunked(paper_id):
            counts["skipped"] += 1
            continue

        if not text_path.exists():
            logger.warning("[%d/%d] Text file missing: %s", idx, total, text_path)
            counts["failed"] += 1
            continue

        try:
            full_text = text_path.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_text(full_text)

            if force:
                store.delete_chunks_for_source(paper_id)

            store.insert_chunks("paper", paper_id, chunks)
            counts["chunked"] += 1

            if idx % 500 == 0 or idx == total:
                logger.info(
                    "[%d/%d] Chunked %d papers so far...", idx, total, counts["chunked"]
                )

        except Exception as exc:
            logger.error("Failed to chunk paper %s: %s", paper_id[:16], exc)
            counts["failed"] += 1

    return counts


def chunk_textbooks(store: ChunkStore, force: bool = False) -> dict:
    """Chunk all extracted textbook text files."""
    conn = sqlite3.connect(TEXTBOOKS_DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT file_path, text_path, title, source FROM textbooks WHERE extraction_status = 'extracted'"
    ).fetchall()
    conn.close()

    total = len(rows)
    counts = {"chunked": 0, "skipped": 0, "failed": 0}

    logger.info("Textbooks to chunk: %d", total)

    for idx, row in enumerate(rows, start=1):
        source_id = row["file_path"]  # use file_path as the stable identifier
        text_path = Path(row["text_path"])

        if not force and store.source_already_chunked(source_id):
            counts["skipped"] += 1
            continue

        if not text_path.exists():
            logger.warning("Text file missing: %s", text_path)
            counts["failed"] += 1
            continue

        try:
            full_text = text_path.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_text(full_text)

            if force:
                store.delete_chunks_for_source(source_id)

            store.insert_chunks("textbook", source_id, chunks)
            counts["chunked"] += 1
            logger.info(
                "[%d/%d] [%s] %s — %d chunks",
                idx, total, row["source"], row["title"][:40], len(chunks)
            )

        except Exception as exc:
            logger.error("Failed to chunk textbook %s: %s", source_id, exc)
            counts["failed"] += 1

    return counts


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Chunk extracted texts into RAG-ready segments")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-chunk all sources even if already chunked"
    )
    args = parser.parse_args()

    with ChunkStore() as store:
        logger.info("=== Chunking papers ===")
        paper_counts = chunk_papers(store, force=args.force)

        logger.info("=== Chunking textbooks ===")
        book_counts = chunk_textbooks(store, force=args.force)

        stats = store.stats()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  CHUNKING COMPLETE")
    print("=" * 55)
    print(f"  Papers   — chunked: {paper_counts['chunked']:,}  "
          f"skipped: {paper_counts['skipped']:,}  "
          f"failed: {paper_counts['failed']:,}")
    print(f"  Textbooks — chunked: {book_counts['chunked']:,}  "
          f"skipped: {book_counts['skipped']:,}  "
          f"failed: {book_counts['failed']:,}")
    print("-" * 55)
    print(f"  Total chunks     : {stats['total_chunks']:,}")
    print(f"  Paper chunks     : {stats['paper_chunks']:,}")
    print(f"  Textbook chunks  : {stats['textbook_chunks']:,}")
    print(f"  Total sources    : {stats['total_sources']:,}")
    print(f"  Total characters : {stats['total_chars']:,}")
    print("=" * 55)


if __name__ == "__main__":
    main()
