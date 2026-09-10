"""
eval/mcq/build_fts_index.py
===========================
Build the BM25 index that the `bm25` and `bm25rerank` arms retrieve over.

Both arms are unreproducible without it, and it is not in version control
because data/db/ is gitignored — so this script is the reproduction path.

Why a lexical index at all: SPECTER2 is trained for document-level citation
similarity over titles and abstracts. Applied to 248,629 arbitrary passages it
places the entire corpus in a ~0.76-0.84 cosine band, so a chunk that answers a
question can score BELOW an unrelated one and the ranking carries little
information. Measured over the 50-item benchmark, the deciding quantity reached
the prompt for 10/50 items under dense top-5 retrieval and 31/50 under BM25.

The index is CONTENTLESS (content=''): it stores the inverted index only, not a
second copy of the text, which matters on a device with little free disk. rowid
is the chunks.db id, so hits resolve straight back against the source table.
chunks.db is opened read-only and never modified.

    python -m eval.mcq.build_fts_index          # ~40 s, ~180 MB
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time

SRC_DEFAULT = "data/db/chunks.db"
DST_DEFAULT = "data/db/chunks_fts.db"


def build(src: str, dst: str, batch: int = 5000) -> None:
    if not os.path.exists(src):
        raise SystemExit(f"ABORT: {src} not found — the corpus database is required.")
    if os.path.exists(dst):
        os.remove(dst)

    con_src = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    con_dst = sqlite3.connect(dst)
    con_dst.execute("PRAGMA journal_mode=OFF")
    con_dst.execute("PRAGMA synchronous=OFF")
    con_dst.execute(
        "CREATE VIRTUAL TABLE ft USING fts5(text, content='', tokenize='porter unicode61')")

    t0, n = time.time(), 0
    cur = con_src.execute("SELECT id, text FROM chunks")
    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            break
        con_dst.executemany("INSERT INTO ft(rowid, text) VALUES (?, ?)",
                            [(i, t or "") for i, t in rows])
        con_dst.commit()
        n += len(rows)
        if n % 50000 == 0:
            print(f"  {n:,} chunks | {os.path.getsize(dst)/1e6:,.0f} MB | "
                  f"{time.time()-t0:.0f}s", flush=True)

    con_dst.execute("INSERT INTO ft(ft) VALUES('optimize')")
    con_dst.commit()
    con_dst.close()
    con_src.close()
    print(f"DONE: {n:,} chunks indexed, {os.path.getsize(dst)/1e6:,.0f} MB, "
          f"{time.time()-t0:.0f}s -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--dst", default=DST_DEFAULT)
    args = ap.parse_args()
    build(args.src, args.dst)
