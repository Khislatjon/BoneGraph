"""
retrieval/retriever.py
=======================
Core RAG retrieval engine for BoneLogic.

How it works
------------
At initialisation, all 200,757 chunk embeddings are loaded from chunks.db
into a single numpy matrix of shape (N, 768).  This costs ~588 MB of RAM
but makes similarity search very fast — a single matrix multiplication
computes the cosine similarity between a query and every chunk in ~50ms.

At query time:
1. The query string is embedded with SPECTER (same model used for chunks).
2. Cosine similarity is computed between the query vector and all chunk vectors.
3. The top-k highest-scoring chunks are retrieved.
4. Metadata (title, authors, year, journal/source) is looked up from
   papers.db and textbooks.db and attached to each result.

Why no vector store (ChromaDB/FAISS)?
--------------------------------------
At 200K chunks, numpy cosine similarity on CPU takes ~50ms per query —
fast enough for an interactive research tool.  A vector store adds
operational complexity (separate process, index files, serialisation) with
no meaningful speed benefit at this scale.  If the corpus grows to several
million chunks, migrating to FAISS would be straightforward since the
embedding format is standard float32.

Result format
-------------
Each result is a dict:
    rank          : int   — 1-indexed position in results
    score         : float — cosine similarity (0–1, higher = more relevant)
    chunk_id      : int   — row ID in chunks.db
    source_type   : str   — "paper" or "textbook"
    source_id     : str   — paper_id or textbook file_path
    chunk_index   : int   — position within the document
    page_number   : int   — page this chunk starts on
    text          : str   — the chunk text
    title         : str   — paper title or textbook title
    authors       : str   — authors (papers only)
    year          : int   — publication year (papers only)
    venue         : str   — journal name (papers only)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

from config.settings import (
    CHUNKS_DB_PATH,
    PAPERS_DB_PATH,
    TEXTBOOKS_DB_PATH,
    SPECTER2_BASE_MODEL,
    SPECTER2_ADAPTER,
)

logger = logging.getLogger(__name__)


class BoneLogicRetriever:
    """
    RAG retrieval engine. Load once, query many times.

    Usage
    -----
        retriever = BoneLogicRetriever()
        retriever.load()   # loads embeddings into RAM (~588 MB, ~10 sec)

        results = retriever.query("cortical bone fracture toughness", top_k=10)
        for r in results:
            print(r['score'], r['title'], r['text'][:200])
    """

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._embeddings: np.ndarray | None = None  # shape (N, 768)
        self._chunk_ids: list[int] = []             # chunk DB ids in same order as rows
        self._metadata: dict[int, dict] = {}        # chunk_id → metadata dict
        self._loaded = False

    def load(self) -> None:
        """
        Load SPECTER2 query model and all chunk embeddings into memory.
        Call this once before making any queries.
        """
        t0 = time.time()

        # Load SPECTER2 with the adhoc_query adapter for query encoding.
        # Documents were embedded with the proximity adapter (allenai/specter2).
        # Queries must use the adhoc_query adapter — this asymmetry is what
        # makes SPECTER2 more accurate than SPECTER1.
        logger.info("Loading SPECTER2 tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(SPECTER2_BASE_MODEL)

        logger.info("Loading SPECTER2 base model...")
        self._model = AutoAdapterModel.from_pretrained(SPECTER2_BASE_MODEL)

        query_adapter = "allenai/specter2_adhoc_query"
        logger.info("Loading adhoc_query adapter: %s", query_adapter)
        self._model.load_adapter(query_adapter, source="hf", load_as="specter2_query", set_active=True)
        self._model.eval()

        # Load all embeddings from chunks.db into a numpy matrix.
        logger.info("Loading chunk embeddings from chunks.db...")
        conn = sqlite3.connect(CHUNKS_DB_PATH)
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """SELECT id, source_type, source_id, chunk_index, page_number, text, embedding
               FROM chunks
               WHERE embedding IS NOT NULL
               ORDER BY id"""
        ).fetchall()
        conn.close()

        self._chunk_ids = []
        embeddings_list = []
        chunk_meta = {}

        for row in rows:
            cid = row["id"]
            self._chunk_ids.append(cid)
            # Deserialise the stored float32 bytes back to a numpy vector.
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            embeddings_list.append(vec)
            chunk_meta[cid] = {
                "source_type": row["source_type"],
                "source_id":   row["source_id"],
                "chunk_index": row["chunk_index"],
                "page_number": row["page_number"],
                "text":        row["text"],
            }

        # Stack into a single matrix (N, 768) and L2-normalise rows so that
        # dot product == cosine similarity (avoids re-normalising at query time).
        self._embeddings = np.vstack(embeddings_list).astype(np.float32)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid division by zero
        self._embeddings /= norms

        # Load source metadata for papers and textbooks.
        logger.info("Loading source metadata...")
        self._metadata = self._load_metadata(chunk_meta)

        self._loaded = True
        elapsed = time.time() - t0
        logger.info(
            "Retriever ready — %d chunks loaded in %.1fs", len(self._chunk_ids), elapsed
        )

    def _load_metadata(self, chunk_meta: dict) -> dict:
        """
        For each chunk, look up its source document's metadata and merge it
        into the chunk metadata dict.
        """
        # Collect unique source IDs by type.
        paper_ids = {v["source_id"] for v in chunk_meta.values() if v["source_type"] == "paper"}
        textbook_ids = {v["source_id"] for v in chunk_meta.values() if v["source_type"] == "textbook"}

        # Fetch paper metadata.
        paper_meta = {}
        if paper_ids:
            conn = sqlite3.connect(PAPERS_DB_PATH)
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" * len(paper_ids))
            rows = conn.execute(
                f"SELECT paper_id, title, authors_json, year, venue FROM papers WHERE paper_id IN ({placeholders})",
                list(paper_ids),
            ).fetchall()
            conn.close()
            for row in rows:
                authors = []
                try:
                    authors_raw = json.loads(row["authors_json"] or "[]")
                    authors = [a.get("name", "") for a in authors_raw[:3]]
                    if len(authors_raw) > 3:
                        authors.append("et al.")
                except Exception:
                    pass
                paper_meta[row["paper_id"]] = {
                    "title":   row["title"] or "Unknown title",
                    "authors": ", ".join(authors) if authors else "Unknown authors",
                    "year":    row["year"],
                    "venue":   row["venue"] or "",
                }

        # Fetch textbook metadata.
        textbook_meta = {}
        if textbook_ids:
            conn = sqlite3.connect(TEXTBOOKS_DB_PATH)
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" * len(textbook_ids))
            rows = conn.execute(
                f"SELECT file_path, title, source FROM textbooks WHERE file_path IN ({placeholders})",
                list(textbook_ids),
            ).fetchall()
            conn.close()
            for row in rows:
                textbook_meta[row["file_path"]] = {
                    "title":   row["title"] or "Unknown textbook",
                    "authors": "",
                    "year":    None,
                    "venue":   row["source"],
                }

        # Merge into chunk metadata.
        merged = {}
        for cid, meta in chunk_meta.items():
            source_id = meta["source_id"]
            if meta["source_type"] == "paper":
                doc_meta = paper_meta.get(source_id, {"title": "Unknown", "authors": "", "year": None, "venue": ""})
            else:
                doc_meta = textbook_meta.get(source_id, {"title": "Unknown", "authors": "", "year": None, "venue": ""})
            merged[cid] = {**meta, **doc_meta}

        return merged

    def query(self, query_text: str, top_k: int = 10) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Parameters
        ----------
        query_text : str
            Natural language question or search phrase.
        top_k : int
            Number of results to return (default 10).

        Returns
        -------
        List of result dicts ordered by relevance (highest score first).
        """
        if not self._loaded:
            raise RuntimeError("Call retriever.load() before querying.")

        # Embed the query with SPECTER2 adhoc_query adapter.
        inputs = self._tokenizer(
            [query_text],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        query_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0].astype(np.float32)

        # Normalise query vector.
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm

        # Cosine similarity = dot product (since both sides are L2-normalised).
        # Shape: (N,) — one score per chunk.
        scores = self._embeddings @ query_vec

        # Get indices of top-k scores (unsorted), then sort them.
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            cid = self._chunk_ids[idx]
            meta = self._metadata[cid]
            results.append({
                "rank":        rank,
                "score":       float(scores[idx]),
                "chunk_id":    cid,
                "source_type": meta["source_type"],
                "source_id":   meta["source_id"],
                "chunk_index": meta["chunk_index"],
                "page_number": meta["page_number"],
                "text":        meta["text"],
                "title":       meta["title"],
                "authors":     meta["authors"],
                "year":        meta["year"],
                "venue":       meta["venue"],
            })

        return results
