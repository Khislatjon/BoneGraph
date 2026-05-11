"""
reasoning/novelty.py
====================
Corpus-grounded novelty classification for BoneMind hypotheses.

Overview
--------
A hypothesis is labelled NOVEL only when the literature (248,629
indexed chunks) does not already contain a close match.

Two-tier approach
-----------------
Tier 1 — Keyword co-occurrence (always runs, zero extra dependencies)
    Extracts key terms from the hypothesis chain and searches chunks.db
    using SQLite LIKE queries.  Counts how many chunks contain:
      - ALL key terms together  → strong coverage → GROUNDED
      - MOST key terms          → partial coverage → SPECULATIVE
      - FEW or NONE             → weak coverage   → NOVEL

Tier 2 — SPECTER2 semantic similarity (runs when embeddings are available)
    Encodes the hypothesis summary using the stored SPECTER2 embeddings
    in chunks.db.  Loads a random sample of N embeddings into memory
    (default 8,000) and computes cosine similarity.  The maximum
    similarity score refines the Tier 1 label.

    Tier 2 is optional — if the sentence-transformers library is not
    installed or the model is unavailable, Tier 1 result is returned.

Novelty thresholds
------------------
                   Keyword coverage   Semantic similarity
    GROUNDED       ≥ 5 chunks, all    max_sim ≥ 0.82
    SPECULATIVE    1–4 chunks or      0.60 ≤ max_sim < 0.82
                   most terms match
    NOVEL          0 full matches     max_sim < 0.60

Corpus disclaimer
-----------------
The classifier is honest about its limitations: all corpus chunks come
from open-access papers.  A "NOVEL" label means the hypothesis was not
found in THIS corpus — not that it has never been published.
The UI surfaces this with an explicit disclaimer banner.

Usage::

    from reasoning.novelty import NoveltyClassifier
    from reasoning.lrm import LRM

    lrm        = LRM()
    classifier = NoveltyClassifier()

    results = lrm.query_physics("How does porosity affect fatigue?")
    for h in results:
        nr = classifier.classify(h)
        print(nr)
"""

from __future__ import annotations

import logging
import random
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from config.settings import CHUNKS_DB_PATH

if TYPE_CHECKING:
    from reasoning.physics_gen import PhysicsHypothesis

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

# Keyword tier
_KW_GROUNDED_CHUNKS    = 5      # ≥ this many chunks with ALL key terms → GROUNDED
_KW_SPECULATIVE_CHUNKS = 1      # ≥ 1 chunk with most terms → SPECULATIVE
_KW_MIN_TERM_LENGTH    = 4      # ignore very short tokens (the, of, …)

# Semantic tier
_SEM_GROUNDED_SIM    = 0.82    # cosine similarity → GROUNDED
_SEM_SPECULATIVE_SIM = 0.60    # cosine similarity → SPECULATIVE
_SEM_SAMPLE_SIZE     = 8_000   # random chunks to load for similarity search
_SEM_TOP_K           = 5       # number of top matches to return as evidence

# Corpus disclaimer — shown by UI whenever novelty is NOVEL or SPECULATIVE.
# Stored here so it is a single source of truth.
CORPUS_DISCLAIMER = (
    "⚠️  Novelty is assessed against our open-access corpus only. "
    "This hypothesis may have already been stated in papers outside that corpus, "
    "including behind paywalls or in non-English literature."
)


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class NoveltyResult:
    """
    Output of the novelty classifier.

    Attributes
    ----------
    label : str
        "GROUNDED" | "SPECULATIVE" | "NOVEL"
    score : float
        Novelty score 0–1.  0 = completely covered in corpus.
        1 = no match found. Derived from 1 - max_similarity.
    max_similarity : float
        Highest similarity found (keyword or semantic). 0–1.
    keyword_hits : int
        Number of corpus chunks containing ALL key terms.
    top_chunk_ids : list[str]
        Up to 5 most similar chunk_ids (evidence / counter-evidence).
    explanation : str
        Human-readable explanation of the verdict.
    tier : str
        "keyword" if only Tier 1 ran; "semantic" if Tier 2 also ran.
    """

    label: str
    score: float
    max_similarity: float
    keyword_hits: int
    top_chunk_ids: list[str] = field(default_factory=list)
    explanation: str = ""
    tier: str = "keyword"

    def __str__(self) -> str:
        return (
            f"[NOVELTY]  {self.label:12}  "
            f"score={self.score:.3f}  "
            f"kw_hits={self.keyword_hits}  "
            f"max_sim={self.max_similarity:.3f}  "
            f"({self.tier})\n"
            f"           {self.explanation}"
        )

    @property
    def show_disclaimer(self) -> bool:
        """True when the UI should show the corpus disclaimer."""
        return self.label == "NOVEL"


# ── NoveltyClassifier ─────────────────────────────────────────────────────────


class NoveltyClassifier:
    """
    Corpus-grounded novelty classifier for BoneMind hypotheses.

    Parameters
    ----------
    db_path : Path
        Path to chunks.db (default: from settings).
    use_semantic : bool
        Whether to attempt Tier 2 semantic similarity (default True).
        Set False to force keyword-only mode (faster, no model needed).
    sample_size : int
        Number of random embeddings to load for semantic search
        (default 8,000 — balances accuracy vs memory).

    Usage::

        classifier = NoveltyClassifier()
        result = classifier.classify(hypothesis_result)
        print(result)
        print(result.score)         # 0 = covered, 1 = novel
        print(result.show_disclaimer)  # True → show corpus warning in UI
    """

    def __init__(
        self,
        db_path: Path = CHUNKS_DB_PATH,
        use_semantic: bool = True,
        sample_size: int = _SEM_SAMPLE_SIZE,
        model=None,
        tokenizer=None,
        device=None,
    ) -> None:
        self.db_path      = db_path
        self.use_semantic = use_semantic
        self.sample_size  = sample_size

        # Allow caller to pass in a pre-loaded SPECTER2 model to avoid
        # loading it twice (retriever already loads the query adapter).
        if model is not None:
            self._model     = model
            self._tokenizer = tokenizer
            self._device    = device
            logger.info("NoveltyClassifier: reusing pre-loaded SPECTER2 model.")
        else:
            self._model     = None
            self._tokenizer = None
            self._device    = None

        # Lazy-loaded semantic index
        self._emb_matrix:  np.ndarray | None = None   # (N, 768) float32, L2-normalised
        self._emb_ids:     list[str] | None  = None   # chunk_ids matching rows
        self._semantic_ok: bool = False               # True once index is loaded

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, hypothesis: "PhysicsHypothesis") -> NoveltyResult:
        """
        Classify the novelty of a hypothesis against the corpus.

        Parameters
        ----------
        hypothesis : PhysicsHypothesis
            Output from LRM.query_physics().

        Returns
        -------
        NoveltyResult
        """
        # Extract key terms from the hypothesis chain
        terms = self._extract_terms(hypothesis)
        if not terms:
            return NoveltyResult(
                label="UNCERTAIN",
                score=0.5,
                max_similarity=0.5,
                keyword_hits=0,
                explanation="No key terms could be extracted from hypothesis chain.",
            )

        # ── Tier 1: keyword search ─────────────────────────────────────────
        kw_hits, partial_hits, top_ids = self._keyword_search(terms)
        kw_result = self._kw_label(kw_hits, partial_hits, terms)

        # ── Tier 2: semantic similarity (optional) ─────────────────────────
        if self.use_semantic:
            sem_result = self._semantic_search(hypothesis.summary or hypothesis.chain_str())
        else:
            sem_result = None

        # ── Merge results ──────────────────────────────────────────────────
        return self._merge(kw_result, sem_result, kw_hits, top_ids, terms)

    def classify_text(self, text: str) -> NoveltyResult:
        """
        Classify novelty of a free-form text string.

        Useful for checking a hypothesis written as plain text rather
        than a PhysicsHypothesis.

        Parameters
        ----------
        text : str
            Hypothesis or claim to check.

        Returns
        -------
        NoveltyResult
        """
        terms = [
            t for t in re.findall(r"[a-z][a-z0-9_]{3,}", text.lower())
            if len(t) >= _KW_MIN_TERM_LENGTH
        ]
        terms = list(dict.fromkeys(terms))[:8]  # deduplicate, cap at 8

        kw_hits, partial_hits, top_ids = self._keyword_search(terms)
        kw_result = self._kw_label(kw_hits, partial_hits, terms)

        sem_result = None
        if self.use_semantic:
            sem_result = self._semantic_search(text)

        return self._merge(kw_result, sem_result, kw_hits, top_ids, terms)

    # ── Term extraction ───────────────────────────────────────────────────────

    def _extract_terms(self, hypothesis: "PhysicsHypothesis") -> list[str]:
        """
        Extract meaningful search terms from a hypothesis.

        Pulls tokens from node_ids in the chain, ignoring very short words
        and stop words.  Returns up to 8 terms.
        """
        _STOP = frozenset({
            "bone", "the", "and", "for", "with", "from", "that", "this",
            "are", "were", "has", "have", "been", "its", "into", "onto",
        })

        terms: list[str] = []
        for nid in hypothesis.chain:
            # Split snake_case node_id into tokens
            for tok in nid.split("_"):
                if (
                    len(tok) >= _KW_MIN_TERM_LENGTH
                    and tok not in _STOP
                    and tok not in terms
                ):
                    terms.append(tok)

        return terms[:8]

    # ── Tier 1: keyword search ────────────────────────────────────────────────

    def _keyword_search(
        self, terms: list[str]
    ) -> tuple[int, int, list[str]]:
        """
        Count chunks containing all / most key terms.

        Returns
        -------
        tuple[int, int, list[str]]
            (full_hits, partial_hits, top_chunk_ids)
            full_hits    — chunks containing ALL terms
            partial_hits — chunks containing ≥ ceil(len/2) terms
            top_chunk_ids — up to 5 chunk ids with most term coverage
        """
        if not terms:
            return 0, 0, []

        conn = sqlite3.connect(self.db_path)
        try:
            # Build query: each term as a LIKE condition on lowercased text
            like_clauses = " AND ".join(
                f"lower(text) LIKE ?" for _ in terms
            )
            params = [f"%{t}%".lower() for t in terms]

            # Full match — all terms present
            full_sql = f"SELECT id FROM chunks WHERE {like_clauses} LIMIT 50"
            full_rows = conn.execute(full_sql, params).fetchall()
            full_hits = len(full_rows)
            top_ids   = [str(r[0]) for r in full_rows[:_SEM_TOP_K]]

            # Partial match — most terms present (≥ half)
            if len(terms) > 1:
                half = max(1, len(terms) // 2)
                partial_clauses = " OR ".join(
                    f"lower(text) LIKE ?" for _ in terms[:half]
                )
                partial_params = params[:half]
                partial_rows = conn.execute(
                    f"SELECT COUNT(*) FROM chunks WHERE {partial_clauses}",
                    partial_params,
                ).fetchone()
                partial_hits = partial_rows[0]
            else:
                partial_hits = full_hits

        finally:
            conn.close()

        return full_hits, partial_hits, top_ids

    def _kw_label(
        self, full_hits: int, partial_hits: int, terms: list[str]
    ) -> tuple[str, float, str]:
        """
        Convert keyword hit counts to (label, similarity, explanation).
        """
        n = len(terms)
        if full_hits >= _KW_GROUNDED_CHUNKS:
            sim = min(0.90, 0.70 + full_hits * 0.01)
            return (
                "GROUNDED",
                sim,
                f"{full_hits} chunks matched all {n} key terms — "
                f"well-covered in the literature.",
            )
        elif full_hits >= _KW_SPECULATIVE_CHUNKS:
            sim = 0.65 + full_hits * 0.02
            return (
                "SPECULATIVE",
                sim,
                f"{full_hits} chunks matched all {n} key terms, "
                f"{partial_hits} matched partially — "
                f"partially discussed in the literature.",
            )
        elif partial_hits >= _KW_GROUNDED_CHUNKS:
            return (
                "SPECULATIVE",
                0.62,
                f"0 chunks matched all {n} key terms, "
                f"{partial_hits} matched partially — "
                f"component concepts exist but this combination is less documented.",
            )
        else:
            sim = max(0.0, 0.30 - partial_hits * 0.02)
            return (
                "NOVEL",
                sim,
                f"0 chunks matched all {n} key terms, "
                f"{partial_hits} matched partially — "
                f"this combination appears under-represented in the corpus.",
            )

    # ── Tier 2: semantic similarity ───────────────────────────────────────────

    def _load_semantic_index(self) -> bool:
        """
        Load a random sample of chunk embeddings into memory.

        Returns True if index loaded successfully, False on error.
        """
        if self._semantic_ok:
            return True

        logger.info(
            "Loading semantic index: %d random embeddings from chunks.db…",
            self.sample_size,
        )
        try:
            conn = sqlite3.connect(self.db_path)
            # Random sample using SQLite ORDER BY RANDOM()
            rows = conn.execute(
                f"SELECT id, embedding FROM chunks "
                f"WHERE embedding IS NOT NULL "
                f"ORDER BY RANDOM() LIMIT {self.sample_size}"
            ).fetchall()
            conn.close()

            if not rows:
                logger.warning("No embeddings found in chunks.db.")
                return False

            ids  = [str(r[0]) for r in rows]
            embs = np.stack([
                np.frombuffer(r[1], dtype=np.float32) for r in rows
            ])  # shape (N, 768)

            # L2-normalise for cosine similarity via dot product
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embs  = embs / norms

            self._emb_matrix  = embs
            self._emb_ids     = ids
            self._semantic_ok = True
            logger.info(
                "Semantic index ready: %d embeddings, dim=%d",
                len(ids), embs.shape[1],
            )
            return True

        except Exception as exc:
            logger.warning("Failed to load semantic index: %s", exc)
            return False

    def _embed_text(self, text: str) -> np.ndarray | None:
        """
        Encode text using the SPECTER2 query adapter (allenai/specter2_adhoc_query).

        Matches exactly the encoding used by retrieval/retriever.py so that
        cosine similarity against stored chunk embeddings is meaningful.

        Returns L2-normalised (768,) float32 array, or None on failure.
        Model is loaded once and cached on self._model / self._tokenizer.
        """
        # Return cached result immediately if model already loaded
        if not hasattr(self, "_model") or self._model is None:
            try:
                import torch
                from adapters import AutoAdapterModel
                from transformers import AutoTokenizer

                logger.info("Loading SPECTER2 query adapter for novelty scoring…")
                tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
                model     = AutoAdapterModel.from_pretrained("allenai/specter2_base")
                model.load_adapter(
                    "allenai/specter2_adhoc_query",
                    source="hf",
                    load_as="specter2_query",
                    set_active=True,
                )
                model.set_active_adapters("specter2_query")
                # Use MPS on Apple Silicon if available
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                elif torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")
                model.to(device)
                model.eval()

                self._tokenizer = tokenizer
                self._model     = model
                self._device    = device
                logger.info("SPECTER2 query adapter loaded on %s.", device)

            except Exception as exc:
                logger.warning(
                    "SPECTER2 load failed (%s) — Tier 2 unavailable.", exc
                )
                self._model = None
                return None

        try:
            import torch
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model(**inputs)
            vec  = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec

        except Exception as exc:
            logger.warning("SPECTER2 encoding failed: %s", exc)
            return None

    def _semantic_search(
        self, text: str
    ) -> tuple[str, float, list[str]] | None:
        """
        Encode text and find most similar corpus chunks.

        Returns
        -------
        tuple[str, float, list[str]] | None
            (label, max_similarity, top_chunk_ids) or None if unavailable.
        """
        if not self._load_semantic_index():
            return None

        query_vec = self._embed_text(text)
        if query_vec is None:
            return None

        # Cosine similarity = dot product (both are L2-normalised)
        sims  = self._emb_matrix @ query_vec        # shape (N,)
        top_k = np.argsort(sims)[::-1][:_SEM_TOP_K]

        max_sim  = float(sims[top_k[0]])
        top_ids  = [self._emb_ids[i] for i in top_k]

        if max_sim >= _SEM_GROUNDED_SIM:
            label = "GROUNDED"
        elif max_sim >= _SEM_SPECULATIVE_SIM:
            label = "SPECULATIVE"
        else:
            label = "NOVEL"

        return label, max_sim, top_ids

    # ── Merge tiers ───────────────────────────────────────────────────────────

    def _merge(
        self,
        kw_result:  tuple[str, float, str],
        sem_result: tuple[str, float, list[str]] | None,
        kw_hits:    int,
        top_ids:    list[str],
        terms:      list[str],
    ) -> NoveltyResult:
        """
        Combine Tier 1 and Tier 2 results into a final NoveltyResult.

        Merge rule:
        - If semantic is available, take the more conservative (higher
          similarity) verdict — we prefer to under-claim novelty.
        - If semantic is unavailable, use keyword result.
        """
        kw_label, kw_sim, kw_explanation = kw_result

        if sem_result is None:
            # Keyword only
            score = max(0.0, 1.0 - kw_sim)
            return NoveltyResult(
                label=kw_label,
                score=round(score, 4),
                max_similarity=round(kw_sim, 4),
                keyword_hits=kw_hits,
                top_chunk_ids=top_ids,
                explanation=kw_explanation,
                tier="keyword",
            )

        sem_label, sem_sim, sem_ids = sem_result

        # Take the higher similarity (more conservative about novelty)
        if sem_sim >= kw_sim:
            final_label = sem_label
            final_sim   = sem_sim
            final_ids   = sem_ids
            explanation = (
                f"{kw_explanation} "
                f"Semantic similarity: {sem_sim:.3f} (semantic prevails)."
            )
            tier = "semantic"
        else:
            final_label = kw_label
            final_sim   = kw_sim
            final_ids   = top_ids or sem_ids
            explanation = (
                f"{kw_explanation} "
                f"Semantic similarity: {sem_sim:.3f} (keyword prevails)."
            )
            tier = "semantic"

        score = max(0.0, 1.0 - final_sim)
        return NoveltyResult(
            label=final_label,
            score=round(score, 4),
            max_similarity=round(final_sim, 4),
            keyword_hits=kw_hits,
            top_chunk_ids=final_ids,
            explanation=explanation,
            tier=tier,
        )

    # ── Batch API ─────────────────────────────────────────────────────────────

    def classify_batch(
        self, hypotheses: "list[PhysicsHypothesis]"
    ) -> "list[NoveltyResult]":
        """
        Classify a list of hypotheses.

        Loads the semantic index once then classifies all hypotheses.
        More efficient than calling classify() repeatedly when the
        semantic index is used.

        Parameters
        ----------
        hypotheses : list[PhysicsHypothesis]

        Returns
        -------
        list[NoveltyResult]
            Same order as input.
        """
        if self.use_semantic:
            self._load_semantic_index()   # load once
        return [self.classify(h) for h in hypotheses]
