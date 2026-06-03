"""
reasoning/semantic_anchor.py
────────────────────────────
SPECTER2-backed semantic top-k retrieval for Variables in the
equation-graph reasoner.

A free-text user query is embedded with the same SPECTER2 adhoc-query
adapter that the retriever and novelty classifier already use; cosine
similarity against pre-embedded ``Variable.description`` vectors gives
a top-k list of plausible anchor variables.

This module owns no model loading of its own — instead it borrows the
tokenizer/model/device the :class:`retrieval.retriever.BoneMindRetriever`
loads at app startup.  That keeps memory usage flat (one SPECTER2 in
RAM, not two) and avoids a cold-start delay on the first reasoning
query.  If the retriever has not been loaded the anchor falls back to
its own lazy load, mirroring the pattern in
:meth:`reasoning.novelty.NoveltyClassifier._embed_text`.

The anchor stays deterministic: for the same query and the same
registry it always returns the same ranking.  The reasoning engine in
:class:`reasoning.relation.RelationRegistry` is unchanged — only the
*mapping from natural language to Variable symbols* moves from regex
keywords to cosine similarity over learned embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from reasoning.relation import RelationRegistry, Variable

logger = logging.getLogger(__name__)


# ── Encoder protocol ─────────────────────────────────────────────────────────


@dataclass
class _EncoderHandle:
    """Bag of the three SPECTER2 objects we need to encode a string."""

    tokenizer: object
    model: object
    device: object

    @classmethod
    def from_retriever(cls, retriever) -> "_EncoderHandle | None":
        """Borrow SPECTER2 from a loaded :class:`BoneMindRetriever`."""
        tok = getattr(retriever, "_tokenizer", None)
        mod = getattr(retriever, "_model", None)
        dev = getattr(retriever, "_device", None)
        if tok is None or mod is None or dev is None:
            return None
        return cls(tokenizer=tok, model=mod, device=dev)

    @classmethod
    def lazy_load(cls) -> "_EncoderHandle | None":
        """
        Fallback path — load SPECTER2 ourselves.

        This exists so the anchor still works in unit tests or in any
        deployment where the retriever isn't running.  In production
        :meth:`from_retriever` is the hot path.
        """
        try:
            import torch
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer

            logger.info("Loading SPECTER2 for semantic variable anchor…")
            tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
            model     = AutoAdapterModel.from_pretrained("allenai/specter2_base")
            model.load_adapter(
                "allenai/specter2_adhoc_query",
                source="hf",
                load_as="specter2_query",
                set_active=True,
            )
            model.set_active_adapters("specter2_query")
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            model.to(device)
            model.eval()
            return cls(tokenizer=tokenizer, model=model, device=device)
        except Exception as exc:
            logger.warning("Semantic anchor SPECTER2 load failed: %s", exc)
            return None


def _encode(encoder: _EncoderHandle, text: str) -> np.ndarray | None:
    """Encode a single string into an L2-normalised (768,) float32 vector."""
    try:
        import torch
        inputs = encoder.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(encoder.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = encoder.model(**inputs)
        vec = (
            out.last_hidden_state[:, 0, :]
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec
    except Exception as exc:
        logger.warning("SPECTER2 encoding failed for %r: %s", text[:60], exc)
        return None


# ── Semantic anchor ──────────────────────────────────────────────────────────


@dataclass
class VariableMatch:
    """One row of a top-k semantic match against the variable graph."""

    symbol: str
    name: str
    unit: str
    description: str
    score: float                 # cosine similarity in [-1, 1]
    display_symbol: str = ""     # Unicode form for the UI; defaults to symbol


class SemanticVariableAnchor:
    """
    Map a free-text query to the most relevant Variable symbols in a
    :class:`RelationRegistry`.

    Queries like *"bone stiffness under cyclic load"* should anchor
    both Currey (stiffness → E) and Paris (cyclic load → ΔK / da/dN)
    in the top-k.  A pure-keyword anchor would only catch whichever
    phrase happens to overlap a curated keyword list; SPECTER2
    embeddings handle paraphrases and synonyms naturally.

    Construction
    ------------
    ``encoder`` is a tuple of (tokenizer, model, device).  Pass
    ``retriever=...`` to borrow them from a loaded retriever, or omit
    both to lazy-load a private SPECTER2.  Either way the variable
    embeddings are computed once on first use and cached.
    """

    def __init__(
        self,
        registry: RelationRegistry,
        *,
        retriever=None,
    ) -> None:
        self._registry = registry
        self._retriever = retriever
        self._encoder: _EncoderHandle | None = None
        self._symbols: list[str] = []
        self._matrix: np.ndarray | None = None       # (V, 768)
        self._variables: list[Variable] = []

    # ── Lazy init ─────────────────────────────────────────────────────────

    def _ensure_encoder(self) -> bool:
        if self._encoder is not None:
            return True
        if self._retriever is not None:
            self._encoder = _EncoderHandle.from_retriever(self._retriever)
        if self._encoder is None:
            self._encoder = _EncoderHandle.lazy_load()
        return self._encoder is not None

    def _ensure_index(self) -> bool:
        """Encode every Variable once and cache the embedding matrix."""
        if self._matrix is not None:
            return True
        if not self._ensure_encoder():
            return False
        assert self._encoder is not None

        # Pull every variable registered in the graph that has metadata.
        variables: list[Variable] = []
        for sym in sorted(self._registry.variables_in_graph()):
            v = self._registry.variable(sym)
            if v is not None:
                variables.append(v)
        if not variables:
            return False

        vecs: list[np.ndarray] = []
        for v in variables:
            text = self._encode_target_text(v)
            vec = _encode(self._encoder, text)
            if vec is None:
                logger.warning("Failed to embed variable %r; skipping.", v.symbol)
                continue
            logger.debug("Anchor text for %r: %s", v.symbol, text)
            self._symbols.append(v.symbol)
            self._variables.append(v)
            vecs.append(vec)

        if not vecs:
            return False
        self._matrix = np.vstack(vecs).astype(np.float32)
        logger.info(
            "SemanticVariableAnchor: indexed %d variables (dim=%d).",
            len(self._symbols), self._matrix.shape[1],
        )
        return True

    def _encode_target_text(self, v: Variable) -> str:
        """
        Build the per-variable string fed to SPECTER2.

        SPECTER2 is a passage encoder trained on scientific abstracts,
        so it discriminates much better given context than from a bare
        noun.  The per-variable text concatenates:

        * The human-readable name, unit, description.
        * Every Relation the variable participates in (name plus the
          relation's own one-line description).  This is what lets
          ``rho`` (apparent density, structural) be distinguished from
          ``dBMD_dt`` (bone adaptation rate, change-over-time) even
          though both descriptions mention "density".

        Listing the relations is a deterministic, registry-derived bit
        of context — no hand-curated alias lists required.  When new
        Relations are added the anchor text picks them up automatically.
        """
        unit_str = f" [{v.unit}]" if v.unit else ""
        head = f"{v.name}{unit_str}. {v.description}".strip()
        if not head.endswith("."):
            head += "."

        # Append the relations this variable participates in.
        contexts: list[str] = []
        for rel in self._registry.relations():
            if v.symbol == rel.output or v.symbol in rel.inputs:
                role = "output of" if v.symbol == rel.output else "input to"
                desc = rel.description.rstrip(".") if rel.description else rel.name
                contexts.append(f"{role} {rel.name}: {desc}")
        if contexts:
            head += " Appears in: " + "; ".join(contexts) + "."

        return head

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self._matrix is not None

    def top_k(self, query: str, k: int = 5) -> list[VariableMatch]:
        """
        Return the top-``k`` variables most similar to ``query``.

        ``score`` is raw cosine similarity in ``[-1, 1]``; the values
        are not calibrated probabilities but they are stable for the
        same model + index so downstream code can threshold them.
        Returns an empty list if SPECTER2 is unavailable.
        """
        if not query.strip():
            return []
        if not self._ensure_index() or self._matrix is None:
            return []
        assert self._encoder is not None

        q_vec = _encode(self._encoder, query)
        if q_vec is None:
            return []

        sims = self._matrix @ q_vec        # (V,)
        order = np.argsort(sims)[::-1][: max(1, k)]
        out: list[VariableMatch] = []
        for idx in order:
            v = self._variables[int(idx)]
            out.append(VariableMatch(
                symbol=v.symbol,
                display_symbol=v.render(),
                name=v.name,
                unit=v.unit,
                description=v.description,
                score=float(sims[int(idx)]),
            ))
        return out

    def best_match(
        self, query: str, *, min_score: float = 0.0,
    ) -> VariableMatch | None:
        """Single best variable for ``query`` or ``None`` below the floor."""
        if not query.strip():
            return None
        top = self.top_k(query, k=1)
        if not top:
            return None
        return top[0] if top[0].score >= min_score else None
