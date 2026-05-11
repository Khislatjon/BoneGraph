"""
reasoning/lrm.py
================
Latent Reasoning Module (LRM) — physics-driven hypothesis pipeline.

The LRM is the orchestrator for the Reasoning tab.  It loads the bone
knowledge graph (used only as a source of canonical node identifiers)
and exposes a single public entry point — :meth:`LRM.query_physics` —
which composes the physics generator, the adversarial critic, and an
optional corpus-grounded novelty classifier into one ranked list of
hypotheses.

Pipeline per query
------------------
1. :class:`PhysicsGenerator` runs every applicable physical law over a
   modest perturbation grid and emits candidate hypotheses with
   quantitative ΔY/Y predictions.
2. :class:`PhysicsCritic` runs three independent falsification rounds
   on each candidate.  Falsified candidates are dropped unless
   ``keep_falsified=True``.
3. Optional :class:`NoveltyClassifier` labels survivors GROUNDED /
   SPECULATIVE / NOVEL based on corpus presence.
4. Survivors are scored and sorted by descending score.

Note
----
The previous graph-walk reasoner (``query``, ``find_gaps``,
``neighbours_of`` and the heuristic novelty classifier) has been
removed.  See the ``v0-physics-grid`` tag for that history.

Usage::

    from reasoning.lrm import LRM
    from reasoning.novelty import NoveltyClassifier

    lrm = LRM()
    classifier = NoveltyClassifier()

    results = lrm.query_physics(
        "How does porosity affect fatigue crack growth?",
        novelty_classifier=classifier,
    )
    for h in results:
        print(h)
"""

from __future__ import annotations

import logging

from reasoning.critic import PhysicsCritic
from reasoning.graph_db import OntologyStore
from reasoning.ontology import BoneKnowledgeGraph
from reasoning.physics import PhysicsEngine
from reasoning.physics_gen import PhysicsGenerator, PhysicsHypothesis

logger = logging.getLogger(__name__)


class LRM:
    """
    Latent Reasoning Module — physics-driven hypothesis orchestrator.

    Loads the bone knowledge graph from ``ontology.db`` once at
    construction.  The graph is consulted only for the set of available
    node identifiers (used by :class:`PhysicsGenerator` to pick canonical
    proxies) and for label rendering in the API layer.

    Usage::

        lrm = LRM()
        results = lrm.query_physics("How does porosity affect E?")
    """

    def __init__(self) -> None:
        self._physics = PhysicsEngine()
        self._graph = self._load_graph()

        # The graph node-ID set is captured here so the generator can
        # pick canonical proxies that actually exist in the graph.
        self._available_nodes: set[str] = {
            n.node_id for n in self._graph.iter_nodes()
        }
        self._generator = PhysicsGenerator(self._available_nodes)
        self._critic = PhysicsCritic(self._physics)

        logger.info(
            "LRM ready — graph: %d nodes, %d edges | physics rules: %d",
            len(self._graph._nodes),
            len(self._graph._edges),
            self._physics.rule_count(),
        )

    # ── Graph loading ─────────────────────────────────────────────────────────

    def _load_graph(self) -> BoneKnowledgeGraph:
        """Load the current graph from ontology.db."""
        with OntologyStore() as store:
            graph = store.load_graph()
        return graph

    def reload(self) -> None:
        """Reload the graph from disk."""
        self._graph = self._load_graph()
        self._available_nodes = {n.node_id for n in self._graph.iter_nodes()}
        self._generator = PhysicsGenerator(self._available_nodes)
        logger.info("Graph reloaded: %s", self._graph)

    # ── Public API ────────────────────────────────────────────────────────────

    def query_physics(
        self,
        question: str,
        max_results: int = 8,
        novelty_classifier=None,
        keep_falsified: bool = False,
    ) -> list[PhysicsHypothesis]:
        """
        Generate physics-derived hypotheses for ``question``.

        Parameters
        ----------
        question : str
            Free-text user query.
        max_results : int
            Cap on the returned list.
        novelty_classifier : NoveltyClassifier | None
            If provided, used to label corpus presence.
        keep_falsified : bool
            If True, falsified hypotheses are retained (annotated with
            their failure reason) so the UI can show *why* they failed.

        Returns
        -------
        list[PhysicsHypothesis]
            Ranked by descending score.
        """
        candidates = self._generator.generate(question)
        if not candidates:
            logger.info(
                "PhysicsGenerator returned 0 candidates for query %r", question
            )
            return []

        kept: list[PhysicsHypothesis] = []
        for h in candidates:
            verdict = self._critic.critique(h)
            h.critique = verdict
            if not verdict.survived and not keep_falsified:
                continue

            # Novelty pass — corpus presence check (best-effort).
            if novelty_classifier is not None:
                try:
                    nov = novelty_classifier.classify(h)
                    h.novelty = nov.label
                    h.novelty_reason = nov.explanation
                    h.corpus_disclaimer = (
                        getattr(nov, "corpus_disclaimer", None)
                        if getattr(nov, "show_disclaimer", False) else None
                    )
                except Exception as exc:        # pragma: no cover
                    logger.warning("Novelty classification failed: %s", exc)

            h.score = self._score_physics(h)
            kept.append(h)

        kept.sort(key=lambda x: x.score, reverse=True)
        return kept[:max_results]

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_physics(self, h: PhysicsHypothesis) -> float:
        """
        Composite score for a physics-derived hypothesis.

        Components (each contributes 0–1, weighted):
          • critic survival    (60%)  — fraction of rounds passed
          • magnitude          (25%)  — |ΔY/Y| capped at 50%, normalised
          • novelty bonus      (15%)  — NOVEL > SPECULATIVE > GROUNDED
        """
        if h.critique:
            survival = h.critique.rounds_passed / max(h.critique.rounds_total, 1)
        else:
            survival = 0.0

        delta_pct = abs(h.delta_output.get("change_pct", 0.0))
        magnitude = min(delta_pct / 50.0, 1.0)

        novelty_bonus = {
            "NOVEL":       1.0,
            "SPECULATIVE": 0.6,
            "GROUNDED":    0.2,
            "":            0.5,    # unlabelled — neutral
        }.get(h.novelty, 0.5)

        return 0.60 * survival + 0.25 * magnitude + 0.15 * novelty_bonus

    # ── Stats ─────────────────────────────────────────────────────────────────

    def graph_stats(self) -> dict:
        """Return current graph statistics."""
        return self._graph.stats()
