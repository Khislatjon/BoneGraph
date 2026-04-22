"""
reasoning/lrm.py
================
Step 4.4 — Latent Reasoning Module (LRM) for BoneLogic.

Overview
--------
The LRM is the reasoning core of Phase 4.  It takes a natural-language
question, maps it onto the bone knowledge graph, traverses causal chains,
validates them with the physics engine, and returns structured hypotheses
that go beyond what any single paper explicitly states.

Pipeline per query
------------------
1. Anchor      — map query keywords to node_ids in the graph
2. Traverse    — find causal paths between anchor nodes (shortest paths,
                 multi-hop chains, neighbour expansion)
3. Gap detect  — identify high-betweenness nodes with sparse edges
                 (candidate research gaps)
4. Validate    — pass every chain through PhysicsEngine
5. Rank        — score by query relevance × edge weight × chain length × novelty
6. Generate    — format structured hypothesis output

Output format
-------------
Each hypothesis is a HypothesisResult dataclass::

    [CHAIN]    aging → porosity → elastic_modulus → fracture_risk
    [EDGES]    increases | decreases | decreases
    [PHYSICS]  PLAUSIBLE  (Currey's law, Bone aging)
    [NOVELTY]  SPECULATIVE  (chain spans 3 hops — unlikely in one paper)
    [SUMMARY]  Aging-driven porosity increase reduces elastic modulus,
               elevating fracture risk via reduced energy absorption.

Usage::

    from reasoning.lrm import LRM

    lrm = LRM()

    # Single query
    results = lrm.query("How does aging affect fracture risk?")
    for h in results:
        print(h)

    # Research gap detection
    gaps = lrm.find_gaps(top_n=10)
    for g in gaps:
        print(g)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterator

from reasoning.graph_db import OntologyStore
from reasoning.ontology import BoneKnowledgeGraph, Edge, Node
from reasoning.physics import PhysicsEngine, ValidationResult

logger = logging.getLogger(__name__)


# ── Novelty classifier ────────────────────────────────────────────────────────
# Simple heuristic until Step 4.5 (novelty.py) is built.
# A chain is GROUNDED if all edges are seed edges (well-established).
# SPECULATIVE if it spans 3+ hops or contains extracted edges.
# NOVEL if it contains extracted edges AND crosses concept type boundaries.

def _classify_novelty(
    chain: list[str],
    edges: list[Edge],
    graph: BoneKnowledgeGraph,
) -> tuple[str, str]:
    """
    Return (novelty_label, reason).

    Labels
    ------
    GROUNDED     — all seed edges and chain ≤ 2 hops, OR high-evidence
                   extracted chain (mean weight ≥ 5.0, chain ≤ 3 hops).
                   Weight accumulates per chunk that confirms a triple, so
                   mean_weight ≥ 5 means the relationship appears in 5+
                   independent corpus passages.
    SPECULATIVE  — extracted edges with moderate evidence, or seed edges
                   in a longer chain.  Plausible but not fully attested.
    NOVEL        — long chain (≥ 3 hops) of low-evidence extracted edges
                   (mean weight < 2.0) crossing ≥ 3 distinct node-type
                   boundaries.  Connections not yet consolidated in the
                   literature.

    Design note
    -----------
    The original classifier used type_changes ≥ 2 as the NOVEL signal.
    After full paper extraction the graph has ~35K nodes, most typed as
    "concept" (LLM default), while seed nodes have specific types.  Any
    mixed seed+extracted path crosses ≥ 2 boundaries trivially, making
    NOVEL fire for everything.  Weight-based thresholds are more stable.
    """
    n_hops = len(edges)
    n_extracted = sum(1 for e in edges if e.edge_source == "extracted")
    mean_weight = sum(e.weight for e in edges) / n_hops if n_hops else 0.0

    # Count distinct node-type boundaries crossed along the chain
    type_changes = 0
    for i in range(1, len(chain)):
        n1 = graph.get_node(chain[i - 1])
        n2 = graph.get_node(chain[i])
        if n1 and n2 and n1.node_type != n2.node_type:
            type_changes += 1

    # GROUNDED — all-seed short chain (canonical case)
    if n_extracted == 0 and n_hops <= 2:
        return "GROUNDED", "all seed edges; ≤2 hops"

    # GROUNDED — well-attested extracted chain (≥5 corpus confirmations)
    if mean_weight >= 5.0 and n_hops <= 3:
        return "GROUNDED", (
            f"high-evidence chain (mean weight {mean_weight:.1f}); "
            f"{n_hops} hops"
        )

    # NOVEL — long, low-evidence, cross-domain chain
    if n_extracted >= 2 and n_hops >= 3 and mean_weight < 2.0 and type_changes >= 3:
        return "NOVEL", (
            f"{n_hops}-hop cross-domain chain; "
            f"{n_extracted} extracted edges; "
            f"mean weight {mean_weight:.1f}; "
            f"{type_changes} type boundaries"
        )

    # SPECULATIVE — everything else
    return "SPECULATIVE", (
        f"{n_hops}-hop chain; {n_extracted}/{n_hops} extracted; "
        f"mean weight {mean_weight:.1f}"
    )


# ── HypothesisResult ──────────────────────────────────────────────────────────


@dataclass
class HypothesisResult:
    """
    A single validated causal hypothesis produced by the LRM.

    Attributes
    ----------
    chain : list[str]
        Ordered list of node_ids forming the causal path.
    edges : list[Edge]
        Edge objects along the chain (len = len(chain) - 1).
    physics : ValidationResult
        Result of physics engine validation.
    novelty : str
        "GROUNDED" | "SPECULATIVE" | "NOVEL"
    novelty_reason : str
        Plain-English explanation of novelty classification.
    score : float
        Composite ranking score (higher = more interesting).
    summary : str
        One-sentence human-readable description of the hypothesis.
    """

    chain: list[str]
    edges: list[Edge]
    physics: ValidationResult
    novelty: str = "SPECULATIVE"
    novelty_reason: str = ""
    score: float = 0.0
    summary: str = ""

    # ── Formatting ────────────────────────────────────────────────────────────

    def chain_str(self, graph: BoneKnowledgeGraph | None = None) -> str:
        """
        Return the chain as a readable arrow string.
        Uses node labels if graph is supplied, node_ids otherwise.
        """
        if graph:
            labels = [
                graph.get_node(n).label if graph.get_node(n) else n
                for n in self.chain
            ]
        else:
            labels = [n.replace("_", " ") for n in self.chain]
        return " → ".join(labels)

    def relations_str(self) -> str:
        """Return edge relations joined by ' | '."""
        return " | ".join(e.relation for e in self.edges)

    def __str__(self) -> str:
        lines = [
            f"[CHAIN]    {self.chain_str()}",
            f"[EDGES]    {self.relations_str()}",
            f"[PHYSICS]  {self.physics.status}"
            + (f"  ({self.physics.law})" if self.physics.law else ""),
            f"[NOVELTY]  {self.novelty}  ({self.novelty_reason})",
        ]
        if self.summary:
            lines.append(f"[SUMMARY]  {self.summary}")
        return "\n".join(lines)


# ── GapResult ─────────────────────────────────────────────────────────────────


@dataclass
class GapResult:
    """
    A candidate research gap: a high-betweenness node with few edges.

    High betweenness means many reasoning paths pass through this node,
    yet low edge count means it is poorly connected — a conceptual bridge
    that the literature has not fully explored.
    """

    node_id: str
    label: str
    node_type: str
    betweenness: float
    n_edges: int
    gap_score: float   # betweenness / (n_edges + 1) — higher = bigger gap

    def __str__(self) -> str:
        return (
            f"[GAP]  {self.label:35} "
            f"type={self.node_type:12} "
            f"betweenness={self.betweenness:.4f}  "
            f"edges={self.n_edges}  "
            f"gap_score={self.gap_score:.4f}"
        )


# ── LRM ───────────────────────────────────────────────────────────────────────


class LRM:
    """
    Latent Reasoning Module — the reasoning core of BoneLogic Phase 4.

    Loads the bone knowledge graph from ontology.db on construction,
    then answers queries by traversing causal chains and generating
    physics-validated hypotheses.

    Parameters
    ----------
    max_hops : int
        Maximum path length to consider (default 5).
        Longer chains are less likely to be physically coherent.
    min_edge_weight : float
        Minimum edge weight to include in traversal (default 0.5).
        Filters out very low-confidence extracted edges.
    physics_filter : bool
        If True (default), drop IMPLAUSIBLE chains from results.
        Set False to see everything including implausible chains
        (useful for debugging).

    Usage::

        lrm = LRM()
        results = lrm.query("How does aging affect fracture risk?")
        gaps    = lrm.find_gaps(top_n=10)
    """

    def __init__(
        self,
        max_hops: int = 5,
        min_edge_weight: float = 0.5,
        physics_filter: bool = True,
    ) -> None:
        self.max_hops = max_hops
        self.min_edge_weight = min_edge_weight
        self.physics_filter = physics_filter
        self._physics = PhysicsEngine()
        self._graph = self._load_graph()
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
        """
        Reload the graph from disk.

        Call this after the extractor has run for a while to pick up
        newly extracted nodes and edges without restarting.
        """
        self._graph = self._load_graph()
        logger.info("Graph reloaded: %s", self._graph)

    # ── Anchoring — map query text to node_ids ────────────────────────────────

    def _anchor(self, query: str) -> list[str]:
        """
        Find node_ids in the graph that match keywords in the query.

        Matching strategy (in priority order):
        1. Exact node_id match (after query normalisation)
        2. Node label substring match (case-insensitive)
        3. Node_id substring match

        Parameters
        ----------
        query : str
            Natural-language question or keyword string.

        Returns
        -------
        list[str]
            node_ids found in the graph, ordered by match quality.
        """
        # Normalise query to tokens
        query_lower = query.lower()
        tokens = re.findall(r"[a-z0-9]+", query_lower)
        query_snake = "_".join(tokens)
        sig_tokens = [t for t in tokens if len(t) > 3]

        # node_id → (priority, -token_hits) for fine-grained ranking
        seen: dict[str, tuple[int, float]] = {}

        for node in self._graph.iter_nodes():
            nid   = node.node_id
            label = node.label.lower()

            # 1 — exact node_id match
            if nid == query_snake:
                seen[nid] = (0, 0.0)
                continue

            # Count how many distinct query tokens appear in this node_id
            token_hits = sum(1 for t in sig_tokens if t in nid)

            # 2 — node_id contains at least one query token
            if token_hits > 0:
                seen.setdefault(nid, (1, -token_hits))

            # 3 — label words appear in query
            label_words = re.findall(r"[a-z0-9]+", label)
            matches = sum(1 for w in label_words if w in tokens and len(w) > 3)
            if matches >= 1:
                seen.setdefault(nid, (2, -matches))

        # Ensure coverage: for each significant token, keep its single best match
        # (highest token_hits, then shortest node_id as tiebreak).
        # Then pad with the next best anchors up to a global cap.
        ordered = sorted(seen, key=lambda k: (seen[k][0], seen[k][1], len(k)))

        per_token_best: list[str] = []
        for tok in sig_tokens:
            for nid in ordered:
                if tok in nid and nid not in per_token_best:
                    per_token_best.append(nid)
                    break

        # Fill remaining slots from the ranked list (skip already included)
        combined = list(per_token_best)
        for nid in ordered:
            if nid not in combined:
                combined.append(nid)
            if len(combined) >= 30:
                break

        return combined

    # ── Path traversal ────────────────────────────────────────────────────────

    def _edges_for_path(self, path: list[str]) -> list[Edge] | None:
        """
        Return the Edge objects along a node path, or None if any edge missing.

        Picks the highest-weight edge between each consecutive pair.
        """
        edges: list[Edge] = []
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            # Find all edges from src → tgt
            candidates = [
                e for e in self._graph.outgoing_edges(src)
                if e.target == tgt and e.weight >= self.min_edge_weight
            ]
            if not candidates:
                return None
            # Pick highest weight
            edges.append(max(candidates, key=lambda e: e.weight))
        return edges

    def _iter_chains(
        self, anchors: list[str], max_results: int = 20
    ) -> Iterator[tuple[list[str], list[Edge]]]:
        """
        Yield (node_path, edge_list) chains between all pairs of anchor nodes.

        For each ordered pair (a, b) of anchor nodes:
        - Shortest path a → b
        - Up to 3 alternative shortest paths

        Deduplicates by path string.
        """
        seen_paths: set[str] = set()

        for i, src in enumerate(anchors):
            for tgt in anchors[i + 1:]:
                if src == tgt:
                    continue

                paths = self._graph.all_shortest_paths(src, tgt, max_paths=3)
                # Also try reverse direction
                paths += self._graph.all_shortest_paths(tgt, src, max_paths=2)

                for path in paths:
                    if len(path) < 2 or len(path) - 1 > self.max_hops:
                        continue
                    key = "→".join(path)
                    if key in seen_paths:
                        continue
                    seen_paths.add(key)

                    edge_list = self._edges_for_path(path)
                    if edge_list is None:
                        continue

                    yield path, edge_list
                    if len(seen_paths) >= max_results:
                        return

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(
        self,
        chain: list[str],
        edges: list[Edge],
        physics: ValidationResult,
        novelty: str,
        query_tokens: set[str] | None = None,
    ) -> float:
        """
        Compute a composite ranking score for a hypothesis.

        Score components (physics is guard-rail only — penalty if IMPLAUSIBLE,
        neutral otherwise):
        - Query relevance (chain nodes ∩ query tokens)  weight 0.35
        - Mean edge weight (evidence strength)           weight 0.40
        - Chain length bonus (2–5 hops)                  weight 0.15
        - Novelty bonus                                  weight 0.10
          NOVEL=1.0, SPECULATIVE=0.5, GROUNDED=0.0
        """
        if physics.is_implausible:
            return 0.0
        length_score = min((len(edges) - 1) / max(self.max_hops - 1, 1), 1.0)
        weight_score = (
            sum(e.weight for e in edges) / len(edges) if edges else 0.0
        )
        weight_score = min(weight_score / 5.0, 1.0)   # normalise (cap at 5)
        # GROUNDED (established) ranks highest; NOVEL (low-evidence) ranks lowest.
        # This surfaces reliable chains first and pushes speculative hypotheses down.
        novelty_score = {"GROUNDED": 1.0, "SPECULATIVE": 0.5, "NOVEL": 0.0}.get(
            novelty, 0.5
        )
        if query_tokens:
            rel_hits = sum(
                1 for nid in chain
                if any(t in nid for t in query_tokens if len(t) > 3)
            )
            relevance_score = rel_hits / len(chain) if chain else 0.0
        else:
            relevance_score = 0.0

        return (
            0.35 * relevance_score
            + 0.40 * weight_score
            + 0.15 * length_score
            + 0.10 * novelty_score
        )

    # ── Summary generation ────────────────────────────────────────────────────

    def _summarise(
        self,
        chain: list[str],
        edges: list[Edge],
        graph: BoneKnowledgeGraph,
    ) -> str:
        """
        Generate a one-sentence summary of the causal chain.

        Uses node labels and relation verbs to construct a readable sentence.
        """
        if not chain or not edges:
            return ""

        def label(nid: str) -> str:
            n = graph.get_node(nid)
            return n.label if n else nid.replace("_", " ")

        # Build verb phrases from relation types
        _VERBS = {
            "increases":      "increases",
            "decreases":      "decreases",
            "determines":     "determines",
            "activates":      "activates",
            "inhibits":       "inhibits",
            "leads_to":       "leads to",
            "is_part_of":     "is part of",
            "predicts":       "predicts",
            "analogous_to":   "is analogous to",
            "correlates_with":"correlates with",
            "measures":       "measures",
        }

        parts: list[str] = [label(chain[0]).capitalize()]
        for edge, nid in zip(edges, chain[1:]):
            verb = _VERBS.get(edge.relation, edge.relation)
            parts.append(f"{verb} {label(nid)}")

        # Join with commas and "which" for readability
        if len(parts) == 2:
            return f"{parts[0]} {parts[1]}."
        elif len(parts) == 3:
            return f"{parts[0]} {parts[1]}, which {parts[2]}."
        else:
            mid = ", which ".join(parts[1:-1])
            return f"{parts[0]} {mid}, ultimately {parts[-1]}."

    # ── Public API ────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        max_results: int = 10,
        include_uncertain: bool = True,
    ) -> list[HypothesisResult]:
        """
        Answer a bone science question with physics-validated causal hypotheses.

        Parameters
        ----------
        question : str
            Natural-language question, e.g. "How does aging affect fracture risk?"
        max_results : int
            Maximum number of hypotheses to return (default 10).
        include_uncertain : bool
            If True (default), include UNCERTAIN physics results.
            If False, return only PLAUSIBLE chains.

        Returns
        -------
        list[HypothesisResult]
            Ranked list of hypotheses, best first.
        """
        # 1 — anchor query to graph nodes
        anchors = self._anchor(question)
        if not anchors:
            logger.warning("No graph nodes found for query: %r", question)
            return []
        logger.info(
            "Query %r anchored to %d nodes: %s",
            question, len(anchors), anchors[:5],
        )
        query_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))

        # 2 — traverse chains
        results: list[HypothesisResult] = []

        for chain, edges in self._iter_chains(anchors, max_results=max_results * 3):

            # 3 — physics validation
            edge_triples = [(e.source, e.relation, e.target) for e in edges]
            physics = self._physics.validate_chain(edge_triples)

            if self.physics_filter and physics.is_implausible:
                continue
            if not include_uncertain and physics.status == "UNCERTAIN":
                continue

            # 4 — novelty classification
            novelty, novelty_reason = _classify_novelty(chain, edges, self._graph)

            # 5 — score
            score = self._score(chain, edges, physics, novelty, query_tokens)

            # 6 — summarise
            summary = self._summarise(chain, edges, self._graph)

            results.append(
                HypothesisResult(
                    chain=chain,
                    edges=edges,
                    physics=physics,
                    novelty=novelty,
                    novelty_reason=novelty_reason,
                    score=score,
                    summary=summary,
                )
            )

        # Sort by score descending
        results.sort(key=lambda h: h.score, reverse=True)
        return results[:max_results]

    def find_gaps(self, top_n: int = 10) -> list[GapResult]:
        """
        Identify research gaps: high-betweenness nodes with few edges.

        A node with high betweenness sits on many causal paths but has
        few direct connections — it is a conceptual bridge that the
        literature has not yet fully explored.

        Parameters
        ----------
        top_n : int
            Number of gap candidates to return.

        Returns
        -------
        list[GapResult]
            Sorted by gap_score descending (highest gap first).
        """
        centrality = self._graph.betweenness_centrality()

        gaps: list[GapResult] = []
        for node in self._graph.iter_nodes():
            nid = node.node_id
            btw = centrality.get(nid, 0.0)
            if btw == 0.0:
                continue

            n_out = len(self._graph.outgoing_edges(nid))
            n_in  = len(self._graph.incoming_edges(nid))
            n_edges = n_out + n_in

            # Gap score: betweenness divided by connectivity
            # High betweenness + low connectivity = big gap
            gap_score = btw / (n_edges + 1)

            gaps.append(
                GapResult(
                    node_id=nid,
                    label=node.label,
                    node_type=node.node_type,
                    betweenness=round(btw, 6),
                    n_edges=n_edges,
                    gap_score=round(gap_score, 6),
                )
            )

        gaps.sort(key=lambda g: g.gap_score, reverse=True)
        return gaps[:top_n]

    def neighbours_of(
        self, concept: str, depth: int = 1
    ) -> list[HypothesisResult]:
        """
        Return all direct relationships around a concept as hypotheses.

        Useful for exploring what the graph knows about a specific node.

        Parameters
        ----------
        concept : str
            A node_id or keyword (anchored the same way as query()).
        depth : int
            Neighbourhood radius (default 1 = direct edges only).

        Returns
        -------
        list[HypothesisResult]
            One HypothesisResult per outgoing/incoming edge.
        """
        anchors = self._anchor(concept)
        if not anchors:
            return []

        centre = anchors[0]
        neighbour_ids = self._graph.neighbours(centre, depth=depth)

        results: list[HypothesisResult] = []
        for nid in neighbour_ids:
            # Try centre → neighbour
            for direction in [(centre, nid), (nid, centre)]:
                src, tgt = direction
                path = self._graph.shortest_path(src, tgt)
                if not path or len(path) < 2:
                    continue
                edges = self._edges_for_path(path)
                if not edges:
                    continue
                edge_triples = [(e.source, e.relation, e.target) for e in edges]
                physics = self._physics.validate_chain(edge_triples)
                if self.physics_filter and physics.is_implausible:
                    continue
                novelty, novelty_reason = _classify_novelty(path, edges, self._graph)
                score = self._score(path, edges, physics, novelty)
                summary = self._summarise(path, edges, self._graph)
                results.append(
                    HypothesisResult(
                        chain=path,
                        edges=edges,
                        physics=physics,
                        novelty=novelty,
                        novelty_reason=novelty_reason,
                        score=score,
                        summary=summary,
                    )
                )

        # Deduplicate by chain string
        seen: set[str] = set()
        unique: list[HypothesisResult] = []
        for h in results:
            key = "→".join(h.chain)
            if key not in seen:
                seen.add(key)
                unique.append(h)

        unique.sort(key=lambda h: h.score, reverse=True)
        return unique

    def graph_stats(self) -> dict:
        """Return current graph statistics."""
        return self._graph.stats()
