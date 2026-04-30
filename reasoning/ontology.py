"""
reasoning/ontology.py
=====================
Data model and in-memory graph for the BoneMind bone knowledge graph.

Design
------
Two thin dataclasses — Node and Edge — represent the typed concepts and
causal relationships in bone science.  BoneKnowledgeGraph wraps a
NetworkX DiGraph and exposes only the operations needed by the LRM:
adding nodes/edges, shortest-path traversal, betweenness centrality for
gap detection, community detection, and sub-graph extraction.

NetworkX stores the graph in memory.  graph_db.py handles persistence to
SQLite (ontology.db).  The two are deliberately separate — you can work
with BoneKnowledgeGraph objects in pure Python without touching the
database.

Node types
----------
structure   Bone structural components at any scale (osteon, collagen_fibril …)
property    Measurable mechanical / physical quantities (elastic_modulus …)
process     Dynamic biological or mechanical processes (bone_remodelling …)
pathology   Diseases and pathological conditions (osteoporosis …)
mechanism   Theoretical frameworks (Wolff_law, Frost_mechanostat …)
material    Biomaterials and implants (hydroxyapatite_scaffold …)
factor      Signaling molecules, hormones, drugs, growth factors (RANKL …)
clinical    Clinical measurements and risk scores (FRAX_score …)
scale       Hierarchical length scales (nanoscale → macroscale)
cell        Bone cells (osteoblast, osteoclast, osteocyte)

Edge relation types
-------------------
determines          structural cause of a property
increases           positive causal influence on a property / process
decreases           negative causal influence on a property / process
activates           triggers or upregulates a process
inhibits            suppresses or downregulates a process
leads_to            consequence (less strictly causal than determines)
is_part_of          hierarchical composition
predicts            statistical / clinical prediction
analogous_to        cross-domain structural mapping (bone ↔ engineered materials)
correlates_with     statistical association without established directionality
measures            clinical instrument measures a quantity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

import networkx as nx

logger = logging.getLogger(__name__)

# ── Valid controlled vocabularies ────────────────────────────────────────────

NODE_TYPES: frozenset[str] = frozenset(
    {
        "structure",
        "property",
        "process",
        "pathology",
        "mechanism",
        "material",
        "factor",
        "clinical",
        "scale",
        "cell",
        "concept",   # catch-all for LLM-extracted nodes not yet classified
    }
)

RELATION_TYPES: frozenset[str] = frozenset(
    {
        "determines",
        "increases",
        "decreases",
        "activates",
        "inhibits",
        "leads_to",
        "is_part_of",
        "predicts",
        "analogous_to",
        "correlates_with",
        "measures",
    }
)


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class Node:
    """
    A concept in the bone knowledge graph.

    Parameters
    ----------
    node_id : str
        Canonical snake_case identifier, e.g. "fracture_toughness".
        Used as the NetworkX node key and the SQLite primary key.
    label : str
        Human-readable display name, e.g. "fracture toughness".
    node_type : str
        One of NODE_TYPES.
    description : str
        One- to two-sentence definition of the concept in bone science context.
    source : str
        How this node entered the graph: "seed", "extracted", or "manual".
    """

    node_id: str
    label: str
    node_type: str
    description: str = ""
    source: str = "seed"

    def __post_init__(self) -> None:
        if self.node_type not in NODE_TYPES:
            raise ValueError(
                f"Unknown node_type '{self.node_type}'. "
                f"Must be one of: {sorted(NODE_TYPES)}"
            )


@dataclass
class Edge:
    """
    A directed causal or structural relationship between two nodes.

    Parameters
    ----------
    source : str
        node_id of the origin concept.
    relation : str
        One of RELATION_TYPES.
    target : str
        node_id of the destination concept.
    weight : float
        Confidence / frequency score.  Seed edges start at 1.0.
        Extracted edges accumulate weight as more chunks confirm them.
    evidence : list[str]
        Chunk IDs (from chunks.db) that support this edge.
        Empty for seed edges.
    edge_source : str
        How this edge entered the graph: "seed", "extracted", or "manual".
    """

    source: str
    relation: str
    target: str
    weight: float = 1.0
    evidence: list[str] = field(default_factory=list)
    edge_source: str = "seed"

    @property
    def edge_id(self) -> str:
        """Stable identifier: '<source>|<relation>|<target>'."""
        return f"{self.source}|{self.relation}|{self.target}"

    def __post_init__(self) -> None:
        if self.relation not in RELATION_TYPES:
            raise ValueError(
                f"Unknown relation '{self.relation}'. "
                f"Must be one of: {sorted(RELATION_TYPES)}"
            )


# ── BoneKnowledgeGraph ────────────────────────────────────────────────────────


class BoneKnowledgeGraph:
    """
    In-memory bone knowledge graph backed by a NetworkX DiGraph.

    Node and Edge objects are stored in parallel dicts for typed access.
    The underlying nx.DiGraph is used for all graph-theoretic operations.

    Typical usage::

        from reasoning.ontology import BoneKnowledgeGraph, Node, Edge

        g = BoneKnowledgeGraph()
        g.add_node(Node("cortical_bone", "cortical bone", "structure", "Dense outer bone layer"))
        g.add_node(Node("fracture_toughness", "fracture toughness", "property", "Resistance to crack propagation"))
        g.add_edge(Edge("cortical_bone", "determines", "fracture_toughness"))

        path = g.shortest_path("cortical_bone", "fracture_toughness")
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        # Typed look-up dicts so callers get Node / Edge objects back,
        # not raw NetworkX attribute dicts.
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_node(self, node: Node) -> None:
        """
        Add a node to the graph, replacing any existing node with the same ID.

        Parameters
        ----------
        node : Node
            The concept to add.
        """
        self._nodes[node.node_id] = node
        self._graph.add_node(
            node.node_id,
            label=node.label,
            node_type=node.node_type,
            description=node.description,
            source=node.source,
        )

    def add_edge(self, edge: Edge) -> None:
        """
        Add a directed edge, raising KeyError if either endpoint is missing.

        Parameters
        ----------
        edge : Edge
            The relationship to add.

        Raises
        ------
        KeyError
            If source or target node has not been added to the graph yet.
        """
        if edge.source not in self._nodes:
            raise KeyError(
                f"Source node '{edge.source}' not in graph. Add it first."
            )
        if edge.target not in self._nodes:
            raise KeyError(
                f"Target node '{edge.target}' not in graph. Add it first."
            )
        self._edges[edge.edge_id] = edge
        self._graph.add_edge(
            edge.source,
            edge.target,
            relation=edge.relation,
            weight=edge.weight,
            evidence=edge.evidence,
            edge_source=edge.edge_source,
        )

    # ── Read — nodes ──────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Node | None:
        """Return the Node for node_id, or None if not found."""
        return self._nodes.get(node_id)

    def nodes_by_type(self, node_type: str) -> list[Node]:
        """Return all nodes of a given type."""
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def iter_nodes(self) -> Iterator[Node]:
        """Iterate over all Node objects."""
        return iter(self._nodes.values())

    # ── Read — edges ──────────────────────────────────────────────────────────

    def get_edge(self, source: str, relation: str, target: str) -> Edge | None:
        """Return the Edge for (source, relation, target), or None if not found."""
        return self._edges.get(f"{source}|{relation}|{target}")

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        """Return all edges leaving node_id."""
        return [
            e for e in self._edges.values() if e.source == node_id
        ]

    def incoming_edges(self, node_id: str) -> list[Edge]:
        """Return all edges arriving at node_id."""
        return [
            e for e in self._edges.values() if e.target == node_id
        ]

    def iter_edges(self) -> Iterator[Edge]:
        """Iterate over all Edge objects."""
        return iter(self._edges.values())

    # ── Graph algorithms ──────────────────────────────────────────────────────

    def shortest_path(self, source: str, target: str) -> list[str] | None:
        """
        Return the shortest directed path from source to target as a list of
        node_ids, or None if no path exists.

        Uses inverse edge weight (1 / weight) so high-confidence edges
        are preferred over low-confidence ones.

        Parameters
        ----------
        source : str
            Starting node_id.
        target : str
            Destination node_id.

        Returns
        -------
        list[str] or None
            Ordered list of node_ids from source to target (inclusive),
            or None if no directed path exists.
        """
        try:
            # Build a weight-inverted copy so Dijkstra prefers strong edges.
            weight_graph = nx.DiGraph()
            for u, v, data in self._graph.edges(data=True):
                w = data.get("weight", 1.0)
                weight_graph.add_edge(u, v, weight=1.0 / max(w, 1e-6))
            return nx.shortest_path(weight_graph, source, target, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def all_shortest_paths(
        self, source: str, target: str, max_paths: int = 5
    ) -> list[list[str]]:
        """
        Return up to max_paths shortest directed paths from source to target.

        Useful for the LRM when multiple reasoning routes exist between
        two concepts and the best path is ambiguous.

        Parameters
        ----------
        source : str
            Starting node_id.
        target : str
            Destination node_id.
        max_paths : int
            Maximum number of paths to return (default 5).

        Returns
        -------
        list[list[str]]
            Each inner list is an ordered sequence of node_ids.
        """
        try:
            paths = nx.all_shortest_paths(self._graph, source, target)
            return [p for _, p in zip(range(max_paths), paths)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def betweenness_centrality(self) -> dict[str, float]:
        """
        Compute normalised betweenness centrality for all nodes.

        Nodes with high betweenness sit on many shortest paths between
        other nodes — they are conceptual bridge points and are prime
        candidates for research gaps when their edge density is low.

        Returns
        -------
        dict[str, float]
            Mapping of node_id → betweenness centrality score (0–1).
        """
        return nx.betweenness_centrality(self._graph, normalized=True)

    def degree_centrality(self) -> dict[str, float]:
        """Return normalised degree centrality for all nodes."""
        return nx.degree_centrality(self._graph)

    def communities(self) -> list[set[str]]:
        """
        Detect communities using the greedy modularity algorithm on the
        undirected projection of the graph.

        Returns
        -------
        list[set[str]]
            Each set contains the node_ids belonging to one community.
        """
        undirected = self._graph.to_undirected()
        return list(nx.community.greedy_modularity_communities(undirected))

    def subgraph(self, node_ids: list[str]) -> BoneKnowledgeGraph:
        """
        Extract a sub-graph induced by node_ids (including all edges
        between them).

        Parameters
        ----------
        node_ids : list[str]
            Nodes to include.

        Returns
        -------
        BoneKnowledgeGraph
            A new graph containing only the specified nodes and their
            interconnecting edges.
        """
        sub = BoneKnowledgeGraph()
        for nid in node_ids:
            if nid in self._nodes:
                sub.add_node(self._nodes[nid])
        for edge in self._edges.values():
            if edge.source in sub._nodes and edge.target in sub._nodes:
                sub.add_edge(edge)
        return sub

    def neighbours(self, node_id: str, depth: int = 1) -> list[str]:
        """
        Return node_ids reachable from node_id within `depth` hops
        (following edges in either direction).

        Parameters
        ----------
        node_id : str
            Centre node.
        depth : int
            Number of hops (default 1 = direct neighbours only).

        Returns
        -------
        list[str]
            node_ids of neighbours, excluding node_id itself.
        """
        undirected = self._graph.to_undirected()
        try:
            ego = nx.ego_graph(undirected, node_id, radius=depth)
            return [n for n in ego.nodes() if n != node_id]
        except nx.NodeNotFound:
            return []

    # ── Statistics ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int | float]:
        """
        Return a summary of graph size and structure.

        Returns
        -------
        dict with keys:
            n_nodes         — total node count
            n_edges         — total edge count
            n_components    — number of weakly connected components
            giant_component — size of the largest component
            density         — graph density (edges / possible edges)
            type_counts     — dict mapping node_type → count
        """
        n = self._graph.number_of_nodes()
        e = self._graph.number_of_edges()
        components = list(
            nx.weakly_connected_components(self._graph)
        )
        type_counts: dict[str, int] = {}
        for node in self._nodes.values():
            type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1

        return {
            "n_nodes": n,
            "n_edges": e,
            "n_components": len(components),
            "giant_component": max((len(c) for c in components), default=0),
            "density": round(nx.density(self._graph), 6),
            "type_counts": type_counts,
        }

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"BoneKnowledgeGraph("
            f"{s['n_nodes']} nodes, "
            f"{s['n_edges']} edges, "
            f"{s['n_components']} components)"
        )
