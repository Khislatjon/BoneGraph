"""
eval/evidence_audit.py
======================

Phase 0.1 — de-risk the evidence layer BEFORE wiring it into the critic.

Two questions this script answers, on a fixed set of fracture-domain probes:

  1. LITERATURE — does BoneGraphRetriever return passages that are actually
     relevant to a fracture-reasoning query? (low expected risk — it's the
     same retriever the Ask tab uses daily)

  2. KNOWLEDGE GRAPH — does ontology.db support meaningful "shortcut" paths
     between two named concepts (Gianluca's node-to-node jump), or are the
     paths noise? (higher expected risk — machine-extracted + pruned graph,
     unverified for the fracture domain)

Output is a human-readable report. The decision gate:
  - Literature relevant?      → proceed with A1 (literature → critic)
  - KG paths meaningful?      → proceed with A4 (KG shortcut → critic)
  - KG paths garbage?         → drop KG, lean on literature only

Run:
    python -m eval.evidence_audit
    python -m eval.evidence_audit --no-literature   # KG only (skip model load)
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import networkx as nx

ONTOLOGY_DB = Path(__file__).resolve().parent.parent / "data" / "db" / "ontology.db"


# ── Probe sets ────────────────────────────────────────────────────────────────

# Literature probes: realistic fracture-reasoning questions.
LITERATURE_PROBES = [
    "Why does trabecular bone fail before cortical bone in osteoporosis?",
    "How does cortical porosity affect whole-bone fracture strength?",
    "What is the effect of a lytic metastatic lesion on vertebral failure load?",
    "How does apparent density relate to trabecular bone strength?",
    "Why does mechanical loading increase bone strength (Wolff's law)?",
]

# KG shortcut probes: (source_node, target_node) pairs the critic might want
# to connect. Node IDs verified to exist in ontology.db.
KG_PAIRS = [
    ("cortical_bone", "trabecular_bone"),
    ("osteoporosis", "fragility_fracture"),
    ("cortical_bone", "bone_strength"),
    ("trabecular_bone", "osteoporosis"),
    ("porosity", "bone_strength"),
]


# ── KG loading ────────────────────────────────────────────────────────────────

def load_graph() -> tuple[nx.DiGraph, dict]:
    """Load ontology.db into a directed graph. Returns (graph, node_labels)."""
    con = sqlite3.connect(ONTOLOGY_DB)
    con.row_factory = sqlite3.Row
    G = nx.DiGraph()
    labels: dict[str, str] = {}
    for r in con.execute("SELECT node_id, label FROM nodes"):
        G.add_node(r["node_id"])
        labels[r["node_id"]] = r["label"]
    for r in con.execute("SELECT source_node, target_node, relation, weight FROM edges"):
        # Skip edges referencing pruned/absent nodes.
        if r["source_node"] in labels and r["target_node"] in labels:
            G.add_edge(r["source_node"], r["target_node"],
                       relation=r["relation"], weight=r["weight"] or 1.0)
    con.close()
    return G, labels


def describe_path(G: nx.DiGraph, labels: dict, path: list[str]) -> str:
    """Render a node path as 'A --[rel]--> B --[rel]--> C'."""
    if len(path) == 1:
        return labels.get(path[0], path[0])
    parts = [labels.get(path[0], path[0])]
    for a, b in zip(path[:-1], path[1:]):
        # Path may come from the undirected view; the stored edge could be b->a.
        if G.has_edge(a, b):
            rel = G.edges[a, b].get("relation", "?")
            arrow = f" --[{rel}]--> "
        elif G.has_edge(b, a):
            rel = G.edges[b, a].get("relation", "?")
            arrow = f" <--[{rel}]-- "
        else:
            arrow = " --- "
        parts.append(f"{arrow}{labels.get(b, b)}")
    return "".join(parts)


def shortcut(G: nx.DiGraph, src: str, dst: str) -> list[str] | None:
    """Shortest directed path src->dst; if none, try the undirected graph
    (a connection may exist but point the other way)."""
    if src not in G or dst not in G:
        return None
    try:
        return nx.shortest_path(G, src, dst)
    except nx.NetworkXNoPath:
        pass
    UG = G.to_undirected(as_view=True)
    try:
        return nx.shortest_path(UG, src, dst)
    except nx.NetworkXNoPath:
        return None


def one_hop(G: nx.DiGraph, labels: dict, node: str, k: int = 6) -> list[str]:
    """Top-k outgoing neighbours by edge weight, as readable bullets."""
    if node not in G:
        return []
    edges = sorted(G.out_edges(node, data=True),
                   key=lambda e: e[2].get("weight", 0), reverse=True)[:k]
    return [f"{labels.get(node,node)} --[{d['relation']}]--> {labels.get(t,t)} (w={d.get('weight',0):g})"
            for _, t, d in edges]


# ── Audit sections ──────────────────────────────────────────────────────────

def audit_kg() -> None:
    print("=" * 74)
    print("KNOWLEDGE-GRAPH AUDIT  (ontology.db)")
    print("=" * 74)
    G, labels = load_graph()
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"connected components (undirected): "
          f"{nx.number_connected_components(G.to_undirected(as_view=True))}\n")

    print("── Shortcut paths (Gianluca's node-to-node jump) ──\n")
    for src, dst in KG_PAIRS:
        path = shortcut(G, src, dst)
        if path is None:
            print(f"  ✗ {src}  →  {dst}   NO PATH")
        else:
            hops = len(path) - 1
            flag = "✓" if hops <= 3 else "⚠"  # >3 hops = likely meandering
            print(f"  {flag} {src} → {dst}  ({hops} hop{'s' if hops!=1 else ''})")
            print(f"      {describe_path(G, labels, path)}")
        print()

    print("── 1-hop neighbourhoods (entity-anchored context) ──\n")
    for node in ("cortical_bone", "osteoporosis", "fragility_fracture"):
        print(f"  [{node}]")
        hops = one_hop(G, labels, node)
        if not hops:
            print("      (no outgoing edges)")
        for h in hops:
            print(f"      • {h}")
        print()

    print("VERDICT GUIDANCE:")
    print("  · Paths ≤3 hops with sensible relations → KG shortcut is usable (A4).")
    print("  · NO PATH / >3 meandering hops / nonsense relations → drop KG, lean on literature.\n")


def audit_literature() -> None:
    print("=" * 74)
    print("LITERATURE AUDIT  (BoneGraphRetriever)")
    print("=" * 74)
    try:
        from retrieval.retriever import BoneGraphRetriever
    except Exception as e:
        print(f"  Could not import retriever: {e}\n  (run with --no-literature to skip)\n")
        return
    print("  Loading retriever (this loads SPECTER2 — may take a moment)…\n")
    retriever = BoneGraphRetriever()
    retriever.load()

    for q in LITERATURE_PROBES:
        print(f"  Q: {q}")
        try:
            results = retriever.query(q, top_k=3)
        except Exception as e:
            print(f"      ✗ query failed: {e}\n")
            continue
        for r in results:
            title = (r.get("title") or "(untitled)").strip()
            score = r.get("score")
            score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "?"
            snippet = (r.get("text") or "").strip().replace("\n", " ")
            print(f"      [{score_s}] {title[:80]}")
            print(f"             {snippet[:140]}…")
        print()

    print("VERDICT GUIDANCE:")
    print("  · Passages on-topic for each query → proceed with A1 (literature → critic).")
    print("  · Off-topic / low scores → tighten retrieval before wiring into the critic.\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-literature", action="store_true",
                    help="Skip the literature audit (avoids loading SPECTER2).")
    ap.add_argument("--no-kg", action="store_true", help="Skip the KG audit.")
    args = ap.parse_args()

    if not ONTOLOGY_DB.exists():
        print(f"ERROR: {ONTOLOGY_DB} not found.")
        return 1

    if not args.no_kg:
        audit_kg()
    if not args.no_literature:
        audit_literature()
    return 0


if __name__ == "__main__":
    sys.exit(main())
