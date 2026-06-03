"""
reasoning/kg_context.py
=======================

A4 — knowledge-graph "shortcut" context for the critic.

Gianluca's 28 May ask: bring the bone knowledge graph (ontology.db) back into
the Reasoning tab, but use *shortcuts* — anchor on the concepts a question
mentions and pull their immediate neighbourhood, rather than walking the whole
graph.

CRITICAL DESIGN RULE (see docs/reasoning/evidence_layer.md, "Composed paths"):
we hand the critic **raw, individually-labelled edges**, never a pre-composed
multi-hop "therefore" chain. Each edge is one fact the graph is confident about;
stitching several into a causal sentence can imply something the graph never
claimed (sign/direction does not compose). The LLM does any composing.

So this module returns 1-hop neighbourhoods of the anchored concepts as a flat
list of discrete edges. No path narratives.

Public API:
    anchor_nodes(question)              -> [node_id, ...]
    kg_facts(question, ...)             -> {"facts": [{subject, relation, object, weight}],
                                            "anchors": [labels]}
    format_facts(kg)                    -> str  (for the critic prompt)
"""

from __future__ import annotations

import re
import sqlite3
from functools import lru_cache
from pathlib import Path

ONTOLOGY_DB = Path(__file__).resolve().parent.parent / "data" / "db" / "ontology.db"

# Labels too generic to anchor on — they'd match almost any bone question and
# pull noisy neighbourhoods.
_ANCHOR_STOPLIST = {"bone", "bones", "tissue", "bone tissue", "structure",
                    "bone structure", "cell", "cells", "system"}
_MIN_ANCHOR_LEN = 5     # skip very short labels
_MAX_ANCHORS = 4
_MAX_FACTS = 12
_FACT_CHAR_BUDGET = 1200   # ~300 tokens


@lru_cache(maxsize=1)
def _load() -> tuple[dict, dict, dict]:
    """Load ontology.db once. Returns:
       labels:   node_id -> label
       out_edges: node_id -> [(relation, target_id, weight)]
       in_edges:  node_id -> [(source_id, relation, weight)]
    """
    labels: dict[str, str] = {}
    out_edges: dict[str, list] = {}
    in_edges: dict[str, list] = {}
    if not ONTOLOGY_DB.exists():
        return labels, out_edges, in_edges
    con = sqlite3.connect(ONTOLOGY_DB)
    con.row_factory = sqlite3.Row
    for r in con.execute("SELECT node_id, label FROM nodes"):
        labels[r["node_id"]] = r["label"]
    for r in con.execute("SELECT source_node, target_node, relation, weight FROM edges"):
        s, t = r["source_node"], r["target_node"]
        if s not in labels or t not in labels:
            continue
        w = r["weight"] or 1.0
        out_edges.setdefault(s, []).append((r["relation"], t, w))
        in_edges.setdefault(t, []).append((s, r["relation"], w))
    con.close()
    return labels, out_edges, in_edges


def anchor_nodes(question: str) -> list[str]:
    """Find KG nodes whose label appears in the question. Prefers longer labels,
    skips generic stoplist labels, caps at _MAX_ANCHORS."""
    labels, _, _ = _load()
    q = question.lower()
    hits: list[tuple[int, str]] = []
    for node_id, label in labels.items():
        lab = label.lower().strip()
        if len(lab) < _MIN_ANCHOR_LEN or lab in _ANCHOR_STOPLIST:
            continue
        # word-boundary-ish containment so "bone" inside "bone strength" is fine
        # but we don't match across unrelated substrings
        if re.search(r"(?<![a-z])" + re.escape(lab) + r"(?![a-z])", q):
            hits.append((len(lab), node_id))
    # longest labels first (most specific), dedupe, cap
    hits.sort(reverse=True)
    chosen: list[str] = []
    for _, node_id in hits:
        if node_id not in chosen:
            chosen.append(node_id)
        if len(chosen) >= _MAX_ANCHORS:
            break
    return chosen


def kg_facts(question: str, max_facts: int = _MAX_FACTS,
             char_budget: int = _FACT_CHAR_BUDGET) -> dict:
    """Raw 1-hop edges around the anchored concepts, highest-weight first.
    No composed paths — each fact is one independent edge."""
    labels, out_edges, in_edges = _load()
    anchors = anchor_nodes(question)
    if not anchors:
        return {"facts": [], "anchors": []}

    # Gather candidate edges (both directions), as (weight, subject, rel, object).
    cand: list[tuple[float, str, str, str]] = []
    for a in anchors:
        for rel, tgt, w in out_edges.get(a, []):
            cand.append((w, labels[a], rel, labels[tgt]))
        for src, rel, w in in_edges.get(a, []):
            cand.append((w, labels[src], rel, labels[a]))

    # Highest weight first; dedupe identical triples.
    cand.sort(key=lambda e: e[0], reverse=True)
    seen: set[tuple[str, str, str]] = set()
    facts: list[dict] = []
    budget = char_budget
    for w, s, rel, o in cand:
        key = (s, rel, o)
        if key in seen:
            continue
        seen.add(key)
        line_len = len(s) + len(rel) + len(o) + 12
        if budget - line_len < 0:
            break
        budget -= line_len
        facts.append({"subject": s, "relation": rel, "object": o, "weight": w})
        if len(facts) >= max_facts:
            break

    return {"facts": facts, "anchors": [labels[a] for a in anchors]}


def format_facts(kg: dict) -> str:
    """Render facts as a flat list of independent edges for the critic prompt.
    Deliberately NOT a narrative chain."""
    facts = kg.get("facts") or []
    if not facts:
        return "(no knowledge-graph facts for this question)"
    lines = [f"- {f['subject']} [{f['relation']}] {f['object']}" for f in facts]
    return "\n".join(lines)
