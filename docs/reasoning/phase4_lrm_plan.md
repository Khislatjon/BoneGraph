# Phase 4 — Bone Knowledge Graph

**Status: ✅ Knowledge graph built and maintained · reasoning layer is the live Reasoning tab**

Phase 4 moves BoneGraph from retrieval to reasoning. It has two parts:

1. **The bone knowledge graph** (this document) — a structured ontology of bone
   concepts and causal edges, extracted from the corpus and cleaned into a
   high-signal `ontology.db`.
2. **The Reasoning tab** — the agent + critic + physical-grounding loop that
   answers questions and improves from feedback. It reads the knowledge graph
   for supporting facts. See [`reasoning_tab.md`](reasoning_tab.md) and
   [`reasoning/architecture.md`](architecture.md).

> **History.** An earlier reasoning engine — a graph-walk LRM, then an
> equation-graph reasoner with a Proposer/Critic hypothesis-generation agent
> loop over 12 variables and 7 relations — was built on top of this graph and
> **retired in May 2026** following the 21 May supervision direction. Its code
> and design docs were removed; the clean-slate Reasoning tab replaced it. The
> knowledge graph below is unaffected and is what the live tab draws on.

---

## Motivation

Phases 1–2 built a strong retrieval system: 248,629 sentence-aware chunks
embedded with SPECTER2, MRR 0.928, Recall@5 1.000. But retrieval can only
surface what has already been written — it cannot connect concepts across papers
that no single source covers end-to-end, or reason about cause and effect.

A structured knowledge graph adds that missing layer: concepts as typed nodes,
causal relations as weighted edges. The approach is inspired by Buehler (MIT,
arXiv:2403.11996), which turns biomaterials papers into an ontological graph and
reasons over it; BoneGraph's corpus is larger and bone-specific, and the graph is
consumed by a reasoning agent with a deterministic physical-grounding layer.

---

## Knowledge-graph build (steps 4.1–4.3)

| Step | What it produced | Code · docs |
|------|------------------|-------------|
| **4.1 — Seed ontology** | ~200 bone-science concepts + ~80 hand-curated causal edges bootstrapped into `ontology.db`. | `reasoning/seed.py` |
| **4.2 — Triple extraction** | `huatuogpt-bone` extracts `(node_1, relation, node_2)` triples from corpus chunks. Textbooks (1,983 chunks → 2,935 triples) + full paper corpus → **35,338 nodes · 34,265 edges** (raw). Resumable; progress tracked in `extraction_progress`. | `reasoning/extractor.py` |
| **4.3 — Graph cleanup + reclassification** | Six-stage cleanup (cross-domain filter, sentence-fragment filter, affix canonicalisation, reverse-pair resolution, low-weight edge drop, orphan removal) → **1,597 nodes · 1,699 edges**, then rule-based concept reclassification. | `scripts/clean_graph.py`, `scripts/reclassify_concepts.py` · [`reasoning/graph_cleanup.md`](graph_cleanup.md), [`reasoning/graph_concept_reclassification.md`](graph_concept_reclassification.md) |

The cleaned `ontology.db` is read by the Reasoning tab's critic for 1-hop facts
(`reasoning/kg_context.py`) and by `/api/stats` for graph counts
(`reasoning/graph_db.py::OntologyStore`).

### Rebuilding the graph

```bash
# Seed the base ontology
python -m reasoning.seed

# Extract triples (resumable; textbooks first, then full corpus)
python -m reasoning.extractor --source textbooks
python -m reasoning.extractor

# Clean + reclassify
python scripts/clean_graph.py
python scripts/reclassify_concepts.py

# Inspect
python -m reasoning.visualize_ontology
```

---

## Reasoning layer

The reasoning that runs on top of this graph is the live **Reasoning tab**:
an agent → physical-grounding → critic loop with bounded revision, a
feedback-driven user-rule registry, and a Quick/Deep mode toggle. It is
documented in full in [`reasoning_tab.md`](reasoning_tab.md) (build log + demo
flow) and [`reasoning/architecture.md`](architecture.md) (pipeline,
components, API, data model). Evidence-layer details (literature, conflict-aware
critic, KG shortcut) are in [`reasoning/evidence_layer.md`](evidence_layer.md).
