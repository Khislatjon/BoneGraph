# BoneLogic LRM Benchmark

## Overview

The LRM benchmark evaluates the reasoning layer of BoneLogic across three components. It is analogous to the retrieval benchmark (`eval/benchmark.json` / `eval/run_eval.py`) but targets the knowledge graph reasoning engine rather than the embedding retriever.

```
eval/lrm_benchmark.json    — annotated benchmark dataset (75 items)
eval/run_lrm_eval.py       — evaluation runner
eval/lrm_results.json      — latest results (written by runner)
```

---

## Run

```bash
# Default: max_results=10, reads eval/lrm_benchmark.json
python eval/run_lrm_eval.py

# Custom settings
python eval/run_lrm_eval.py --max-results 5
python eval/run_lrm_eval.py --out eval/lrm_results.json
```

The physics component runs instantly (pure rule lookup). The chain coverage and novelty components require loading the full graph (~35K nodes) — allow ~30–60 seconds for startup.

---

## Components

### Component 1 — Physics Engine Accuracy

**Purpose:** Verify that the physics engine correctly classifies known-plausible and known-implausible edge directions.

**Dataset:** 30 edge pairs — 15 IMPLAUSIBLE + 15 PLAUSIBLE — derived directly from the 71 directional rules in `reasoning/physics.py`. Each pair has a single correct expected status grounded in established bone mechanics.

**Evaluation:** For each pair, `PhysicsEngine.validate_edge(source, relation, target)` is called and the returned status is compared against the expected label.

**Metrics:**

| Metric | Formula | Target |
|---|---|---|
| Physics Accuracy (PA) | % of 30 pairs correctly classified | ≥ 0.90 |
| Sensitivity | accuracy on IMPLAUSIBLE pairs only | — |
| Specificity | accuracy on PLAUSIBLE pairs only | — |

Physics accuracy is deterministic (rule-based), so PA = 1.00 is expected unless node fragments in the benchmark don't match the rule keys. Any failure here indicates a mismatch between benchmark node naming and the physics rule fragment table.

---

### Component 2 — Chain Coverage

**Purpose:** Test whether the LRM finds causal chains that pass through the expected bone science concepts for 25 domain queries spanning 7 sub-domains.

**Dataset:** 25 queries covering mechanics (8), pathology (4), cell biology (4), mechanobiology (4), biomaterials (2), morphology (2), simulation (1). Each query has 2–3 `expected_nodes` — lowercase fragments that should appear somewhere in the node_ids of the top-3 returned chains.

**Evaluation:**
1. `lrm.query(query, max_results=10)` is called.
2. **Anchor Success Rate (ASR):** whether ≥1 chain was returned (anchoring worked).
3. **Chain Coverage Rate (CCR@3):** whether EACH expected node fragment appears in the node_ids of at least one chain across the top-3 results collectively.

Each fragment is checked independently — it does not need to appear in the same chain as the others.  The LRM finds shortest paths between anchor pairs, so a query with three expected concepts typically returns separate chains per pair rather than one chain spanning all three.  Fragment matching is substring-based (e.g., `"porosity"` matches `"cortical_porosity"`).

**Metrics:**

| Metric | Formula | Target |
|---|---|---|
| Anchor Success Rate (ASR) | % of 25 queries where ≥1 chain returned | ≥ 0.80 |
| Chain Coverage Rate CCR@3 | % of 25 queries where expected nodes in top-3 | ≥ 0.60 |

CCR@3 target of 0.60 reflects that the graph was built by automated extraction and some expected chains may be missing or fragmented. A score of 1.00 would require every expected concept to be connected in the extracted graph.

---

### Component 3 — Novelty Calibration

**Purpose:** Verify that the novelty classifier assigns the expected label for 20 calibration cases spanning the three categories.

**Dataset:** 20 cases — 6 GROUNDED, 7 SPECULATIVE, 7 NOVEL.

| Label | Expected characteristics |
|---|---|
| GROUNDED | Direct 1-hop seed edge, no extracted edges |
| SPECULATIVE | 3+ hop chain or any extracted edges |
| NOVEL | Extracted edges crossing ≥2 concept-type boundaries |

**Evaluation:** `lrm.query(query, max_results=5)` is called and a match is recorded if the expected label appears in any of the top-3 chains.  Top-1 alone is insufficient because GROUNDED chains (1–2 hop seed edges) score lower than longer extracted chains by design (length weight = 0.5 in the scorer), so they rarely reach position 1.

**Metrics:**

| Metric | Formula | Target |
|---|---|---|
| Novelty Agreement Rate (NAR) | % of 20 cases matching expected label | ≥ 0.50 |

NAR target of 0.50 is intentionally moderate — novelty classification is a heuristic (see `reasoning/lrm.py: _classify_novelty`). GROUNDED cases are the hardest to achieve since they require the LRM to find and return a seed edge as the top result rather than a longer extracted chain.

---

## Readiness thresholds

All four must pass to consider the LRM layer benchmark-complete:

| Metric | Threshold | Meaning |
|---|---|---|
| PA ≥ 0.90 | Physics engine is reliable | Core guard-rail function works |
| ASR ≥ 0.80 | Graph covers bone science concepts | Anchoring is robust post-extraction |
| CCR@3 ≥ 0.60 | Reasoning finds expected chains | Graph connectivity is meaningful |
| NAR ≥ 0.50 | Novelty classifier is calibrated | Hypothesis labels are informative |

---

## Output format

Results are saved to `eval/lrm_results.json`:

```json
{
  "benchmark_version": "1.0",
  "graph_stats": { "n_nodes": 35338, "n_edges": 34265 },
  "aggregate": {
    "physics_accuracy": 1.000,
    "anchor_success_rate": 0.880,
    "chain_coverage_rate_3": 0.680,
    "novelty_agreement_rate": 0.550
  },
  "thresholds": { "pa_pass": true, "asr_pass": true, "ccr3_pass": true, "nar_pass": true },
  "physics": [...],
  "chain_coverage": [...],
  "novelty": [...]
}
```

---

## Interpreting failures

**PA < 1.00** — A benchmark node fragment doesn't match any physics rule key. Check the `law` field in the result; if it's empty the rule didn't fire. Either the node naming differs from the fragment in `_DIRECTIONAL_RULES`, or the benchmark pair is genuinely outside the rule table.

**ASR < 0.80** — The graph doesn't contain nodes for the query concepts. Check whether the concept appears in `ontology.db` using `lrm.graph_stats()` or a direct SQLite query. The 35K-node graph from full extraction should cover most bone science terms.

**CCR@3 < 0.60** — The expected concepts exist as nodes but aren't connected by edges. The relevant triples weren't extracted from the corpus, or the path is too long and exceeds `max_hops=5`. Inspect `top_chain` in the per-query results to understand what the LRM found instead.

**NAR < 0.50** — The novelty classifier isn't calibrated. After full paper extraction the graph has ~35K nodes, most typed as `"concept"` (LLM default), while seed nodes have specific types (`structure`, `cell`, `property`, etc.). The old classifier used `type_changes ≥ 2` as the NOVEL signal; any mixed seed+extracted path crosses ≥2 boundaries trivially, causing everything to return NOVEL. The revised classifier uses mean edge weight instead: GROUNDED requires mean weight ≥ 5.0 (confirmed by 5+ corpus passages) or all-seed ≤2-hop chains; NOVEL requires long chains (≥3 hops) with low-evidence edges (mean weight < 2.0) crossing ≥3 type boundaries.
