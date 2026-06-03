# BoneMind Reasoning Benchmark

This document describes the benchmark design for evaluating BoneMind's
reasoning layer — the equation-graph reasoner backed by a
`RelationRegistry` of 12 Variables and 7 Relations (Currey, Paris–Vashishth,
beam bending, Frost mechanostat, etc.) with forward, abductive, and
counterfactual inference modes, plus the Proposer/Critic agent loop.

The original annotated benchmark dataset `eval/lrm_benchmark.json` (75 items)
was designed against the earlier LRM graph-walk reasoner. The dataset
survives and is reusable as seed material; the evaluation design below
targets the current equation-graph system.

See [`docs/reasoning/equation_graph.md`](reasoning/equation_graph.md) for the
full reasoner design and API reference.

## Overview

The benchmark evaluates three independent capabilities of the current
reasoning layer: inference correctness, agent hypothesis quality, and
novelty classification. The legacy annotated items in
`eval/lrm_benchmark.json` (75 items, originally targeting the graph-walk
LRM) are reusable as seed queries; they need new expected-output annotations
for the equation-graph components below.

```
eval/lrm_benchmark.json    — seed annotated dataset (75 items); reuse as query source
eval/run_eval.py           — retrieval benchmark runner (separate)
```

---

## Components

### Component 1 — Forward / Abductive / Counterfactual Inference

**Purpose:** Verify that the equation-graph reasoner produces numerically
and conceptually correct outputs for all three inference modes.

**Endpoint:** `POST /api/reason` with `mode` ∈ `{forward, abductive, counterfactual}`.

**Dataset:** 30 queries — 10 per mode — drawn from the preset library
(`/api/reason/presets`) plus 20 novel hand-crafted inputs covering all
7 registered Relations (Currey, Paris–Vashishth, beam bending, Frost
mechanostat, compositional rule, adaptation rule, clinical rule).

**Evaluation per mode:**

| Mode | Correct if… |
|---|---|
| `forward` | Computed output value lies within ±10 % of ground-truth calculation from the same inputs |
| `abductive` | Returned `given` variables are a valid minimal set that could produce the stated observation |
| `counterfactual` | Direction of predicted change (increase / decrease / unchanged) matches domain expectation |

**Metrics:**

| Metric | Formula | Target |
|---|---|---|
| Forward Accuracy (FA) | % of 10 forward queries within tolerance | ≥ 0.90 |
| Abductive Validity (AV) | % of 10 abductive queries with valid variable sets | ≥ 0.70 |
| Counterfactual Direction Accuracy (CDA) | % of 10 counterfactual queries with correct direction | ≥ 0.80 |

---

### Component 2 — Agent Hypothesis Quality

**Purpose:** Test whether the Proposer + Critic agent loop generates
hypotheses that are both physically grounded and surprising relative to
the direct equation output.

**Dataset:** 20 queries from `eval/lrm_benchmark.json` that include
a domain-expert reference hypothesis (the `expected_chain` field, repurposed).

**Evaluation:**
1. `POST /api/reason` with `agent=true` is called for each query.
2. **Hypothesis Rate (HR):** whether the agent returns a non-empty hypothesis.
3. **Novelty Pass Rate (NPR):** the novelty classifier labels the hypothesis
   SPECULATIVE or NOVEL (not GROUNDED — a GROUNDED result means the agent
   merely restated a known equation output).
4. **Critic Acceptance Rate (CAR):** whether the Critic agent accepted the
   Proposer's hypothesis without requesting revision (proxy for physical
   coherence).

**Metrics:**

| Metric | Formula | Target |
|---|---|---|
| Hypothesis Rate (HR) | % of 20 queries returning a hypothesis | ≥ 0.85 |
| Novelty Pass Rate (NPR) | % of returned hypotheses labelled SPECULATIVE or NOVEL | ≥ 0.60 |
| Critic Acceptance Rate (CAR) | % of hypotheses accepted by Critic on first pass | ≥ 0.50 |

---

### Component 3 — Novelty Calibration

**Purpose:** Verify that the novelty classifier assigns the expected label
for calibration cases spanning the three categories.

**Dataset:** 20 cases — 6 GROUNDED, 7 SPECULATIVE, 7 NOVEL — reused from
`eval/lrm_benchmark.json` with updated expected labels for the
SPECTER2-based classifier in `reasoning/novelty.py`.

| Label | Expected characteristics |
|---|---|
| GROUNDED | Hypothesis text closely paraphrases a corpus sentence (cosine ≥ 0.82) |
| SPECULATIVE | Hypothesis consistent with corpus but not verbatim (0.60–0.82) |
| NOVEL | Hypothesis not supported by corpus retrieval (< 0.60) |

**Evaluation:** `NoveltyClassifier.classify(hypothesis_text)` is called and
the returned label is compared against the expected annotation.

**Metrics:**

| Metric | Formula | Target |
|---|---|---|
| Novelty Agreement Rate (NAR) | % of 20 cases matching expected label | ≥ 0.60 |

---

## Readiness thresholds

All metrics must pass to consider the reasoning benchmark complete:

| Metric | Threshold | Meaning |
|---|---|---|
| FA ≥ 0.90 | Forward inference is numerically reliable | Core equation evaluation works |
| AV ≥ 0.70 | Abductive mode returns valid variable sets | Inverse reasoning is coherent |
| CDA ≥ 0.80 | Counterfactual directions are correct | Perturbation reasoning is sound |
| HR ≥ 0.85 | Agent produces hypotheses reliably | Proposer/Critic loop is stable |
| NPR ≥ 0.60 | Agents generate novel content | Loop adds value beyond direct inference |
| NAR ≥ 0.60 | Novelty classifier is calibrated | Hypothesis labels are informative |

---

## Output format

Results should be saved to `eval/reasoning_results.json`:

```json
{
  "benchmark_version": "2.0",
  "registry_stats": { "n_variables": 12, "n_relations": 7 },
  "aggregate": {
    "forward_accuracy": 0.0,
    "abductive_validity": 0.0,
    "counterfactual_direction_accuracy": 0.0,
    "hypothesis_rate": 0.0,
    "novelty_pass_rate": 0.0,
    "critic_acceptance_rate": 0.0,
    "novelty_agreement_rate": 0.0
  },
  "thresholds": {},
  "forward": [],
  "abductive": [],
  "counterfactual": [],
  "agent": [],
  "novelty": []
}
```

---

## Interpreting failures

**FA < 0.90** — Check whether the `RelationRegistry` relation fired for the
query's input variables. If no relation matched, the result falls back to the
LLM estimate. Inspect `reasoning/relation.py` to verify the relevant
`Relation.inputs` and `Relation.output` definitions.

**AV < 0.70** — The abductive mode may be returning too many or too few
variables. Review the `_abductive` path in `reasoning/explorer.py` to check
the variable-set construction logic.

**CDA < 0.80** — Counterfactual direction errors usually indicate that the
perturbation multiplier is being applied to the wrong variable. Check `given`
key filtering in `api/main.py` and the `_counterfactual` branch in
`reasoning/explorer.py`.

**HR < 0.85** — The Proposer agent is failing silently. Check
`reasoning/proposer_agent.py` retry logic and the LLM response parsing in
`_parse_json`.

**NPR < 0.60** — Agents are generating hypotheses that just restate the
equation output. Review the system prompt in `reasoning/proposer_agent.py`
to encourage extrapolation beyond the direct computation.

**NAR < 0.60** — Novelty thresholds may need recalibration against the
current corpus. Adjust `GROUNDED_THRESHOLD` / `SPECULATIVE_THRESHOLD`
in `reasoning/novelty.py`.
