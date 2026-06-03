# Equation-Graph Reasoner

> **⚠️ Superseded — archived May 2026.**
> The Reasoning tab has been rebuilt from clean slate. The current design
> is documented in [`../reasoning_tab.md`](../reasoning_tab.md): a reasoning
> agent + critic loop, an 8-rule physical-grounding filter, and a user-feedback
> rule registry. The equation-graph reasoner described below is retained for
> historical reference only; none of its code is in the live request path.

**Status: 🗄 Archived — historical design.**

This document describes the Reasoning tab's only reasoner: a typed
equation graph supporting forward, abductive, and counterfactual
inference, plus an active-exploration ("Surprises") panel and a
Proposer / Critic agent loop that walks the graph for novel
hypotheses without the user typing a query.

It supersedes an earlier physics-grid pipeline whose code has been
removed; that design — and the four structural limits that motivated
the redesign — is preserved as a retrospective in
[`physics_grid.md`](physics_grid.md) for future paper writing.

---

## The equation graph

### What lives at each node

Each node is a **typed Variable** — a physical quantity with a unit,
a plausible range, and a description. There are 12 variables in the
registry:

| Display     | Internal symbol | Name                   | Unit       | Range            |
|-------------|-----------------|------------------------|------------|------------------|
| `φ`         | `phi`           | porosity               | —          | 0 – 0.95         |
| `ρ`         | `rho`           | apparent density       | g/cm³      | 0.05 – 2.10      |
| `E`         | `E`             | elastic modulus        | GPa        | 0.001 – 30       |
| `ΔK`        | `dK`            | stress-intensity range | MPa·√m     | 0 – 6            |
| `da/dN`     | `da_dN`         | crack growth rate      | m/cycle    | 1e-14 – 1e-3     |
| `R`         | `R`             | outer cortical radius  | mm         | 5 – 25           |
| `t`         | `t`             | cortical thickness     | mm         | 0.5 – 8          |
| `I`         | `I_section`     | second moment of area  | mm⁴        | 1 – 1e5          |
| `M`         | `M`             | applied bending moment | N·mm       | 0 – 1e6          |
| `σ`         | `sigma`         | bending stress         | MPa        | 0 – 300          |
| `ε`         | `eps`           | peak strain            | µε         | 0 – 10 000       |
| `ΔBMD/Δt`   | `dBMD_dt`       | bone adaptation rate   | %/yr       | -5 – 5           |

**Per-bone-type sweep ranges.** Three Variables additionally carry a
`typical_ranges` dict that narrows the sweep window when a tissue regime
is active (set via the Agents panel's Tissue dropdown). The table above
remains the full validity envelope used for clamping and Monte-Carlo
sampling; the table below is used only by the Proposer agent, by
`complete_given` midpoint fills, and for sweep-value clipping in the
agent path. Variables without a `typical_ranges` entry fall back to
`[lo, hi]` in every regime.

| Symbol | Cortical    | Transitional | Trabecular |
|--------|-------------|--------------|------------|
| `φ`    | 0.02 – 0.15 | 0.15 – 0.50  | 0.50 – 0.95 |
| `ρ`    | 1.70 – 2.00 | 1.20 – 1.70  | 0.10 – 0.60 |
| `E`    | 15 – 25     | 5 – 15       | 0.05 – 2    |

This prevents the engine from sweeping `φ` from 0.05 (healthy cortical)
to 0.95 (trabecular foam) in a single hypothesis — a single sweep can
otherwise walk across two distinct tissue regimes and produce
extrapolation artefacts (e.g. +70,000 % changes in `da_dN`) that are
mathematically correct but physically meaningless.

Each Variable carries both an ASCII `symbol` (used in SymPy equations,
JSON payloads, and code paths — easy to type, easy to grep) and a
Unicode `display_symbol` (rendered in the UI, diagrams, and this
document). The two are decoupled on purpose: the API contract stays
`given={"phi": 0.10}`, while the chain pill on screen reads `φ`.
Definitions live in [`reasoning/bone_relations.py`](../../reasoning/bone_relations.py).

### What lives on each edge

Each edge is a **Relation** — a SymPy equation with explicit input
symbols, output symbol, parameters, and a citation. There are 7
relations:

| Name                     | Equation                                         | Citation                              |
|--------------------------|--------------------------------------------------|---------------------------------------|
| `density_from_porosity`  | `ρ = ρ_full · (1 − φ)`                           | Geometric definition                  |
| `currey_modulus`         | `E = a · ρⁿ`                                     | Currey 1988                           |
| `vashishth_paris`        | `da/dN = C₀ · (ρ_ref / ρ)^k_ρ · ΔKᵐ`             | Vashishth 2003 / Paris 1963           |
| `cortical_inertia`       | `I = π/4 · (R⁴ − (R − t)⁴)`                      | Solid mechanics                       |
| `beam_bending`           | `σ = M · R / I`                                  | Euler–Bernoulli                       |
| `hookes_law`             | `ε = 1000 · σ / E`                               | Hooke's law (units bridged)           |
| `frost_mechanostat`      | `dBMD/dt = k · tanh((ε − ε_set)/ε_w)`            | Frost 2003 / Robling 2009             |

### How the graph emerges

There is no "graph build" step. When two Relations mention the same
symbol — e.g. `currey_modulus` uses `ρ` as input, and
`density_from_porosity` produces `ρ` as output — they share that node.
The `RelationRegistry` in [`relation.py`](../../reasoning/relation.py)
just stores Relations in a dict keyed by output symbol; the
topology is implicit.

This is the key architectural difference from the LLM-extracted bone
ontology (`data/db/ontology.db`), which still exists as a separate
artifact for future graph-driven reasoning modes:

| LLM-extracted ontology        | Equation graph                 |
|-------------------------------|--------------------------------|
| Nodes are *concept strings*   | Nodes are *typed variables*    |
| Edges are *qualitative claims* (`increases`, `decreases`) | Edges are *equations* |
| Edges have weights            | Edges have parameters + priors |
| Built by an LLM reading papers | Hand-authored, 7 relations    |
| You can *walk* it             | You can *compute* on it        |

### The full graph

```
       ┌────────────[ density_from_porosity ]──────────┐
       │                                                ▼
      (φ)                                              (ρ) ──┐
                                                              │
                                                              ├──[ currey_modulus ]──▶ (E) ──┐
                                                              │                                │
       (ΔK) ────────────────[ vashishth_paris ]◀──────────────┘                                │
                                    │                                                          │
                                    ▼                                                          │
                                (da/dN)                                                        │
                                                                                               │
                                                                                               ▼
       (R) ─┐                                                                          [ hookes_law ]
            ├──[ cortical_inertia ]──▶ (I) ──┐                                                 │
       (t) ─┘                                 ├──[ beam_bending ]──▶ (σ) ──────────────────────┤
                                              │                                                │
       (M) ──────────────────────────────────┘                                                 ▼
                                                                                              (ε)
                                                                                               │
                                                                                               ▼
                                                                                  [ frost_mechanostat ]
                                                                                               │
                                                                                               ▼
                                                                                          (dBMD/dt)
```

Roots (no producer): `φ`, `ΔK`, `R`, `t`, `M` — these must be supplied
or filled from literature defaults.

Sinks (no consumer): `da/dN`, `dBMD/dt` — these are pure outputs.

Internal nodes: `ρ`, `E`, `I`, `σ`, `ε` — produced by one relation,
consumed by another.

---

## How a query gets answered

### Chain discovery — BFS at query time

Given `forward('da_dN', given={'phi': 0.10, 'dK': 1.0})`:

1. Start at the target, `da_dN`.
2. Find every Relation that produces `da_dN`. Only `vashishth_paris`
   does. Its inputs are `ρ` and `dK`.
3. `dK` is in `given` — leaf.
4. `ρ` is not. Recurse: which Relation produces `ρ`?
   `density_from_porosity`. Its only input is `φ`.
5. `φ` is in `given` — leaf.
6. Topological-sort the visited Relations.
   Forward order: `density_from_porosity` → `vashishth_paris`.

The chain `φ → ρ → da/dN` is *discovered*, not stored. Nothing in the
codebase declares this chain — the BFS finds it from the shared
symbol `ρ`.

### Monte Carlo propagation

Each Relation parameter has a prior. To answer the query, the
reasoner samples 2000 (or however many) values from each prior and
evaluates the chain on the whole array at once (vectorised through
SymPy's `lambdify` to NumPy):

```
       phi = 0.10                            (fixed)
                  │
                  ▼
   ┌──────────────────────────┐
   │  density_from_porosity   │   ρ_full sampled 2000× (here it's fixed at 1.90)
   │  ρ = ρ_full · (1 − φ)    │  ────────▶  ρ: 2000 numbers
   └──────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────┐
   │  vashishth_paris         │   C₀, m, k_ρ, ρ_ref each sampled 2000×
   │  da/dN = C₀·(ρ_ref/ρ)^k_ρ·ΔKᵐ │  ────────▶  da/dN: 2000 numbers
   └──────────────────────────┘
                  │
                  ▼
       mean = 2.24 × 10⁻⁹ m/cycle
       P5   = 1.06 × 10⁻⁹
       P95  = 4.04 × 10⁻⁹
```

The output is a *distribution*, not a single number. The reasoner
returns the mean, 5th and 95th percentiles, and the relative
uncertainty `(P95 − P5) / |mean|`. The Phase 5 explorer uses
`1 / (1 + rel_uncertainty)` as its `physics_confidence` score.

---

## The four inference modes

All four modes run against the same RelationRegistry. They share
chain discovery, state initialisation, parameter sampling, and the
covariate machinery — only the inference loop differs.

### 1. Forward — "predict the answer"

Given some inputs, propagate forward to a target.

```python
registry.forward(
    target='da_dN',
    given={'phi': 0.10, 'dK': 1.0},
    covariates={'age': 75, 'sex': 'F', 'site': 'femur_cortical'},
)
```

Returns: chain trace + per-step intermediates + final distribution +
citations.

### 2. Abductive — "explain a measurement"

Given an *observation* on a downstream variable, infer the posterior
over upstream causes.

```python
registry.abductive(
    target='E',                        # what was measured
    observed=12.0,                     # the value
    observed_std=0.6,                  # measurement noise (optional)
    infer=['phi'],                     # what to infer
    given={},                          # any other roots, if known
)
```

Implementation: importance sampling. Draws uniform samples over
`phi` in its physical range, runs forward, computes a Gaussian
likelihood weight against the observation, and reports the weighted
posterior. The result includes an *effective sample size* so you
know whether the posterior is well-resolved.

This is real Bayesian inversion, not algebraic inversion of the
equation. It handles non-monotonic chains and multi-variable
inference correctly.

### 3. Counterfactual — "what if we intervened?"

Given a baseline state and an intervention (`do(X = x)`), compute
both predictions with the *same* random samples so the delta isolates
the intervention.

```python
registry.counterfactual(
    target='da_dN',
    given={'phi': 0.05, 'dK': 1.0},
    intervention={'phi': 0.30},        # raise porosity to 30%
)
```

Returns: baseline mean + intervened mean + delta mean (paired
samples). Identical RNG seed across both runs guarantees that
parameter draws (Currey's `a`, Paris' `C₀`, etc.) cancel from the
delta. This is Pearl-style do-calculus, not just two separate runs.

### 4. Explore — "find me a surprise" (currently hidden)

Active exploration. No user query needed. Walks a hand-picked but
principled set of sweeps over the variable graph, evaluates each at
3–5 points, asks the corpus how attested the rendered claim is, and
ranks candidates by

```
   surprise = magnitude × physics_confidence × corpus_attestation
```

The UI panel is hidden behind a feature flag
(`V2_SURPRISES_ENABLED = false` in
[`frontend/index.html`](../../frontend/index.html)). The backend
endpoint `GET /api/reason/explore` still works if you `curl` it.
Implementation lives in [`reasoning/explorer.py`](../../reasoning/explorer.py).

---

## Covariate conditioning

The same physical query should yield a different prediction for a
30-year-old femur and a 75-year-old osteoporotic vertebra. The
reasoner achieves this by letting **patient covariates reshape the
parameter priors** before sampling.

### How a shift works

Each Relation's parameters declare a `Prior` with optional `shifts`:

```python
"a": Prior(
    mean=7.0, std=0.8,
    shifts=(
        _shift_currey_a_site(),    # site-dependent prefactor
        _shift_currey_a_age(),     # age-dependent decline
        _shift_currey_a_sex(),     # M/F offset
        _shift_currey_a_oi(),      # osteogenesis imperfecta
    ),
)
```

When the user calls `forward(..., covariates={'age': 80, 'sex': 'F', 'site': 'vertebra'})`:

1. The base prior is `a ~ Normal(7.0, 0.8)`.
2. `currey_a_site` sees `site='vertebra'` → multiplies mean and std
   by 0.55 → `a ~ Normal(3.85, 0.44)`.
3. `currey_a_age` sees `age=80` (50 years past 30) → multiplies by
   `(1 - 0.10 × 5)` = 0.50 → `a ~ Normal(1.93, 0.22)`.
4. `currey_a_sex` sees `sex='F'` → multiplies by 0.95 → `a ~ Normal(1.83, 0.21)`.
5. Sampling happens against the *final* prior.

Shifts compose left-to-right per declaration order; each shift carries
its own citation and is logged in the result's trace block so the UI
can show *why* the prior moved.

### The shift library

About a dozen literature-anchored shifts live in
[`bone_relations.py`](../../reasoning/bone_relations.py):

| Shift                          | Cites                              | Effect                                              |
|--------------------------------|------------------------------------|-----------------------------------------------------|
| `currey_a_age`                 | Burstein 1976, McCalden 1993       | E pre-factor ~10%/decade past age 30                |
| `currey_a_sex`                 | Smith 1976, Riggs 1981             | F ≈ 0.95× M                                         |
| `currey_a_site`                | Carter–Hayes 1977, Keller 1994     | vertebra ≈ 0.55× cortical                           |
| `currey_n_site`                | Keller 1994, Rho 1995              | trabecular exponent ≈ 3.0 vs cortical 2.5           |
| `currey_a_oi`                  | Imbert 2014                        | OI lowers E pre-factor by ≈40%                      |
| `paris_c0_age`                 | Diab & Vashishth 2005              | Crack growth ~2×/decade past 50                     |
| `paris_c0_osteoporosis`        | Vashishth 2003                     | OP bone ~1.5× faster crack growth                   |
| `frost_eps_set_age`            | Frost 2003                         | Mechanostat setpoint rises with age                 |

### Verified example

```
forward('E', given={'phi': 0.10}, covariates={'age': 30}) → 26.8 GPa
forward('E', given={'phi': 0.10}, covariates={'age': 80}) → 13.4 GPa
ratio: 0.50  (matches Burstein 1976 / McCalden 1993 expectation)
```

---

## Natural-language ask

Most users do not want to type `forward('da_dN', given={'phi':0.10, 'dK':1.0})`.
The Reasoning tab exposes a free-text input that handles the translation.

### The pipeline

```
   "bone stiffness under cyclic load"
              │
              ▼
   ┌─────────────────────────────────────┐
   │  QueryRouter (Phase 4)              │   ONE Ollama call, temperature 0,
   │  reasoning/query_router.py          │   JSON-mode output
   │                                     │   ───▶  mode = "forward"
   │                                     │         target_hint = "stiffness"
   │                                     │         confidence = 0.85
   └─────────────────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────────┐
   │  SemanticVariableAnchor (Phase 4)   │   SPECTER2 cosine over Variable
   │  reasoning/semantic_anchor.py       │   embeddings
   │                                     │   ───▶  "stiffness" → E (0.83)
   └─────────────────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────────┐
   │  _v2_build_ask_payload              │   Fill chain roots from literature
   │  api/main.py                        │   defaults so the deterministic
   │                                     │   reasoner has a complete payload
   └─────────────────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────────┐
   │  registry.forward(...)              │   Deterministic.  Same as if the
   │                                     │   user had typed the payload by hand.
   └─────────────────────────────────────┘
              │
              ▼
        ForwardResult + routing block
```

The LLM is **only** responsible for routing — it never computes a
prediction. If Ollama is unreachable, the router falls back to
`mode="forward" + top semantic match`, so the system still works,
just with a lower-confidence routing block.

### Anchor quality (verified)

Six paraphrased queries, all picked the correct variable as top-1:

| Query                                  | Top-1                  | Score |
|----------------------------------------|------------------------|-------|
| "bone stiffness"                       | `E` (elastic modulus)  | 0.83  |
| "cyclic load fatigue"                  | `da_dN` (crack growth) | 0.77  |
| "cortical thinning"                    | `t` (cortical thickness)| 0.81  |
| "mineralisation density"               | `rho` (density)        | 0.84  |
| "how strong is the femur"              | `E`                    | 0.76  |
| "crack growth in osteoporotic bone"    | `da_dN`                | 0.79  |

---

## UI walkthrough

From top to bottom of the Reasoning tab:

### 1. Header — free-text "Ask" bar

```
   ┌─ Hypothesis Generator ──────────────────────────────────────┐
   │  Forward, abductive and counterfactual inference over a     │
   │  typed equation graph. Patient covariates reshape each      │
   │  law's priors before propagation.                           │
   │  ┌──────────────────────────────────────────────────────┐   │
   │  │ 🔍 Ask in plain English…                  [ Reason ] │   │
   │  └──────────────────────────────────────────────────────┘   │
   └─────────────────────────────────────────────────────────────┘
```

The bar uses `.reason-bar` / `.reason-input` / `.reason-btn` classes
and hits `/api/reason/ask`. The routing block is always rendered
above the result card so you can see which mode and target the LLM
picked.

### 2. Patient covariates panel

```
   ┌─ Patient covariates ──────────────────  age=30 · M · femur cortical
   │   Age     [████████│············]  30 yr
   │   Sex     [ Male │ Female ]
   │   Site    femur cortical ▼
   │   Disease  ○ osteoporosis  ○ glucocorticoid  ○ osteogenesis imperfecta
   └────────────────────────────────────────────────────────────
```

Every inference uses these values. The slider's filled track and the
selected Sex / Site / Disease chips all share a solid amber-with-white
"selected" treatment so it's obvious at a glance what the active
profile is. Change the age slider and the same query yields a
different number.

### 3. Try chips *(only shown before a result lands)*

```
   Try:  [ bone stiffness under cyclic load ]
         [ how does porosity affect modulus in a 70-year-old female? ]
         [ why is this patient's E only 12 GPa? ]
         [ what if porosity dropped to 5%? ]
```

Click a chip → the header "Ask" bar fills in and the query fires.
Chips disappear once a result is shown and reappear if the user
clears the result.

### 4. Result cards

The renderer picks a card based on `result.mode`:

- `ForwardCard`: chain visualisation + per-step intermediate values
  with their 90% bands + KaTeX-rendered equation per step + final
  prediction + citations
- `AbductiveCard`: observation + posterior over inferred variables
  with prior-to-posterior shift score + effective sample size +
  citations
- `CounterfactualCard`: baseline / intervened side-by-side + the
  paired delta with its 90% band + citations

A `routing` block above the card lists the LLM rationale, confidence,
source (`llm` or `fallback`), and the SPECTER2 matched-variables
table. KaTeX is loaded from CDN; the step equations render as proper
math (`ρ = ρ_full · (1 − φ)`) rather than raw LaTeX strings.

### 5. Agents panel

A Proposer → Critic loop sits below the result card. The Proposer
walks the variable graph and suggests novel `(target, sweep_var)`
pairs the user hasn't asked about; the deterministic Explorer
evaluates each, and the Critic queries the corpus to verdict each
prediction as *interesting*, *trivial*, *out-of-domain*, or
*needs-more-data*. Fully live; hits `/api/reason/agents`.

### Hidden by feature flag

Two pieces of UI are present in the code but currently hidden:

| Flag (in `frontend/index.html`) | What it hides |
|---|---|
| `SURPRISES_ENABLED = false` | The active-exploration "Surprises" panel — and the auto-fetch on tab mount that would otherwise trigger a 25 s `/api/reason/explore` cold start. |
| `PRESETS_ENABLED = false` | The three-column grid of canonical-form preset buttons (one column per inference mode), hidden because the canonical forms confused early users. |

Flip either flag to `true` to bring the section back; the underlying
backend (`/api/reason/explore`, the preset metadata endpoint) is
live regardless.

---

## Where the code lives

| File                                                                          | Role |
|-------------------------------------------------------------------------------|------|
| [`reasoning/relation.py`](../../reasoning/relation.py)                           | `Variable`, `Prior`, `CovariateShift`, `Relation`, `RelationRegistry`. All inference modes. |
| [`reasoning/bone_relations.py`](../../reasoning/bone_relations.py)               | The seven bone-physics Relations + the covariate-shift library. |
| [`reasoning/semantic_anchor.py`](../../reasoning/semantic_anchor.py)             | SPECTER2 embedding index over Variables. |
| [`reasoning/query_router.py`](../../reasoning/query_router.py)                   | Single Ollama call for mode classification + value extraction. |
| [`reasoning/explorer.py`](../../reasoning/explorer.py)                           | Active exploration + `evaluate_proposal` bridge for the agent loop. |
| [`reasoning/agent_tools.py`](../../reasoning/agent_tools.py)                     | Tool registry + dispatcher (`list_variables`, `list_relations`, `forward`, `corpus_search`). |
| [`reasoning/proposer_agent.py`](../../reasoning/proposer_agent.py)               | LLM Proposer over the variable graph. |
| [`reasoning/critic_agent.py`](../../reasoning/critic_agent.py)                   | LLM Critic — verdicts a proposal as interesting / trivial / out-of-domain / needs-more-data. |
| [`api/main.py`](../../api/main.py)                                               | `/api/reason`, `/api/reason/ask`, `/api/reason/presets`, `/api/reason/explore`, `/api/reason/agents`. |
| [`frontend/index.html`](../../frontend/index.html)                               | `ReasonSection`, `ForwardCard`, `AbductiveCard`, `CounterfactualCard`, `CovariatePanel`, `SurprisesPanel` (flag-hidden), `AgentsPanel`. |

---

## API endpoints

### `GET /api/reason/presets`

Returns the available preset queries plus full Variable and Relation
metadata for the UI to render.

### `POST /api/reason`

Typed entry point. Request:

```json
{
  "mode": "forward" | "abductive" | "counterfactual",
  "target": "da_dN",
  "given": {"phi": 0.10, "dK": 1.0},
  "covariates": {"age": 60, "sex": "F", "site": "femur_cortical"}
}
```

For `abductive`: also `observed`, optional `observed_std`, optional
`infer`. For `counterfactual`: also `intervention`.

### `POST /api/reason/ask`

Free-text entry point. Request:

```json
{
  "query": "bone stiffness under cyclic load",
  "covariates": { ... }
}
```

The routing block in the response describes the mode, target,
confidence, and SPECTER2-matched variables.

### `GET /api/reason/explore`

Active exploration. Optional `?refresh=true` forces a recompute
(default behaviour: serve cached result, ~25 s on first call,
<1 ms thereafter).

### `POST /api/reason/agents`

Run the Proposer → Critic loop. Body: `{"n_proposals": 2}` (clamped
to 1–4). Returns an array of `{hypothesis, physics, critique}`
objects plus the current scratchpad size.

---

## What changed from the physics-grid retrospective

The earlier physics-grid pipeline (see
[`physics_grid.md`](physics_grid.md)) had four structural limits;
the equation graph addresses each:

| Aspect                  | Physics grid (retired)                           | Equation graph (current)                           |
|-------------------------|--------------------------------------------------|----------------------------------------------------|
| Hypothesis source       | Pre-encoded perturbation grid                    | Discovered chains via BFS over variable graph      |
| Composition across laws | None                                             | Automatic via shared symbols                       |
| Inference modes         | Forward only                                     | Forward · abductive · counterfactual · explore     |
| Uncertainty             | Deterministic point estimates                    | Monte Carlo with 90% bands                         |
| Biological context      | Cortical/trabecular only                         | Age, sex, site, disease via covariate shifts       |
| Natural-language input  | Keyword anchor on node IDs                       | SPECTER2 embeddings + single LLM routing call      |
| Hypothesis generation   | Cartesian product of fixed grids                 | Active graph walk + LLM Proposer / Critic agents   |
| Citation per prediction | Optional, attached to law name                   | Per-parameter, per-relation, per-shift             |
