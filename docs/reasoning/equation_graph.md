# v2 — Equation-Graph Reasoner

**Status: ✅ Live — the "equation graph" toggle of the Reasoning tab,
marked `experimental` in the UI.**

This document explains how the v2 reasoner works, how it differs from
the v0 physics-grid system, and how to operate it. It is the
reference for the architecture introduced across commits
`Phase 0` through `Phase 5` on the `chore/reasoning-further` branch.

The v0 pipeline that runs under the **`physics grid`** toggle is
documented in [`physics_reasoning.md`](physics_grid.md). Both
reasoners coexist in the same tab — use the toggle at the top of the
page to switch between them.

---

## Why a second reasoner?

The original v0 pipeline (`physics.py`, `physics_gen.py`, `critic.py`)
ran each physics law over a fixed grid of perturbations and
filtered the output through a deterministic critic. It was reliable
but had four structural limits, all of which motivated the v2 redesign:

1. **No composition.** Currey, Paris, Frost, and beam-bending each
   ran independently. There was no way for the output of one law to
   feed the next.
2. **Brittle anchoring.** A query like *"bone stiffness under cyclic
   load"* never fired Paris law because *"cyclic load"* was not an
   exact keyword in the variable registry.
3. **No biological context.** Age, sex, anatomical site, and disease
   state had no effect on the predictions — every query was answered
   for an idealised "generic bone."
4. **No surprise.** The system could only answer queries the user
   typed. It could not propose hypotheses on its own.

v2 keeps v0 alive for comparison and adds a parallel reasoner that
addresses each of those limits.

---

## The toggle

In the Reasoning tab:

```
   ┌──────────────────────────────────────────────────────┐
   │      Hypothesis Generator   experimental             │
   │                                                      │
   │       [ physics grid ]  [ equation graph ]           │
   └──────────────────────────────────────────────────────┘
```

`physics grid` is the v0 system, untouched.
`equation graph` is v2 — what this document describes. The
`experimental` tag next to the title indicates this is preview-stage
software (changing shape, no clinical validation yet); it does *not*
mean "experimental data input."

The two share nothing at the reasoning layer. They do share the corpus
(`chunks.db`) for evidence retrieval, and the SPECTER2 embedding model
that the v2 anchor and the v0 novelty classifier both use.

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

This is the key architectural difference from v0:

| v0 graph (extracted)          | v2 equation graph              |
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
endpoint `GET /api/reason_v2/explore` still works if you `curl` it.
Implementation lives in [`reasoning/explorer.py`](../../reasoning/explorer.py).

---

## Covariate conditioning

The same physical query should yield a different prediction for a
30-year-old femur and a 75-year-old osteoporotic vertebra. The v2
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
The v2 tab exposes a free-text input that handles the translation.

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

The v2 mode shares the same header skeleton as v0 — only the
description, the toggle state, and what appears below differ. From
top to bottom:

### 1. Header — free-text "Ask" bar

```
   ┌─ Hypothesis Generator   experimental ───────────────────────┐
   │      [ physics grid ]  [ equation graph ]                   │
   │  Forward, abductive and counterfactual inference over a     │
   │  typed equation graph. Patient covariates reshape each      │
   │  law's priors before propagation.                           │
   │  ┌──────────────────────────────────────────────────────┐   │
   │  │ 🔍 Ask in plain English…                  [ Reason ] │   │
   │  └──────────────────────────────────────────────────────┘   │
   └─────────────────────────────────────────────────────────────┘
```

The bar uses the same `.reason-bar` / `.reason-input` / `.reason-btn`
classes as the v0 search bar — same width, same Reason button. Hits
`/api/reason_v2/ask`. The routing block is always rendered above the
result card so you can see which mode and target the LLM picked.

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
Chips disappear once a result is shown (matching v0's behaviour) and
reappear if the user clears the result.

### 4. Result cards

The renderer picks a card based on `result.mode`:

- `V2ForwardCard`: chain visualisation + per-step intermediate values
  with their 90% bands + KaTeX-rendered equation per step + final
  prediction + citations
- `V2AbductiveCard`: observation + posterior over inferred variables
  with prior-to-posterior shift score + effective sample size +
  citations
- `V2CounterfactualCard`: baseline / intervened side-by-side + the
  paired delta with its 90% band + citations

A `routing` block above the card lists the LLM rationale, confidence,
source (`llm` or `fallback`), and the SPECTER2 matched-variables
table. KaTeX is loaded from CDN; the step equations render as proper
math (`ρ = ρ_full · (1 − φ)`) rather than raw LaTeX strings.

### Hidden by feature flag

Two pieces of UI are present in the code but currently hidden:

| Flag (in `frontend/index.html`) | What it hides |
|---|---|
| `V2_SURPRISES_ENABLED = false` | The Phase 5 active-exploration "Surprises" panel — and the auto-fetch on tab mount that would otherwise trigger a 25 s `/api/reason_v2/explore` cold start. |
| `V2_PRESETS_ENABLED = false` | The three-column grid of canonical-form preset buttons (one column per inference mode), hidden because the canonical forms confused early users. |

Flip either flag to `true` to bring the section back; the underlying
backend (`/api/reason_v2/explore`, the preset metadata endpoint) is
live regardless.

---

## Where the code lives

| File                                                                          | Phase  | Role |
|-------------------------------------------------------------------------------|--------|------|
| [`reasoning/relation.py`](../../reasoning/relation.py)                           | 1–3    | `Variable`, `Prior`, `CovariateShift`, `Relation`, `RelationRegistry`. All four inference modes. |
| [`reasoning/bone_relations.py`](../../reasoning/bone_relations.py)               | 1–3    | The seven bone-physics Relations + the covariate-shift library. |
| [`reasoning/semantic_anchor.py`](../../reasoning/semantic_anchor.py)             | 4      | SPECTER2 embedding index over Variables. |
| [`reasoning/query_router.py`](../../reasoning/query_router.py)                   | 4      | Single Ollama call for mode classification + value extraction. |
| [`reasoning/explorer.py`](../../reasoning/explorer.py)                           | 5      | Active exploration (currently UI-hidden). |
| [`api/main.py`](../../api/main.py) (≥ line 970)                                  | 1–5    | `/api/reason_v2`, `/api/reason_v2/ask`, `/api/reason_v2/presets`, `/api/reason_v2/explore`. |
| [`frontend/index.html`](../../frontend/index.html)                               | 1–5    | `V2Section`, `V2ForwardCard`, `V2AbductiveCard`, `V2CounterfactualCard`, `V2CovariatePanel`, `V2Surprises` (flag-hidden). |

---

## API endpoints

### `GET /api/reason_v2/presets`

Returns the available preset queries plus full Variable and Relation
metadata for the UI to render.

### `POST /api/reason_v2`

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

### `POST /api/reason_v2/ask`

Free-text entry point. Request:

```json
{
  "query": "bone stiffness under cyclic load",
  "covariates": { ... }
}
```

The routing block in the response describes the mode, target,
confidence, and SPECTER2-matched variables.

### `GET /api/reason_v2/explore`

Active exploration (Phase 5). Optional `?refresh=true` forces a
recompute (default behaviour: serve cached result, ~25 s on first
call, <1 ms thereafter).

---

## Mapping back to the original critiques

| Original dislike                              | Where it's addressed                                                |
|-----------------------------------------------|---------------------------------------------------------------------|
| "Lookup engine, no surprise"                  | Phase 5 active exploration (`reasoning/explorer.py`, currently hidden). |
| "Keyword matching is brittle"                 | Phase 4 SPECTER2 semantic anchor (`reasoning/semantic_anchor.py`).  |
| "Laws are independent, can't chain"           | Phase 1 — chains emerge from shared symbols via BFS in `RelationRegistry`. |
| "No biological context"                       | Phase 3 — 12 citation-anchored covariate shifts in `bone_relations.py`. |

---

## Comparison with v0 in one table

| Aspect                  | v0 (physics grid)                                | v2 (equation graph)                                |
|-------------------------|--------------------------------------------------|----------------------------------------------------|
| Hypothesis source       | Pre-encoded perturbation grid                    | Discovered chains via BFS over variable graph      |
| Composition across laws | None                                             | Automatic via shared symbols                       |
| Inference modes         | Forward only                                     | Forward · abductive · counterfactual · explore     |
| Uncertainty             | Deterministic point estimates                    | Monte Carlo with 90% bands                         |
| Biological context      | Cortical/trabecular only                         | Age, sex, site, disease via covariate shifts       |
| Natural-language input  | Keyword anchor on node IDs                       | SPECTER2 embeddings + single LLM routing call      |
| Hypothesis generation   | Cartesian product of fixed grids                 | Active graph walk with corpus-grounded scoring     |
| Citation per prediction | Optional, attached to law name                   | Per-parameter, per-relation, per-shift             |

Both reasoners coexist in the Reason tab and the user can compare
their outputs side by side. v0 remains untouched; v2 is the new
substrate.
