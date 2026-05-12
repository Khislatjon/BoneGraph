# Physics-driven hypothesis generation (v0 — "physics grid")

**Status: ✅ Live — the "physics grid" toggle of the Reasoning tab.**

This document describes the v0 reasoner. It runs behind the
**`physics grid`** option in the Reasoning tab's top-of-page toggle.
The other option, **`equation graph`** (v2), is a parallel reasoner
documented in [`v2_equation_graph.md`](equation_graph.md). The two
share nothing at the reasoning layer; they coexist so you can compare
their outputs side by side.

This pipeline supersedes the graph-walk + physics-as-filter approach
described in [`phase4_lrm_plan.md`](../phase4_lrm_plan.md) (steps 4.4–4.6),
which has been removed from the codebase but is preserved in the doc
for historical context.

The Reasoning tab's v0 pipeline generates hypotheses **from physical
laws** and falsifies them with **an adversarial physics critic**
before any human-readable output is produced. The cleaned bone
knowledge graph ([`graph_cleanup.md`](graph_cleanup.md)) is consulted
only for node-id materialisation (the chain cards display extracted
graph labels), not as a source of hypotheses.

---

## Why this exists

The previous design (steps 4.4–4.5 of the original Phase 4 plan)
walked the LLM-extracted knowledge graph between query-anchored nodes
via shortest paths, then used the directional physics rule table as a
filter to drop IMPLAUSIBLE chains. Two structural problems made the
output unsatisfying:

1. **The graph is the corpus.** Every edge in the graph was extracted
   from a published passage. Walking shortest paths between nodes
   re-traces what papers already say, so the system was a literature
   retrieval engine dressed as a reasoning engine. Genuinely new
   hypotheses cannot exist in a graph derived solely from the corpus.

2. **Physics was decorative.** The 70-odd directional rules were
   consulted only as a final IMPLAUSIBLE filter. They could not
   propose anything, predict magnitudes, or reason about regimes.
   For most chains no rule applied, so the filter was effectively a
   no-op.

The new design inverts the roles: **physics laws generate the
candidates with quantitative predictions; the graph and the corpus
serve as grounding evidence; an adversarial critic falsifies before
display.** No LLM is involved at runtime — every result is
deterministic and reproducible.

---

## Pipeline overview

```
                   user query (free text)
                            │
                            ▼
            ┌─────────────────────────────┐
            │  variables_in_query()       │   reasoning/physics_vars.py
            │  (curated keyword regex)    │
            └──────────────┬──────────────┘
                           │   physics variables touched
                           ▼
            ┌─────────────────────────────┐
            │  PhysicsGenerator           │   reasoning/physics_gen.py
            │   ├ Currey's law            │
            │   ├ Frost mechanostat       │
            │   ├ Paris law               │
            │   └ Beam bending            │
            │  one perturbation grid /law │
            └──────────────┬──────────────┘
                           │   list[PhysicsHypothesis]  (with ΔY/Y predictions)
                           ▼
            ┌─────────────────────────────┐
            │  PhysicsCritic              │   reasoning/critic.py
            │   round 1: directional rules│
            │   round 2: magnitude bounds │
            │   round 3: law-aware domain │
            └──────────────┬──────────────┘
                           │   survivors only (or all, if keep_falsified)
                           ▼
            ┌─────────────────────────────┐
            │  NoveltyClassifier          │   reasoning/novelty.py
            │  GROUNDED · SPECULATIVE ·   │   (corpus presence check)
            │  NOVEL                      │
            └──────────────┬──────────────┘
                           │   composite-scored, sorted
                           ▼
                  /api/reason JSON
                           │
                           ▼
                Reasoning-tab chain cards
```

The orchestrator is `LRM.query_physics()` in `reasoning/lrm.py`. The
legacy `LRM.query()` (graph-walk) was removed in Phase 0; the
companion `eval/run_lrm_eval.py` benchmark moved to
`eval/legacy/run_lrm_eval.py` at the same time and is not maintained
against the current LRM constructor.

---

## Variable registry (`reasoning/physics_vars.py`)

The registry is the bridge between physical equations and graph node
IDs. Each `PhysicsVariable` declares:

| Field | Purpose |
|---|---|
| `name`, `symbol`, `units`, `description` | UI / human-readable metadata |
| `cortical_range`, `trabecular_range` | Numerical bounds for the magnitude critic round |
| `node_ids` | Graph node_ids that proxy this variable, in priority order — first match in the cleaned graph wins |
| `keywords` | Curated whole-phrase triggers for query gating |
| `inverse_of` | Set when one variable is the inverse of another (e.g. porosity ↔ apparent density) |

The current registry covers nine variables, grouped by which law uses
them:

| Variable | Symbol | Used by | Cortical range | Trabecular range |
|---|---|---|---|---|
| `apparent_density` | ρ | Currey | 1.5–2.0 g/cm³ | 0.1–0.9 g/cm³ |
| `porosity` | φ | Currey (inverse of ρ) | 0.03–0.15 | 0.50–0.95 |
| `elastic_modulus` | E | Currey | 15–25 GPa | 0.05–5 GPa |
| `strength` | σ | Currey | 100–200 MPa | 1–20 MPa |
| `peak_strain` | ε | Frost mechanostat | 50–5,000 µε | 50–5,000 µε |
| `bone_adaptation_rate` | ΔBMD/yr | Frost mechanostat | −3 to +3 %/yr | −5 to +5 %/yr |
| `stress_intensity_range` | ΔK | Paris law | 0.3–2.0 MPa·√m | 0.1–1.0 MPa·√m |
| `crack_growth_rate` | da/dN | Paris law | 1e-12 to 1e-6 m/cycle | (same) |
| `cortical_thickness` | t | Beam bending | 1.5–6.0 mm | 0.1–0.5 mm |
| `bending_resistance` | EI | Beam bending | 5–1,000 % of baseline | (same) |

### Query gating

`variables_in_query()` matches each variable's `keywords` list with a
case-insensitive word-boundary regex (specifically `(?<!\w)kw(?!\w)` to
handle non-ASCII characters such as `Δ` and `µ`). This replaced the
earlier substring-match-on-node-IDs implementation, which produced a
lot of cross-bleed — e.g. the word *"strength"* alone would activate
beam bending in a Currey query. The keywords list is hand-curated; do
not auto-derive it from `node_ids`.

A query activates a law only when at least one of the law's input or
output variables matches by keyword. The result is tight per-law
dispatch:

| Query | Laws activated |
|---|---|
| *porosity and elastic modulus in cortical bone* | Currey only |
| *mechanical loading and bone formation* | Frost only |
| *cyclic loading and crack growth* | Paris only |
| *cortical thickness and bending strength* | Beam only |

---

## The four laws

Each law lives as a `_apply_<law>()` branch on `PhysicsGenerator` plus
a differential predictor in `reasoning/physics.py`. All four follow the
same shape: pick a baseline from the variable's range, sweep a small
perturbation grid, materialise a `PhysicsHypothesis` per perturbation
with input/output graph nodes pulled from the registry.

### Currey's law — composition → mechanics

| | |
|---|---|
| **Equation** | E ∝ ρⁿ with ρ = ρ_full · (1 − φ) |
| **Inputs** | `porosity`, `apparent_density` |
| **Outputs** | `elastic_modulus` (n = 2.5), `strength` (n = 2.0) |
| **Predictor** | `currey_delta_from_porosity()`, `currey_delta_from_density()` in `physics.py` |
| **Perturbation grid** | Δφ ∈ {+5, +10, +20} pp; Δρ ∈ {±5, ±10, ±20} % |
| **Baseline** | Geometric mean of the variable's range (cortical or trabecular) |
| **Relation semantics** | Porosity *decreases* E and σ; density *increases* E and σ — the variable-level relation is fixed by the law, not by the perturbation sign |

A "+10 pp porosity" perturbation in cortical bone (φ baseline ≈ 0.07)
predicts ΔE/E ≈ −24.7%; a "+20 pp porosity" perturbation drops E
below the cortical floor and is correctly falsified at Round 2.

### Frost mechanostat — loading → adaptation regime

| | |
|---|---|
| **Equation** | ε → adaptation regime, with literature midpoint ΔBMD/yr per zone |
| **Inputs** | `peak_strain` |
| **Outputs** | `bone_adaptation_rate` (chain output node varies per zone: `bone_formation`, `bone_resorption`, `bone_remodeling`) |
| **Predictor** | `mechanostat_adaptation()` wraps the existing `mechanostat_zone()` and adds `bmd_pct_per_year` and `chain_output_node` |
| **Perturbation grid** | ε ∈ {100, 1000, 2200, 5000} µε — one strain inside each canonical Frost zone |
| **BMD/yr midpoints** | acute disuse −3.0; chronic disuse −1.5; adapted 0.0; mild overload +1.5; pathological overload −1.0; fracture −5.0 |

Zones come from Frost (2003), Robling (2009), and Burr (2002). The
chain's output node is selected per-zone, so a mild-overload strain
emits *"strain → bone_formation"*, while a chronic-disuse strain emits
*"strain → bone_resorption"*. The variable-level relation is always
*increases* (more loading positively drives the adaptation rate within
the working range); the prediction string makes any harmful regime
explicit.

### Paris law — fatigue → crack growth

| | |
|---|---|
| **Equation** | da/dN = C · ΔKᵐ |
| **Inputs** | `stress_intensity_range` |
| **Outputs** | `crack_growth_rate` |
| **Predictor** | `paris_delta(ΔK_baseline, ΔK_new)` returns absolute da/dN at both points plus the rate ratio and log₁₀ ratio |
| **Constants** | C = 1.7 × 10⁻⁹, m = 3.9 — Vashishth et al. (2004), human cortical bone |
| **Perturbation grid** | ΔK ∈ {0.5, 1.0, 1.5} MPa·√m around a baseline of 0.6 |
| **Domain bound** | ΔK must stay below the cortical KIc (~6 MPa·√m) — Round 3 rejects anything at or above that |

A 2.5× increase in ΔK predicts a ~36× increase in da/dN — the steep
exponent is the whole point of Paris. The hypothesis carries both the
ratio and the absolute m/cycle figure for easy comparison with
literature da/dN-vs-ΔK plots.

### Beam bending — geometry → bending capacity

| | |
|---|---|
| **Equation** | σ = M·rₒ / I, with I = π/4 · (rₒ⁴ − rᵢ⁴) for a hollow cylinder |
| **Inputs** | `cortical_thickness` |
| **Outputs** | `bending_resistance` (∝ I, scaled to a baseline of 100 %) |
| **Predictor** | `beam_bending_thickness_delta(r_o, t_baseline, Δt%)` returns the new thickness, both I values, and σ_new/σ_old |
| **Held constant** | Outer radius r_o = 16 mm (femoral midshaft midpoint); thickness baseline t = 5 mm |
| **Perturbation grid** | Δt ∈ {−10, −20, −30} % — progressive osteoporotic thinning |
| **Scenario** | Cortical only — the hollow-cylinder long-bone model is not appropriate for trabecular tissue |

The output variable was deliberately framed as *bending resistance*
(positive correlation with thickness), not *bending stress* (negative).
This avoids a directional rule conflict in the rule table — the rule
*"cortical_thickness decreases bending_strength"* is correctly marked
IMPLAUSIBLE there, and a stress-framed hypothesis would have failed
Round 1 even though the underlying physics is sound. Framing the output
as resistance keeps the variable-level relation as *increases* and
lets the prediction string carry the I-ratio (or, equivalently, the
σ-ratio at fixed M).

---

## The three critic rounds (`reasoning/critic.py`)

Each candidate hypothesis runs through every round in sequence. A
hypothesis survives only if all three pass; the first failure is
recorded so the UI can show *why* a candidate was rejected. A fourth
"scenario consistency" round existed in earlier versions but was
removed in Phase 0 — the generator's own invariants made it a no-op.

### Round 1 — directional consistency

`PhysicsEngine.validate_chain()` looks the chain's edges up in the
40-rule directional table from `reasoning/physics.py`. IMPLAUSIBLE
rules veto; UNCERTAIN or absent rules pass through. After cleanup and
the variable-registry refactor, this round is mostly a sanity check
that the generator's relation labels match the rule table's reading
of "increases" / "decreases" as a *variable-level* effect (not a
perturbation direction).

### Round 2 — magnitude in physical range

For each output variable, the critic checks that the predicted absolute
value falls inside the bone-physical range for the chosen tissue
scenario (cortical or trabecular), with a ±20 % tolerance on the range.

The generator may supply `delta_output["predicted_value"]` directly
(absolute units — used by Frost, Paris and Beam, where %-changes
aren't relative). For Currey, which emits a relative ΔE/E, the critic
falls back to the geometric-mean baseline times (1 + Δ%/100). This
fallback is why a "+20 pp porosity" Currey hypothesis correctly
falsifies in cortical bone: predicted E ≈ 10.6 GPa is below the
15–25 GPa cortical band.

### Round 3 — law-aware domain check

Round 3 dispatches by law name:

| Law | Domain check |
|---|---|
| Currey | ρ_new / ρ_old must lie in [0.3, 2.0] — outside this band the power-law fit isn't calibrated |
| Paris | ΔK must be below cortical KIc ≈ 6 MPa·√m — at or above, failure is single-cycle and Paris no longer applies |
| Frost mechanostat | strain must be in [0, 25 000] µε — above 25 000 the adaptation framework gives way to the fracture threshold |
| Beam bending | t_new must be > 0 — kept positive automatically by the perturbation grid, defensive check only |

---

## Scoring and novelty

`LRM._score_physics()` ranks survivors with a composite score:

```
score = 0.60 · (rounds_passed / rounds_total)
      + 0.25 · min(|ΔY/Y| / 50 %, 1)
      + 0.15 · novelty_bonus
```

Novelty bonus weights NOVEL > SPECULATIVE > GROUNDED, so the
ranking is biased toward physics-predicted hypotheses the literature
hasn't fully established. Currey predictions for cortical bone tend to
land as GROUNDED (the corpus does cover them); Frost regime predictions
in less-studied conditions and Paris predictions for trabecular bone
tend to surface as SPECULATIVE or NOVEL.

The novelty classifier itself is the same two-tier (keyword + SPECTER2)
implementation from Step 4.5. `PhysicsHypothesis` exposes a `summary`
property that the classifier consumes, so the existing classifier
worked unchanged.

---

## API and UI

### `/api/reason` response shape

A single chain (per surviving hypothesis) carries:

```jsonc
{
  "nodes":         ["porosity", "elastic modulus"],
  "relations":     ["decreases"],
  "law":           "Currey's law",
  "law_form":      "E ∝ ρ^2.5  (with ρ = ρ_full·(1−φ))",
  "scenario":      "cortical",
  "perturbation":  "+10 pp porosity (φ: 0.07 → 0.17)",
  "prediction":    "-24.7% ΔE/E",
  "input_var":     "porosity",
  "output_var":    "elastic_modulus",
  "delta_input":   { "variable": "porosity", "change_abs": 0.10, "from": 0.07, "to": 0.17, "magnitude": "moderate" },
  "delta_output":  { "variable": "elastic_modulus", "change_pct": -24.7, "rho_ratio": 0.93, "exponent": 2.5 },
  "assumed_inputs":{ "phi_baseline": 0.07, "rho_baseline": 1.73, "exponent": 2.5, "scenario": "cortical" },

  "validity":              "PLAUSIBLE",
  "critic_rounds_total":   3,
  "critic_rounds_passed":  3,
  "critic_failure":        "",
  "critic_checks": [
    { "name": "directional_consistency", "passed": true,  "detail": "Hypothesis direction matches established physics rules." },
    { "name": "magnitude_range",         "passed": true,  "detail": "Predicted E ≈ 14.5 GPa within cortical bone range (predicted)." },
    { "name": "powerlaw_domain",         "passed": true,  "detail": "Density ratio 0.93 within Currey bounds." }
  ],

  "novelty":         "GROUNDED",
  "novelty_display": "Grounded",
  "explanation":     "...",
  "disclaimer":      false,
  "corpus_disclaimer": null,
  "summary":         "Currey's law predicts that +10 pp porosity in cortical bone yields -24.7% ΔE/E (porosity → elastic modulus).",
  "score":           0.753
}
```

A `keep_falsified=true` query parameter returns falsified candidates
too, with `validity = "IMPLAUSIBLE"` and `critic_failure` populated by
the first failing round's detail.

### Chain card layout

Each card in the Reasoning tab shows:

- Hypothesis number, score, and a `cortical` / `trabecular` scenario tag
- Law header — name (e.g. *Currey's law*) and law-form (`E ∝ ρ²·⁵`)
- Chain — graph nodes connected by an arrow (`porosity → elastic modulus`)
- Two-pane prediction box:
  - **Perturbation** (left) — the input perturbation in plain language and units
  - **Predicted change** (right, amber) — the law's quantitative prediction
- Critic-rounds pill — green *"Survived 3/3 physics rounds"* or red *"Falsified at round N"*
- Falsification detail (red, only when applicable)
- Novelty badge + corpus disclaimer

CSS for the card lives under `.chain-law`, `.chain-scenario`,
`.pred-box`, `.pred-half`, `.critic-pill` and `.critic-fail-detail`
in `frontend/index.html`.

---

## What's novel relative to Buehler 2024

[Buehler (arXiv:2403.11996)](https://arxiv.org/abs/2403.11996)
extracts an ontological knowledge graph from ~1,000 biomaterials
papers and uses graph reasoning (transitive chains, isomorphic
mapping, path sampling) plus an LLM to suggest novel material designs.
The "physics" in that pipeline is decorative vocabulary in the prompt;
the LLM is asked to be creative and the results are evaluated by
inspection.

The BoneMind Reasoning tab is a deliberate departure on three points:

1. **Physics generates, doesn't decorate.** The four laws (Currey,
   Frost mechanostat, Paris, beam bending) actively *produce*
   candidate hypotheses with quantitative predictions before any
   graph traversal happens. The graph is consulted only afterwards,
   for grounding evidence and to materialise human-readable chains.

2. **Predictions are quantitative.** Every hypothesis carries an
   absolute or relative magnitude — "+26.9 % ΔE/E" or "da/dN = 8.3e-9
   m/cycle" — derivable from the law's equation, not generated by an
   LLM. This makes results reproducible and directly comparable to
   experimental measurements.

3. **An adversarial physics critic falsifies before display.** Each
   candidate runs through three independent rounds (directional
   consistency, magnitude bounds, law-aware domain). The role
   Buehler delegates to a creative LLM agent is filled here by a
   deterministic falsification loop. The first failure is recorded
   so the UI can show *why* a candidate was rejected — turning
   falsifications into teaching signals rather than silent discards.

No LLM is involved at runtime. The pipeline is fully deterministic and
auditable, which matters for a system whose outputs are intended to be
defensible in a thesis context.

---

## How to add a fifth law

The four laws cover what we believe are the canonical macroscopic axes
of bone mechanics: composition→mechanics, loading→adaptation,
fatigue→damage, geometry→capacity. A fifth law should target a
genuinely distinct concept space; otherwise it will overlap and add
noise. The most defensible candidate is AGE-crosslink degradation as
a critic-only rule, not a generator. If a generator-class law is
warranted:

1. Register the variables in `physics_vars.py` — node_ids (priority
   order), cortical and trabecular ranges, curated `keywords`. Add an
   `inverse_of` relationship if applicable.
2. Add a differential predictor in `physics.py` — given a baseline and
   a perturbation, return absolute and relative output magnitudes plus
   any law-specific metadata.
3. Add an `_apply_<law>()` branch on `PhysicsGenerator`. Activate only
   when the query touches the law's variables. Set the variable-level
   relation by the law's monotonic sign, not the perturbation sign.
   Always populate `delta_output["predicted_value"]` with the absolute
   prediction.
4. Extend `PhysicsCritic._round_powerlaw_domain()` with a law-name
   case. Choose a single bound that captures the calibration range or
   the regime breakdown.
5. Wire the new law into `PhysicsGenerator.generate()`'s dispatch.
6. Add to the smoke test query list and confirm:
   - The new law fires only on its intended queries
   - At least one perturbation in each cortical/trabecular scenario
     survives all three rounds
   - At least one extreme perturbation is correctly falsified

---

## Known limits

- **The variable registry is hand-curated.** Keyword lists are tight
  but conservative; some legitimate phrasings may not anchor. Ranges
  are conservative literature midpoints, not patient-specific bounds.
- **Frost BMD/yr midpoints are point estimates.** Real adaptation rates
  vary 2–3× across populations. The generator deliberately predicts
  the midpoint; the critic's magnitude check uses a wide range to
  accommodate variation.
- **The directional rule table overlaps the variable registry.** The
  Currey-family rules are now redundant (the generator subsumes them),
  but they remain because they still serve as the Round 1 check for
  hand-extracted edges. A future refactor should treat
  `PhysicalLaw` instances as the source of truth and derive the rule
  table from them, with hand-curated rules reserved for non-equation
  biology (RANKL/OPG, AGE crosslinks).
- **No automated eval.** The Phase 0 cleanup moved
  `eval/run_lrm_eval.py` to `eval/legacy/` and dropped the legacy
  `LRM.query()` path that it called. A fresh evaluator covering
  `query_physics()` (and the v2 reasoner alongside it) is pending.
- **Single-step chains.** Each hypothesis chains one input to one
  output via one law. Multi-step chains across laws cannot be
  generated here — the v2 equation graph reasoner
  ([`v2_equation_graph.md`](equation_graph.md)) was built to address
  that limit and coexists in the same Reasoning tab.

---

## File map

| File | Role |
|---|---|
| `reasoning/physics_vars.py` | Variable registry, query gating, range checks |
| `reasoning/physics.py` | Equations + differential predictors (`*_delta_*` functions) and the 40-rule directional table |
| `reasoning/physics_gen.py` | `PhysicsGenerator`, perturbation grids, `PhysicsHypothesis` dataclass |
| `reasoning/critic.py` | `PhysicsCritic`, `CritiqueResult`, `CheckRecord` |
| `reasoning/lrm.py` | `LRM.query_physics()` orchestrator + `_score_physics()` |
| `reasoning/novelty.py` | Two-tier corpus-presence classifier (unchanged) |
| `api/main.py` | `/api/reason` endpoint, response serialisation |
| `frontend/index.html` | Reasoning tab UI, chain-card markup and CSS |
