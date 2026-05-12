# Physics-grid reasoner (deprecated)

> **Status: ❌ Removed from the codebase.** The physics-grid reasoner has
> been replaced by the [equation-graph reasoner](equation_graph.md). The
> source files (`reasoning/physics.py`, `reasoning/physics_gen.py`,
> `reasoning/critic.py`, `reasoning/lrm.py`, `reasoning/physics_vars.py`)
> and its `/api/reason` endpoint have been deleted. This document is
> preserved as a retrospective of the design and its limitations for
> future writing about why the system was redesigned.

The physics grid was BoneMind's first deterministic reasoner. It
superseded an even earlier graph-walk-plus-physics-filter design
(documented in [phase4_lrm_plan.md](../phase4_lrm_plan.md), steps 4.4–4.6)
and was itself superseded by the equation-graph reasoner. The bone
knowledge graph (`data/db/ontology.db`) survives — the physics grid only
consulted it for display labels on chain cards, not as a source of
hypotheses, and the cleaned graph remains the substrate for the next
generation of graph-driven features.

## What it did

The reasoner generated hypotheses **from physical laws** and falsified
them with **an adversarial physics critic** before any output. No LLM
was involved at runtime — every result was deterministic and
reproducible.

The pipeline (`PhysicsGenerator` → `PhysicsCritic` → `NoveltyClassifier`)
ran each applicable physical law over a fixed perturbation grid and
emitted candidate hypotheses with quantitative `ΔY/Y` predictions:

| Law | Implemented as |
|---|---|
| Currey 1988 — *E ∝ ρⁿ* | porosity → elastic modulus |
| Frost mechanostat | strain → adaptation rate |
| Paris–Vashishth | stress-intensity range → crack growth rate |
| Beam bending | bending moment, geometry → stress |

Each candidate was scored against a four-round adversarial critic that
checked direction agreement with established literature, magnitude
plausibility against an internal range table, regime validity, and
consistency with shared-variable couplings. Survivors were classified
for corpus grounding (`GROUNDED` / `SPECULATIVE` / `NOVEL`).

## Why it worked and why it didn't

What the physics grid got right and what later writing should preserve:

- **Determinism.** No LLM at runtime. Identical inputs produced
  identical outputs, making the pipeline auditable in a way that is
  unusual for hypothesis-generation systems.
- **Quantitative predictions.** Each hypothesis carried a `ΔY/Y`
  magnitude, not just a direction — the physics actually predicted
  something falsifiable.
- **Honest filtering.** The adversarial critic rejected outputs the
  laws themselves could not justify, rather than wrapping speculation
  in confident prose.

What it couldn't do, and why each limit motivated the redesign:

1. **No composition.** Currey, Paris, Frost, and beam bending each
   ran independently. There was no way for the output of one law
   (`elastic_modulus` from Currey) to flow into the input of another
   (`strain` for Frost) or to traverse a chain — every hypothesis was
   one law deep, even when the underlying biomechanics is many.

2. **Brittle anchoring.** A query like *"bone stiffness under cyclic
   load"* never fired Paris law because *"cyclic load"* was not an
   exact keyword in the variable registry. The keyword-anchor layer
   in `reasoning.physics_vars.variables_in_query` caught whichever
   phrase happened to be hard-coded; everything else was silently
   dropped.

3. **No biological context.** Age, sex, anatomical site, and disease
   state had no effect on the predictions — every query was answered
   for an idealised "generic bone." Two 30-year-old men with healthy
   femur cortices and two 75-year-old women with osteoporotic
   vertebrae got the same `ΔE/E` for the same `Δφ`.

4. **No surprise.** The system could only answer queries the user
   typed. It could not propose hypotheses on its own, scan for
   under-explored regions of the variable graph, or seek out
   relationships the user had not yet asked about.

5. **No abduction or counterfactual.** The forward direction (cause
   → effect) was the only mode. The reasoner could not be inverted
   to ask "what value of `φ` is consistent with the observed `E`?"
   or "what would `E` look like under intervention?".

6. **Decorative graph use.** The cleaned ontology graph was used only
   for display labels on the chain cards. It was never queried as a
   reasoning substrate — a structural waste of an artifact that had
   taken substantial effort to clean.

## What replaced it

The [equation-graph reasoner](equation_graph.md) addresses each of the
six limits directly:

| Physics-grid limit | Equation-graph response |
|---|---|
| No composition | A typed equation graph where laws share variables; chains are discovered by traversal, not pre-encoded. |
| Brittle anchoring | SPECTER2-based semantic variable anchoring on free-text queries. |
| No biological context | Per-relation covariate shifts on priors before forward propagation (age, sex, site, disease). |
| No surprise | An active-exploration ("Surprises") panel that walks the graph without a query, scoring candidates by corpus surprise. |
| Forward only | Forward, abductive, and counterfactual inference modes. |
| Decorative graph use | The ontology graph remains, and its future role is graph-driven reasoning modes (gap finding, mechanism explanation) built on top of the cleaned + reclassified data. |

A proposer/critic agent loop runs on top of the equation graph to
generate hypotheses without the user typing a query, walking the
variable graph for novel `(target, sweep_var)` pairs and grading them
against the corpus.

## When to reach for this retrospective

This document exists because the physics grid is the clearest
demonstration in the project history that a deterministic,
physics-driven hypothesis pipeline is *insufficient on its own* —
the four-year arc from the original Phase 4 graph walk to the current
equation-graph reasoner is a sequence of "this doesn't compose" /
"this doesn't anchor" / "this doesn't see the patient" lessons. When
writing the methods or related-work section of a paper, the limits
enumerated above are the motivation for everything that came after.
