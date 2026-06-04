# Concept-typing reclassification

After the six-stage cleanup (see [graph_cleanup.md](graph_cleanup.md)) the bone
knowledge graph still had a serious problem: the LLM extractor defaulted ~75%
of surviving nodes into the catch-all `concept` type. The typed-ontology schema
was effectively unused, breaking every type-aware downstream query.

This document describes `scripts/reclassify_concepts.py`, the rule-based pass
that converts `concept`-typed nodes into the correct types from `NODE_TYPES`.

## Why reclassification is needed

The ontology defines ten meaningful node types
(`structure`, `property`, `process`, `pathology`, `mechanism`, `material`,
`factor`, `clinical`, `scale`, `cell`) plus `concept` as a catch-all for
unclassified nodes. The extractor LLM (`huatuogpt-bone:latest`) was given the
type vocabulary but routinely fell back to `concept` whenever it was uncertain
— which was most of the time.

Audit of the cleaned graph (1,597 nodes), before reclassification:

| Type | Count | Share |
|---|---:|---:|
| concept | 1,200 | 75.1% |
| structure | 117 | 7.3% |
| property | 98 | 6.1% |
| process | 89 | 5.6% |
| pathology | 30 | 1.9% |
| material | 18 | 1.1% |
| factor | 16 | 1.0% |
| clinical | 12 | 0.8% |
| mechanism | 11 | 0.7% |
| cell | 3 | 0.2% |
| scale | 3 | 0.2% |

In practice this meant:

- `nodes_by_type("factor")` returned 16 nodes — but the graph contained 30+
  drugs alone (denosumab, alendronate, the bisphosphonate family, raloxifene,
  teriparatide), all hiding inside `concept`.
- Type-constrained reasoning (e.g. "find a `factor` that affects a `property`
  leading to a `pathology`") collapsed because the search space lost most of
  its causal vocabulary.
- The colour-by-type visualisation showed 75% of nodes as the catch-all grey,
  obscuring whatever structure was actually there.

A typed ontology where 75% of nodes share one type is not really typed.

## How the script works

`scripts/reclassify_concepts.py` operates on `concept`-typed nodes only.
All other types are trusted (they came from either the hand-curated seed or
a confident LLM classification). The script makes a decision in three layers,
in priority order:

1. **`EXACT_OVERRIDES`** — hand-curated mapping for high-leverage nodes whose
   names would either trick the rules or genuinely need judgement. Examples:
   `physical_activity → factor`, `mechanical_load → process`, `temperature →
   property`, `hydroxyapatite → material`.
2. **`BARE_WORDS`** — single-word nodes that suffix-based regex can't match.
   Examples: `osseointegration → process`, `biocompatibility → property`,
   `fractures → pathology`, `connective_tissue → structure`.
3. **`RULES`** — ordered regex patterns; first match wins. The rules are
   grouped by target type and proceed from most-specific to most-general
   (drug-name suffixes before generic suffix patterns, etc.).

Anything none of the three layers match stays as `concept`. The script is
conservative by design — it would rather leave a node mis-typed as `concept`
than wrongly promote it to a real type.

### Rule classes

| Target type | Pattern shape | Examples that match |
|---|---|---|
| **factor** | drug suffixes (`*mab`, `*nib`, `*fene`, `*dronate`, `*paratide`) | denosumab, romosozumab, alendronate, teriparatide, raloxifene |
| **factor** | gene/protein family + suffix (`il_*`, `tnf_*`, `bmp_*`, `wnt_*`, `fgf_*`, `mir_*`, `mmp_*`, …) | il_6, bmp_2, wnt3a, mir_214_3p, mmp_9 |
| **factor** | known bone-specific proteins (exact-name set) | rankl, opg, sost, runx2, fgf23, dkk1, ctsk, sclerostin |
| **factor** | hormones, vitamins, minerals, `anti_*`, `*_factors`, `*_ions`, `*_lifestyle` | estrogen, vitamin_d, calcium, anti_rankl, growth_factors, sedentary_lifestyle |
| **cell** | `*blasts`, `*clasts`, `*cytes`, `*_cell(s)` | osteoblasts, osteoclasts, chondrocytes, stem_cells |
| **pathology** | `*itis`, `*_fractures?`, `*_deficiency`, `*_insufficiency`, `*_pain`, `*_injury`, `*_complications`, `*_disease`, `*_syndrome`, `*_occurrence`, `*_mortality`, `*_disability` | peri_implantitis, hip_fractures, vitamin_d_deficiency, fracture_occurrence |
| **process** | `*_activity`, `*_expression`, `*_proliferation`, `*_differentiation`, `*_polarization`, `*_apoptosis`, `*_senescence`, … | osteoblast_activity, vegf_expression, macrophage_polarization |
| **process** | bone-specific dynamics: `*_resorption`, `*_formation`, `*_remodelling`, `*_mineralization`, `*genesis`, `*_osseointegration`, `*_ingrowth` | bone_resorption, bone_formation, osteogenesis, osseointegration |
| **process** | metabolism/signalling: `*_synthesis`, `*_degradation`, `*_uptake`, `*_release`, `*_signaling`, `*_phosphorylation`, `*_binding`, `*_response`, `*_inflammation` | collagen_synthesis, dkk1_expression, wnt_signalling |
| **process** | mechanical stimuli: `*_stress`, `*_pressure`, `*_force(s)`, `*_loading`, `*_loss`, `*_repair`, `*_healing` | shear_stress, bone_loss, fracture_healing |
| **property** | `*_risk`, `*_probability`, `*_score`, `*_index`, `*_burden`, `*_prevalence` | fracture_risk, osteoporosis_risk, frax_score |
| **property** | mechanical/material: `*_strength`, `*_stiffness`, `*_toughness`, `*_modulus`, `*_stability`, `*_solubility`, `*_roughness`, `*_biocompatibility` | fracture_resistance, surface_roughness, mechanical_stability |
| **property** | morphometric/quantitative: `*_density`, `*_porosity`, `*_thickness`, `*_volume`, `*_concentration`, `*_level`, `*_rate`, `*_bmd`, `*_bmc`, `*_age` | trabecular_thickness, serum_calcium_concentration, lumbar_bmd |
| **mechanism** | `*_pathway(s)` | wnt_pathway, rankl_opg_pathway, smad_signaling_pathways |
| **clinical** | `*_treatment`, `*_therapy`, `*_intervention`, `*_supplementation`, `*_assessment`, `*_marker(s)`, `*_diagnosis`, `*_cost(s)`, `serum_*`, `blood_*` | alendronate_treatment, calcium_supplementation, serum_calcium, biochemical_markers |
| **clinical** | imaging/risk tools (exact-name set) | frax, dxa, dexa, qct, hr_pqct |
| **structure** | anatomy (exact-name set + `*_bone`, `*_cortex`, `*_trabeculae`, `*_structure`) | cortical_bone, lumbar_spine, vertebral_column, hierarchical_structure |
| **material** | biomaterials by suffix or exact name: `*hydrogel`, `*scaffold`, `*cement`, `*nanoparticle`, `*composite`, `*biomaterial`, `*calcium_phosphate`, plus hyaluronic_acid, bcp, tcp, plla, pcl, plga, cmc, alginate, chitosan, bioglass | hyaluronic_acid, biphasic_calcium_phosphate, carboxymethyl_cellulose, bioglass |

The full pattern set lives in
[`scripts/reclassify_concepts.py`](../../scripts/reclassify_concepts.py).

## Result of the full pass

Running `python scripts/reclassify_concepts.py` on `data/db/ontology.db`:

```
concept-typed nodes:    1200
reclassified:           571 (47.6%)
remain as 'concept':    629
```

| Type | Before | After | Change |
|---|---:|---:|---:|
| concept | 1,200 | 629 | **−571 (−48%)** |
| process | 89 | 247 | **2.8×** |
| property | 98 | 220 | **2.2×** |
| factor | 16 | 141 | **8.8×** |
| structure | 117 | 132 | +15 |
| pathology | 30 | 79 | **2.6×** |
| clinical | 12 | 70 | **5.8×** |
| material | 18 | 39 | 2.2× |
| mechanism | 11 | 19 | 1.7× |
| cell | 3 | 18 | **6×** |
| scale | 3 | 3 | — |

The `concept` share of the whole graph drops from **75% → 39%**.

### Quality indicators after reclassification

- **`factor` covers actual factors.** Up from 16 to 141 — drug families
  (bisphosphonates, monoclonals, SERMs, parathyroids), signalling proteins
  (RANKL/OPG/SOST, BMP/WNT/FGF families, microRNAs), hormones, vitamins, and
  modifiable risk factors (smoking, physical activity, calcium intake) are
  now reachable through `nodes_by_type("factor")`.
- **`process` covers actual processes.** Up from 89 to 247 — all of the
  `*_activity`, `*_expression`, `*_resorption`, `*_formation`,
  `*_differentiation`, `*_signaling`, `osteo*genesis` nodes are now typed
  correctly. Mechanotransduction stimuli (`mechanical_load`,
  `mechanical_stress`, `shear_stress`, `loading`) are processes too.
- **`pathology` covers actual disease state.** Hip/vertebral/atypical
  fractures, deficiencies/insufficiencies, peri-implantitis, nonunion, and
  mortality/disability outcomes are now queryable as pathologies rather than
  hiding in `concept`.
- **Top hubs are now correctly typed.** `fracture_risk` (property),
  `bone_mineral_density` (property), `bone_resorption` (process),
  `bone_formation` (process), `denosumab` (factor) — the heaviest nodes in
  the graph all have the type a reader would expect.

## Switching to the reclassified graph

The script writes to a separate file so the original is preserved.

```bash
# Run the reclassification
python scripts/reclassify_concepts.py

# Inspect the result
sqlite3 data/db/ontology_typed.db \
  "SELECT node_type, COUNT(*) FROM nodes GROUP BY node_type ORDER BY COUNT(*) DESC;"

# When satisfied, swap (keeping the old one as backup)
mv data/db/ontology.db       data/db/ontology_pre_typed.db
mv data/db/ontology_typed.db data/db/ontology.db
```

The Reasoning tab's critic reads `ontology.db` via `reasoning/kg_context.py`
(and `/api/stats` via `OntologyStore`); both pick up the reclassified types on
the next API restart.

## Known residual issues

The reclassification is rule-based and deliberately conservative; it leaves
629 nodes (≈39% of the graph) typed as `concept`. These break down roughly
as follows:

- **Genuinely ambiguous bare-word nouns** — `function`, `nutrient_diffusion`,
  `hormonal_signals`, `leverage_power`, `respiratory_role`. No pattern can
  resolve these without semantic context.
- **Surface-treatment and alloy variants** — `turned_surface`, `zirti_surface`,
  `mg_sc_sr_alloy`, `bcp_with_cmc`, `bcp_block`. These are materials but the
  naming is too irregular to capture with a small rule set.
- **Long tail of one-off compounds** — `melatonin`, `hypoxia`, `endoplasmic_reticulum`,
  `sella_turcica`, `dermal_elements`, etc. Each would need to be added as a
  bare-word override.
- **Plurals not in the bare-word list** — a few processes/pathologies whose
  singular forms are covered.

Two follow-up passes can close most of this gap:

1. **An LLM batch pass over the residual 629.** Feed each node's id +
   description (and a couple of neighbours) to a small model and ask for a
   type. One-off and cheap.
2. **Hand-curation of the top-degree leftovers.** A few hundred high-degree
   nodes account for most reasoning paths; hand-fixing those by appending to
   `EXACT_OVERRIDES` or `BARE_WORDS` is fast and disproportionately effective.

## Reproducibility

```bash
# Default — writes data/db/ontology_typed.db
python scripts/reclassify_concepts.py

# Preview without writing
python scripts/reclassify_concepts.py --dry-run

# Custom paths
python scripts/reclassify_concepts.py \
    --input  data/db/ontology.db \
    --output data/db/ontology_v3.db
```

The script is deterministic — re-running it on the same input produces an
identical output DB. Adding entries to `EXACT_OVERRIDES`, `BARE_WORDS`, or
`RULES` is the supported way to extend coverage; existing entries should not
be removed lightly because they encode review decisions.
