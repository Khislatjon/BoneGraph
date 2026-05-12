# Knowledge graph cleanup

The bone knowledge graph in `data/db/ontology.db` is built by an LLM
(`huatuogpt-bone:latest` via `reasoning/extractor.py`) which extracts
`(source, relation, target)` triples from corpus chunks. The raw extraction is
noisy: surface text fragments become nodes, the same concept appears under
many spellings, single-mention triples dominate, and unrelated domains
(cancer, viral disease, food/silk research) leak in from background mentions
in the corpus.

This document describes `scripts/clean_graph.py`, the six-stage cleanup that
turns the raw graph into a usable substrate for the Reasoning tab.

## Why cleanup is needed

Audit of the raw graph (35,338 nodes / 34,265 edges):

| Symptom | Raw count | Impact on reasoning |
|---|---|---|
| Mean node degree | 1.94 | Sparse — most paths dead-end |
| Leaf nodes (degree 1) | 26,538 (75%) | Most nodes are unreachable for chain-finding |
| Edges with weight = 1 (single mention) | 32,768 (96%) | Almost no corroboration |
| Nodes typed as `concept` (LLM default) | 30,555 (86%) | Type system collapsed |
| Direction-encoded nodes (`increased_*`, `low_*`, …) | 830 | Semantic info lives in node names instead of edges |
| Same-relation reverse pairs (A→B and B→A) | 217 | Causes reasoning cycles |
| Cross-domain contamination (cancer/covid/silk/etc.) | ~500+ | Off-topic chains pollute results |
| Sentence-fragment node labels | ~570 | Junk concepts (e.g. *"10 year probability of a hip fracture 3"*) |
| Near-duplicate clusters | hundreds | e.g. 26 distinct nodes for "porosity", 25 for BMD |

The raw graph is essentially a noisy text dump. The clean graph aims to be a
proper knowledge graph.

## The six stages

The cleanup is implemented in `scripts/clean_graph.py`. It reads
`data/db/ontology.db` and writes a separate `data/db/ontology_clean.db`,
leaving the original untouched so you can A/B test.

### Stage 1 — Drop cross-domain nodes

Any node whose `node_id` contains a substring from `CROSS_DOMAIN_KEYWORDS`
(in `clean_graph.py`) is removed, along with all its edges. The list targets
obvious off-topic contamination:

- Cancer / oncology — `tumor`, `tumour`, `cancer`, `carcinoma`, `oncolog`,
  `malignan`, `metasta`, `leukemia`, `lymphoma`, `melanoma`
- Viral / pathogen — `covid`, `sars_cov`, `ebola`, `hiv_`,
  `viral_infection`, `influenza`
- Non-bone tissue — `breast_`, `_breast`, `prostate`, `ovarian`,
  `uterine`, `cervical`, `pancreatic`
- Unrelated materials in the corpus — `silk_`, `_silk`, `cheese`,
  `dough`, `hake_`, `whiting_`, `pacific_whiting`, `beethoven`,
  `symphony`, `fishmeal`

An `ALLOW_LIST` preserves bone-relevant terms that would otherwise be
caught by the keywords (`osteosarcoma`, `bone_cancer`, `bone_metastasis`,
`skeletal_metastas`, `spinal_metastas`, `bone_tumor`, `bone_tumour`).

### Stage 2 — Drop sentence-fragment nodes

Nodes whose label has more than 5 words OR whose `node_id` exceeds 50
characters are removed. The LLM extractor sometimes returned full sentences
as "concepts", e.g. `evidence_based_approaches_for_management_and_tailored_treatment_plans`.
These are not concepts and they break reasoning.

### Stage 3 — Canonicalise direction-encoded variants

The LLM produced many node variants where the *direction* of a relationship
is baked into the node name itself (`increased_fracture_risk`,
`low_bmd_levels`, `porosity_decrease`). These should be edges, not nodes.

`canonical_id()` strips the affixes iteratively until stable:

| Affix kind | Examples |
|---|---|
| Direction prefixes | `increased_`, `decreased_`, `reduced_`, `elevated_`, `higher_`, `lower_`, `low_`, `high_`, `abnormal_`, `new_`, `future_`, `subsequent_`, `old_`, `loss_of_`, `lack_of_`, `absence_of_`, `presence_of_`, `degree_of_`, `level_of_`, `accuracy_of_` |
| Direction suffixes | `_increase`, `_decrease`, `_reduction`, `_decline`, `_increases`, `_decreases`, `_levels`, `_level` |

Examples of canonicalisation:

```
increased_fracture_risk     → fracture_risk
low_bmd_levels              → bmd
fracture_risk_increase      → fracture_risk
abnormal_collagen_decrease  → collagen
```

After mapping, all edges are rewritten with canonical endpoints. Edges that
collapse onto the same `(source, relation, target)` triple have their
weights summed, so evidence scattered across variants is properly
consolidated. Self-loops created by canonicalisation are dropped.

### Stage 4 — Resolve same-relation reverse-pair contradictions

For any pair `(A, r, B)` and `(B, r, A)` where `r` is the same relation,
keep only the higher-weight edge. This eliminates obvious LLM-extraction
contradictions like:

```
bone_formation increases bone_mineral_density   # weight 14
bone_mineral_density increases bone_formation   # weight 2 — dropped
```

Opposite-direction relations between two nodes (e.g. `is_part_of` vs
`leads_to`) are kept — those are not contradictions.

### Stage 5 — Drop low-weight extracted edges

Extracted edges with weight below `--min-weight` (default `2.0`) are
removed. This eliminates the 96% of edges that came from a single corpus
mention. Seed edges are preserved regardless of weight.

### Stage 6 — Drop orphan nodes

Any node with no incoming or outgoing edges after stages 1-5 is removed.
After stage 5 in particular this drops a large number of leaves whose
single supporting edge was below the weight threshold.

## Result of the full cleanup

Running `python scripts/clean_graph.py` (default settings, `min_weight=2.0`)
on the raw graph:

| Stage | Nodes (before → after) | Edges (before → after) |
|---|---|---|
| Initial | — | — | 35,338 / 34,265 |
| 1. Drop cross-domain | 35,338 → 34,812 (-526) | 34,265 → 33,452 (-813) |
| 2. Drop sentence fragments | 34,812 → 34,238 (-574) | 33,452 → 32,850 (-602) |
| 3. Canonicalise variants | 34,238 → 33,339 (-899; 1,496 merged into existing canonicals) | 32,850 → 32,754 (-96 from dedup/self-loop) |
| 4. Resolve reverse pairs | 33,339 → 33,339 | 32,754 → 32,684 (-70) |
| 5. Drop low-weight edges | 33,339 → 33,339 | 32,684 → 1,699 (-30,985) |
| 6. Drop orphans | 33,339 → **1,597** (-31,742) | 1,699 → 1,699 |

**Final: 1,597 nodes / 1,699 edges** — a 95% reduction.

### Quality indicators after cleanup

- **0 isolated nodes** (was 16)
- **Mean degree 2.13** (was 1.94 — slight improvement; still typical for
  knowledge graphs of this size)
- **All extracted edges have weight ≥ 2** corpus mentions
- **Top hubs are bone-physics central concepts**: `fracture_risk` (deg 161),
  `bone_mineral_density` (84), `bone_resorption` (72), `bone_formation` (43),
  `porosity` (42), `fracture_toughness` (35), `elastic_modulus` (21)
- **Heaviest edges are the seed bone-physics laws**:
  `collagen_crosslink_density → determines → fracture_toughness` (weight 2,036),
  `bone_mineral_density → predicts → fracture_risk` (1,893),
  `porosity → decreases → elastic_modulus` (1,511)
- **Porosity cluster collapsed** from 26 variants → 5 anatomically-meaningful
  ones (cortical, vascular, trabecular, subchondral, micro)
- **776 nodes** reachable from `fracture_risk` within 6 hops (≈49% of
  the graph forms a connected core, vs ~5% in the raw graph)

## Switching the LRM to the cleaned graph

The cleanup writes to a separate file so the original is preserved.

```bash
# Run cleanup
python scripts/clean_graph.py

# Inspect the result
sqlite3 data/db/ontology_clean.db "SELECT COUNT(*) FROM nodes;"

# When you're satisfied, swap it in
mv data/db/ontology.db       data/db/ontology_raw.db
mv data/db/ontology_clean.db data/db/ontology.db
```

The LRM (`reasoning/lrm.py`) loads `ontology.db` via `OntologyStore.load_graph()`
and will pick up the cleaned graph on the next API restart.

## Known residual issues

Even after cleanup, a few classes of issue remain. These are tradeoffs of
the conservative cleanup rules and can be addressed in future passes:

- **BMD anatomical sites still split**: `bmd_at_lumbar_spine`,
  `lumbar_spine_bmd`, `lumbar_bmd`, `spine_bmd` survive as distinct nodes
  because the affix-stripping rules don't merge non-direction variants.
- **A few low-weight relation contradictions remain**: e.g.
  `bone_mineral_density → increases → fracture_risk` (weight 24) coexists
  with `bone_mineral_density → decreases → fracture_risk` (weight 109).
  These are different relations so stage 4 doesn't catch them; the LRM's
  weight-based scoring will favour the correct edge.
- **Some `concept`-typed nodes** that should be `property` or `process`
  remain mislabelled — an LLM-default typing problem the cleanup does not
  address.

These can be fixed in a v2 pass with a small hand-curated canonical
mapping, but the current cleanup is enough for the physics-driven
hypothesis generator to work against.

## Reproducibility

```bash
# Default — strict
python scripts/clean_graph.py

# Looser weight threshold (keep edges with at least 1 mention)
python scripts/clean_graph.py --min-weight 1

# Custom paths
python scripts/clean_graph.py \
    --input  data/db/ontology.db \
    --output data/db/ontology_v2.db \
    --min-weight 3
```

The script is deterministic — re-running it on the same input produces an
identical output DB.
