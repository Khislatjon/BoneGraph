# Phase 4 — LRM Reasoning Layer

> **⚠️ Partially superseded (May 2026).** The reasoning engine described
> below — graph-walk over corpus-extracted nodes with physics as an
> IMPLAUSIBLE-only post-filter — was replaced after audit of the
> LLM-extracted graph (see [`graph_cleanup.md`](graph_cleanup.md)) and
> a full rework of the hypothesis pipeline (see
> [`physics_reasoning.md`](reasoning/physics_grid.md)).
>
> The current Reasoning tab generates hypotheses **from physical laws**
> (Currey, Frost mechanostat, Paris, beam bending) with quantitative
> predictions, then runs them through a four-round adversarial physics
> critic. The cleaned graph is used for grounding evidence and chain
> materialisation, not as the source of hypotheses.
>
> Steps 4.1–4.3 (seed ontology, triple extraction, physics-engine
> implementation) are still accurate and feed the new pipeline.
> Steps 4.4–4.6 (graph-walk reasoning, novelty integration, original
> Reasoning tab) are kept here for historical reference but do not
> match the running system. See [`architecture.md`](architecture.md)
> for the current Phase 4 step list.

**Status: 🔶 In Progress — Steps 4.1–4.6 complete · paper corpus extraction pending**

Move from retrieval to reasoning. A structured bone ontology (knowledge graph) connects concepts causally. The LRM traverses the ontology to construct multi-hop reasoning chains and generate grounded hypotheses — not just summaries of what the literature says.

---

## Motivation

Phases 1–2 built a state-of-the-art retrieval system: 248,629 sentence-aware chunks embedded with SPECTER2, MRR 0.928, Recall@5 1.000. But retrieval has a fundamental ceiling — it can only surface what has already been written. It cannot:

- Construct a causal chain across concepts that no single paper covers end-to-end
- Identify a *gap* in the knowledge graph (a missing edge between two nodes that should be connected)
- Generate a hypothesis that is grounded but novel — i.e. consistent with known bone physics but not explicitly stated in the corpus

Phase 4 adds Layer 2 of the BoneMind architecture: a reasoning engine that operates over a structured bone knowledge graph to do all three.

The concept is directly inspired by two converging ideas:

1. **Unreasonable Labs' "Living World Model"** — the argument that the bottleneck in scientific AI is not access to information but context, causal structure, and the ability to validate hypotheses against physical reality, not just statistical plausibility.

2. **Buehler (MIT, arXiv:2403.11996)** — a working prototype that transforms 1,000 biological materials papers into an ontological knowledge graph and uses graph reasoning (transitive chains, isomorphic mapping, path sampling) to generate novel material designs. BoneMind's corpus is 7× larger, domain-specific to bone, and will add a physics validation layer that Buehler's system lacks.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1 (Phases 2–3)                                               │
│  248,629 chunks · SPECTER2 embeddings · RAG retrieval               │
│  LLaVA VLM reports · cross-modal retrieval · HuatuoGPT answers      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  structured knowledge (triples)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — LRM Reasoning Engine (Phase 4)                           │
│                                                                     │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐   │
│  │  Bone Knowledge Graph│    │  Bone Physics Engine             │   │
│  │                      │    │                                  │   │
│  │  Nodes: concepts     │    │  Scaling laws (E ∝ ρ²)           │   │
│  │  Edges: causal links │    │  Frost mechanostat               │   │
│  │  ~10K–15K nodes      │    │  Fracture mechanics (KIc, Paris) │   │
│  │  Scale-free graph    │    │  Beam theory (long bones)        │   │
│  └──────────┬───────────┘    └────────────┬─────────────────────┘   │
│             │                             │                         │
│             ▼                             ▼                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Reasoning Engine                                            │   │
│  │                                                              │   │
│  │  • Transitive chains:  A→B, B→C  -→  A→C                     │   │
│  │  • Shortest-path traversal between query concepts            │   │
│  │  • Betweenness centrality → research gap detection           │   │
│  │  • Community detection → sub-domain clustering               │   │
│  │  • Cross-domain isomorphism (bone ↔ bio-inspired materials)  │   │
│  └──────────────────────────────────┬───────────────────────────┘   │
│                                     │                               │
│                                     ▼                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Hypothesis Generator + Validator                            │   │
│  │                                                              │   │
│  │  • Sub-graph → LLM prompt → candidate hypothesis             │   │
│  │  • Physics check: is the direction/magnitude plausible?      │   │
│  │  • Novelty check: is this already stated in the corpus?      │   │
│  │  • Output: KNOWN / SPECULATIVE-GROUNDED / IMPLAUSIBLE        │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Example output

A Phase 4 query — *"What is the mechanical consequence of osteocyte lacunar density loss in cortical bone?"* — should produce:

```
[REASONING CHAIN]
osteocyte_lacunar_density ──(acts as)──► stress_concentrators
stress_concentrators ──(elevate)──► local_stress_around_voids
local_stress_around_voids ──(accelerates)──► fatigue_crack_initiation
fatigue_crack_initiation ──(reduces)──► fatigue_life
fatigue_life ──(increases)──► fracture_risk

[HYPOTHESIS]
A 15% reduction in osteocyte lacunar density in cortical bone would redistribute
stress around the remaining lacunae, increasing the local stress concentration
factor and accelerating fatigue crack initiation under cyclic loading.

[PHYSICS CHECK]
Stress concentration factor for a spherical void in an elastic solid: Kt ≈ 3.0.
Consistent with known range for lacunar geometry (Vashishth et al. framework).
Direction: PLAUSIBLE. Magnitude: CONSISTENT WITH LITERATURE RANGE.

[NOVELTY]
The specific 15% threshold is not stated in the corpus.
Related papers: [3 on lacunar mechanics, 2 on cortical fatigue].
Classification: SPECULATIVE / GROUNDED.
```

This is the output that retrieval cannot produce and that a plain LLM without the graph cannot ground.

---

## Step 1 — Bone knowledge graph bootstrap

**Goal:** Seed the graph with ~200 core bone science concepts and their causal relationships before running automated extraction.

### Ontology schema

Nodes represent typed concepts. Edges represent typed, directional relationships.

**Node types:**

| Type | Examples |
|---|---|
| `structure` | cortical_bone, trabecular_bone, collagen_fibril, hydroxyapatite, osteon, osteocyte_lacuna, Haversian_canal |
| `property` | elastic_modulus, fracture_toughness, yield_strength, porosity, mineral_density, collagen_crosslink_density |
| `process` | bone_remodelling, crack_propagation, mineralisation, creep, fatigue, osteoclast_resorption |
| `pathology` | osteoporosis, osteopenia, Paget_disease, osteogenesis_imperfecta, stress_fracture, avascular_necrosis |
| `mechanism` | Wolff_law, Frost_mechanostat, crack_deflection, crack_bridging, stress_shielding |
| `scale` | nanoscale, microscale, mesoscale, macroscale |
| `material` | bone_scaffold, hydroxyapatite_coating, bioglass, PEEK, collagen_sponge |

**Edge types (causal / structural):**

| Relation | Meaning | Example |
|---|---|---|
| `determines` | structural cause of property | `collagen_crosslink_density` → `determines` → `fracture_toughness` |
| `increases` | positive causal influence | `mineralisation` → `increases` → `elastic_modulus` |
| `decreases` | negative causal influence | `porosity` → `decreases` → `yield_strength` |
| `activates` | biological trigger | `mechanical_loading` → `activates` → `bone_remodelling` |
| `leads_to` | consequential relationship | `osteoclast_resorption` → `leads_to` → `trabecular_thinning` |
| `is_part_of` | hierarchical composition | `collagen_fibril` → `is_part_of` → `osteon` |
| `predicts` | clinical/mechanical prediction | `low_BMD` → `predicts` → `fracture_risk` |
| `analogous_to` | cross-domain mapping | `Haversian_system` → `analogous_to` → `fibre_reinforced_composite` |

### Seed concepts (first 50 — starting point)

```python
SEED_NODES = [
    # Structures
    "cortical_bone", "trabecular_bone", "collagen_fibril", "hydroxyapatite_crystal",
    "osteon", "Haversian_canal", "osteocyte_lacuna", "canalicular_network",
    "periosteum", "endosteum", "growth_plate", "bone_marrow",

    # Properties
    "elastic_modulus", "fracture_toughness", "yield_strength", "ultimate_strength",
    "fatigue_life", "bone_mineral_density", "porosity", "anisotropy",
    "viscoelasticity", "toughness", "stiffness", "hardness",

    # Processes
    "bone_remodelling", "mineralisation", "crack_propagation", "fatigue_crack_initiation",
    "creep", "stress_relaxation", "osteoblast_formation", "osteoclast_resorption",

    # Pathologies
    "osteoporosis", "osteopenia", "stress_fracture", "osteogenesis_imperfecta",
    "Paget_disease", "avascular_necrosis", "bone_metastasis",

    # Mechanisms
    "Wolff_law", "Frost_mechanostat", "crack_deflection", "crack_bridging",
    "stress_shielding", "piezoelectric_effect",

    # Materials / biomaterials
    "hydroxyapatite_scaffold", "bioglass", "PEEK_implant", "collagen_membrane",
    "calcium_phosphate_cement", "bone_graft",
]
```

### New files

```
reasoning/
    ontology.py         # Node / Edge dataclasses, GraphBuilder, save/load
    graph_db.py         # SQLite-backed graph persistence (ontology.db)
```

### Database schema (`ontology.db`)

```sql
CREATE TABLE nodes (
    node_id     TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    node_type   TEXT,               -- structure, property, process, pathology, mechanism, material
    description TEXT,
    source      TEXT,               -- 'seed', 'extracted', 'manual'
    created_at  TEXT
);

CREATE TABLE edges (
    edge_id     TEXT PRIMARY KEY,
    source_node TEXT NOT NULL REFERENCES nodes(node_id),
    target_node TEXT NOT NULL REFERENCES nodes(node_id),
    relation    TEXT NOT NULL,      -- determines, increases, decreases, leads_to, ...
    weight      REAL DEFAULT 1.0,   -- confidence / frequency
    evidence    TEXT,               -- chunk_ids that support this edge (JSON list)
    source      TEXT,               -- 'seed', 'extracted'
    created_at  TEXT
);

CREATE INDEX idx_edges_source ON edges(source_node);
CREATE INDEX idx_edges_target ON edges(target_node);
```

---

## Step 2 — Triple extraction from corpus

**Goal:** Use an LLM to extract `(node_1, relation, node_2)` triples from the existing 248,629 chunks in `chunks.db` and populate the knowledge graph.

This is the approach validated by Buehler (arXiv:2403.11996) — applied here to 7× more domain-specific content.

### Extraction pipeline

```python
# reasoning/extractor.py

SYSTEM_PROMPT = """
You are an expert in bone science (mechanics, morphology, pathology, biomaterials).
Extract causal and structural relationships from the text as triples.

Each triple has:
- node_1: a bone science concept (noun phrase, snake_case)
- relation: one of [determines, increases, decreases, activates, leads_to,
                    is_part_of, predicts, analogous_to, inhibits, correlates_with]
- node_2: a bone science concept (noun phrase, snake_case)

Rules:
- Only extract relationships explicitly stated or strongly implied in the text.
- Use consistent, canonical node names (e.g. always "elastic_modulus" not "Young's modulus").
- Return 5–10 triples per chunk as a JSON array.
- If no clear relationships exist, return an empty array.
"""

USER_PROMPT = """
Text: "{chunk_text}"

Extract bone science triples as JSON:
[{{"node_1": "...", "relation": "...", "node_2": "..."}}]
"""
```

### Extraction strategy

1. **Prioritise high-quality chunks** — process chunks from textbooks and highly-cited papers first (quality signal before volume)
2. **Batch processing** — process 10 chunks per LLM call to reduce API overhead
3. **Deduplication** — edges are merged by `(source_node, relation, target_node)` with weight = frequency count
4. **Evidence tracking** — each edge stores the chunk IDs that support it

```bash
python -m reasoning.extract_triples          # extract from all chunks (resumable)
python -m reasoning.extract_triples --limit 5000   # test run on 5,000 chunks
python -m reasoning.extract_triples --source textbooks  # textbooks only (highest quality)
```

### Expected graph size

Based on Buehler's result (1,000 papers → 12,319 nodes / 15,752 edges) and our corpus size:
- Bone-specific focus will reduce concept diversity but increase edge density per node
- Estimated: **8,000–15,000 nodes · 12,000–20,000 edges** after deduplication
- Scale-free structure expected (few hub nodes like "bone_mineral_density", "fracture_toughness")

---

## Step 3 — Bone physics engine

**Goal:** A lightweight, domain-specific physics layer that can evaluate whether a generated hypothesis is physically plausible. This is the differentiator over Buehler's system.

### Implemented relationships

```python
# reasoning/physics.py

class BonePhysicsEngine:

    # Currey's law: elastic modulus from apparent density
    # E (GPa) = a * rho^b  where a=6.95, b=1.49 for cortical (Currey 1988)
    def elastic_modulus_from_density(self, rho_g_cm3: float, bone_type="cortical") -> float: ...

    # Yield strength from modulus (empirical regression)
    # sigma_y (MPa) ≈ 0.0824 * E^1.0 (Kopperdahl & Keaveny 1998)
    def yield_strength_from_modulus(self, E_GPa: float) -> float: ...

    # Stress concentration factor for spherical void (Kirsch, elastic)
    # Kt = 1 + 2*(1 - 2*nu) / (2 - nu)  for uniaxial loading  ≈ 2.0–3.0
    def stress_concentration_void(self, nu_poisson: float = 0.3) -> float: ...

    # Fracture toughness from porosity (power law)
    # KIc decreases ~linearly with porosity increase in cortical bone
    def fracture_toughness_from_porosity(self, porosity_fraction: float) -> float: ...

    # Frost mechanostat: strain threshold for adaptive remodelling
    # <1000 με  → disuse/resorption
    # 1000–3000 με → homeostasis
    # >3000 με  → modelling / hypertrophy
    # >25000 με → pathological overload / fracture
    def mechanostat_zone(self, microstrain: float) -> str: ...

    # Paris law: fatigue crack growth rate
    # da/dN = C * (ΔK)^m  where C≈2.4×10⁻¹², m≈4.2 for cortical bone
    def fatigue_crack_growth_rate(self, delta_K: float) -> float: ...

    # Long bone bending: maximum stress from 3-point bending
    # sigma_max = M * c / I  (standard beam theory)
    def bending_stress(self, moment_Nm: float, outer_r_m: float, inner_r_m: float) -> float: ...
```

The physics engine does not need to be numerically precise — it needs to evaluate *direction* and *order of magnitude* for hypothesis plausibility checks.

---

## Step 4 — Reasoning engine

**Goal:** Graph traversal and LLM-over-subgraph for multi-hop reasoning, hypothesis construction, and research gap detection.

### Core operations

```python
# reasoning/lrm.py

class BoneLRM:

    def query(self, question: str) -> ReasoningResult:
        """
        Main entry point. Given a natural language question:
        1. Extract key concepts from the question
        2. Locate them in the knowledge graph
        3. Extract the connecting sub-graph
        4. Generate a reasoning chain
        5. Run physics plausibility check
        6. Check novelty against corpus
        7. Return structured result
        """

    def reasoning_chain(self, concept_a: str, concept_b: str) -> list[Edge]:
        """
        Find and return the shortest causal path from concept_a to concept_b.
        Uses Dijkstra on edge weights (weight = 1/frequency, so
        well-supported edges are preferred).
        """

    def detect_gaps(self, top_k: int = 20) -> list[GapReport]:
        """
        Find nodes with high betweenness centrality that have sparse
        inbound/outbound edges — these are under-studied bridge concepts.
        Returns ranked list of research gap candidates.
        """

    def generate_hypothesis(self, subgraph: nx.DiGraph) -> Hypothesis:
        """
        Prompt the LLM with a sub-graph rendered as a structured context.
        Returns a Hypothesis with: statement, reasoning_chain,
        physics_check, novelty_label, confidence, supporting_chunk_ids.
        """

    def cross_domain_analogy(self, bone_concept: str, target_domain: str) -> Analogy:
        """
        Find the isomorphic sub-graph in target_domain (e.g. 'bio_inspired_materials',
        'structural_engineering') that maps to the bone_concept neighbourhood.
        Useful for biomaterials design predictions.
        """
```

### Sub-graph context format fed to LLM

Rather than dumping the whole graph into the prompt, the engine extracts a targeted sub-graph and renders it as structured text:

```
BONE KNOWLEDGE SUB-GRAPH
========================
Query concepts: osteocyte_lacuna, fracture_risk

Causal chain (shortest path):
  osteocyte_lacuna  ──[acts_as]──►  stress_concentrator  (evidence: 14 chunks)
  stress_concentrator  ──[elevates]──►  local_stress  (evidence: 9 chunks)
  local_stress  ──[accelerates]──►  fatigue_crack_initiation  (evidence: 21 chunks)
  fatigue_crack_initiation  ──[reduces]──►  fatigue_life  (evidence: 17 chunks)
  fatigue_life  ──[increases]──►  fracture_risk  (evidence: 31 chunks)

Neighbouring nodes (degree-1):
  osteocyte_lacuna: is_part_of cortical_bone · connected_to canalicular_network
  fracture_risk: predicted_by bone_mineral_density · predicted_by cortical_thickness

Physics constraints:
  Kt (spherical void) ≈ 3.0 | fatigue crack threshold ΔK₀ ≈ 1.5 MPa√m

Generate a precise bone science hypothesis connecting these concepts.
Flag it as KNOWN, SPECULATIVE-GROUNDED, or IMPLAUSIBLE.
```

---

## Step 5 — Novelty classifier

**Goal:** Determine whether a generated hypothesis is already stated in the corpus (KNOWN), logically grounded but unstated (SPECULATIVE-GROUNDED), or contradicted by the literature (IMPLAUSIBLE).

### Method

```python
# reasoning/novelty.py

class NoveltyClassifier:

    def classify(self, hypothesis_text: str) -> NoveltyResult:
        """
        1. Embed the hypothesis with SPECTER2 (adhoc_query adapter)
        2. Retrieve top-5 most similar chunks from chunks.db
        3. If max cosine similarity > 0.92: KNOWN (essentially stated)
        4. If max cosine similarity > 0.75: RELATED (consistent but novel framing)
        5. If max cosine similarity < 0.75: NOVEL
        6. Cross-check against physics engine for IMPLAUSIBLE flag
        Returns: label, similarity score, supporting chunk IDs
        """
```

This reuses the SPECTER2 embeddings already computed in Phase 2 — no additional computation needed.

---

## Step 6 — UI integration

Add a sidebar tab to `frontend/index.html` — **"Reasoning"** — alongside Ask, Search, and Vision.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Ask BoneMind  │  Search BoneScholar  │  Reasoning  │  Vision  │     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Question or hypothesis to reason about:                             │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │ How does increased cortical porosity affect fracture risk? │      │
│  └────────────────────────────────────────────────────────────┘      │
│  [ 🧠 Reason ]                                                       │
│                                                                      │
├──────────────────────────────┬───────────────────────────────────────┤
│  Reasoning chain             │  Generated hypothesis                 │
│  (graph path visualisation)  │  + physics check                      │
│                              │  + novelty label                      │
│                              │  + supporting chunks                  │
├──────────────────────────────┴───────────────────────────────────────┤
│  Research gaps in this area (betweenness centrality analysis)        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Step 7 — Evaluation

### Benchmark design (`eval/benchmark_lrm.json`)

Three evaluation tasks:

**Task A — Reasoning chain correctness (20 pairs)**
Given `(concept_a, concept_b)`, does the graph produce a biologically valid causal path?
Scored by a domain expert against a reference chain.

**Task B — Hypothesis novelty vs. validity (20 questions)**
Does the system generate a hypothesis that is:
- Physically plausible (passes physics check)
- Not already verbatim in the corpus (novelty score)
- Consistent with what a bone scientist would consider reasonable

**Task C — Research gap detection (10 sub-domains)**
Does betweenness centrality correctly identify known under-studied topics in bone science?
Ground truth: manually identified gaps from review papers.

### Metrics

| Metric | What it measures |
|---|---|
| Chain validity rate | % of reasoning chains rated valid by domain expert |
| Hypothesis plausibility rate | % of hypotheses that pass the physics engine check |
| Novelty rate | % of hypotheses classified SPECULATIVE-GROUNDED (not KNOWN) |
| Gap precision@5 | Do the top-5 betweenness gaps match known under-studied topics? |

```bash
python eval/run_eval_lrm.py        # runs all three task benchmarks
```

---

## Repository structure (as built)

```
reasoning/
    __init__.py
    ontology.py          # Node/Edge dataclasses, GraphBuilder, NetworkX wrappers, NODE_TYPES
    graph_db.py          # SQLite-backed persistence + extraction_progress tracking
    seed.py              # ~200 seed concepts + ~80 hand-curated causal edges
    extractor.py         # LLM triple extraction pipeline from chunks.db (resumable)
    physics.py           # Bone physics engine: 40 guard-rail rules (IMPLAUSIBLE filter) + 5 numerical laws
    lrm.py               # Core reasoning engine: anchor → traverse → score → summarise
    novelty.py           # Novelty classifier: keyword Tier 1 + SPECTER2 Tier 2

data/db/
    ontology.db          # Knowledge graph: 3,001 nodes · 2,339 edges (after textbook extraction)

eval/
    benchmark_lrm.json   # 50-item benchmark (pending)
    run_eval_lrm.py      # Evaluation runner (pending)
    results_lrm.json     # Latest results (pending)

docs/
    phase4_lrm_plan.md   # This file
```

### Physics engine implementation (`reasoning/physics.py`)

**Layer 1 — Directional guard-rail rules (40 rules)**

Rules are stored as `(src_fragment, relation, tgt_fragment) → (status, law, explanation)`. Node matching uses substring lookup so extracted node names like `"cortical_bone_porosity"` match the rule fragment `"porosity"`.

Rules act as **guard rails only**: an `IMPLAUSIBLE` match on any edge causes the entire chain to score 0 and be filtered out. There is no `PLAUSIBLE` reward — chains that pass (or are not covered by any rule) are scored identically. This avoids systematic bias toward topics with many encoded rules.

Example IMPLAUSIBLE guard rails:
```python
("porosity",   "increases", "elastic_modulus"): ("IMPLAUSIBLE", "Currey's law",              "..."),
("OPG",        "activates", "osteoclast"):      ("IMPLAUSIBLE", "RANK/RANKL/OPG signalling", "..."),
("sclerostin", "activates", "bone_formation"):  ("IMPLAUSIBLE", "Wnt/β-catenin signalling",  "..."),
("disuse",     "leads_to",  "bone_formation"):  ("IMPLAUSIBLE", "Frost mechanostat",         "..."),
```

**Chain validation**

A chain is immediately rejected (score = 0) if any single edge matches an IMPLAUSIBLE rule. All other chains pass through to scoring unchanged — PLAUSIBLE and UNCERTAIN are treated identically.

**Layer 2 — Numerical functions**

| Function | Law | Formula |
|---|---|---|
| `check_currey(rho)` | Currey's law | E = 7·ρ² (GPa, ρ in g/cm³) |
| `check_mechanostat(με)` | Frost mechanostat | 6 zones: disuse · remodelling · homeostasis · modelling · overload · fracture |
| `check_paris(ΔK)` | Paris crack growth | da/dN = 1.7×10⁻⁹ · ΔK³·⁹ |
| `check_beam_bending(M, c, I)` | Euler–Bernoulli beam | σ = Mc/I |
| `check_stress_concentration(a, ρ)` | Inglis / Kirsch | Kt = 1 + 2√(a/ρ) |

### Novelty classifier implementation (`reasoning/novelty.py`)

**Tier 1 — Keyword search**

Tokenises the hypothesis into keywords, runs `LIKE` queries against `chunks.db`. Thresholds: ≥5 chunk hits → GROUNDED; 1–4 → SPECULATIVE; 0 → NOVEL.

**Tier 2 — SPECTER2 semantic similarity**

Embeds the hypothesis with SPECTER2 adhoc_query adapter. Computes cosine similarity against 8,000 randomly sampled L2-normalised corpus embeddings. Thresholds: ≥0.82 → GROUNDED; 0.60–0.82 → SPECULATIVE; <0.60 → NOVEL.

**Merge rule:** take the label corresponding to the higher similarity score (more conservative = less likely to overclaim novelty).

**SPECTER2 model sharing:** the model, tokenizer, and device are passed in from the retriever at startup so the 1.6 GB model is loaded only once.

**Corpus disclaimer:** shown only for NOVEL results:
> ⚠️ Novelty is assessed against our open-access corpus only — some results may already appear in paywalled literature.

---

## Execution order

| Step | Work | Status | Output |
|---|---|---|---|
| 4.1 | Seed ontology with ~200 bone concepts + hand-curated edges | ✅ Done | `ontology.db` populated |
| 4.2 | Triple extraction on textbooks (highest quality) | ✅ Done | 2,935 triples · 3,001 nodes · 2,339 edges |
| 4.3 | Physics engine — directional guard-rail rules + numerical laws | ✅ Done | 40 rules · 5 numerical functions |
| 4.4 | LRM reasoning engine — anchor, traverse, score | ✅ Done | Multi-hop chains · gap detection |
| 4.5 | Novelty classifier — keyword + SPECTER2 semantic tiers | ✅ Done | GROUNDED / SPECULATIVE / NOVEL |
| 4.6 | Gradio "Reason" tab integration | ✅ Done | End-to-end UI with physics + novelty badges |
| 4.7 | Triple extraction on full papers corpus | ⏳ Pending | ~50,000–100,000 additional triples |
| 4.8 | LRM benchmark evaluation, fix gaps | ⏳ Pending | `results_lrm.json` |

### Actual extraction results (textbooks, April 2026)

| Metric | Value |
|---|---|
| Textbook chunks processed | 1,983 / 1,983 |
| Chunks with extracted triples | 727 (37%) |
| Empty chunks (figures, TOC, captions) | 1,256 (63%) |
| Raw triples extracted | 2,935 |
| Unique nodes after deduplication | 3,001 |
| Unique edges after deduplication | 2,339 |
| Extraction speed | ~8.4 s/chunk (CPU-only, huatuogpt-bone = HuatuoGPT-o1-8B + custom prompt) |

---

## Novel contributions (for the paper)

1. **First domain-specific reasoning graph for bone science** — Buehler's GraphReasoning system covers biological materials broadly. BoneMind builds a bone-only graph from a 7× larger corpus with typed causal edges and quantitative node attributes (units, ranges, literature values), not just relational triples.

2. **Physics-validated hypothesis generation** — The key gap in all existing graph-based AI discovery systems (including Buehler 2024) is the absence of physical plausibility checking. BoneMind's physics engine filters hypotheses against known bone mechanics (Currey's law, Frost's mechanostat, Paris law) before they are presented — categorically reducing hallucination risk for bone science claims.

3. **Multi-scale causal reasoning** — The ontology explicitly encodes scale (nanoscale → microscale → mesoscale → macroscale) as a node attribute, enabling the LRM to construct reasoning chains that cross hierarchical levels. This is distinct from flat knowledge graphs used in prior work.

4. **Cross-modal grounding** — Phase 3 VLM reports are embedded in the same SPECTER2 space as text chunks. Phase 4 can therefore ground a hypothesis not only in the text corpus but also in visual evidence (X-ray / MRI findings), making it the first reasoning system that integrates radiological image understanding with causal graph reasoning in bone science.

5. **Research gap detection as a first-class output** — Betweenness centrality analysis on the bone graph surfaces structurally under-studied bridge concepts — providing a systematic method for research agenda setting rather than relying on expert intuition.

---

## References

- Buehler, M.J. (2024). *Accelerating Scientific Discovery with Generative Knowledge Extraction, Graph-Based Representation, and Multimodal Intelligent Graph Reasoning.* arXiv:2403.11996
- Code reference: https://github.com/lamm-mit/GraphReasoning
- Currey, J.D. (1988). The effect of porosity and mineral content on the Young's modulus of bone. *Journal of Biomechanics*, 21(2), 131–139.
- Frost, H.M. (1987). Bone "mass" and the "mechanostat." *Anatomical Record*, 219(1), 1–9.
- Vashishth, D. et al. (2000). Influence of nonenzymatic glycation on biomechanical properties of cortical bone. *Bone*, 28(2), 195–201.
- Kopperdahl, D.L. & Keaveny, T.M. (1998). Yield strain behavior of trabecular bone. *Journal of Biomechanics*, 31(7), 601–608.
- Unreasonable Labs. (2026). *Why We Are Building Unreasonable Labs.* https://www.unreasonablelabs.ai/news/why-we-are-building-unreasonable-labs
