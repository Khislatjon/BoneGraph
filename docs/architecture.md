# BoneGraph — System Architecture

## What is BoneGraph?

BoneGraph is an intelligent reasoning system for bone science. It is deliberately restricted to the bone domain because of the richness of available data: published papers, textbooks, mechanical measurements, X-ray and MRI images.

The goal is not to build a simple question-answering chatbot. The goal is to build a system that can:

1. **Understand** bone science at the level of morphology, structure-function relationships, mechanics, and pathology.
2. **Reason** over that knowledge to answer clinical and research questions.
3. **Generate hypotheses** — propose new ideas about bone behaviour, bio-inspired materials, or biomechanical predictions that are not simply recalled from training data.
4. **Improve** over time as users interact with it and provide feedback.

---

## System architecture (2 layers)

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Perception & Understanding                               │
│                                                                     │
│  ┌──────────────────────────┐   ┌──────────────────────────────┐    │
│  │  LLM                     │   │  VLM                         │    │
│  │  (text understanding)    │   │  (image understanding)       │    │
│  │                          │   │                              │    │
│  │  Input: abstracts,       │   │  Input: X-ray, MRI images    │    │
│  │  papers, textbooks,      │   │                              │    │
│  │  clinical notes          │   │  Output: structured visual   │    │
│  │                          │   │  descriptions (density,      │    │
│  │  Output: embeddings,     │   │  fracture lines, erosions…)  │    │
│  │  structured knowledge    │   │                              │    │
│  └──────────────────────────┘   └──────────────────────────────┘    │
│                      │                        │                     │
│                      └──────────┬─────────────┘                     │
│                                 │                                   │
│                    Fused multimodal representation                  │
│                                 │                                   │
├─────────────────────────────────┼───────────────────────────────────┤
│  LAYER 2 — Reasoning & Hypothesis Generation                        │
│                                 │                                   │
│  ┌──────────────────────────────▼──────────────────────────────┐    │
│  │  LRM (Large Reasoning Model)                                │    │
│  │                                                             │    │
│  │  • Structured bone ontology (knowledge graph)               │    │
│  │  • Causal inference: structure → function → mechanics       │    │
│  │  • Hypothesis formulation (beyond recall)                   │    │
│  │  • Bio-inspired material prediction                         │    │
│  │  • Feedback-driven ontology updates                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

**Layer 1** handles perception — converting raw text and images into structured, machine-readable representations of bone knowledge.

**Layer 2** handles reasoning — operating over those representations to answer questions, make connections between concepts, and generate new hypotheses that are constrained by the ontology (so they are grounded, not hallucinated).

---

## Domain scope

BoneGraph is intentionally restricted to bone science. This is not a limitation — it is a design decision that enables depth over breadth.

| Domain | What BoneGraph learns |
|---|---|
| **Morphology** | Shape, size, internal geometry of different bones; cortical/trabecular organisation; osteocyte lacunar networks; Haversian systems |
| **Structure-function** | How hierarchical structure (from nanoscale collagen-mineral to macroscale geometry) determines mechanical behaviour |
| **Mechanics** | Elastic modulus, fracture toughness, fatigue, viscoelasticity, crack propagation, finite element models |
| **Pathology** | Osteoporosis, Paget's disease, osteogenesis imperfecta, bone metastasis, avascular necrosis, stress fractures |
| **Imaging** | Interpreting X-ray and MRI findings; relating radiological appearance to underlying structural changes |
| **Biomaterials** | Bone scaffolds, grafts, tissue engineering, 3D-printed bone substitutes |
| **Simulation** | Computational modelling, multiscale FEA, bone remodelling models, molecular dynamics of mineral |

---

## Build phases

| Phase | Status | Detail |
|---|---|---|
| Phase 1 — Paper ingestion | ✅ Complete (March 2026) | [papers_ingestion_pipeline.md](papers_ingestion_pipeline.md) |
| Phase 1b — Textbook ingestion | ✅ Complete (April 2026) | [textbooks_ingestion_pipeline.md](chat/textbooks_ingestion_pipeline.md) |
| Phase 2 — Text processing & RAG | ✅ Complete (April 2026) | [phase2_rag_pipeline.md](phase2_rag_pipeline.md) |
| Phase 3 — VLM integration | 🔶 Partial (April 2026) | LLaVA 1.6 tab live · cross-modal retrieval pending · [phase3_vlm_plan.md](vision/phase3_vlm_plan.md) |
| Phase 4 — Bone knowledge graph | ✅ Complete (May 2026) | Seed ontology · triple extraction · cleanup → 1,597 / 1,699 · [phase4_lrm_plan.md](reasoning/phase4_lrm_plan.md) |
| Phase 4b — Reasoning tab (rebuild) | 🟢 Live (May 2026) | Agent + critic loop · physical grounding · feedback-driven user rules · [reasoning_tab.md](reasoning/reasoning_tab.md) |
| Phase 5 — Feedback loop | 🟢 Live for the Reasoning tab (May 2026); other tabs pending | Reasoning-tab feedback flow documented in [reasoning_tab.md](reasoning/reasoning_tab.md) |

---

## Phase summaries

### Phase 1 — Data ingestion ✅
Collect the knowledge base. 54,634 papers from OpenAlex · 7,674 PDFs · 16 textbooks. 2-pass PDF download pipeline (OpenAlex OA URLs + CrossRef + Wiley TDM).

### Phase 2 — Text processing & RAG ✅
Extract, chunk, and embed the text corpus. 248,629 sentence-aware chunks embedded with SPECTER2 (768-dim). RAG retrieval via cosine similarity. HuatuoGPT-o1-8B (served locally via Ollama as `huatuogpt-bone` — HuatuoGPT-o1-8B with a custom bone science system prompt) answers questions. Retrieval benchmark: MRR 0.928, Recall@5 1.000. Multi-turn chat supported via a sliding window (last 3 question/answer pairs) with thinking-block and References stripping to keep prompts within the 8,192-token context window. References section always sorted in ascending `[N]` order.

### Phase 3 — VLM integration 🔶 Partial
Add image understanding for X-ray and MRI inputs. LLaVA 1.6 "Analyse Image" tab is live in both the React UI (`frontend/index.html`) and the legacy Gradio UI — users can upload an image and receive a structured radiological report. Full cross-modal retrieval (reports embedded with SPECTER2 and used to search the text corpus) is planned but not yet implemented.

### Phase 4 — Bone knowledge graph ✅
Build the structured knowledge layer the Reasoning tab draws on. Steps 4.1–4.3 are complete:

- **4.1 — Seed ontology**: ~200 bone science concepts and ~80 hand-curated causal edges bootstrapped into `ontology.db`.
- **4.2 — Triple extraction**: `huatuogpt-bone` (HuatuoGPT-o1-8B with a custom bone science system prompt, via Ollama) extracts `(node_1, relation, node_2)` triples from corpus chunks into the knowledge graph. Two extraction runs completed:
  - **Textbooks** (April 2026): 1,983 chunks → 2,935 triples → 3,001 nodes, 2,339 edges
  - **Full paper corpus** (April 2026): 17,381 chunks attempted (9,258 yielded triples · 8,120 empty · 3 failed) · 2,310 min runtime → 41,359 triples → **35,338 nodes · 34,265 edges** (raw combined graph)
- **4.3 — Graph cleanup** (`scripts/clean_graph.py` + [`reasoning/graph_cleanup.md`](reasoning/graph_cleanup.md)): six-stage cleanup of the LLM-extracted graph (cross-domain filter, sentence-fragment filter, affix canonicalisation, reverse-pair resolution, low-weight edge drop, orphan removal). 35,338 / 34,265 → **1,597 / 1,699** nodes / edges. Followed by a rule-based concept reclassification pass (`scripts/reclassify_concepts.py` + [`reasoning/graph_concept_reclassification.md`](reasoning/graph_concept_reclassification.md)) that retypes 571 of the 1,200 `concept`-typed nodes into their correct ontology types.
The cleaned `ontology.db` (1,597 nodes / 1,699 edges) is the graph the live
Reasoning tab's critic reads from. An earlier equation-graph reasoner and a
Proposer/Critic hypothesis-generation agent loop were built on top of this graph
and **retired in May 2026** in favour of the clean-slate Reasoning tab below;
their code and design docs have been removed (recoverable via git history).

### Phase 4b — Reasoning tab (clean-slate rebuild) 🟢
Following the 21 May supervision direction, the Reasoning tab was rebuilt around three pillars: an agentic reasoning + critic loop, a deterministic fracture-scoped physical-grounding filter, and a user-feedback rule registry. Endpoints: `/api/reason/chat`, `/api/reason/feedback`, `/api/reason/rules/*`. Full architecture, data model, API surface, and demo flow in [`reasoning_tab.md`](reasoning/reasoning_tab.md).

### Phase 5 — Feedback loop 🟢 (Reasoning tab) · ⏳ (other tabs)
Make the system improve with use. **For the Reasoning tab** this is live: thumbs-down + free-text corrections feed an LLM rule extractor whose proposals the user confirms into a personal SQLite-backed rule registry, merged into the physical-grounding check on every future request. See [`reasoning_tab.md`](reasoning/reasoning_tab.md) §"Feedback loop". For the Chat and Vision tabs, a feedback channel into the ontology / retrieval is still planned.

---

## Key design principles

**1. Papers are never shown by default.**
Responses should read like a knowledgeable colleague, not a literature search. References are surfaced only when explicitly requested.

**2. Hypothesis generation, not just recall.**
Every response has the potential to contain a hypothesis marker — a structured claim that goes beyond the training data, flagged as speculative but grounded in the ontology.

**3. Domain boundary enforcement.**
Queries outside bone science are politely redirected. This is what makes the system authoritative within its domain.

**4. Modular, replaceable components.**
The LLM, VLM, and LRM are separate modules with defined interfaces. Each can be upgraded independently as better models become available.

---

## Repository layout

```
BoneGraph/
│
├── config/                      Phase-independent settings
│   └── settings.py              All constants, paths, API config
│
├── ingestion/                   Phase 1: data collection
│   ├── papers/
│   │   ├── keywords.py          133 bone-domain search queries (17 groups)
│   │   ├── openalex.py          OpenAlex API client (free, no key required)
│   │   ├── storage.py           SQLite metadata store
│   │   ├── downloader.py        2-pass PDF downloader (OpenAlex + CrossRef + Wiley TDM)
│   │   └── pipeline.py          CLI entrypoint
│   └── textbooks/
│       ├── storage.py           SQLite textbook store
│       ├── scanner.py           Folder scanner + PyMuPDF metadata extraction
│       └── pipeline.py          CLI entrypoint
│
├── scripts/
│   ├── inspect_db.py            Database statistics + progress inspector
│   ├── download_pass1.py        Pass 1: OpenAlex OA URLs + CrossRef
│   ├── download_pass2.py        Pass 2: DOI → CrossRef retry
│   └── filter_english.py        Two-pass language filter (body-text detection, skips abstract)
│
├── processing/                  Phase 2: text extraction, chunking, embedding
│   ├── extractor.py             PDF → plain text via PyMuPDF (strips reference sections)
│   ├── extract_papers.py        Extract all paper PDFs → .txt files
│   ├── extract_textbooks.py     Extract all textbook PDFs → .txt files
│   ├── chunker.py               Sentence-aware chunker (NLTK · 400 tokens · 2-sentence overlap)
│   ├── chunk_all.py             Chunk English papers + textbooks → chunks.db
│   └── embed.py                 Embed chunks with SPECTER2 proximity adapter → chunks.db
│
├── retrieval/                   Phase 2: RAG retrieval engine + CLI/web query interface
│   ├── retriever.py             BoneGraphRetriever — loads all embeddings, cosine search
│   └── query.py                 CLI entrypoint (single query + interactive mode)
│
├── reasoning/                   Phase 4: bone knowledge graph + the live Reasoning tab
│   ├── __init__.py
│   ├── ontology.py              Node/Edge dataclasses, GraphBuilder, NetworkX wrappers
│   ├── graph_db.py              SQLite-backed graph persistence + extraction progress tracking
│   ├── seed.py                  ~200 seed concepts + ~80 hand-curated causal edges
│   ├── extractor.py             LLM triple extraction from chunks.db (resumable, huatuogpt-bone = HuatuoGPT-o1-8B + custom prompt)
│   ├── visualize_ontology.py    Render ontology.db to an interactive graph
│   ├── physical_grounding.py    Fracture-scoped Tier-1 rules + Tier-2 user-rule compiler
│   ├── feedback_store.py        SQLite store: feedback events, corrections, user rules
│   ├── rule_extractor.py        llama3.2:3b extraction of a structured rule from feedback
│   ├── rule_import.py           Bulk CSV/XLSX user-rule import
│   └── kg_context.py            1-hop knowledge-graph facts fed to the critic
│
├── eval/                        Retrieval quality benchmarks
│   ├── benchmark.json           30 questions across 7 domains with expected keywords
│   ├── run_eval.py              Computes MRR and Recall@k · saves results.json
│   └── results.json             Latest benchmark results
│
├── models/                      Phases 3-4: LLM, VLM, LRM wrappers
│
├── data/
│   ├── raw/papers/              Downloaded PDFs — 7,674 files (gitignored)
│   ├── raw/textbooks/           Textbook PDFs by source (gitignored)
│   ├── processed/text/          Extracted .txt files (gitignored)
│   └── db/
│       ├── papers.db            54,634 paper metadata rows (gitignored)
│       ├── textbooks.db         16 textbook metadata rows (gitignored)
│       ├── chunks.db            248,629 chunks + SPECTER2 embeddings (gitignored)
│       └── ontology.db          Knowledge graph: 35,338 nodes · 34,265 edges (gitignored)
│
├── docs/                                   one folder per tab + shared top-level docs
│   ├── architecture.md                    ← this file (system-wide overview)
│   ├── papers_ingestion_pipeline.md       shared — papers ingestion (feeds Chat + Search)
│   ├── textbooks_ingestion_pipeline.md    shared — textbooks ingestion (feeds Chat + Search)
│   ├── phase2_rag_pipeline.md             shared — text processing, RAG, evaluation
│   ├── chat/                              Chat tab (RAG question-answering)
│   ├── search/                            Search tab (semantic corpus search)
│   ├── reasoning/                         Reasoning tab (agent + critic + grounding + feedback)
│   │   ├── architecture.md                  authoritative pipeline reference
│   │   ├── reasoning_tab.md                 build log + demo flow
│   │   ├── evidence_layer.md · consistency_bench.md · feedback_demo.md
│   │   ├── phase4_lrm_plan.md               knowledge-graph background
│   │   └── graph_cleanup.md · graph_concept_reclassification.md
│   └── vision/                            Vision tab (image understanding)
│       └── phase3_vlm_plan.md               VLM integration plan
│
├── tests/
│   └── test_ingestion.py
│
├── api/                         FastAPI backend
│   ├── __init__.py
│   └── main.py                  Endpoints: /api/ask · /api/search · /api/reason · /api/analyse · /api/gaps · /api/stats
│
├── frontend/                    React web UI
│   ├── index.html               Single-file React app (Babel in-browser transpilation)
│   └── static/                  React, ReactDOM, Babel bundles (vendored)
│
├── serve.py                     Uvicorn launcher — FastAPI at http://localhost:8000
├── .env                         API keys (gitignored — never commit)
├── .env.example                 Template showing which keys are needed
└── requirements.txt
```
