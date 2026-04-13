# BoneLogic — System Architecture

## What is BoneLogic?

BoneLogic is an intelligent reasoning system for bone science. It is deliberately restricted to the bone domain because of the richness of available data: published papers, textbooks, mechanical measurements, X-ray and MRI images.

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

BoneLogic is intentionally restricted to bone science. This is not a limitation — it is a design decision that enables depth over breadth.

| Domain | What BoneLogic learns |
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
| Phase 1b — Textbook ingestion | ✅ Complete (April 2026) | [textbooks_ingestion_pipeline.md](textbooks_ingestion_pipeline.md) |
| Phase 2 — Text processing & RAG | ✅ Complete (April 2026) | [phase2_rag_pipeline.md](phase2_rag_pipeline.md) |
| Phase 3 — VLM integration | ⏳ Planned | [phase3_vlm_plan.md](phase3_vlm_plan.md) |
| Phase 4 — LRM reasoning layer | ⏳ Planned | — |
| Phase 5 — Feedback loop | ⏳ Planned | — |

---

## Phase summaries

### Phase 1 — Data ingestion ✅
Collect the knowledge base. 54,634 papers from OpenAlex · 7,674 PDFs · 16 textbooks. 2-pass PDF download pipeline (OpenAlex OA URLs + CrossRef + Wiley TDM).

### Phase 2 — Text processing & RAG ✅
Extract, chunk, and embed the text corpus. 248,629 sentence-aware chunks embedded with SPECTER2 (768-dim). RAG retrieval via cosine similarity. HuatuoGPT-o1-8B answers via Ollama. Retrieval benchmark: MRR 0.928, Recall@5 1.000.

### Phase 3 — VLM integration ⏳
Add image understanding for X-ray and MRI inputs. LLaVA 1.6 generates structured radiological reports; reports are embedded with SPECTER2 and used to retrieve relevant literature from the text corpus (cross-modal retrieval in a shared embedding space). New "Analyse Image" tab in the web UI.

### Phase 4 — LRM reasoning layer ⏳
Move from retrieval to reasoning. A structured bone ontology (knowledge graph) connects concepts causally. The LRM traverses the ontology to construct reasoning chains and generate grounded hypotheses — not just summaries.

### Phase 5 — Feedback loop ⏳
Make the system improve with use. User feedback (corrections, confirmations) updates the ontology, refining retrieval and reasoning in subsequent queries.

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
BoneLogic/
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
│   ├── retriever.py             BoneLogicRetriever — loads all embeddings, cosine search
│   └── query.py                 CLI entrypoint (single query + interactive mode)
│
├── eval/                        Retrieval quality benchmarks
│   ├── benchmark.json           30 questions across 7 domains with expected keywords
│   ├── run_eval.py              Computes MRR and Recall@k · saves results.json
│   └── results.json             Latest benchmark results
│
├── models/                      Phases 3-4: LLM, VLM, LRM wrappers (coming in Phase 3)
├── reasoning/                   Phase 4: ontology and LRM (coming in Phase 4)
├── api/                         Phase 5: user-facing interface (coming in Phase 5)
│
├── data/
│   ├── raw/papers/              Downloaded PDFs — 7,674 files (gitignored)
│   ├── raw/textbooks/           Textbook PDFs by source (gitignored)
│   ├── processed/text/          Extracted .txt files (gitignored)
│   └── db/
│       ├── papers.db            54,634 paper metadata rows (gitignored)
│       ├── textbooks.db         16 textbook metadata rows (gitignored)
│       └── chunks.db            248,629 chunks + SPECTER2 embeddings (gitignored)
│
├── docs/
│   ├── architecture.md                    ← this file (high-level overview)
│   ├── papers_ingestion_pipeline.md       Phase 1 — papers ingestion deep-dive
│   ├── textbooks_ingestion_pipeline.md    Phase 1b — textbooks ingestion deep-dive
│   ├── phase2_rag_pipeline.md             Phase 2 — text processing, RAG, evaluation
│   └── phase3_vlm_plan.md                 Phase 3 — VLM integration plan
│
├── tests/
│   └── test_ingestion.py
│
├── app.py                       Gradio web UI (http://localhost:7860)
├── .env                         API keys (gitignored — never commit)
├── .env.example                 Template showing which keys are needed
└── requirements.txt
```
