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

The system is built incrementally. Each phase delivers a working product, and the next phase extends it.

### Phase 1 — Data ingestion (current)
Collect the knowledge base that will ground the LLM and LRM.

```
Semantic Scholar API  →  paper metadata (SQLite)  →  PDF downloads
Textbooks             →  raw PDF files
```

**Output:** A corpus of ~15,000–25,000 bone-domain papers with abstracts and open-access PDFs, stored locally.

**Status:** ✅ Complete — pipeline running.

---

### Phase 2 — RAG (Retrieval-Augmented Generation)
Chunk and embed the text corpus so the LLM can retrieve relevant passages at query time.

```
Paper abstracts + PDF text  →  text chunks  →  embeddings  →  vector store
                                                                     │
User query  ──────────────────────────────────────────────────►  retrieve top-k
                                                                     │
                                                              LLM answers using
                                                              retrieved context
```

**Why RAG before fine-tuning:** RAG gives the LLM access to the entire corpus without retraining. It also makes the knowledge updatable (add new papers → re-embed) without changing the model.

**Novel contribution here:** The retrieval is not generic — it is guided by a bone ontology. When a user asks about "femoral neck fracture", the ontology expands the query to related concepts (cortical thinning, reduced BMD, trabecular connectivity) before retrieval, improving coverage.

---

### Phase 3 — VLM integration
Add image understanding for X-ray and MRI inputs.

```
X-ray / MRI image  →  VLM  →  structured visual report
                                    │
                              "cortical thinning at femoral neck,
                               decreased trabecular density,
                               consistent with Grade 2 osteoporosis"
                                    │
                              fed into Layer 2 reasoning
```

**Novel contribution here:** Cross-modal linking — the VLM output is mapped to the same bone ontology concepts as the text data, so a visual finding ("cortical thinning") automatically connects to the mechanical literature on how cortical thickness affects fracture risk.

---

### Phase 4 — LRM reasoning layer
Move from retrieval to reasoning.

```
Retrieved context + visual report + user question
        │
        ▼
  Bone Ontology Graph
  (structured relationships between concepts)
        │
        ▼
  LRM: causal reasoning, hypothesis generation
        │
        ▼
  Response: answer + (optionally) new hypothesis + confidence
```

**Novel contribution here:** The LRM does not just retrieve and summarise. It traverses the ontology to construct reasoning chains:

> "Cortical thinning at the femoral neck → reduced cross-sectional moment of inertia → lower bending stiffness → increased fracture risk under hip loading. Novel hypothesis: a graded periosteal coating (inspired by the calcified cartilage interface in bone) could restore stiffness without adding bulk."

---

### Phase 5 — Feedback loop
Make the system improve with use.

```
User provides feedback on a response
        │
        ▼
  Feedback classifier: correction / confirmation / new fact
        │
        ▼
  Ontology updater: add / modify / weight knowledge graph nodes
        │
        ▼
  Updated retrieval and reasoning in subsequent queries
```

This is what distinguishes BoneLogic from a static RAG system. Each interaction is a data point that refines the system's bone knowledge model.

---

## Key design principles

**1. Papers are never shown by default.**
Responses should read like a knowledgeable colleague, not a literature search. References are surfaced only when explicitly requested. The LRM answers from internalised knowledge, backed by retrieved evidence behind the scenes.

**2. Hypothesis generation, not just recall.**
Every response has the potential to contain a hypothesis marker — a structured claim that goes beyond the training data, flagged as speculative but grounded in the ontology.

**3. Domain boundary enforcement.**
Queries outside bone science are politely redirected. This is not a limitation — it is what makes the system authoritative within its domain.

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
│   │   ├── semantic_scholar.py  S2 API client
│   │   ├── storage.py           SQLite metadata store
│   │   ├── downloader.py        Open-access PDF downloader
│   │   └── pipeline.py          CLI entrypoint
│   └── textbooks/               Phase 1b: textbook loading (placeholder)
│
├── processing/                  Phase 2: text extraction and chunking
│   └── (coming in Phase 2)
│
├── retrieval/                   Phase 2: vector store and RAG
│   └── (coming in Phase 2)
│
├── models/                      Phases 2-4: LLM, VLM, LRM wrappers
│   └── (coming in Phase 2)
│
├── reasoning/                   Phase 4: ontology and LRM
│   └── (coming in Phase 4)
│
├── api/                         Phase 5: user-facing interface
│   └── (coming in Phase 5)
│
├── data/
│   ├── raw/papers/              Downloaded PDFs (gitignored)
│   ├── raw/textbooks/           Textbook files (gitignored)
│   ├── processed/               Chunks, embeddings (gitignored)
│   └── db/papers.db             SQLite metadata (gitignored)
│
├── docs/
│   ├── architecture.md          ← this file
│   └── ingestion_pipeline.md    Detailed walkthrough of Phase 1 code
│
├── tests/
│   └── test_ingestion.py
│
├── .env                         Your API keys (gitignored — never commit)
├── .env.example                 Template showing which keys are needed
└── requirements.txt
```
