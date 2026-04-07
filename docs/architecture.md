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

### Phase 1 — Data ingestion ✅ Complete (March 2026)
Collect the knowledge base that will ground the LLM and LRM.

```
Semantic Scholar API  →  paper metadata (SQLite)  →  URL resolver  →  PDF downloads
Textbooks             →  raw PDF files
```

**Delivered:**

| Item | Result |
|---|---|
| Papers collected | **55,277** unique papers |
| Open-access PDF URLs | 21,571 (39% of corpus) |
| PDFs downloaded | **6,125** out of 21,571 (rest require institutional access) |
| Year range | 1900–2025 |
| Search keywords | 133 across 17 topic groups |
| Top journal | *Bone* — 2,452 papers |
| PDF resolution | 3-tier chain: direct link → publisher transform → Unpaywall API |
| PDF coverage | ~56% resolved without any API call; ~100% with Unpaywall |
| Database | SQLite at `data/db/papers.db` — browsable with PyCharm or DB Browser |
| Inspection tool | `scripts/inspect_db.py` — stats + progress bar |

**Novel engineering decision here:** Rather than accepting Semantic Scholar's mixed bag of direct links, DOI redirects, and viewer pages, we built a publisher-specific URL resolver (`resolvers.py`) that transforms 7 known publisher URL patterns to direct PDF links before download, and falls back to the free Unpaywall API for DOI-based lookup. This significantly increases the actual downloadable PDF yield compared to a naive download attempt.

---

### Phase 2 — Text Processing & RAG (Retrieval-Augmented Generation) 🔄 In Progress (April 2026)
Chunk and embed the text corpus so the LLM can retrieve relevant passages at query time.

```
PDF text  →  extract_papers.py / extract_textbooks.py  →  .txt files
                                                               │
                                                          chunk_all.py
                                                               │
                                                         chunks.db (SQLite)
                                                               │
                                                           embed.py
                                                               │
                                                    768-dim SPECTER vectors
                                                               │
User query  ──────────────────────────────────────────►  retrieve top-k
                                                               │
                                                       LLM answers using
                                                       retrieved context
```

**Delivered so far:**

| Item | Result |
|---|---|
| Papers extracted | **6,085** (40 scanned — no text layer) |
| Textbooks extracted | **16 / 16** |
| Total chunks | **200,757** (188,042 paper + 12,715 textbook) |
| Chunk size | 2,048 chars (~512 tokens) with 200-char overlap |
| Embedding model | SPECTER (`allenai-specter`) — trained on 146M S2 citations |
| Embedding dimensions | 768 |
| Embeddings complete | **200,757 chunks** embedded (127 min, CPU-only) |
| Language filtering | **763 non-English papers** removed (564 by abstract + 199 with English abstract but non-English body) |
| English chunks remaining | **193,537** |
| Retrieval interface | CLI (`python -m retrieval.query`) + Gradio web UI (`python app.py`) |

**Why RAG before fine-tuning:** RAG gives the LLM access to the entire corpus without retraining. It also makes the knowledge updatable (add new papers → re-embed) without changing the model.

**Why SPECTER:** Trained by Allen AI specifically on Semantic Scholar paper citations — produces embeddings that capture scientific meaning, making it ideal for a corpus collected from Semantic Scholar.

**Language filtering — two-pass approach:** Many papers in Semantic Scholar have English abstracts (translated for indexing) but non-English full text. A single abstract-based filter missed 199 such papers. The final implementation runs two passes: (1) detect from abstract/title for unprocessed papers; (2) re-detect from the first 2,000 characters of the extracted `.txt` file for any paper previously classified as English via abstract alone. This catches the abstract-English / body-foreign class of papers reliably.

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
│   │   ├── resolvers.py         3-tier PDF URL resolver
│   │   ├── downloader.py        Open-access PDF downloader
│   │   └── pipeline.py          CLI entrypoint
│   └── textbooks/
│       ├── storage.py           SQLite textbook store
│       ├── scanner.py           Folder scanner + PyMuPDF metadata extraction
│       └── pipeline.py          CLI entrypoint
│
├── scripts/
│   ├── inspect_db.py            Database statistics + progress inspector
│   └── filter_english.py        Two-pass language filter (abstract + text-file detection)
│
├── processing/                  Phase 2: text extraction, chunking, embedding
│   ├── extractor.py             PDF → plain text via PyMuPDF
│   ├── extract_papers.py        Extract all paper PDFs → .txt files
│   ├── extract_textbooks.py     Extract all textbook PDFs → .txt files
│   ├── chunker.py               Page-aware sliding window chunker
│   ├── chunk_all.py             Chunk all texts → chunks.db
│   └── embed.py                 Embed chunks with SPECTER → chunks.db
│
├── retrieval/                   Phase 2: RAG retrieval engine + CLI query interface
│   ├── retriever.py             BoneLogicRetriever — loads all embeddings, cosine search
│   └── query.py                 CLI entrypoint (single query + interactive mode)
│
├── models/                      Phases 2-4: LLM, VLM, LRM wrappers
│   └── (coming in Phase 3)
│
├── reasoning/                   Phase 4: ontology and LRM
│   └── (coming in Phase 4)
│
├── api/                         Phase 5: user-facing interface
│   └── (coming in Phase 5)
│
├── data/
│   ├── raw/papers/              Downloaded PDFs (gitignored)
│   ├── raw/textbooks/           Textbook PDFs by source (gitignored)
│   ├── processed/text/          Extracted .txt files (gitignored)
│   └── db/
│       ├── papers.db            Paper metadata (gitignored)
│       ├── textbooks.db         Textbook metadata (gitignored)
│       └── chunks.db            Text chunks + SPECTER embeddings (gitignored)
│
├── docs/
│   ├── architecture.md                      ← this file
│   ├── papers_ingestion_pipeline.md         Detailed walkthrough of papers ingestion
│   └── textbooks_ingestion_pipeline.md      Detailed walkthrough of textbooks ingestion
│
├── tests/
│   └── test_ingestion.py
│
├── app.py                       Gradio web UI (http://localhost:7860)
├── .env                         Your API keys (gitignored — never commit)
├── .env.example                 Template showing which keys are needed
└── requirements.txt
```
