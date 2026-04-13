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
OpenAlex API  →  paper metadata (SQLite)  →  URL resolver  →  PDF downloads
Textbooks     →  raw PDF files
```

**Delivered:**

| Item | Result |
|---|---|
| Papers collected | **54,634** unique English papers (1970–2026) |
| PDFs downloaded | **7,674** (Pass 1: OpenAlex OA URLs + CrossRef · Pass 2: DOI → CrossRef) |
| Year range | 1970–2026 |
| Search keywords | 133 across 17 topic groups |
| PDF sources | OpenAlex OA URLs · EuropePMC · PMC · PLOS · Frontiers · Springer · MDPI · Wiley TDM |
| Database | SQLite at `data/db/papers.db` — browsable with PyCharm or DB Browser |
| Inspection tool | `scripts/inspect_db.py` — stats + progress bar |

**Source switch — Semantic Scholar → OpenAlex:** The original pipeline used Semantic Scholar, but the API key expired mid-project. We switched to OpenAlex — completely free, no API key required, and richer open-access metadata. CrossRef replaced Unpaywall for PDF resolution (more reliable direct PDF links).

**Novel engineering decision here:** A 2-pass PDF download pipeline. Pass 1 uses OpenAlex OA URLs and publisher-specific URL transforms plus CrossRef as a fallback. Pass 2 re-attempts DOI-based CrossRef lookup for any paper that failed Pass 1. The Wiley TDM API (token-based) bypasses Cloudflare for Wiley journals. This layered approach maximises download yield without institutional access.

---

### Phase 2 — Text Processing & RAG (Retrieval-Augmented Generation) ✅ Complete (April 2026)
Chunk and embed the text corpus so the LLM can retrieve relevant passages at query time.

```
PDF text  →  extract_papers.py / extract_textbooks.py  →  .txt files (refs stripped)
                                                               │
                                                     filter_english.py (2-pass)
                                                               │
                                                          chunk_all.py
                                                     (English only · sentence-aware)
                                                               │
                                                         chunks.db (SQLite)
                                                               │
                                                           embed.py
                                                     (SPECTER2 · proximity adapter)
                                                               │
                                                    768-dim SPECTER2 vectors
                                                               │
User query  ──────────────────────────────────────────►  retrieve top-k
                                                    (SPECTER2 · adhoc_query adapter)
                                                               │
                                                       LLM answers using
                                                       retrieved context
```

**Delivered:**

| Item | Result |
|---|---|
| Papers extracted | **7,433** English papers (28 scanned — no text layer) |
| Textbooks extracted | **16 / 16** |
| Non-English flagged | **336 papers** (detected via 2-pass body-text analysis) |
| Reference sections | Stripped at extraction — not included in any chunk |
| Total chunks | **248,629** (246,646 paper + 1,983 textbook) |
| Chunk strategy | Sentence-aware · target 400 tokens · 2-sentence overlap · never cuts mid-sentence |
| Embedding model | SPECTER2 (`allenai/specter2_base`) — trained on 164M citation relationships |
| Adapter (documents) | `allenai/specter2` (proximity) — used at embedding time |
| Adapter (queries) | `allenai/specter2_adhoc_query` — used at retrieval time |
| Embedding dimensions | 768 |
| Embeddings complete | **248,629 / 248,629** chunks embedded |
| Retrieval interface | CLI (`python -m retrieval.query`) + Gradio web UI (`python app.py`) |
| LLM | HuatuoGPT-o1-8B served via Ollama — streams grounded answers from retrieved context |
| Web UI tabs | **Ask BoneLogic** (RAG + LLM streaming) · **Search Corpus** (raw retrieval, no LLM) |

**Phase 2 polish — also complete:**

| Item | Result |
|---|---|
| Retrieval benchmark | 30 questions · 7 domains (mechanics, morphology, pathology, biomaterials, simulation, imaging, mechanobiology) |
| MRR | **0.928** |
| Recall@3 | **1.000** — all 30 questions have a relevant chunk in top 3 |
| Recall@5 | **1.000** — Phase 3 readiness threshold passed |
| Citation behaviour | Few-shot example in system prompt · mandatory `[N]` inline citations · DOI links auto-injected into References section |
| Reference formatting | Each `[N]` entry rendered on its own line with clickable Open paper link |
| Eval script | `eval/run_eval.py` — rerun any time corpus or retriever changes |

**Why RAG before fine-tuning:** RAG gives the LLM access to the entire corpus without retraining. It also makes the knowledge updatable — add new papers, re-embed, done.

**Why SPECTER2:** Allen AI's 2023 successor to SPECTER, trained on 164M citation relationships. Key advantage over SPECTER1: asymmetric encoding — documents and queries use different task-specific adapters (proximity vs adhoc_query), improving retrieval precision on scientific text.

**Sentence-aware chunking:** The original implementation used a character-based sliding window that cut text arbitrarily. The final implementation uses NLTK sentence tokenisation — chunks always end at a sentence boundary, with the last 2 sentences of each chunk carried into the next as overlap. This produces cleaner, more semantically coherent passages for embedding.

**Language filtering — two-pass approach:** Many papers have English abstracts (translated for indexing) but non-English body text. The filter skips the first 1,000 characters (title/abstract) and detects language from the next 4,000 characters of body text, catching the abstract-English / body-foreign class reliably. Pass A processes all new papers; Pass B re-checks any previously abstract-classified English papers that now have an extracted text file.

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
│   │                            Query encoding uses SPECTER2 adhoc_query adapter
│   └── query.py                 CLI entrypoint (single query + interactive mode)
│
├── models/                      Phases 3-4: LLM, VLM, LRM wrappers
│   └── (coming in Phase 3)
│
├── reasoning/                   Phase 4: ontology and LRM
│   └── (coming in Phase 4)
│
├── api/                         Phase 5: user-facing interface
│   └── (coming in Phase 5)
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
