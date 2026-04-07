# BoneLogic

An intelligent reasoning system for bone science — morphology, mechanics, pathology, and biomaterials.

## Documentation

| Document | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Full system architecture, all build phases, design principles |
| [docs/papers_ingestion_pipeline.md](docs/papers_ingestion_pipeline.md) | Papers ingestion deep-dive: how data flows from API → SQLite → PDFs |
| [docs/textbooks_ingestion_pipeline.md](docs/textbooks_ingestion_pipeline.md) | Textbooks ingestion pipeline — 16 curated open-access books |

## Architecture (summary)

```
Layer 1  │  LLM (text) + VLM (X-ray / MRI)      ← perception & understanding
Layer 2  │  LRM (reasoning model)                ← hypothesis generation & prediction
```

## Progress

| Phase | Status | Key results |
|---|---|---|
| Phase 1 — Paper ingestion | ✅ Complete | 55,277 papers, 6,125 PDFs downloaded |
| Phase 1b — Textbook ingestion | ✅ Complete | 16 textbooks, 7,203 pages |
| Phase 2 — Text extraction | ✅ Complete | 6,085 papers + 16 textbooks extracted |
| Phase 2 — Chunking | ✅ Complete | 200,757 chunks (2,048 chars, 200 overlap) |
| Phase 2 — Embedding | 🔄 Running | SPECTER 768-dim vectors |
| Phase 2 — RAG retrieval | ⏳ Next | |
| Phase 3 — VLM integration | ⏳ Planned | |
| Phase 4 — LRM reasoning | ⏳ Planned | |

---

## Phase 1 — Data Ingestion

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your SEMANTIC_SCHOLAR_API_KEY and UNPAYWALL_EMAIL to .env
```

### Run paper ingestion

```bash
# Collect metadata for all keyword groups
python -m ingestion.papers.pipeline

# Also download open-access PDFs
python -m ingestion.papers.pipeline --download

# Specific groups only
python -m ingestion.papers.pipeline --groups mechanics pathology imaging

# Custom keyword + small cap (useful for testing)
python -m ingestion.papers.pipeline --keywords "bone fracture toughness" --max 50
```

### Run textbook ingestion

```bash
# Place PDFs in data/raw/textbooks/<Source Name>/book.pdf
# Then register them in the database:
python -m ingestion.textbooks.pipeline
```

### Inspect the database

```bash
python scripts/inspect_db.py
```

---

## Phase 2 — Text Processing & RAG

### Extract text from PDFs

```bash
python -m processing.extract_papers       # Extract all paper PDFs
python -m processing.extract_textbooks    # Extract all textbook PDFs
```

### Chunk extracted text

```bash
python -m processing.chunk_all
```

### Embed chunks with SPECTER

```bash
python -m processing.embed    # Runs for ~1-2 hours on CPU
```

---

## Project structure

```
BoneLogic/
├── config/                  # Central settings (paths, model names, constants)
├── ingestion/
│   ├── papers/              # Semantic Scholar API client, storage, downloader
│   └── textbooks/           # Textbook scanner and storage
├── processing/              # Text extraction, chunking, embedding
├── retrieval/               # RAG query interface (coming next)
├── data/
│   ├── raw/papers/          # Downloaded paper PDFs (gitignored)
│   ├── raw/textbooks/       # Textbook PDFs by source (gitignored)
│   ├── processed/text/      # Extracted .txt files (gitignored)
│   └── db/                  # SQLite databases (gitignored)
│       ├── papers.db
│       ├── textbooks.db
│       └── chunks.db
├── docs/                    # Detailed documentation per phase
├── mypaper/                 # Paper draft (gitignored)
└── tests/
```

---

## Run tests

```bash
pytest tests/ -v
```
