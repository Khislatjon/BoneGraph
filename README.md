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
| Phase 1 — Paper ingestion | ✅ Complete | 54,634 papers · 7,674 PDFs downloaded |
| Phase 1b — Textbook ingestion | ✅ Complete | 16 textbooks |
| Phase 2 — Text extraction | ✅ Complete | 7,433 English papers + 16 textbooks extracted |
| Phase 2 — Language filtering | ✅ Complete | 336 non-English papers flagged |
| Phase 2 — Chunking | ✅ Complete | 248,629 chunks (sentence-aware · ~400 tokens · 2-sentence overlap) |
| Phase 2 — Embedding | ✅ Complete | SPECTER2 768-dim · proximity adapter · 248,629 chunks |
| Phase 2 — RAG retrieval | ✅ Complete | CLI + Gradio web UI (Ask + Search tabs) |
| Phase 2 — LLM integration | ✅ Complete | HuatuoGPT-o1-8B via Ollama · streaming RAG answers |
| Phase 3 — VLM integration | ⏳ Planned | |
| Phase 4 — LRM reasoning | ⏳ Planned | |

---

## Phase 1 — Data Ingestion

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add CROSSREF_EMAIL and WILEY_TDM_TOKEN to .env
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
python -m processing.extract_papers       # Extract all paper PDFs (strips reference sections)
python -m processing.extract_textbooks    # Extract all textbook PDFs
```

### Filter non-English papers

```bash
python -m scripts.filter_english --dry-run   # Preview — no changes written
python -m scripts.filter_english             # Run both passes
```

### Chunk extracted text

```bash
python -m processing.chunk_all    # English papers only · sentence-aware · resumable
```

### Embed chunks with SPECTER2

```bash
python -m processing.embed        # Resumes from last embedded chunk if interrupted
python -m processing.embed --force  # Re-embed everything from scratch
```

### Query the corpus

```bash
# CLI — single query
python -m retrieval.query "cortical bone fracture toughness"

# CLI — interactive mode (embeddings loaded once, fast repeated queries)
python -m retrieval.query --interactive

# Gradio web UI (two tabs: Ask BoneLogic + Search Corpus)
python app.py    # Opens automatically at http://localhost:7860
```

**Ask BoneLogic tab** requires Ollama running with the fine-tuned model:

```bash
ollama serve          # start the server (separate terminal if not running as a service)
ollama run huatuogpt-bone
```

**Search Corpus tab** works without Ollama — pure semantic retrieval only.

---

## Project structure

```
BoneLogic/
├── config/                  # Central settings (paths, model names, constants)
├── ingestion/
│   ├── papers/              # OpenAlex API client, storage, downloader
│   └── textbooks/           # Textbook scanner and storage
├── processing/              # Text extraction, chunking, embedding
├── retrieval/               # RAG retrieval engine + CLI/web query interface
├── scripts/                 # Utilities: DB inspector, language filter
├── data/
│   ├── raw/papers/          # Downloaded paper PDFs (gitignored)
│   ├── raw/textbooks/       # Textbook PDFs by source (gitignored)
│   ├── processed/text/      # Extracted .txt files (gitignored)
│   └── db/                  # SQLite databases (gitignored)
│       ├── papers.db        # 54,634 paper metadata rows
│       ├── textbooks.db     # 16 textbook metadata rows
│       └── chunks.db        # 248,629 chunks + SPECTER2 embeddings
├── docs/                    # Detailed documentation per phase
├── app.py                   # Gradio web UI
├── mypaper/                 # Paper draft (gitignored)
└── tests/
```

---

## Run tests

```bash
pytest tests/ -v
```
