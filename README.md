# BoneLogic

An intelligent reasoning system for bone science — morphology, mechanics, pathology, and biomaterials.

## Architecture

```
Layer 1  │  LLM (text) + VLM (X-ray / MRI)      ← perception & understanding
Layer 2  │  LRM (reasoning model)                ← hypothesis generation & prediction
```

## Phase 1 — Data Ingestion

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your SEMANTIC_SCHOLAR_API_KEY to .env
```

### Run paper ingestion

```bash
# All keyword groups (≈60 queries, up to 1000 papers each)
python -m ingestion.papers.pipeline

# Specific groups only
python -m ingestion.papers.pipeline --groups mechanics pathology imaging

# Restrict to recent papers and also download open-access PDFs
python -m ingestion.papers.pipeline --year 2015-2024 --download

# Custom keyword + small cap (useful for testing)
python -m ingestion.papers.pipeline --keywords "bone fracture toughness" --max 50
```

### Run tests

```bash
pytest tests/ -v
```

## Project structure

```
BoneLogic/
├── config/                  # Central settings
├── ingestion/
│   ├── papers/
│   │   ├── semantic_scholar.py   # S2 API client
│   │   ├── keywords.py           # Bone keyword taxonomy
│   │   ├── storage.py            # SQLite metadata store
│   │   ├── downloader.py         # Open-access PDF downloader
│   │   └── pipeline.py           # CLI entrypoint
│   └── textbooks/               # (Phase 1b) textbook loaders
├── data/
│   ├── raw/papers/          # Downloaded PDFs (gitignored)
│   └── db/papers.db         # SQLite metadata DB (gitignored)
└── tests/
```
