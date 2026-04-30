# BoneMind

An intelligent reasoning system for bone science — morphology, mechanics, pathology, and biomaterials.

## Documentation

| Document | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | System overview, 2-layer architecture, phase summaries, design principles, repo layout |
| [docs/papers_ingestion_pipeline.md](docs/papers_ingestion_pipeline.md) | Phase 1 — papers ingestion deep-dive: API → SQLite → PDFs |
| [docs/textbooks_ingestion_pipeline.md](docs/textbooks_ingestion_pipeline.md) | Phase 1b — textbooks ingestion: 16 curated open-access books |
| [docs/phase2_rag_pipeline.md](docs/phase2_rag_pipeline.md) | Phase 2 — text extraction, chunking, embedding, RAG, LLM, evaluation |
| [docs/phase3_vlm_plan.md](docs/phase3_vlm_plan.md) | Phase 3 — VLM integration plan: datasets, pipeline, cross-modal retrieval, UI |
| [docs/phase4_lrm_plan.md](docs/phase4_lrm_plan.md) | Phase 4 — LRM reasoning layer: bone knowledge graph, physics engine, hypothesis generation |
| [docs/lrm_benchmark.md](docs/lrm_benchmark.md) | LRM benchmark: chain coverage, physics accuracy, novelty calibration — methodology and thresholds |

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
| Phase 2 — RAG retrieval | ✅ Complete | CLI + React web UI (Ask · Search · Analyse Image · Reason tabs) |
| Phase 2 — LLM integration | ✅ Complete | HuatuoGPT-o1-8B via Ollama (`huatuogpt-bone` — HuatuoGPT-o1-8B with custom bone science system prompt) · streaming RAG answers |
| Phase 2 — Retrieval evaluation | ✅ Complete | MRR 0.928 · Recall@5 1.000 · 30-question benchmark |
| Phase 2 — Citation behaviour | ✅ Complete | Few-shot system prompt · inline [N] citations · DOI links |
| Phase 3 — VLM integration | 🔶 Partial | LLaVA 1.6 tab implemented · cross-modal retrieval pending |
| Phase 4 — LRM reasoning | 🔶 In Progress | Steps 4.1–4.6 complete · 35,338 nodes · 34,265 edges (textbooks + papers) |

---

## Phase 1 — Data Ingestion

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add CROSSREF_EMAIL and WILEY_TDM_TOKEN to .env
```

#### Known issue — gradio_client crash on startup (legacy UI only)

Some versions of `gradio_client` crash with `TypeError: argument of type 'bool' is not iterable` when building API info for components. This only affects `app.py` (the legacy Gradio UI). If you see this error, apply the following two-line patch:

**File:** `.venv/lib/python3.9/site-packages/gradio_client/utils.py`

**Fix 1** — in `get_type()` (around line 862), add a guard at the top of the function:
```python
def get_type(schema: dict):
    if not isinstance(schema, dict):   # ← add this line
        return "unknown"
    if "const" in schema:
```

**Fix 2** — in `_json_schema_to_python_type()` (around line 955), guard the `additionalProperties` branch:
```python
# change this:
if "additionalProperties" in schema:
# to this:
if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
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

# React web UI (tabs: Ask BoneMind · Search Corpus · Analyse Image · Reason)
python serve.py    # FastAPI backend + React frontend at http://localhost:8000
```

**Ask BoneMind tab** requires Ollama running with the fine-tuned model:

```bash
ollama serve          # start the server (separate terminal if not running as a service)
ollama run huatuogpt-bone  # HuatuoGPT-o1-8B with custom bone science system prompt
```

**Search Corpus tab** works without Ollama — pure semantic retrieval only.

> Legacy Gradio UI is still available via `python app.py` (port 7860) but is no longer the primary interface.

### Run the retrieval benchmark

```bash
python eval/run_eval.py              # default top_k=10
python eval/run_eval.py --top-k 20  # custom top_k
```

Results are saved to `eval/results.json`. Current scores (top_k=10):

| Metric | Score |
|---|---|
| MRR | **0.928** |
| Recall@1 | **0.867** (26/30) |
| Recall@3 | **1.000** (30/30) |
| Recall@5 | **1.000** (30/30) |

---

## Phase 4 — LRM Reasoning Layer

Steps 4.1–4.6 are complete. The knowledge graph is seeded, triple extraction has run on all 16 textbooks, and the full reasoning stack (physics engine, LRM, novelty classifier, Gradio "Reason" tab) is live.

### Check graph stats

```bash
# Quick SQLite inspection
python -c "
import sqlite3, pathlib
con = sqlite3.connect('data/db/ontology.db')
nodes = con.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]
edges = con.execute('SELECT COUNT(*) FROM edges').fetchone()[0]
print(f'Nodes: {nodes}  Edges: {edges}')
"
```

### Run triple extraction (resumable)

```bash
# Textbooks only (highest quality, ~1,983 chunks, ~7 hours CPU-only)
.venv/bin/python -m reasoning.extractor --source textbooks

# All chunks including papers (long-running, use caffeinate / nohup on macOS)
caffeinate -i nohup .venv/bin/python -m reasoning.extractor > logs/extractor.log 2>&1 &

# Tail progress
tail -f logs/extractor.log
```

> Extraction is **resumable** — if interrupted, re-run the same command and it picks up from the last processed chunk (tracked in `extraction_progress` table in `ontology.db`).

### Current graph stats (after textbook extraction)

| Metric | Value |
|---|---|
| Chunks processed | 1,983 / 1,983 textbook chunks |
| Non-empty chunks | 727 (37%) |
| Raw triples extracted | 2,935 |
| Nodes (unique concepts) | 3,001 |
| Edges (unique relations) | 2,339 |

### Launch the Reason tab

```bash
python serve.py   # React UI at http://localhost:8000 — Reason tab in sidebar
```

The Reason tab supports:
- **Natural language hypothesis queries** — anchors to graph nodes, traverses multi-hop causal chains
- **Physics validation** — 71 directional rules + numerical checks (Currey's law, Frost mechanostat, Paris crack growth, beam bending, stress concentration)
- **Novelty classification** — Tier 1 keyword search + Tier 2 SPECTER2 semantic similarity → GROUNDED / SPECULATIVE / NOVEL
- **Research gap detection** — betweenness centrality analysis ranks under-studied bridge concepts

Example queries:
- `"aging fracture risk"`
- `"cortical porosity elastic modulus"`
- `"collagen crosslink toughness"`
- `"osteocyte lacuna fatigue crack"`
- `"bone mineral density osteoporosis"`

### Run the LRM benchmark

```bash
python eval/run_lrm_eval.py              # default max_results=10
python eval/run_lrm_eval.py --max-results 5
```

Three components — physics accuracy, chain coverage, novelty calibration. Results saved to `eval/lrm_results.json`. See [docs/lrm_benchmark.md](docs/lrm_benchmark.md) for full methodology and pass thresholds.

---

## Project structure

```
BoneMind/
├── config/                  # Central settings (paths, model names, constants)
├── ingestion/
│   ├── papers/              # OpenAlex API client, storage, downloader
│   └── textbooks/           # Textbook scanner and storage
├── processing/              # Text extraction, chunking, embedding
├── retrieval/               # RAG retrieval engine + CLI/web query interface
├── reasoning/               # Phase 4: bone knowledge graph + LRM reasoning layer
│   ├── __init__.py
│   ├── ontology.py          # Node/Edge dataclasses, GraphBuilder, NetworkX wrappers
│   ├── graph_db.py          # SQLite-backed graph persistence (ontology.db)
│   ├── seed.py              # ~200 seed concepts + hand-curated causal edges
│   ├── extractor.py         # LLM triple extraction pipeline from chunks.db (resumable)
│   ├── physics.py           # Bone physics engine: 71 directional rules + numerical laws
│   ├── lrm.py               # Core reasoning engine: path-finding, gap detection, scoring
│   └── novelty.py           # Novelty classifier: keyword tier + SPECTER2 semantic tier
├── scripts/                 # Utilities: DB inspector, language filter
├── data/
│   ├── raw/papers/          # Downloaded paper PDFs (gitignored)
│   ├── raw/textbooks/       # Textbook PDFs by source (gitignored)
│   ├── processed/text/      # Extracted .txt files (gitignored)
│   └── db/                  # SQLite databases (gitignored)
│       ├── papers.db        # 54,634 paper metadata rows
│       ├── textbooks.db     # 16 textbook metadata rows
│       ├── chunks.db        # 248,629 chunks + SPECTER2 embeddings
│       └── ontology.db      # Knowledge graph: 35,338 nodes · 34,265 edges
├── eval/                    # Retrieval quality benchmark
│   ├── benchmark.json       # 30 questions across 7 domains with expected keywords
│   ├── run_eval.py          # Eval script — computes MRR and Recall@k
│   └── results.json         # Latest benchmark results
├── api/                     # FastAPI backend (endpoints: ask, search, reason, analyse, gaps, stats)
│   └── main.py
├── frontend/                # React web UI (sidebar design: Ask · Search · Analyse Image · Reason)
│   ├── index.html           # Single-file React app (Babel in-browser transpilation)
│   └── static/              # React, ReactDOM, Babel bundles
├── serve.py                 # Uvicorn launcher — starts FastAPI at http://localhost:8000
├── app.py                   # Legacy Gradio UI (4 tabs) — kept for reference
├── docs/                    # Detailed documentation per phase
├── mypaper/                 # Paper draft (gitignored)
└── tests/
```

---

## Run tests

```bash
pytest tests/ -v
```
