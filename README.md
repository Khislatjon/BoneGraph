# BoneGraph

**An intelligent research assistant for bone science** — morphology, mechanics, pathology, imaging, and biomaterials.

BoneGraph unifies four research workflows behind a single web UI: conversational question-answering over a curated literature corpus, raw semantic search, knowledge-graph-grounded reasoning with a self-correcting critic, and vision-language analysis of medical images. It runs locally on top of a 248,629-chunk embedding index, a hand-cleaned bone knowledge graph, and open-weight language models served through Ollama.

> ⚠️ **For research and educational use only. BoneGraph is not a clinical tool and does not provide medical diagnoses.**

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python serve.py          # http://localhost:8000
```

Pull the models BoneGraph depends on (via [Ollama](https://ollama.com)):

```bash
ollama pull huatuogpt-bone   # Chat + Reasoning — HuatuoGPT-o1-8B with a bone-science system prompt
ollama pull llava:13b        # Vision tab
ollama pull llama3.2:3b      # bone-relevance classifier + rule extraction
```

The **Chat**, **Search**, and **Reasoning** tabs are fully offline once the models are pulled and the embedding index is built. The **Vision** tab additionally requires `llava:13b`.

---

## The four tabs

BoneGraph's UI is organised into four tabs, each backed by its own API surface in [`api/main.py`](api/main.py).

### 1 · Chat — conversational RAG

The default tab (labelled **Chat**) is a retrieval-augmented question-answering interface over the **BoneScholar** corpus.

**How it works**
1. Each question is first screened by a lightweight two-stage **bone-relevance guard** — a fast lexical pass over bone vocabulary, falling back to `llama3.2:3b` for ambiguous wording (and failing open on network errors). Off-topic questions are politely declined so the model stays in domain.
2. The question is embedded with SPECTER2's ad-hoc query adapter and used to retrieve the top-k most relevant chunks (default `top_k=8`) from the 248,629-chunk index.
3. The retrieved passages are assembled into a grounded context and streamed through **HuatuoGPT-o1-8B**, a medical reasoning model running under a bone-science system prompt.
4. The answer streams back token-by-token with inline `[N]` citations that are rewritten into clickable DOI links pointing at the exact source paper.

**Multi-turn chat** is supported with a bounded sliding context window (the last 3 question/answer pairs). Thinking blocks and reference sections are stripped from history before re-injection so the context stays small and on-topic across a conversation.

### 2 · Search — raw semantic retrieval

Direct semantic search over the full BoneScholar corpus with **no LLM in the loop** — what you see is exactly what the retriever returns.

**Controls**
- **Source filter** — papers, textbooks, or all.
- **Year range** — restrict to a publication window (default 1970–2026).
- **top-k** — how many ranked passages to return (default 10).

Each result shows the passage text, paper/textbook metadata, a DOI link where available, and the raw similarity score. This tab is the fastest way to do literature review and to sanity-check retrieval quality without generation overhead — it's effectively a window onto what the Chat and Reasoning tabs are reading from.

### 3 · Reasoning — agent + critic, grounded in the knowledge graph

A self-correcting reasoning loop grounded in the **bone knowledge graph** (1,597 nodes · 1,699 edges).

**How it works**
1. A **reasoning agent** drafts an answer to the question, supported by retrieved literature.
2. A **critic** independently checks that draft against two grounding sources:
   - **Tier-1 deterministic rules** — fracture-mechanics relationships encoded in [`reasoning/physical_grounding.py`](reasoning/physical_grounding.py), checked programmatically rather than by an LLM.
   - **1-hop knowledge-graph facts** — relevant nodes and edges pulled from the graph and fed to the critic as ground truth ([`reasoning/kg_context.py`](reasoning/kg_context.py)).
3. If the critic flags violations, the agent revises. The loop runs under **Quick** and **Deep** modes, with a hard 2-iteration cap so it always terminates.

**It learns from feedback.** When you 👎 an answer and describe what's wrong, `llama3.2:3b` extracts a candidate **user rule** ([`reasoning/rule_extractor.py`](reasoning/rule_extractor.py)). Once you confirm it, the rule is persisted (Tier-2) and applied by the critic on every future request. Rules can be listed, toggled on/off, deleted, and imported/exported as a template — all from the in-app **Rules** manager. 👍/👎 events and corrections are stored in a SQLite feedback store ([`reasoning/feedback_store.py`](reasoning/feedback_store.py)).

### 4 · Vision — image analysis with a feedback loop

Upload an X-ray, MRI, or histology image and analyse it with a vision-language model.

**How it works**
1. The uploaded image is encoded ([`vision/encoder.py`](vision/encoder.py)) and sent to **LLaVA 13b** via Ollama.
2. LLaVA produces a structured identification of the visible structures, which is parsed into a clean summary, plus a free-form description.
3. You can then **chat about the image** in a multi-turn conversation — asking follow-up questions while the model retains the image context.

**It learns from corrections.** Submitting a 👎 with a description of what's wrong stores a correction ([`vision/correction_store.py`](vision/correction_store.py)) that is recalled to shape future responses. Corrections can be reviewed and deleted from the in-app corrections manager.

> The Vision tab is explicitly **research and educational only — not a clinical diagnostic tool.**

---

## BoneScholar corpus

The literature backbone behind the Chat and Search tabs.

| Stat | Value |
|---|---|
| Papers collected | 54,634 |
| Papers with full text | 7,433 |
| Open-access textbooks | 16 |
| Chunks (sentence-aware, ~400 tokens, 2-sentence overlap) | 248,629 |
| Embedding model | SPECTER2 · 768-dim · proximity adapter |

Papers were collected via the **OpenAlex API** across targeted keyword groups spanning bone morphology, mechanics, pathology, imaging, and biomaterials. Non-English papers (336) are flagged and excluded from retrieval. Chunks are sentence-aware (~400 tokens with 2-sentence overlap) to stay safely under SPECTER2's 512-token limit while preserving context across boundaries.

---

## Bone knowledge graph

The reasoning backbone behind the Reasoning tab.

| Stat | Value |
|---|---|
| Nodes (unique concepts) | 1,597 |
| Edges (unique relations) | 1,699 |
| Source | 16 open-access textbooks · 1,983 chunks |

The graph was built by running LLM triple extraction over the textbook corpus, then cleaning and reclassifying the raw triples down to a high-signal graph. It is read by the Reasoning tab's critic as a 1-hop fact source for grounding agent drafts.

---

## Models

| Role | Model | Used by |
|---|---|---|
| Answer generation / reasoning agent | HuatuoGPT-o1-8B (`huatuogpt-bone`) | Chat, Reasoning |
| Embeddings (documents + queries) | SPECTER2 (`allenai/specter2` + ad-hoc query adapter) | Chat, Search, Reasoning |
| Vision-language analysis | LLaVA 13b (`llava:13b`) | Vision |
| Bone-relevance classifier · rule extraction · triple extraction | `llama3.2:3b` / `huatuogpt-bone` | Chat, Reasoning |

Model names, the Ollama base URL (`OLLAMA_URL`), timeouts, and chunking parameters are centralised in [`config/settings.py`](config/settings.py) and overridable via environment variables.

---

## Retrieval benchmark

A 30-question benchmark across 7 bone-science domains (morphology, mechanics, pathology, imaging, biomaterials, remodelling, fracture).

| Metric | Score |
|---|---|
| MRR | **0.928** |
| Recall@1 | **0.867** (26 / 30) |
| Recall@3 | **1.000** (30 / 30) |
| Recall@5 | **1.000** (30 / 30) |

```bash
python eval/run_eval.py           # default top_k=10
python eval/run_eval.py --top-k 5
```

Results are saved to `eval/results.json`.

---

## Building the pipeline from scratch

### 1 — Collect papers

```bash
cp .env.example .env
# Set CROSSREF_EMAIL and WILEY_TDM_TOKEN

python -m ingestion.papers.pipeline              # metadata only
python -m ingestion.papers.pipeline --download   # + open-access PDFs
```

### 2 — Ingest textbooks

```bash
# Place PDFs in data/raw/textbooks/<Source Name>/book.pdf
python -m ingestion.textbooks.pipeline
```

### 3 — Extract, chunk, embed

```bash
python -m processing.extract_papers
python -m processing.extract_textbooks
python -m scripts.filter_english
python -m processing.chunk_all
python -m processing.embed        # resumable; use --force to re-embed from scratch
```

### 4 — Build the knowledge graph

```bash
# Triple extraction (resumable — picks up from last processed chunk)
python -m reasoning.extractor --source textbooks

# Check graph stats
python -c "
import sqlite3
con = sqlite3.connect('data/db/ontology.db')
n = con.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]
e = con.execute('SELECT COUNT(*) FROM edges').fetchone()[0]
print(f'Nodes: {n}  Edges: {e}')
"
```

---

## Project structure

```
BoneGraph/
├── api/                     # FastAPI backend (ask, search, reason, vision, stats)
│   └── main.py
├── frontend/                # Single-file React app (Babel in-browser transpilation)
│   ├── index.html           # Chat · Search · Reasoning · Vision tabs
│   └── static/
├── ingestion/
│   ├── papers/              # OpenAlex API client, storage, downloader
│   └── textbooks/           # Textbook scanner and storage
├── processing/              # Text extraction, chunking, embedding
├── retrieval/               # SPECTER2 retrieval engine + CLI query interface
├── reasoning/               # Knowledge graph + Reasoning tab backend
│   ├── ontology.py          # Node/Edge dataclasses, GraphBuilder
│   ├── graph_db.py          # SQLite-backed graph persistence
│   ├── seed.py              # ~200 seed concepts + hand-curated causal edges
│   ├── extractor.py         # LLM triple extraction pipeline (resumable)
│   ├── physical_grounding.py# Fracture-scoped Tier-1 rules + Tier-2 user-rule compiler
│   ├── feedback_store.py    # Feedback events, corrections, user rules (SQLite)
│   ├── rule_extractor.py    # Rule extraction from 👎 feedback via llama3.2:3b
│   └── kg_context.py        # 1-hop KG facts fed to the critic
├── vision/                  # Vision tab backend
│   ├── encoder.py           # Image encoding for LLaVA
│   └── correction_store.py  # Vision feedback and correction storage
├── config/                  # Central settings (paths, model names, constants)
├── scripts/                 # Utilities: DB inspector, language filter, rule import
├── eval/                    # Retrieval benchmark (30 questions, MRR + Recall@k)
├── data/
│   └── db/
│       ├── papers.db        # 54,634 paper metadata rows
│       ├── chunks.db        # 248,629 chunks + SPECTER2 embeddings
│       └── ontology.db      # Knowledge graph: 1,597 nodes · 1,699 edges
├── docs/                    # Per-phase architecture documentation
├── serve.py                 # Uvicorn launcher → http://localhost:8000
└── tests/
```

---

## Documentation

| Document | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | System overview, 2-layer architecture, design principles |
| [docs/papers_ingestion_pipeline.md](docs/papers_ingestion_pipeline.md) | Phase 1 — OpenAlex API → SQLite → PDFs |
| [docs/textbooks_ingestion_pipeline.md](docs/chat/textbooks_ingestion_pipeline.md) | Phase 1b — 16 curated open-access textbooks |
| [docs/phase2_rag_pipeline.md](docs/phase2_rag_pipeline.md) | Phase 2 — extraction, chunking, embedding, RAG, evaluation |
| [docs/vision/architecture.md](docs/vision/architecture.md) | Vision tab — LLaVA pipeline, feedback loop, correction store |
| [docs/reasoning/architecture.md](docs/reasoning/architecture.md) | Reasoning tab — agent + critic loop, rule tiers, KG grounding |

---

## Tests

```bash
pytest tests/ -v
```
