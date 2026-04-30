# Phase 2 — Text Processing & RAG Pipeline

**Status: ✅ Complete (April 2026)**

This document explains how the Phase 2 pipeline works: text extraction, language filtering, chunking, embedding, retrieval, and LLM integration.

---

## Data flow

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
                                                    (HuatuoGPT-o1-8B via Ollama)
```

---

## Delivered

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
| LLM | HuatuoGPT-o1-8B served via Ollama as `huatuogpt-bone` (HuatuoGPT-o1-8B with a custom bone science system prompt) — streams grounded answers from retrieved context |
| Web UI tabs | **Ask BoneMind** (RAG + LLM streaming) · **Search Corpus** (raw retrieval, no LLM) |

---

## Phase 2 polish

| Item | Result |
|---|---|
| Retrieval benchmark | 30 questions · 7 domains (mechanics, morphology, pathology, biomaterials, simulation, imaging, mechanobiology) |
| MRR | **0.928** |
| Recall@3 | **1.000** — all 30 questions have a relevant chunk in top 3 |
| Recall@5 | **1.000** — Phase 3 readiness threshold passed |
| Citation behaviour | Few-shot example in system prompt · mandatory `[N]` inline citations · DOI links auto-injected into References section |
| Reference formatting | Each `[N]` entry rendered on its own line with clickable Open paper link |
| Eval script | `eval/run_eval.py` — rerun any time corpus or retriever changes |

---

## How to run

### Extract text from PDFs

```bash
python -m processing.extract_papers       # strips reference sections
python -m processing.extract_textbooks
```

### Filter non-English papers

```bash
python -m scripts.filter_english --dry-run   # preview only
python -m scripts.filter_english             # run both passes
```

### Chunk extracted text

```bash
python -m processing.chunk_all    # English papers only · sentence-aware · resumable
```

### Embed chunks

```bash
python -m processing.embed          # resumes from last embedded chunk
python -m processing.embed --force  # re-embed everything from scratch
```

### Query

```bash
# CLI
python -m retrieval.query "cortical bone fracture toughness"
python -m retrieval.query --interactive

# Web UI (requires Ollama + huatuogpt-bone — HuatuoGPT-o1-8B with custom bone science system prompt)
python app.py
```

### Run retrieval benchmark

```bash
python eval/run_eval.py              # default top_k=10
python eval/run_eval.py --top-k 20
```

---

## Key design decisions

### Why RAG before fine-tuning
RAG gives the LLM access to the entire corpus without retraining. Knowledge is updatable — add new papers, re-embed, done. Fine-tuning bakes knowledge into weights and requires retraining whenever the corpus changes.

### Why SPECTER2
Allen AI's 2023 successor to SPECTER, trained on 164M citation relationships. Key advantage over SPECTER1: asymmetric encoding — documents and queries use different task-specific adapters (proximity vs adhoc_query), improving retrieval precision on scientific text compared to general-purpose sentence transformers.

### Sentence-aware chunking
The original implementation used a character-based sliding window that cut text mid-sentence. The final implementation uses NLTK sentence tokenisation — chunks always end at a sentence boundary, with the last 2 sentences of each chunk carried into the next as overlap. This produces cleaner, more semantically coherent passages for embedding.

### Language filtering — two-pass approach
Many papers have English abstracts (translated for indexing) but non-English body text. The filter skips the first 1,000 characters (title/abstract) and detects language from the next 4,000 characters of body text, catching the abstract-English / body-foreign class reliably. Pass A processes all new papers; Pass B re-checks any previously abstract-classified English papers that now have an extracted text file.

### Citation enforcement
The system prompt includes a few-shot example using real corpus references to demonstrate the required `[N]` inline citation format and mandatory `## References` section. The citation rule is placed at the top of the prompt as the "MOST IMPORTANT RULE" because smaller models (8B) weight earlier instructions more strongly.
