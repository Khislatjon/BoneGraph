# BoneGraph

**An intelligent research assistant for bone science** — morphology, mechanics, pathology, imaging, and biomaterials.

[![Live](https://img.shields.io/badge/live-bonegraph.org-2ea44f)](https://bonegraph.org)
[![Status](https://img.shields.io/badge/status-public%20beta-blue)]()
[![Python](https://img.shields.io/badge/python-3.12-3776ab)]()

BoneGraph unifies four research workflows behind one web app: **conversational
question-answering** over a curated literature corpus, **raw semantic search**,
**knowledge-graph-grounded reasoning** with a self-correcting critic, and
**vision-language analysis** of medical images that learns from your corrections.
It runs on a 248,629-chunk embedding index, a hand-cleaned bone knowledge graph,
and open-weight models served through [Ollama](https://ollama.com) — fully local,
with no third-party API calls at inference time.

🔗 **Live beta:** [bonegraph.org](https://bonegraph.org) (running on an NVIDIA Jetson AGX Orin)

> ⚠️ **For research and educational use only. BoneGraph is not a clinical tool and does not provide medical diagnoses.**

<!-- Tip: drop a screenshot or short GIF of the four tabs here, e.g. ![BoneGraph UI](docs/assets/ui.png) -->

---

## Why BoneGraph

Most "chat with your papers" tools stop at retrieval + generation. BoneGraph adds
the two things a research assistant actually needs to be trusted:

- **Deterministic grounding.** The Reasoning tab checks every answer against
  hard-coded fracture-mechanics rules and a bone knowledge graph — physics that
  doesn't depend on what the model felt like saying.
- **It learns from you, honestly.** Correct it once and the correction sticks:
  Reasoning turns your 👎 into a durable rule applied on every future answer;
  Vision remembers a misidentified scan and gets it right the next time a similar
  image appears. This is retrieval and rule-application, **not** opaque weight
  updates — you can list, edit, toggle, and delete everything it has learned.

The whole system is deliberately scoped to **bone science** — depth over breadth.

---

## The four tabs

Each tab is a distinct pipeline over a shared corpus + knowledge graph, exposed by [`api/main.py`](api/main.py).

### 1 · Chat — conversational RAG
Retrieval-augmented question-answering over the **BoneScholar** corpus. A
two-stage bone-relevance guard (lexical pass → `llama3.2:3b` fallback) keeps
questions in domain; SPECTER2 retrieves the top passages; **`huatuogpt-bone`**
streams an answer with inline `[N]` citations that link to the exact source DOI.
The References section is rebuilt server-side so citations never drift.
Multi-turn via a bounded sliding window.

### 2 · Search — raw semantic retrieval
Semantic search over the full corpus with **no LLM in the loop** — what you see
is exactly what the retriever returns. Filter by source (papers/textbooks) and
year; each hit shows the passage, metadata, a DOI link, and its similarity score.
The fastest way to do literature review and to sanity-check what Chat and
Reasoning are reading.

### 3 · Reasoning — agent + critic, grounded in physics and a knowledge graph
A self-correcting loop. A **reasoning agent** (`huatuogpt-bone`) drafts a
Point/Basis answer; it is then checked two ways:
- **Deterministic physical grounding** — fracture-mechanics rules in
  [`reasoning/physical_grounding.py`](reasoning/physical_grounding.py), evaluated in code, not by an LLM.
- **A critic** (`huatuogpt-bone`) that weighs the draft against retrieved
  literature and **1-hop knowledge-graph facts** ([`reasoning/kg_context.py`](reasoning/kg_context.py)),
  returning `accept` / `dispute` / `conflicting_evidence`.

Disputes trigger a revision (hard 2-iteration cap). **Quick** and **Deep** modes
trade latency for the full evidence loop. When you 👎 and explain, `llama3.2:3b`
proposes a structured **user rule**; once you confirm it, it's enforced on every
future request — the "second chat is better" loop. Rules can be listed, toggled,
deleted, and bulk-imported from the in-app Rules manager.

### 4 · Vision — image analysis with correction memory
Upload an X-ray, MRI, micro-CT, or histology image and **`llava:13b`** returns a
structured identification. Its one learning surface is **correction memory**: a
👎 + note is stored keyed by a **BiomedCLIP image embedding** (the original plus
rotated/flipped augments, so a re-windowed or rotated copy of the same scan still
matches). The next time a similar image appears, the prior correction is recalled
and fed to the model — *don't make the same misidentification twice*. Unlike
Reasoning, the Vision tab has **no critic and no grounding rules** by design.

> The Vision tab is explicitly **research and educational only — not a clinical diagnostic tool.**

---

## Accounts & privacy

BoneGraph is behind a lightweight **email/password login** ([`api/auth_store.py`](api/auth_store.py)).
Your account scopes everything you teach it: your Reasoning rules and Vision
corrections are tied to your email and never leak into another user's session.
The 8 built-in physical-grounding rules are global and ship with the app.
Passwords are PBKDF2-HMAC-SHA256 with per-user salts (stdlib only) — a sensible
threat model for a single-install research tool, not a public identity provider.

A **"Send feedback"** button posts bug reports / ideas to a separate store
([`api/beta_feedback.py`](api/beta_feedback.py)); a token-gated **admin dashboard**
at `/admin` shows signups and feedback for the maintainer.

---

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python serve.py          # http://localhost:8000
```

Pull the three Ollama models BoneGraph depends on:

```bash
ollama pull llava:13b        # Vision tab
ollama pull llama3.2:3b      # bone-relevance guard + rule extraction
# huatuogpt-bone is a CUSTOM model (HuatuoGPT-o1-8B + a bone-science system prompt)
# with no public pull — recreate it from a Modelfile (see docs/deployment.md, Step 3).
```

SPECTER2 (text embeddings) and BiomedCLIP (Vision correction memory) download
automatically from HuggingFace on first use. Chat, Search, and Reasoning work
fully offline once the models are present and the embedding index is built;
Vision additionally needs `llava:13b`.

> **Note:** the corpus and graph databases (`data/db/`, ~1.7 GB) are gitignored.
> A fresh clone has the code but not the data — either build the pipeline from
> scratch (below) or copy `data/db/` from an existing install.

---

## Models

| Role | Model | Used by |
|---|---|---|
| Answer generation · reasoning agent · critic | HuatuoGPT-o1-8B (`huatuogpt-bone`, custom Ollama model) | Chat, Reasoning |
| Text embeddings (documents + queries) | SPECTER2 (`allenai/specter2_base` + proximity / ad-hoc query adapters) | Chat, Search, Reasoning |
| Vision-language analysis | LLaVA 13B (`llava:13b`) | Vision |
| Bone-relevance guard · feedback → rule extraction | `llama3.2:3b` | Chat, Reasoning |
| Image embeddings for correction recall | BiomedCLIP (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) | Vision |

Model names, the Ollama base URL (`OLLAMA_URL`), timeouts, and chunking
parameters are centralised in [`config/settings.py`](config/settings.py) and
overridable via environment variables. Each model sits behind a single swap
point — changing the model leaves the pipeline untouched.

---

## BoneScholar corpus

The literature backbone behind the Chat and Search tabs.

| Stat | Value |
|---|---|
| Papers indexed (metadata) | 54,634 |
| Full-text PDFs downloaded | 7,674 |
| Open-access textbooks | 16 |
| Embedded chunks (sentence-aware, ~400 tokens, 2-sentence overlap) | 248,629 |
| Text embedding | SPECTER2 · 768-dim · proximity adapter |

Papers were collected via the **OpenAlex API** (PDFs resolved through OpenAlex OA
URLs, CrossRef, and Wiley TDM) across keyword groups spanning bone morphology,
mechanics, pathology, imaging, and biomaterials. English-language papers are
retained for retrieval; non-English are flagged and excluded. Chunks are
sentence-aware to stay safely under SPECTER2's 512-token limit while preserving
context across boundaries.

---

## Bone knowledge graph

The reasoning backbone behind the Reasoning tab.

| Stat | Value |
|---|---|
| Nodes (unique concepts) | 1,597 |
| Edges (unique relations) | 1,699 |

Seeded with ~200 hand-curated concepts and ~80 causal edges, then grown by LLM
triple extraction over the corpus and **cleaned and reclassified** down to a
high-signal graph (a six-stage cleanup + rule-based concept retyping). The
Reasoning critic reads 1-hop edges around a question's concepts as a grounding
fact source. Browsable renders live in [`visualisation/graph/`](visualisation/graph/).

---

## Retrieval benchmark

A 30-question benchmark across 7 bone-science domains (morphology, mechanics,
pathology, imaging, biomaterials, remodelling, fracture).

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

## Deployment

The public beta runs 24/7 on an **NVIDIA Jetson AGX Orin 64 GB**, exposed at
**bonegraph.org** through a **Cloudflare Tunnel** — no open ports, automatic
HTTPS, near-zero running cost. `uvicorn api.main:app` on `127.0.0.1:8000` +
Ollama serving the three models. Full runbook (systemd units, model recreation,
DB transfer, pre-launch checklist) in **[docs/deployment.md](docs/deployment.md)**.

---

## Building the pipeline from scratch

```bash
# 1 — Collect papers (set CROSSREF_EMAIL + WILEY_TDM_TOKEN in .env first)
cp .env.example .env
python -m ingestion.papers.pipeline              # metadata only
python -m ingestion.papers.pipeline --download   # + open-access PDFs

# 2 — Ingest textbooks (PDFs in data/raw/textbooks/<Source>/book.pdf)
python -m ingestion.textbooks.pipeline

# 3 — Extract → filter → chunk → embed
python -m processing.extract_papers
python -m processing.extract_textbooks
python -m scripts.filter_english
python -m processing.chunk_all
python -m processing.embed                       # resumable; --force re-embeds

# 4 — Build the knowledge graph
python -m reasoning.extractor --source textbooks # LLM triple extraction (resumable)
python -m scripts.clean_graph                    # six-stage cleanup
python -m scripts.reclassify_concepts            # concept retyping
```

`scripts/inspect_db.py` reports progress and statistics at any stage.

---

## Project structure

```
BoneGraph/
├── api/                     # FastAPI backend
│   ├── main.py              #   all endpoints, prompts, critic loop, auth wiring
│   ├── auth_store.py        #   email/password accounts + sessions (auth.db)
│   └── beta_feedback.py     #   "Send feedback" store (beta_feedback.db)
├── frontend/                # Single-file React app (in-browser Babel)
│   ├── index.html           #   Chat · Search · Reasoning · Vision tabs + login
│   ├── admin.html           #   server-rendered admin dashboard (/admin)
│   └── static/
├── ingestion/
│   ├── papers/              # OpenAlex client, resolvers, storage, downloader
│   └── textbooks/           # scanner + storage
├── processing/              # extract → chunk → embed (SPECTER2)
├── retrieval/               # SPECTER2 retrieval engine + CLI
├── reasoning/               # knowledge graph + Reasoning tab backend
│   ├── ontology.py · graph_db.py · seed.py · extractor.py
│   ├── physical_grounding.py# deterministic rules + user-rule compiler
│   ├── feedback_store.py    # events · corrections · user rules (SQLite)
│   ├── rule_extractor.py    # 👎 → proposed rule (llama3.2:3b)
│   └── kg_context.py        # 1-hop KG facts for the critic
├── vision/                  # Vision tab backend
│   ├── encoder.py           # BiomedCLIP image embeddings (+ augments)
│   └── correction_store.py  # image-embedding correction memory (SQLite)
├── config/settings.py       # central settings (paths, model names, constants)
├── scripts/                 # pipeline utilities + graph cleanup/reclassify
├── eval/                    # retrieval + reasoning benchmarks
├── visualisation/graph/     # rendered interactive knowledge-graph HTML
├── data/db/                 # papers · chunks · ontology · feedback · auth (gitignored)
├── docs/                    # per-tab architecture docs + deployment guide
├── serve.py                 # Uvicorn launcher → http://localhost:8000
└── tests/
```

---

## Documentation

| Document | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | System-wide overview — the four tabs, shared substrate, models, design principles |
| [docs/deployment.md](docs/deployment.md) | Jetson AGX Orin + Cloudflare Tunnel deployment runbook |
| [docs/chat/](docs/chat/) | Chat tab + shared ingestion / RAG pipeline docs |
| [docs/search/README.md](docs/search/README.md) | Search tab |
| [docs/reasoning/architecture.md](docs/reasoning/architecture.md) | Reasoning tab — agent + critic loop, rule tiers, KG grounding |
| [docs/vision/architecture.md](docs/vision/architecture.md) | Vision tab — VLM + correction memory (read the scope banner first) |

---

## Tests

```bash
pytest tests/ -v
```

---

> **Disclaimer.** BoneGraph is a research and educational tool. It does not
> provide medical advice, diagnosis, or treatment, and must not be used for
> clinical decision-making.
