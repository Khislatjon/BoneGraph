# BoneGraph

**An intelligent research assistant for bone science** — morphology, mechanics, pathology, imaging, and biomaterials.

[![Live](https://img.shields.io/badge/live-bonegraph.org-2ea44f)](https://bonegraph.org)
[![Status](https://img.shields.io/badge/status-public%20beta-blue)]()
[![Python](https://img.shields.io/badge/python-3.12-3776ab)]()

BoneGraph unifies five research workflows behind one web app: **conversational
question-answering** over a curated literature corpus, **raw semantic search**,
**knowledge-graph-grounded reasoning** with a self-correcting critic,
**vision-language analysis** of medical images that learns from your corrections,
and **deep-learning image mechanics** that predicts bone displacement and strain
fields from a single scan.
It runs on a 248,629-chunk embedding index, a hand-cleaned bone knowledge graph,
and open-weight models served through [Ollama](https://ollama.com) — fully local,
with no third-party API calls at inference time.

🔗 **Live beta:** [bonegraph.org](https://bonegraph.org) (running on an NVIDIA Jetson AGX Orin)

> ⚠️ **For research and educational use only. BoneGraph is not a clinical tool and does not provide medical diagnoses.**

<!-- Tip: drop a screenshot or short GIF of the five tabs here, e.g. ![BoneGraph UI](docs/assets/ui.png) -->

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

## The five tabs

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

### 4 · Vision — image analysis, trained grounding + correction memory
Upload an X-ray, MRI, micro-CT, or histology image and **`llava:13b`** returns a
structured identification, anchored two ways:

- **A trained region head** ([`vision/classifier.py`](vision/classifier.py)) — an
  MLP over **frozen BiomedCLIP** features, trained on **MURA** upper-limb X-rays
  (~36.8k train / 3.2k val). It scores **92.6% accuracy / 0.918 macro-F1** on the
  7-way region task over unseen validation data, and its prediction is passed to
  the VLM as a hint, so the answer is anchored to a model trained on bone data
  rather than the VLM guessing unaided. Inputs outside that distribution (spine,
  MRI, CT, micro-CT) are out of scope, and an image detected as out-of-scope has
  its label **withheld** rather than injected — the confident-wrong failure mode
  came from injecting it anyway.
  Training write-up: [`docs/vision/training.md`](docs/vision/training.md).
- **Correction memory** — a 👎 + note is stored keyed by a **BiomedCLIP image
  embedding** (the original plus rotated/flipped augments, so a re-windowed or
  rotated copy of the same scan still matches). The next time a similar image
  appears, the prior correction is recalled and fed to the model — *don't make
  the same misidentification twice*.

The two are ordered, not blended: a recalled correction **suppresses** the region
hint, because a user correction outranks a trained guess. The hint runs on the
first turn only. Unlike Reasoning, the Vision tab has **no critic and no
deterministic rule tier** by design, and both components degrade gracefully — if
the head's weights or the encoder are absent, the tab still answers from the VLM
alone.

> The Vision tab is explicitly **research and educational only — not a clinical diagnostic tool.**

### 5 · Mechanics — deep-learning displacement & strain prediction
The quantitative counterpart to Vision. Upload one **undeformed micro-CT slice**
and **D2IM** (Soar, Palanca, Dall'Ara & Tozzi, *J. Orthop. Translat.* 2024 —
[paper](https://www.sciencedirect.com/science/article/pii/S2352431624000828),
[code](https://github.com/PeterSoar/D2IM_Prototype)) predicts the **displacement
field** (u, v, w) and, by differentiating the axial component, the **axial strain
field** ε_zz — from the greyscale image alone, no FE model or DVC at inference.
The tab returns a labelled figure (input · displacement · strain) plus summary
statistics in physical units (peak strain, displacement range). A bone mask can
be supplied; otherwise an approximate one is derived from the scan.

D2IM is a fully isolated, swappable adapter ([`mechanics/d2im.py`](mechanics/d2im.py)) —
TensorFlow is lazy-imported and optional, so the other four tabs run without it,
and the [**D2IM-Strain**](https://www.biorxiv.org/content/10.64898/2026.03.31.715417v2)
follow-up drops in by changing the weights path. Architecture:
[`docs/mechanics/architecture.md`](docs/mechanics/architecture.md).

> The Mechanics tab is **research and educational only — not a clinical tool.**

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
at `bonegraph.org/admin` — a static page served from the edge, reading
`/api/admin/*` — shows signups and feedback for the maintainer.

---

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python serve.py          # API on http://localhost:8000 — see the note below
```

`serve.py` starts the **API only**. Since the [edge split](#deployment) the
backend no longer serves the frontend: every route it exposes is under `/api/*`,
and `http://localhost:8000/` is a 404 by design. Browse the endpoints at
`http://localhost:8000/docs`.

To run the UI against it, serve `frontend/` from any static server — the same
files Cloudflare publishes, no build step:

```bash
python -m http.server 5173 --directory frontend    # → http://localhost:5173
```

`API_BASE` in `frontend/index.html` resolves to `http://localhost:8000` off the
public hosts, and CORS is already `allow_origins=["*"]`, so the two talk across
ports with no proxy. To point the UI at another box (a LAN IP, or the Jetson),
set `localStorage.setItem('bg_api_base', 'http://<host>:8000')` in the console.

Pull the Ollama models BoneGraph depends on:

```bash
ollama pull llava:13b        # Vision tab
ollama pull llama3.2:3b      # bone-relevance guard + rule extraction
# huatuogpt-bone is a CUSTOM model (HuatuoGPT-o1-8B + a bone-science system prompt)
# with no public pull. Its definition is in this repo:
ollama create huatuogpt-bone -f huatuogpt-bone.Modelfile
```

> The Modelfile's `FROM` line points at `./huatuogpt-bone.base.gguf`, the
> HuatuoGPT-o1-8B weights — **not** in the repo (too large). Supply that file
> next to the Modelfile, or repoint `FROM` at a base you already have, before
> running `ollama create`. Full walk-through in
> [docs/deployment.md](docs/deployment.md), Step 3.

SPECTER2 (text embeddings) and BiomedCLIP (Vision — correction memory *and* the
region head's frozen features) download automatically from HuggingFace on first
use. Chat, Search, and Reasoning work fully offline once the models are present
and the embedding index is built; Vision additionally needs `llava:13b`, plus
`data/models/vision_region_head.pt` for the trained grounding hint — without it
the tab runs on the VLM alone.

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
| Bone-region grounding head | MLP over frozen BiomedCLIP features, trained on MURA (`data/models/vision_region_head.pt`) | Vision |
| Displacement & strain field prediction | D2IM (TensorFlow/Keras CNN, `D2IM_trained.h5`) | Mechanics |

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

## Evaluation

Benchmarks live under [`eval/`](eval/), one subfolder per target.

### Retrieval ([`eval/retrieval/`](eval/retrieval/))

A 30-question benchmark across 7 bone-science domains (morphology, mechanics,
pathology, imaging, biomaterials, remodelling, fracture). Each question has a
known target passage; the metric is where the retriever ranks it.

| Metric | Score |
|---|---|
| MRR | **0.928** |
| Recall@1 | **0.867** (26 / 30) |
| Recall@3 | **1.000** (30 / 30) |
| Recall@5 | **1.000** (30 / 30) |

```bash
python eval/retrieval/run_eval.py           # default top_k=10
python eval/retrieval/run_eval.py --top-k 5
```

Results are saved to `eval/retrieval/results.json`.

> **Read this alongside the MCQ results below.** These scores measure *topical*
> retrieval, which is what Chat and Search need. They do not transfer to
> quantitative reasoning: SPECTER2 is trained for document-level citation
> similarity, and applied to 248,629 arbitrary passages it compresses the whole
> corpus into a ~0.76–0.84 cosine band. Measured over the 50-item MCQ set, the
> passage carrying the *deciding quantity* reached the prompt for **10/50** items
> under dense top-5 retrieval versus **31/50** under BM25. See the rationale in
> [`eval/mcq/build_fts_index.py`](eval/mcq/build_fts_index.py).

### Grounded reasoning — MCQ ([`eval/mcq/`](eval/mcq/))

A 50-item multiple-choice benchmark of questions that must be answered by
**calculating with reported quantities**, not by recognising a familiar phrase.
Each item is built from a quantity stated somewhere in the corpus
(`source_chunks` records exactly which passage), but the answer itself appears
nowhere — it takes a unit conversion and an arithmetic step — so an item is
solved only if the deciding number actually reaches the prompt. That makes the
set a probe of **grounding**, not recall.

Grading is deterministic letter extraction from `\boxed{}` — no LLM judge, which
would have to be stronger in-domain than the system it grades. Options are
shuffled per item under a fixed seed, `temperature=0`, and every raw generation
is stored so refusals and hedges can be diagnosed rather than silently scored
wrong.

The arms share one system prompt and one grading path — **only the evidence
appended to the user message changes**, so any difference is attributable to
retrieval and nothing else. Accuracy at `seed=17`, n=50:

| Arm | What it sees | prompt v1 | prompt v2 |
|---|---|---|---|
| `bare` (closedbook) | nothing — the model alone | 0.36 | 0.42 |
| `bm25` | 14 whole chunks (~1,700 chars each) by lexical rank | 0.46 | 0.52 |
| `bm25rerank` | a 150-deep BM25 pool reranked down to 4 | 0.46 | **0.58** |
| `oracle` | the passage known to hold the answer, by chunk id | 0.66 | **0.78** |

Three readings, and the first is not flattering. **The shipped Chat/Reasoning
retrieval path scored 13/50 (26%) on these items — below closed-book's 18/50.**
The oracle arm exists to disambiguate that: at 66% it proves the model *can* use
the evidence, so the whole gap was retrieval failing to deliver it. Three causes
were found and fixed in the `bm25` arms — per-passage truncation at 840 chars
(the deciding quantity sat past that cut in 7 of 10 oracle items), no lexical
channel at all, and question-shaped queries embedding far from property tables.
See the failure analyses in the [`run_bm25.py`](eval/mcq/run_bm25.py) and
[`run_bm25rerank.py`](eval/mcq/run_bm25rerank.py) docstrings. Both lexical arms
run with the dense channel **off**: reciprocal-rank fusion with SPECTER2 is
available behind `--use-vector` and measured net negative on this set.

Second: a large ceiling remains. Oracle at 0.78 against the best real retrieval
at 0.58 means the outstanding 20 points are a *retrieval* problem, not a model
one. Third: the prompt carries as much weight as the evidence. v2 (state values
→ convert units → calculate → match) buys +6 to +12 points over v1 in every arm,
because most failures are unit-mixing rather than ignorance — and precision only
converts under v2, where the model uses evidence when it has it (61% with the
needle vs 37% without, against a near-flat 48/42 under v1).

```bash
python -m eval.mcq.build_fts_index          # BM25 index over chunks.db (~40 s, ~180 MB)

# every runner takes --prompt / --items / --seed / --out; --prompt defaults to v1
python -m eval.mcq.run_arms --arm bare --prompt eval/mcq/prompt_v2.txt
python -m eval.mcq.run_bm25        --prompt eval/mcq/prompt_v2.txt
python -m eval.mcq.run_bm25rerank  --prompt eval/mcq/prompt_v2.txt
python -m eval.mcq.run_oracle      --prompt eval/mcq/prompt_v2.txt
```

`run_arms.py` also carries the original `rag`, `graph` and `full` arms — the
shipped pipeline's own evidence path, kept so the 26% result above stays
reproducible.

Run outputs land in `eval/mcq/results/` (gitignored). The BM25 index is not in
version control — `chunks.db` isn't either — so `build_fts_index.py` is the
reproduction path for the two lexical arms.

### Other harnesses

[`eval/evidence/`](eval/evidence/) audits what evidence reaches the Reasoning
critic; [`eval/feedback/`](eval/feedback/) demonstrates the 👎 → rule → enforced
loop end to end.

---

## Deployment

The public beta is a **split deployment** — no open ports, automatic HTTPS,
near-zero running cost:

- **Frontend** — `frontend/` ships as an **assets-only Cloudflare Worker**, served
  from the global edge at **bonegraph.org**. No build step (JSX is transpiled in
  the browser); `git push` redeploys it. It stays up even when the Jetson doesn't.
- **API** — `run_api.sh` → `uvicorn api.main:app` on `127.0.0.1:8000` plus Ollama,
  on an **NVIDIA Jetson AGX Orin 64 GB** at home, reached through a **Cloudflare
  Tunnel** at **api.bonegraph.org**. The frontend calls it cross-origin via
  `API_BASE`; the backend serves no HTML at all.

The split exists because the Jetson's Wi-Fi drops several times an hour. When one
process served both, every drop took the whole domain down; now a blip degrades a
single in-flight query instead of the site. Full runbook (systemd units, tunnel
config, Worker setup, model recreation, DB transfer, pre-launch checklist) in
**[docs/deployment.md](docs/deployment.md)**.

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
├── api/                     # FastAPI backend — /api/* only, serves no HTML
│   ├── main.py              #   all endpoints, prompts, critic loop, auth wiring
│   ├── auth_store.py        #   email/password accounts + sessions (auth.db)
│   └── beta_feedback.py     #   "Send feedback" store (beta_feedback.db)
├── frontend/                # Single-file React app (in-browser Babel), Cloudflare-served
│   ├── index.html           #   Chat · Search · Reasoning · Vision · Mechanics tabs + login
│   ├── admin.html           #   admin dashboard (/admin) — static, calls /api/admin/*
│   ├── wrangler.jsonc       #   assets-only Worker config (build root = frontend/)
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
│   ├── rule_import.py       # bulk import of user rules
│   └── kg_context.py        # 1-hop KG facts for the critic
├── vision/                  # Vision tab backend
│   ├── encoder.py           # BiomedCLIP image embeddings (+ augments)
│   ├── classifier.py        # MURA-trained region head → VLM grounding hint
│   ├── correction_store.py  # image-embedding correction memory (SQLite)
│   └── training/            # MURA dataset · feature cache · head training
├── mechanics/               # Mechanics tab backend
│   └── d2im.py              # D2IM adapter: preprocess → predict → strain → render
├── config/settings.py       # central settings (paths, model names, constants)
├── scripts/                 # pipeline utilities + graph cleanup/reclassify
├── eval/                    # benchmarks, one folder per target
│   ├── retrieval/           #   ranking benchmark (MRR, Recall@k)
│   ├── mcq/                 #   grounded-reasoning MCQ, four retrieval arms
│   ├── evidence/            #   what evidence reaches the critic
│   └── feedback/            #   👎 → rule → enforced, end to end
├── visualisation/graph/     # rendered interactive knowledge-graph HTML
├── data/db/                 # papers · chunks · ontology · feedback · auth (gitignored)
├── docs/                    # per-tab architecture docs + deployment guide
├── huatuogpt-bone.Modelfile # custom Ollama model definition (base GGUF not in git)
├── run_api.sh               # production launcher (aarch64 workarounds) — the Jetson
├── serve.py                 # Uvicorn launcher → API on http://localhost:8000
└── tests/
```

---

## Documentation

| Document | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | System-wide overview — the tabs, shared substrate, models, design principles |
| [docs/deployment.md](docs/deployment.md) | Cloudflare edge frontend + Jetson API deployment runbook |
| [docs/chat/](docs/chat/) | Chat tab + shared ingestion / RAG pipeline docs |
| [docs/search/README.md](docs/search/README.md) | Search tab |
| [docs/reasoning/architecture.md](docs/reasoning/architecture.md) | Reasoning tab — agent + critic loop, rule tiers, KG grounding |
| [docs/vision/architecture.md](docs/vision/architecture.md) | Vision tab — VLM + correction memory (read the scope banner first) |
| [docs/vision/training.md](docs/vision/training.md) | Vision tab — MURA region head: data, training, results |
| [docs/mechanics/architecture.md](docs/mechanics/architecture.md) | Mechanics tab — D2IM adapter, preprocessing, strain derivation |

---

## Tests

```bash
pip install pytest        # not in requirements.txt — dev-only
pytest tests/ -v
```

`tests/test_ingestion.py` mixes offline storage tests with **integration** tests
that call a live paper API, so the network-facing half needs
`SEMANTIC_SCHOLAR_API_KEY` in the environment and will fail without it.

---

> **Disclaimer.** BoneGraph is a research and educational tool. It does not
> provide medical advice, diagnosis, or treatment, and must not be used for
> clinical decision-making.
