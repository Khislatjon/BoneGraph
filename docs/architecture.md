# BoneGraph — System Architecture

**Status: 🟢 Public beta (June 2026).** Live at **bonegraph.org**, served from an
NVIDIA Jetson AGX Orin behind a Cloudflare Tunnel. See [`deployment.md`](deployment.md).

This is the system-wide overview. Each tab has its own authoritative deep doc —
this file explains how the pieces fit together and points to them.

---

## What is BoneGraph?

BoneGraph is an intelligent reasoning system for bone science, delivered as a
four-tab web application over a curated bone-science corpus and a bone knowledge
graph. It is deliberately restricted to the bone domain because of the richness
of available data: published papers, textbooks, mechanical measurements, X-ray,
MRI, micro-CT, and histology images.

The goal is not a generic question-answering chatbot. The goal is a system that can:

1. **Understand** bone science at the level of morphology, structure–function
   relationships, mechanics, and pathology — over both text and images.
2. **Reason** over that knowledge to answer clinical and research questions, with
   a critic that weighs the answer against literature and a knowledge graph.
3. **Ground** its claims in physics — deterministic checks that catch numeric and
   directional nonsense regardless of what the model says.
4. **Improve** over time as users correct it: corrections become durable, per-user
   rules (Reasoning) and recalled image memories (Vision).

---

## The product — four tabs

BoneGraph is one React app with four tabs, each a distinct pipeline over a shared
substrate (corpus + embeddings + knowledge graph + models). Behind a login.

| Tab | Internal name | What it does | Primary model | LLM in the loop? | Deep doc |
|-----|---------------|--------------|---------------|:---------------:|----------|
| **Chat** | `ask` | RAG question-answering with inline `[N]` citations + References, multi-turn | `huatuogpt-bone` | ✅ | [chat/README.md](chat/README.md) |
| **Search** | `search` | Raw semantic search over the corpus, ranked by cosine similarity | SPECTER2 only | ❌ | [search/README.md](search/README.md) |
| **Reasoning** | `reason` | Agent → physical grounding → critic loop with feedback-driven user rules | `huatuogpt-bone` | ✅ | [reasoning/architecture.md](reasoning/architecture.md) |
| **Vision** | `analyse` | VLM identifies a bone image; a 👎+note is remembered for similar images | `llava:13b` | ✅ | [vision/architecture.md](vision/architecture.md) |

Chat and Search share one retrieval engine and corpus. Reasoning and Vision share
one architectural pattern (agent + feedback memory), but Vision is deliberately
narrower — **no critic and no grounding rules**, correction memory only.

---

## System architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CLIENT — frontend/index.html (single-file React, in-browser Babel)        │
│  Login · Chat · Search · Reasoning · Vision · "Send feedback"              │
│  frontend/admin.html — server-rendered admin dashboard (/admin)           │
└───────────────────────────────────┬────────────────────────────────────────┘
                                     │  HTTPS (Cloudflare Tunnel → 127.0.0.1:8000)
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  API — FastAPI (api/main.py)                                              │
│  Auth (Bearer token → user_id = email) gates every data endpoint          │
│                                                                            │
│   /api/ask        /api/search      /api/reason/*      /api/vision/*        │
│   (Chat: RAG)     (Search: cosine) (agent+critic loop)(VLM + recall)       │
│        │               │                 │                  │             │
└────────┼───────────────┼─────────────────┼──────────────────┼─────────────┘
         │               │                 │                  │
         ▼               ▼                 ▼                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  SHARED SUBSTRATE                                                          │
│                                                                            │
│  Corpus + embeddings        Knowledge graph        Per-user feedback       │
│  ─ papers.db (54,634)       ─ ontology.db          ─ reasoning_feedback.db │
│  ─ textbooks.db (16)          (1,597 nodes /        (rules, corrections)   │
│  ─ chunks.db (248,629         1,699 edges,         ─ vision_feedback.db    │
│    chunks + SPECTER2          cleaned)               (image-embed memory)  │
│    768-d embeddings)                                ─ auth.db · beta_feedback.db │
└───────────────────────────────────┬────────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  MODELS                                                                    │
│  Ollama (local):  huatuogpt-bone · llava:13b · llama3.2:3b                 │
│  HuggingFace:     SPECTER2 (text embeddings) · BiomedCLIP (image embeds)   │
└──────────────────────────────────────────────────────────────────────────┘
```

The original two-layer "perception → reasoning (LLM/VLM/LRM)" framing has been
superseded by what was actually built: four tab pipelines over a shared corpus
and knowledge graph, with a per-user feedback layer. The perception/reasoning
spirit survives — Search/Chat/Vision are perception, Reasoning is the reasoning
layer — but the system is best understood tab-by-tab.

---

## Shared substrate

### Corpus + embeddings (Chat + Search)
- **54,634** paper metadata rows (OpenAlex) · **7,674** downloaded PDFs · **16** textbooks.
- **248,629** sentence-aware chunks (NLTK · 400 tokens · 2-sentence overlap),
  each embedded with **SPECTER2** (`allenai/specter2_base` + the proximity
  adapter, 768-dim) and stored in `chunks.db`.
- Retrieval is cosine similarity via `BoneGraphRetriever`. Benchmark: **MRR 0.928,
  Recall@5 1.000** (`eval/retrieval/run_eval.py`).
- Pipeline detail: [chat/papers_ingestion_pipeline.md](chat/papers_ingestion_pipeline.md),
  [chat/textbooks_ingestion_pipeline.md](chat/textbooks_ingestion_pipeline.md),
  [chat/phase2_rag_pipeline.md](chat/phase2_rag_pipeline.md).

### Knowledge graph (Reasoning critic)
- `ontology.db` — a cleaned bone-science graph of **1,597 nodes / 1,699 edges**.
- Built by seeding ~200 concepts + ~80 hand-curated causal edges, then extracting
  `(node_1, relation, node_2)` triples from the corpus with `huatuogpt-bone`,
  then a six-stage cleanup (`scripts/clean_graph.py`) and a rule-based concept
  reclassification (`scripts/reclassify_concepts.py`). The raw and intermediate
  graphs are kept as `ontology_raw.db` and `ontology_pre_typed.db`.
- The Reasoning critic reads 1-hop edges around question concepts from this graph;
  it is read-only for the app. See [reasoning/phase4_lrm_plan.md](reasoning/phase4_lrm_plan.md),
  [reasoning/graph_cleanup.md](reasoning/graph_cleanup.md),
  [reasoning/graph_concept_reclassification.md](reasoning/graph_concept_reclassification.md).
- Rendered, browsable visualisations live in [`visualisation/graph/`](../visualisation/graph/).

### Models
| Model | Where | Role |
|-------|-------|------|
| `huatuogpt-bone` | Ollama (custom: HuatuoGPT-o1-8B + bone system prompt) | Chat answers; Reasoning agent + critic |
| `llava:13b` | Ollama | Vision tab VLM (image identification / report) |
| `llama3.2:3b` | Ollama | Topic guard (Chat + Reasoning); Reasoning rule extractor |
| SPECTER2 | HuggingFace (`allenai/specter2_base` + adapter) | 768-d text embeddings for retrieval |
| BiomedCLIP | HuggingFace (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) | 512-d image embeddings for Vision correction recall |

`huatuogpt-bone` is a **custom** Ollama model with no public pull — it is recreated
from its base + system prompt on deploy (see deployment Step 3). Everything else
pulls/downloads automatically.

---

## The four tabs in detail

### Chat (`/api/ask`) ✅
Retrieval-augmented question answering. SPECTER2 retrieves the top passages,
`huatuogpt-bone` answers with inline `[N]` citations, and the **References section
is rebuilt server-side** from the retrieved results (the model's own References
block is discarded because it renumbers). Multi-turn via a sliding window of the
last few Q/A pairs, with thinking-blocks and References stripped to stay inside
the 8,192-token context. Topic-guarded (lexical vocab → `llama3.2:3b` fallback).

### Search (`/api/search`) ✅
No LLM. A query is embedded with SPECTER2 and ranked by cosine similarity against
the indexed chunks; results return with source metadata and relevance scores.
This is the retrieval engine of the Chat tab, exposed raw.

### Reasoning (`/api/reason/*`) 🟢
The trustworthy-AI tab. A streamed **agent** (`huatuogpt-bone`, primed with the
user's learned rules, *no* retrieval in its prompt) produces a Point/Basis chain.
Its answer passes a deterministic **physical-grounding** check (8 built-in rules +
the user's own rules). In *deep* mode a **critic** (`huatuogpt-bone`, JSON) then
reviews it against retrieved literature and 1-hop knowledge-graph edges and returns
`accept` / `dispute` / `conflicting_evidence`; disputes trigger an agent revision
(hard cap: 2 critic iterations). A 👎 + correction is turned into a structured rule
proposal by `llama3.2:3b`, which the user confirms into their personal rule
registry — applied on every future request (the "second chat is better" loop).
Full pipeline, agent-vs-critic input matrix, rule kinds, and SSE event schema in
[reasoning/architecture.md](reasoning/architecture.md);
build log and demo flow in [reasoning/reasoning_tab.md](reasoning/reasoning_tab.md).

### Vision (`/api/vision/*`, legacy `/api/analyse`) 🟡
A VLM (`llava:13b`) identifies an uploaded bone image (X-ray, MRI, micro-CT,
histology) and returns a structured identification / report. Its **only** learning
surface is **correction memory**: a 👎 + free-text note is stored keyed by a
**BiomedCLIP image embedding** (the original + four augments — rot90/180/270 +
hflip — so a rotated/re-windowed copy still matches). On the next similar image,
`correction_store.recall` cosine-matches and prepends the prior correction to the
VLM prompt — "don't make the same misidentification twice" (the *Cephalo lesson*).
Unlike Reasoning, the Vision tab has **no critic and no grounding rules** (that
fuller mirror design is documented but deferred). The original one-shot
`/api/analyse` remains as a fallback. See [vision/architecture.md](vision/architecture.md)
(read its scope banner first) and [vision/phase0_reuse_map.md](vision/phase0_reuse_map.md).

---

## Accounts & per-user data

Every data endpoint is behind email/password auth (`api/auth_store.py`):

- Register/login returns a Bearer **session token**; `get_current_user` resolves
  it to the account's **email**, which becomes the `user_id` threaded everywhere.
- That `user_id` scopes per-user data: a clinician's Reasoning rules and Vision
  corrections never leak into another account's. The 8 built-in Tier-1 grounding
  rules stay global and hardcoded.
- Passwords are PBKDF2-HMAC-SHA256 with per-user salts (stdlib only); tokens are
  `secrets.token_urlsafe`. This is a per-install tool, not a public identity
  provider — good enough for that threat model. Stored in `auth.db`.

**Two-tier rules (Reasoning):** Tier 1 = the 8 built-in physical rules, shipped
with releases; Tier 2 = each user's own rules (from feedback + bulk import),
grown locally in `reasoning_feedback.db`.

---

## Feedback & admin

Two completely separate feedback channels:

1. **Per-tab learning feedback** — shapes model behaviour. Reasoning corrections →
   user rules (`reasoning_feedback.db`); Vision corrections → image memory
   (`vision_feedback.db`).
2. **Product feedback** — the "Send feedback" button. Free-text bug reports /
   ideas / questions land in `beta_feedback.db` (`api/beta_feedback.py`); no
   embeddings, no recall — just for the maintainer to read.

**Admin dashboard** (`/admin`, `frontend/admin.html`) shows signups + beta
feedback via `/api/admin/overview`, gated by the `FEEDBACK_ADMIN_TOKEN` env var
(required in production; open when unset locally).

---

## API surface

```
Auth     POST /api/auth/register · /api/auth/login · /api/auth/logout · GET /api/auth/me
Stats    GET  /api/stats
Chat     POST /api/ask                         (SSE)
Search   POST /api/search                      (JSON)
Reason   POST /api/reason/chat                 (SSE)
         POST /api/reason/feedback
         POST /api/reason/rules/confirm · import   · GET /api/reason/rules · /api/reason/rules/template
         POST /api/reason/rules/{id}/enabled   · DELETE /api/reason/rules/{id}
Vision   POST /api/vision/chat                 (SSE, multipart)
         POST /api/vision/feedback             · GET /api/vision/corrections · DELETE /api/vision/corrections/{id}
         POST /api/analyse                     (legacy one-shot VLM)
Feedback POST /api/feedback · GET /api/feedback/list
Admin    GET  /admin · /api/admin/overview     (token-gated)
Pages    GET  / (frontend) · /test
```

---

## Data stores (`data/db/`, all gitignored)

| File | Contents | ~Size |
|------|----------|-------|
| `papers.db` | 54,634 paper metadata rows | 112 MB |
| `textbooks.db` | 16 textbook metadata rows | small |
| `chunks.db` | 248,629 chunks + SPECTER2 embeddings | 1.5 GB |
| `ontology.db` | Cleaned knowledge graph (1,597 nodes / 1,699 edges) | ~1.8 MB |
| `ontology_raw.db` · `ontology_pre_typed.db` | Pre-cleanup / pre-reclassification graph snapshots | — |
| `reasoning_feedback.db` | Reasoning: events, corrections, user_rules | small |
| `vision_feedback.db` | Vision: events, corrections, correction_embeddings | small |
| `auth.db` | Accounts + sessions (created on first signup) | small |
| `beta_feedback.db` | Product feedback (created on first submission) | small |

---

## Domain scope

BoneGraph is intentionally restricted to bone science — a design decision that
enables depth over breadth, and is enforced by the topic guard on Chat/Reasoning.

| Domain | What BoneGraph covers |
|---|---|
| **Morphology** | Bone shape/geometry; cortical/trabecular organisation; osteocyte lacunar networks; Haversian systems |
| **Structure–function** | How hierarchical structure (nanoscale collagen–mineral → macroscale geometry) determines mechanical behaviour |
| **Mechanics** | Elastic modulus, fracture toughness, fatigue, viscoelasticity, crack propagation, FEA |
| **Pathology** | Osteoporosis, Paget's disease, osteogenesis imperfecta, bone metastasis, avascular necrosis, stress fractures |
| **Imaging** | X-ray, MRI, micro-CT, histology interpretation; relating radiological appearance to structural change |
| **Biomaterials** | Bone scaffolds, grafts, tissue engineering, 3D-printed substitutes |
| **Simulation** | Multiscale FEA, bone remodelling models, molecular dynamics of mineral |

---

## Build phases

| Phase | Status | Detail |
|---|---|---|
| 1 — Paper ingestion | ✅ Complete (Mar 2026) | OpenAlex + CrossRef + Wiley TDM · [chat/papers_ingestion_pipeline.md](chat/papers_ingestion_pipeline.md) |
| 1b — Textbook ingestion | ✅ Complete (Apr 2026) | [chat/textbooks_ingestion_pipeline.md](chat/textbooks_ingestion_pipeline.md) |
| 2 — Text processing & RAG | ✅ Complete (Apr 2026) | Chat + Search tabs · [chat/phase2_rag_pipeline.md](chat/phase2_rag_pipeline.md) |
| 4 — Bone knowledge graph | ✅ Complete (May 2026) | Seed → extract → clean → reclassify → 1,597 / 1,699 · [reasoning/phase4_lrm_plan.md](reasoning/phase4_lrm_plan.md) |
| 4b — Reasoning tab | 🟢 Live (May 2026) | Agent + critic + grounding + feedback rules · [reasoning/architecture.md](reasoning/architecture.md) |
| 3 — Vision tab | 🟡 Partial (Jun 2026) | VLM chat ✅ + BiomedCLIP correction memory ✅; critic/rules deferred · [vision/architecture.md](vision/architecture.md) |
| 5 — Feedback loop | 🟢 Reasoning + Vision live | Reasoning user rules; Vision correction memory; Chat/Search feedback into retrieval still planned |
| — Accounts + beta | 🟢 Live (Jun 2026) | Email/password auth, per-user scoping, admin dashboard, deployed at bonegraph.org |

---

## Key design principles

**1. Determinism where it matters.** Physical grounding and user rules are code,
not an LLM — same input, same outcome. Probabilistic, evidence-weighing work lives
only in the critic.

**2. The agent reasons; the critic judges.** In the Reasoning tab the agent never
sees retrieval — that keeps its identity distinct from Chat and its context lean.
All literature/KG evidence reaches the critic only. User rules are the one channel
into both, because they are ground truth, not evidence.

**3. User corrections are non-negotiable and remembered.** A correction the user
taught overrides the model — enforced deterministically (Reasoning rules) or
recalled on a similar input (Vision memory). "The second chat is better." This is
honest retrieval/rule application, **not** weight updates.

**4. Honest scope.** Identification and reasoning over bone science — not clinical
diagnosis. Off-domain questions are politely redirected.

**5. Modular, replaceable models.** Each model sits behind one swap point — change
`OLLAMA_MODEL`/`VLM_MODEL`/the encoder file and the pipeline is unchanged.

---

## Deployment

Production runs as a 24/7 public beta on an **NVIDIA Jetson AGX Orin 64 GB**,
exposed at **bonegraph.org** through a **Cloudflare Tunnel** (no open ports, HTTPS
automatic). `uvicorn api.main:app` on `127.0.0.1:8000` + Ollama serving the three
models; SQLite DBs are transferred to the device (gitignored, not cloned). Full
runbook, systemd units, and pre-launch checklist in [deployment.md](deployment.md).

---

## Repository layout

```
BoneGraph/
│
├── config/
│   └── settings.py             All constants, paths, API + model config
│
├── ingestion/                  Phase 1: data collection
│   ├── papers/                 keywords · openalex · semantic_scholar · resolvers
│   │                           · storage · downloader · pipeline
│   └── textbooks/              scanner · storage · pipeline
│
├── scripts/                    One-off pipeline + maintenance scripts
│   ├── download_pass1.py · download_pass2.py · filter_english.py
│   ├── inspect_db.py
│   ├── clean_graph.py · reclassify_concepts.py   (graph cleanup)
│   └── test_vlm.py
│
├── processing/                 Phase 2: extract → chunk → embed
│   ├── extractor.py · extract_papers.py · extract_textbooks.py
│   ├── chunker.py · chunk_all.py · chunk_store.py
│   └── embed.py                SPECTER2 embeddings → chunks.db
│
├── retrieval/                  Phase 2: RAG engine (Chat + Search)
│   ├── retriever.py            BoneGraphRetriever — load embeddings, cosine search
│   └── query.py                CLI entrypoint
│
├── reasoning/                  Phase 4 + Reasoning tab
│   ├── ontology.py · graph_db.py · seed.py · extractor.py
│   ├── visualize_ontology.py
│   ├── physical_grounding.py   built-in + user-rule compiler + check()
│   ├── feedback_store.py       events · corrections · user_rules
│   ├── rule_extractor.py       👎 correction → proposed rule (llama3.2:3b)
│   ├── rule_import.py          CSV/XLSX bulk rule import
│   └── kg_context.py           1-hop KG facts for the critic
│
├── vision/                     Vision tab
│   ├── encoder.py              BiomedCLIP image embeddings (lazy singleton) + augments
│   └── correction_store.py     SQLite: events · corrections · embedding recall
│
├── eval/                       Benchmarks
│   ├── retrieval/              MRR / Recall@k — run_eval.py · benchmark.json · results.json
│   ├── evidence/               Reasoning: literature + KG quality audit
│   ├── feedback/               "second chat is better" demo harness
│   └── mcq/                    Multiple-choice benchmark + corpus checks
│
├── api/                        FastAPI backend
│   ├── main.py                 All endpoints, prompts, critic loop, auth wiring
│   ├── auth_store.py           Email/password accounts + sessions (auth.db)
│   └── beta_feedback.py        Product-feedback store (beta_feedback.db)
│
├── frontend/                   React web UI
│   ├── index.html              Single-file React app (Babel in-browser); 4 tabs + feedback
│   ├── admin.html              Server-rendered admin dashboard (/admin)
│   └── static/                 Vendored React / ReactDOM / Babel
│
├── data/                       (gitignored)
│   ├── raw/papers/ · raw/textbooks/      Downloaded PDFs
│   ├── processed/text/                   Extracted .txt
│   └── db/                                papers · textbooks · chunks · ontology(+raw/pre_typed)
│                                         · reasoning_feedback · vision_feedback · auth · beta_feedback
│
├── docs/
│   ├── architecture.md          ← this file (system-wide overview)
│   ├── deployment.md            Jetson + Cloudflare Tunnel runbook
│   ├── chat/                    Chat tab + shared ingestion/RAG docs
│   │   ├── README.md
│   │   ├── papers_ingestion_pipeline.md · textbooks_ingestion_pipeline.md
│   │   └── phase2_rag_pipeline.md
│   ├── search/README.md         Search tab
│   ├── reasoning/               Reasoning tab (authoritative architecture + build log + evals)
│   │   ├── architecture.md · reasoning_tab.md
│   │   ├── evidence_layer.md · feedback_demo.md
│   │   ├── phase4_lrm_plan.md · graph_cleanup.md · graph_concept_reclassification.md
│   └── vision/                  Vision tab
│       ├── README.md · architecture.md
│       └── phase0_reuse_map.md · phase3_vlm_plan.md
│
├── visualisation/graph/         Rendered interactive knowledge-graph HTML
├── mypaper/ · presentation/     Research write-up (LaTeX) + slides
├── tests/                       test_ingestion.py
│
├── serve.py                     Uvicorn launcher — FastAPI at http://localhost:8000
├── requirements.txt
├── .env / .env.example          API keys (gitignored) / template
└── README.md
```
