# BoneGraph — System Architecture

**Status: 🟢 Public beta.** Live at **bonegraph.org** as a split deployment: the
frontend from Cloudflare's edge, the API from an NVIDIA Jetson AGX Orin behind a
Cloudflare Tunnel. See [`deployment.md`](deployment.md).

This is the system-wide overview. Each tab has its own authoritative deep doc —
this file explains how the pieces fit together and points to them.

---

## What is BoneGraph?

BoneGraph is an intelligent reasoning system for bone science, delivered as a
five-tab web application over a curated bone-science corpus and a bone knowledge
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
5. **Predict** mechanical behaviour from an image — displacement and strain
   fields from a single scan, with no FE model or DVC at inference (Mechanics).

---

## The product — five tabs

BoneGraph is one React app with five tabs, each a distinct pipeline over a shared
substrate (corpus + embeddings + knowledge graph + models). Behind a login.

| Tab | Internal name | What it does | Primary model | LLM in the loop? | Deep doc |
|-----|---------------|--------------|---------------|:---------------:|----------|
| **Chat** | `ask` | RAG question-answering with inline `[N]` citations + References, multi-turn | `huatuogpt-bone` | ✅ | [chat/README.md](chat/README.md) |
| **Search** | `search` | Raw semantic search over the corpus, ranked by cosine similarity | SPECTER2 only | ❌ | [search/README.md](search/README.md) |
| **Reasoning** | `reason` | Agent → physical grounding → critic loop with feedback-driven user rules | `huatuogpt-bone` | ✅ | [reasoning/architecture.md](reasoning/architecture.md) |
| **Vision** | `vision` | VLM identifies a bone image, grounded by a trained region head; a 👎+note is remembered for similar images | `llava:13b` | ✅ | [vision/architecture.md](vision/architecture.md) |
| **Mechanics** | `mechanics` | Predicts displacement + axial-strain fields from one undeformed micro-CT slice | D2IM (TF/Keras CNN) | ❌ | [mechanics/architecture.md](mechanics/architecture.md) |

Chat and Search share one retrieval engine and corpus. Reasoning and Vision share
one architectural pattern (agent + feedback memory), but Vision is deliberately
narrower — **no critic and no deterministic rule tier**; its grounding is a
trained classifier hint, and its learning surface is correction memory. Mechanics
is the outlier: a pure forward pass over a supervised model, with no LLM, no
retrieval and no feedback loop — the tab exists to put a *quantitative* answer
next to Vision's qualitative one.

---

## System architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│  CLIENT — frontend/ published as an assets-only Cloudflare Worker,         │
│           served from the global edge at bonegraph.org                     │
│                                                                            │
│  index.html   single-file React app (Babel in-browser)                     │
│               Login · Chat · Search · Reasoning · Vision · Mechanics       │
│  admin.html   static admin dashboard (/admin) · "Send feedback" → /api     │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │  HTTPS, cross-origin → api.bonegraph.org
                                    │  (Cloudflare Tunnel → Jetson 127.0.0.1:8000)
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  API — FastAPI (api/main.py) · /api/* ONLY, serves no HTML                 │
│  Auth (Bearer token → user_id = email) gates every data endpoint           │
│                                                                            │
│  /api/ask   /api/search  /api/reason/*  /api/vision/*  /api/mechanics/*    │
│  RAG chat   cosine only  agent+critic   VLM + head +   D2IM CNN            │
│                          + KG grounding correction mem  (no LLM)           │
│     │           │             │              │              │              │
└─────┼───────────┼─────────────┼──────────────┼──────────────┼──────────────┘
      │           │             │              │              │
      ▼           ▼             ▼              ▼              ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  SHARED SUBSTRATE — Chat · Search · Reasoning · Vision                     │
│                     (Mechanics reads none of it)                           │
│                                                                            │
│  Corpus + embeddings      Knowledge graph       Per-user data              │
│  ─ papers.db   54,634     ─ ontology.db         ─ reasoning_feedback.db    │
│  ─ textbooks.db    16       1,597 nodes /         (rules · corrections)    │
│  ─ chunks.db  248,629       1,699 edges          ─ vision_feedback.db      │
│    + SPECTER2 768-d         (cleaned)              (image-embed memory)    │
│                                                  ─ auth.db                 │
│                                                  ─ beta_feedback.db        │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  MODELS                                                                    │
│  Ollama (local)   huatuogpt-bone · llava:13b · llama3.2:3b                 │
│  HuggingFace      SPECTER2 (text 768-d) · BiomedCLIP (image 512-d)         │
│  data/models/     vision_region_head.pt (MURA) · D2IM_trained.h5           │
└────────────────────────────────────────────────────────────────────────────┘
```

The original two-layer "perception → reasoning (LLM/VLM/LRM)" framing has been
superseded by what was actually built: five tab pipelines over a shared corpus
and knowledge graph, with a per-user feedback layer. The perception/reasoning
spirit survives — Search/Chat/Vision are perception, Reasoning is the reasoning
layer, Mechanics is prediction — but the system is best understood tab-by-tab.

Note what the substrate arrow does *not* carry: Mechanics touches neither the
corpus nor the graph. It shares the app, the auth layer and the frontend, and
nothing else.

---

## Shared substrate

### Corpus + embeddings (Chat + Search)
- **54,634** paper metadata rows (OpenAlex) · **7,674** downloaded PDFs · **16** textbooks.
- **248,629** sentence-aware chunks (NLTK · 400 tokens · 2-sentence overlap),
  each embedded with **SPECTER2** (`allenai/specter2_base` + the proximity
  adapter, 768-dim) and stored in `chunks.db`.
- Retrieval is cosine similarity via `BoneGraphRetriever`. Benchmark: **MRR 0.928,
  Recall@5 1.000** (`eval/retrieval/run_eval.py`) — but that measures *topical*
  retrieval. On quantitative items it does not hold: SPECTER2 puts the whole
  corpus in a ~0.76–0.84 cosine band, and the passage holding the deciding number
  reached the prompt for 10/50 MCQ items under dense top-5 against 31/50 under
  BM25 (`eval/mcq/`). Read the two benchmarks together.
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
| BiomedCLIP | HuggingFace (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) | 512-d image embeddings for Vision correction recall *and* region-head features |
| Region head | `data/models/vision_region_head.pt` (MLP over frozen BiomedCLIP, trained on MURA) | Vision grounding hint — 7-way body region |
| D2IM | `data/models/D2IM_trained.h5` (TensorFlow/Keras CNN) | Mechanics: displacement + strain fields |

`huatuogpt-bone` is a **custom** Ollama model with no public pull. Its definition
is in the repo (`huatuogpt-bone.Modelfile`), but the base GGUF it points at is
not, so it is recreated from base + system prompt on deploy (see deployment
Step 3). SPECTER2 and BiomedCLIP download automatically. The two trained
checkpoints under `data/models/` are gitignored and copied to the device — both
tabs degrade gracefully without them.

---

## The five tabs in detail

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

### Vision (`/api/vision/*`, legacy `/api/analyse`) 🟢
A VLM (`llava:13b`) identifies an uploaded bone image (X-ray, MRI, micro-CT,
histology) and returns a structured identification / report, anchored by two
mechanisms that are **ordered, not blended**:

1. **Trained grounding** (`vision/classifier.py`) — an MLP over frozen BiomedCLIP
   features, trained on **MURA** upper-limb radiographs (~36.8k train / 3.2k
   valid), predicts a 7-way body region at **92.6% accuracy / 0.918 macro-F1** on
   unseen validation data, every region ≥88%. The prediction is injected into the
   VLM prompt as a hint with its confidence and runner-up. It runs on the **first
   turn only**. An image the head scores as out-of-scope has its label
   **withheld** rather than injected — the confident-wrong failure mode came from
   injecting it anyway, and the out-of-scope note is deliberately scope-neutral so
   the VLM does not start dismissing bone micro-CT as "not a bone image".
   Data, training and results: [vision/training.md](vision/training.md).
2. **Correction memory** (`vision/correction_store.py`) — a 👎 + free-text note is
   stored keyed by a **BiomedCLIP image embedding** (the original + four augments
   — rot90/180/270 + hflip — so a rotated/re-windowed copy still matches). On the
   next similar image, `correction_store.recall` cosine-matches and prepends the
   prior correction to the VLM prompt — "don't make the same misidentification
   twice" (the *Cephalo lesson*).

A recalled correction **suppresses** the region hint entirely. This follows design
principle 3: a user correction outranks a trained guess, and a contradicting hint
(a MURA "humerus" guess over a recalled femur) would both mislead the VLM and
clash with the recall note shown in the UI.

Unlike Reasoning, the Vision tab has **no critic and no deterministic rule tier**
(that fuller mirror design is documented but deferred). Both components degrade
gracefully: absent weights or encoder, the tab answers from the VLM alone. The
original one-shot `/api/analyse` remains as a fallback. See
[vision/architecture.md](vision/architecture.md) (read its scope banner first)
and [vision/phase0_reuse_map.md](vision/phase0_reuse_map.md).

### Mechanics (`/api/mechanics/*`) 🟢
The quantitative counterpart to Vision, and the only tab with **no LLM anywhere in
it**. One undeformed micro-CT slice goes in; **D2IM** — the group's own CNN (Soar,
Palanca, Dall'Ara & Tozzi, *J. Orthop. Translat.* 2024) — predicts the
displacement field (u, v, w), and differentiating the axial component by finite
difference over the DVC node spacing gives the axial strain field ε_zz. No FE
model and no digital volume correlation at inference: just the greyscale image.
The response is a rendered figure (input · displacement · strain) plus summary
statistics in physical units.

The model contract is fixed by the trained weights — a (1, 256, 256, 1) scan plus
a (1, 20, 20, 1) bone mask in, three flattened 20×20 fields out, in network units
scaled by the voxel size. A mask can be supplied or approximated from the scan.

Everything model-specific is isolated in one swappable adapter
([`mechanics/d2im.py`](../mechanics/d2im.py)): **TensorFlow is lazy-imported**, so
importing the module is cheap and never fails on a machine without TF, and the
other four tabs run untouched while `/api/mechanics/status` reports why the tab is
unavailable. Keras and matplotlib are driven under a single lock — a prediction is
a sub-second forward pass, so serialising avoids TF/pyplot re-entrancy across
FastAPI's worker threads. Swapping in the D2IM-Strain follow-up is a weights-path
change. See [mechanics/architecture.md](mechanics/architecture.md).

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

**Admin dashboard** (`bonegraph.org/admin`, `frontend/admin.html`) is a static
page served from the edge alongside the app. It shows signups + beta feedback by
calling `/api/admin/overview`, gated by the `FEEDBACK_ADMIN_TOKEN` env var
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
Mechanics GET /api/mechanics/status            (TF + weights present?)
         POST /api/mechanics/predict           (multipart: slice [+ mask])
Feedback POST /api/feedback · GET /api/feedback/list
Admin    GET  /api/admin/overview              (token-gated)
```

**Every route is under `/api/`.** The `/`, `/test` and `/admin` page handlers and
the `/static` mount were removed when the frontend moved to Cloudflare's edge —
`http://<host>:8000/` is a 404 by design, and `/admin` is a static page the Worker
serves. Interactive API docs remain at `/docs`.

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
| 3 — Vision tab | 🟢 Live (Jul 2026) | VLM chat ✅ + BiomedCLIP correction memory ✅ + MURA-trained region head ✅ (92.6% / 0.918 macro-F1); critic/rules deferred · [vision/architecture.md](vision/architecture.md) · [vision/training.md](vision/training.md) |
| 6 — Mechanics tab | 🟢 Live (Jun 2026) | D2IM single-slice displacement + strain inference; TF optional and lazy · [mechanics/architecture.md](mechanics/architecture.md) |
| 5 — Feedback loop | 🟢 Reasoning + Vision live | Reasoning user rules; Vision correction memory; Chat/Search feedback into retrieval still planned |
| — Accounts + beta | 🟢 Live (Jun 2026) | Email/password auth, per-user scoping, admin dashboard, deployed at bonegraph.org |
| — Edge split | 🟢 Live (Jul 2026) | Frontend to a Cloudflare Worker, Jetson API-only · [deployment.md](deployment.md) |
| — Evaluation | 🟢 Retrieval + MCQ | Ranking benchmark; 50-item grounded-reasoning MCQ across four retrieval arms (`eval/`) |

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
`OLLAMA_MODEL`/`VLM_MODEL`/the encoder file and the pipeline is unchanged. D2IM is
the strictest case: a fully isolated adapter with a lazy, optional heavy
dependency, so a tab that cannot load stays a degraded panel instead of a failed
import.

**6. A missing component degrades, never crashes.** TensorFlow, the D2IM weights,
the Vision region head and the correction store are each optional at runtime, and
each reports its own absence (`/api/mechanics/status`, `classifier.is_available()`).
A fresh clone runs with whatever it has.

---

## Deployment

Production is a **split deployment** (no open ports, HTTPS automatic):

- **Frontend** — `frontend/` as an assets-only **Cloudflare Worker**, served from
  the global edge at **bonegraph.org**. No build step; `git push` redeploys.
- **API** — `run_api.sh` → `uvicorn api.main:app` on `127.0.0.1:8000` plus Ollama,
  on an **NVIDIA Jetson AGX Orin 64 GB**, reached at **api.bonegraph.org** through
  a **Cloudflare Tunnel**. SQLite DBs and the two model checkpoints are
  transferred to the device (gitignored, not cloned).

The split exists because the Jetson's Wi-Fi drops several times an hour: when one
process served both, every drop took the whole domain down (Cloudflare 530). Now
the page always loads and a blip degrades a single in-flight query. Full runbook,
systemd units, Worker setup and pre-launch checklist in [deployment.md](deployment.md).

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
│   ├── inspect_db.py · download_d2im.py
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
│   ├── classifier.py           MURA-trained region head → VLM grounding hint
│   ├── correction_store.py     SQLite: events · corrections · embedding recall
│   └── training/               mura_dataset · cache_features · train_head
│
├── mechanics/                  Mechanics tab
│   └── d2im.py                 D2IM adapter: preprocess → predict → strain → render
│
├── eval/                       Benchmarks
│   ├── retrieval/              MRR / Recall@k — run_eval.py · benchmark.json · results.json
│   ├── evidence/               Reasoning: literature + KG quality audit
│   ├── feedback/               "second chat is better" demo harness
│   └── mcq/                    50-item grounded-reasoning MCQ, four retrieval arms
│
├── api/                        FastAPI backend
│   ├── main.py                 All endpoints, prompts, critic loop, auth wiring
│   ├── auth_store.py           Email/password accounts + sessions (auth.db)
│   └── beta_feedback.py        Product-feedback store (beta_feedback.db)
│
├── frontend/                   React web UI — published to Cloudflare's edge
│   ├── index.html              Single-file React app (Babel in-browser); 5 tabs + feedback
│   ├── admin.html              Static admin dashboard (/admin) → /api/admin/*
│   ├── wrangler.jsonc          Assets-only Worker config (build root = frontend/)
│   └── static/                 Vendored React / ReactDOM / Babel
│
├── data/                       (gitignored)
│   ├── raw/papers/ · raw/textbooks/      Downloaded PDFs
│   ├── processed/text/                   Extracted .txt
│   ├── db/                                papers · textbooks · chunks · ontology(+raw/pre_typed)
│   │                                     · reasoning_feedback · vision_feedback · auth · beta_feedback
│   └── models/                            vision_region_head.pt · D2IM_trained.h5
│
├── docs/
│   ├── architecture.md          ← this file (system-wide overview)
│   ├── deployment.md            Cloudflare edge + Jetson API runbook
│   ├── chat/                    Chat tab + shared ingestion/RAG docs
│   │   ├── README.md
│   │   ├── papers_ingestion_pipeline.md · textbooks_ingestion_pipeline.md
│   │   └── phase2_rag_pipeline.md
│   ├── search/README.md         Search tab
│   ├── reasoning/               Reasoning tab (authoritative architecture + build log + evals)
│   │   ├── architecture.md · reasoning_tab.md
│   │   ├── evidence_layer.md · feedback_demo.md
│   │   ├── phase4_lrm_plan.md · graph_cleanup.md · graph_concept_reclassification.md
│   ├── vision/                  Vision tab
│   │   ├── README.md · architecture.md · training.md · roadmap.md
│   │   └── phase0_reuse_map.md · phase3_vlm_plan.md
│   └── mechanics/architecture.md  Mechanics tab (D2IM adapter)
│
├── visualisation/graph/         Rendered interactive knowledge-graph HTML
├── mypaper/ · presentation/     Research write-up (LaTeX) + slides
├── tests/                       test_ingestion.py
│
├── serve.py                     Uvicorn launcher — API at http://localhost:8000
├── run_api.sh                   Production launcher (aarch64 workarounds) — the Jetson
├── huatuogpt-bone.Modelfile     Custom Ollama model definition (base GGUF not in git)
├── bonegraph.service            systemd unit
├── requirements.txt
├── .env / .env.example          API keys (gitignored) / template
└── README.md
```
