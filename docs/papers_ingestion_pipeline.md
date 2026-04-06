# Phase 1 — Data Ingestion Pipeline

This document explains exactly how the paper ingestion pipeline works: what each file does, how data flows between them, and how to operate it.

---

## Phase 1 Results (completed March 2026)

The full metadata collection run completed successfully. Key statistics:

| Metric | Value |
|---|---|
| **Total papers collected** | **55,277** |
| **Unique — no duplicates** | ✅ enforced by SQLite PRIMARY KEY |
| **Papers with open-access PDF URL** | 21,571 (39%) |
| **Year range** | 1900 – 2025 (pre-1900 papers removed) |
| **Peak publication years** | 2015–2021 (~2,400–2,700 papers/year) |
| **Keywords used** | 133 across 17 topic groups |
| **PDFs downloaded to disk** | **6,125** (rest require institutional access) |

**Top journals by paper count:**

| Journal | Papers |
|---|---|
| Bone | 2,452 |
| Journal of Bone and Mineral Research | 1,662 |
| Journal of Bone and Joint Surgery (Am.) | 814 |
| Journal of Biomechanics | 771 |
| Osteoporosis International | 598 |
| Clinical Orthopaedics and Related Research | 460 |
| Calcified Tissue International | 447 |
| J. Mechanical Behavior of Biomedical Materials | 441 |
| Biomaterials | 388 |

**Most cited papers in corpus:**

| Year | Citations | Title |
|---|---|---|
| 1990 | 10,257 | Biomechanics and Motor Control of Human Movement |
| 1969 | 6,894 | Traumatic arthritis of the hip after dislocation… |
| 2003 | 6,483 | Osteoclast differentiation and activation |
| 2005 | 6,244 | Porosity of 3D biomaterial scaffolds and osteogenesis |
| 1997 | 5,228 | Osteoprotegerin: a novel secreted protein… |

**PDF URL resolution analysis** (measured on a 300-paper sample):

| Tier | Mechanism | Coverage | API call? |
|---|---|---|---|
| Tier 1 | URL already a direct `.pdf` link | ~40% | None |
| Tier 2 | Publisher-specific URL rewrites | +16% | None |
| Tier 3 | Unpaywall DOI lookup | remaining ~44% | Yes (free) |
| **Total resolvable** | | **~100% of open-access papers** | |

---

## Overview

The goal of Phase 1 is to build the knowledge base that will power the LLM in later phases. We collect paper metadata and PDFs from Semantic Scholar using a curated set of bone-domain search keywords.

```
                      ┌─────────────────────┐
                      │  keywords.py        │
                      │  133 search queries │
                      │  across 17 groups   │
                      └────────┬────────────┘
                               │ keyword strings
                               ▼
                      ┌─────────────────────┐
                      │  semantic_scholar.py│
                      │  S2 API client      │◄── API key from .env
                      │  pagination         │
                      │  rate limiting      │
                      └────────┬────────────┘
                               │ paper dicts (title, abstract,
                               │ authors, year, PDF URL, …)
                               ▼
                      ┌─────────────────────┐
                      │  storage.py         │
                      │  SQLite database    │◄── data/db/papers.db
                      │  upsert + dedup     │
                      └────────┬────────────┘
                               │ papers with pdf_url
                               ▼
                      ┌─────────────────────┐     ┌──────────────────────┐
                      │  resolvers.py       │     │  Unpaywall API       │
                      │  3-tier URL         │────►│  (free, email only)  │
                      │  resolution chain   │     │  resolves DOI links  │
                      └────────┬────────────┘     └──────────────────────┘
                               │ direct PDF URL
                               ▼
                      ┌─────────────────────┐
                      │  downloader.py      │
                      │  stream PDF files   │◄── data/raw/papers/
                      │  verify %PDF magic  │
                      │  number on disk     │
                      └─────────────────────┘

All of the above is orchestrated by pipeline.py (the CLI entry point).
```

---

## Files in detail

### `config/settings.py`

The single source of truth for all configuration. Every other module imports constants from here rather than hardcoding values.

Key constants:

| Constant | Value | Purpose |
|---|---|---|
| `SEMANTIC_SCHOLAR_API_KEY` | from `.env` | Authenticates API requests (1 req/s limit) |
| `UNPAYWALL_EMAIL` | from `.env` | Email for Unpaywall API — resolves DOI links to direct PDFs |
| `REQUEST_DELAY_SECONDS` | `1.1` | Wait between API calls to stay under rate limit |
| `RETRY_BACKOFF_SECONDS` | `60` | Wait after a 429 (rate-limit) error |
| `SEARCH_BATCH_SIZE` | `100` | Max results per API request (S2 hard cap) |
| `MAX_PAPERS_PER_QUERY` | `500` | Max papers to collect per keyword |
| `PAPERS_DB_PATH` | `data/db/papers.db` | SQLite database location |
| `RAW_PAPERS_DIR` | `data/raw/papers/` | Where PDFs are saved |

All data directories are created automatically when `settings.py` is imported for the first time.

---

### `ingestion/papers/keywords.py`

Defines what we search for. Contains a dictionary called `KEYWORD_GROUPS` with 17 groups and 133 search strings total.

```python
KEYWORD_GROUPS = {
    "mechanics": [
        "bone mechanical properties",
        "cortical bone fracture toughness",
        ...
    ],
    "pathology": [...],
    "bone_types_skull": [...],
    ...
}
```

**Groups:**
- `bone_types_skull` — frontal, parietal, temporal, occipital, sphenoid, ethmoid + facial bones
- `bone_types_ear` — malleus, incus, stapes
- `bone_types_hyoid` — hyoid bone
- `bone_types_spine` — cervical, thoracic, lumbar vertebrae, sacrum, coccyx
- `bone_types_thorax` — ribs, sternum, clavicle, scapula
- `bone_types_upper_limb` — humerus, radius, ulna, carpals, metacarpals, phalanges
- `bone_types_pelvis` — ilium, ischium, pubis, acetabulum, femoral head
- `bone_types_lower_limb` — femur, tibia, fibula, patella, tarsals, metatarsals
- `bone_types_tissue` — cortical, trabecular, long/short/flat/irregular/sesamoid bone types
- `morphology` — microstructure, Haversian systems, osteocyte networks
- `structure_function` — hierarchical structure, collagen-mineral composite
- `mechanics` — elastic modulus, fracture toughness, fatigue, FEA
- `pathology` — osteoporosis, Paget's, metastasis, osteosarcoma
- `imaging` — X-ray, MRI, microCT, DXA
- `biomaterials` — scaffolds, tissue engineering, 3D printing
- `simulation` — computational modelling, molecular dynamics
- `cell_biology` — osteoblasts, osteoclasts, RANKL/OPG signalling

To add more keywords, simply add strings to the appropriate list. To add a new group, add a new key to `KEYWORD_GROUPS`.

---

### `ingestion/papers/semantic_scholar.py`

The API client. Wraps all HTTP communication with Semantic Scholar.

**Class:** `SemanticScholarClient`

```
SemanticScholarClient
├── _build_session()         Creates a persistent HTTP session with retry logic
├── _get(endpoint, params)   Sends one GET request; handles 429s manually
├── search_papers(query)     GENERATOR: yields papers page by page
├── get_paper(paper_id)      Fetches a single paper by ID or DOI
├── get_paper_references()   Returns the papers cited by a given paper
└── get_paper_citations()    Returns the papers that cite a given paper
```

**How `search_papers` works (pagination):**

Semantic Scholar returns at most 100 results per request. To get 500 papers for a keyword, we make 5 requests with `offset=0`, `offset=100`, `offset=200`, etc.

```
Request 1: offset=0,   limit=100  → papers 1-100
Request 2: offset=100, limit=100  → papers 101-200
Request 3: offset=200, limit=100  → papers 201-300
Request 4: offset=300, limit=100  → papers 301-400
Request 5: offset=400, limit=100  → papers 401-500
```

Between each request, the code waits `REQUEST_DELAY_SECONDS` (1.1s) to stay within the 1 req/s rate limit.

`search_papers` is a Python **generator** — it yields one paper at a time as it fetches pages. This means memory usage stays low regardless of how many total papers are collected.

**The paper dict** returned by the API looks like:
```python
{
    "paperId": "abc123...",             # stable S2 identifier
    "title": "Bone fracture mechanics",
    "abstract": "We investigated...",
    "year": 2022,
    "venue": "Journal of Bone Research",
    "citationCount": 45,
    "referenceCount": 62,
    "authors": [{"authorId": "1", "name": "Jane Doe"}],
    "externalIds": {"DOI": "10.1016/...", "PubMed": "123456"},
    "openAccessPdf": {"url": "https://..."},  # or None
    "s2FieldsOfStudy": [{"category": "Medicine"}, ...],
    "publicationTypes": ["JournalArticle"],
    "publicationDate": "2022-06-01"
}
```

---

### `ingestion/papers/storage.py`

The database layer. All paper metadata is stored in a local SQLite file.

**Class:** `PaperStore`

```
PaperStore
├── connect()                Opens the database; creates tables if needed
├── close()                  Closes the connection
├── upsert_paper(paper, kw)  Insert or update one paper; returns True if new
├── upsert_papers(papers, kw) Bulk upsert; returns (new_count, updated_count)
├── record_search_run()      Logs that a keyword query was executed
├── set_local_pdf_path()     Records where a downloaded PDF was saved
├── get_paper(paper_id)      Look up one paper by its S2 ID
├── get_papers_with_pdf()    All papers with PDF URL but not yet downloaded
├── total_papers()           Count of all papers in the database
└── stats()                  Summary: total papers, how many have PDFs
```

**Deduplication — how it works:**

Every paper has a unique `paper_id` (the Semantic Scholar hash). The database uses this as the PRIMARY KEY:

```sql
CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY,
    ...
)
```

When we try to insert a paper that already exists, we use:
```sql
INSERT INTO papers (...) VALUES (...)
ON CONFLICT(paper_id) DO UPDATE SET
    citation_count   = excluded.citation_count,
    keywords_matched = excluded.keywords_matched,
    updated_at       = datetime('now')
```

This means:
- Same paper found by keyword A → inserted (new)
- Same paper found by keyword B → **updated, not duplicated**
- The `keywords_matched` column accumulates `["keyword A", "keyword B"]`

**The database schema:**

| Column | Type | Description |
|---|---|---|
| `paper_id` | TEXT (PK) | S2 stable identifier |
| `title` | TEXT | Full paper title |
| `abstract` | TEXT | Abstract — primary input for RAG |
| `year` | INTEGER | Publication year |
| `venue` | TEXT | Journal or conference |
| `citation_count` | INTEGER | Impact proxy |
| `has_pdf` | INTEGER | 1 if open-access PDF available |
| `pdf_url` | TEXT | URL of free PDF |
| `pdf_local_path` | TEXT | Local path after download |
| `authors_json` | TEXT | JSON list of `{authorId, name}` |
| `external_ids_json` | TEXT | JSON: DOI, PubMed, ArXiv IDs |
| `fields_of_study` | TEXT | JSON list of field labels |
| `keywords_matched` | TEXT | JSON list of queries that found this paper |
| `created_at` | TEXT | First ingestion timestamp |
| `updated_at` | TEXT | Last update timestamp |

---

### `ingestion/papers/resolvers.py`

**Added March 2026** — resolves stored PDF URLs to actual downloadable PDF files.

**Why this was needed:**
When we analysed the 21,571 stored PDF URLs, we found they fell into distinct categories that could not all be downloaded directly:

```
doi.org links           4,424   → redirect to publisher landing page
europepmc.org viewer    2,179   → HTML viewer page, not raw PDF
ncbi.nlm.nih.gov/pmc      984   → HTML article page, needs /pdf/ suffix
publisher landing pages  3,000+ → need Unpaywall to find the PDF
direct .pdf links        ~8,500  → work as-is
```

**The 3-tier resolution chain:**

```
stored_url (from S2)
       │
       ▼
Tier 1: is_direct_pdf(url)
  ├─ URL ends in .pdf                     → return url as-is  ✓
  ├─ URL contains /article/am/pii/        → ScienceDirect AM  ✓
  ├─ URL contains blobtype=pdf            → EuropePMC backend ✓
  └─ URL contains /content/pdf/           → Springer direct   ✓

       │ (if Tier 1 failed)
       ▼
Tier 2: apply_publisher_transforms(url)
  ├─ europepmc.org/articles/pmc{id}       → backend PDF endpoint
  ├─ ncbi.nlm.nih.gov/pmc/articles/PMC{id} → /pdf/ suffix
  ├─ journals.plos.org/article?id=        → /article/file?...&type=printable
  ├─ frontiersin.org/.../full             → .../pdf
  ├─ mdpi.com/{path}                      → {path}/pdf
  ├─ onlinelibrary.wiley.com/doi/{doi}    → /pdfdirect
  └─ link.springer.com/article/{doi}      → /content/pdf/{doi}.pdf

       │ (if Tier 2 failed — doi.org links, unknown publishers)
       ▼
Tier 3: resolve_via_unpaywall(doi)
  └─ GET https://api.unpaywall.org/v2/{doi}?email={email}
     └─ returns best_oa_location.url_for_pdf  (direct PDF URL)
```

**Results on the actual corpus (sampled 300 URLs):**

| Tier | Papers resolved | Without API call |
|---|---|---|
| Tier 1 | ~40% | ✅ |
| Tier 2 | +16% | ✅ |
| Tier 3 (Unpaywall) | ~44% | ❌ (1 call per paper) |

56% of open-access PDFs can be resolved entirely offline with no network calls.

**Safety guard in `download_pdf()`:**
After downloading, the code reads the first 4 bytes of the saved file and checks for the PDF magic number `%PDF`. If the file starts with anything else (e.g. `<html`) — which happens when servers silently redirect to a login page — the file is deleted and the download is marked as failed. This prevents the disk from filling up with useless HTML files.

**Public functions:**

| Function | Description |
|---|---|
| `is_direct_pdf(url)` | Returns True if URL is already a direct PDF link |
| `apply_publisher_transforms(url)` | Tries all 7 publisher rewrites, returns first match |
| `resolve_via_unpaywall(doi, session)` | Calls Unpaywall API, returns direct PDF URL |
| `resolve_pdf_url(paper_id, stored_url, doi, session)` | Main entry point — runs all 3 tiers |

---

### `ingestion/papers/downloader.py`

Downloads the actual PDF files for papers that have a free legal URL.

**Key function:** `download_all_open_access(store)`

Updated steps (as of March 2026):
1. Queries the DB: `WHERE has_pdf = 1 AND pdf_local_path IS NULL`
2. Extracts the DOI from `external_ids_json` for each paper (needed for Tier 3)
3. Calls `resolve_pdf_url()` to find a direct downloadable PDF URL (3-tier chain)
4. If no URL can be resolved, logs it as `unresolvable` and moves on
5. Calls `download_pdf(resolved_url, dest)` which streams the file in 8 KB chunks
6. Verifies the downloaded file starts with `%PDF` — rejects HTML pages silently served as PDFs
7. On success, writes the local path back to `pdf_local_path` in the DB
8. Waits 1.1s between downloads

**Return counters:**

| Counter | Meaning |
|---|---|
| `downloaded` | PDF successfully fetched, verified, and saved |
| `failed` | URL resolved but download errored or returned non-PDF |
| `unresolvable` | No direct PDF URL found after all 3 tiers |
| `skipped` | File already on disk from a previous run |

**File layout on disk:**
```
data/raw/papers/
├── 2020/
│   ├── abc123def456.pdf
│   └── xyz789...pdf
├── 2022/
│   └── ...
└── unknown/          ← papers where year is missing
```

Papers are grouped by year to keep directories manageable and browsable.

Only open-access papers are downloaded — we never attempt to download papers behind paywalls. The `openAccessPdf` field from S2 only contains URLs for legally free copies.

---

### `ingestion/papers/pipeline.py`

The entry point that ties everything together. This is what you run from the command line.

**Function:** `run(keywords, max_per_keyword, year_range, download_pdfs)`

```
for each keyword:
    papers = client.search_papers(keyword, max=max_per_keyword)
    new, updated = store.upsert_papers(papers, keyword)
    store.record_search_run(keyword, ...)

if download_pdfs:
    download_all_open_access(store)
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--groups` | all groups | Run only specific keyword groups |
| `--keywords` | — | Custom search strings (overrides --groups) |
| `--year` | none | Filter by year range, e.g. `2015-2024` |
| `--max` | 500 | Max papers per keyword |
| `--download` | off | Also download open-access PDFs |

---

## Running the pipeline

### Quick test (2 minutes)
```bash
python -m ingestion.papers.pipeline --keywords "bone fracture" --max 10
```

### One topic group (5–10 minutes)
```bash
python -m ingestion.papers.pipeline --groups mechanics --max 100
```

### Full metadata run (20–25 minutes)
```bash
python -m ingestion.papers.pipeline
```

### Full run including PDFs (add overnight — 11–17 hours, ~16–31 GB)
```bash
python -m ingestion.papers.pipeline --download
```

---

## Inspecting what was collected

### Option 1 — `scripts/inspect_db.py` (recommended)

A ready-made script that prints all key statistics in one command:

```bash
python scripts/inspect_db.py              # default top-10 per section
python scripts/inspect_db.py --top 20    # show more rows
python scripts/inspect_db.py --keyword femur   # filter keyword section
```

Output sections:
- Overall counts (total papers, open-access PDFs, download progress)
- Papers by year with a visual bar chart
- Top journals by paper count
- Most cited papers
- Keyword coverage table
- PDF download status with progress bar and time estimate

### Option 2 — PyCharm Database tool

**View → Tool Windows → Database → `+` → Data Source → SQLite** → point to `data/db/papers.db`.
Browse all 55,277 rows, filter, sort, and write SQL queries inside your IDE. No extra install needed.

### Option 3 — DB Browser for SQLite (GUI)

```bash
brew install --cask db-browser-for-sqlite
```
Open `data/db/papers.db` directly. Full table browsing, SQL editor, CSV export.

---

## What comes next (Phase 2)

Once the corpus is collected, the next step is to make it searchable by the LLM:

1. **Text extraction** — parse PDF text from downloaded papers using `pdfminer` or `pypdf`.
2. **Chunking** — split each paper into overlapping text windows (~500 tokens each).
3. **Embedding** — convert each chunk into a vector using a sentence-transformer model.
4. **Vector store** — index all chunk vectors in a database like ChromaDB or FAISS so we can retrieve the top-k most relevant chunks for any query.
5. **RAG loop** — wire retrieval into the LLM so it answers questions grounded in the actual papers.
