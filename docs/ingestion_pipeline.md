# Phase 1 — Data Ingestion Pipeline

This document explains exactly how the paper ingestion pipeline works: what each file does, how data flows between them, and how to operate it.

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
                      ┌─────────────────────┐
                      │  downloader.py      │
                      │  stream PDF files   │◄── data/raw/papers/
                      │  to disk            │
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

### `ingestion/papers/downloader.py`

Downloads the actual PDF files for papers that have a free legal URL.

**Key function:** `download_all_open_access(store)`

Steps:
1. Queries the DB: `WHERE has_pdf = 1 AND pdf_local_path IS NULL`
2. For each paper, constructs the local path: `data/raw/papers/<year>/<paper_id>.pdf`
3. Calls `download_pdf(url, dest)` which streams the file in 8 KB chunks
4. On success, writes the path back to `pdf_local_path` in the DB
5. Waits 1.1s between downloads

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

You can query the SQLite database directly to explore what was collected:

```bash
python3 -c "
import sqlite3, json
conn = sqlite3.connect('data/db/papers.db')
conn.row_factory = sqlite3.Row

# Overall stats
row = conn.execute('SELECT COUNT(*) as n, SUM(has_pdf) as pdfs FROM papers').fetchone()
print(f'Total papers: {row[\"n\"]},  with open PDF: {row[\"pdfs\"]}')

# Top 10 most-cited
print()
for r in conn.execute('SELECT title, year, citation_count FROM papers ORDER BY citation_count DESC LIMIT 10'):
    print(f'  [{r[\"year\"]}] {r[\"title\"][:65]}  ({r[\"citation_count\"]} cites)')
"
```

Or open `data/db/papers.db` directly in **DB Browser for SQLite** (free GUI app) to browse and filter interactively.

---

## What comes next (Phase 2)

Once the corpus is collected, the next step is to make it searchable by the LLM:

1. **Text extraction** — parse PDF text from downloaded papers using `pdfminer` or `pypdf`.
2. **Chunking** — split each paper into overlapping text windows (~500 tokens each).
3. **Embedding** — convert each chunk into a vector using a sentence-transformer model.
4. **Vector store** — index all chunk vectors in a database like ChromaDB or FAISS so we can retrieve the top-k most relevant chunks for any query.
5. **RAG loop** — wire retrieval into the LLM so it answers questions grounded in the actual papers.
