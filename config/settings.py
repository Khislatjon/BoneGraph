"""
config/settings.py
==================
Central configuration for the entire BoneGraph project.

All tunable constants live here so every other module imports from one place.
If you need to change a behaviour (rate limits, file paths, which S2 fields to
fetch) you only need to touch this file.

Environment variables are loaded from a .env file in the repo root via
python-dotenv.  The .env file is gitignored — copy .env.example and fill in
your keys.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# load_dotenv() reads the .env file that sits next to this repo root and injects
# each KEY=VALUE pair into os.environ so that os.getenv() calls below work.
# If .env does not exist the function does nothing (no error).
load_dotenv()


# ── Paths ─────────────────────────────────────────────────────────────────────

# ROOT_DIR is the absolute path to the repo root, computed by going two levels
# up from this file (config/settings.py → config/ → repo root).
ROOT_DIR = Path(__file__).resolve().parent.parent

# DATA_DIR is where all persistent data lives (raw downloads, processed output,
# the SQLite database).  Keeping data outside the source tree makes it easier
# to .gitignore large files without complex rules.
DATA_DIR = ROOT_DIR / "data"

# RAW_PAPERS_DIR — downloaded PDFs land here, organised into sub-folders by
# publication year (e.g. data/raw/papers/2022/<paper_id>.pdf).
RAW_PAPERS_DIR = DATA_DIR / "raw" / "papers"

# RAW_TEXTBOOKS_DIR — manually supplied textbook PDFs / EPUBs go here for
# Phase 1b ingestion.
RAW_TEXTBOOKS_DIR = DATA_DIR / "raw" / "textbooks"

# PROCESSED_DIR — text chunks, embeddings, and any other derived artefacts
# produced by the processing pipeline are stored here.
PROCESSED_DIR = DATA_DIR / "processed"

# DB_DIR — the SQLite database file lives here.
DB_DIR = DATA_DIR / "db"

# Create all directories on import so downstream code never has to check.
# exist_ok=True means no error if the directory already exists.
for _dir in (RAW_PAPERS_DIR, RAW_TEXTBOOKS_DIR, PROCESSED_DIR, DB_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ── CrossRef API ───────────────────────────────────────────────────────────────

# Email for CrossRef polite pool — increases rate limit from 50 to ~150 req/s.
# No API key needed. Set in .env as: CROSSREF_EMAIL=you@example.com
CROSSREF_EMAIL = os.getenv("CROSSREF_EMAIL", os.getenv("UNPAYWALL_EMAIL", ""))

# Wiley TDM (Text & Data Mining) token for programmatic PDF access.
# Request at: https://tdm.wiley.com
WILEY_TDM_TOKEN = os.getenv("WILEY_TDM_TOKEN", "")


# ── OpenAlex API ───────────────────────────────────────────────────────────────

# Base URL for the OpenAlex REST API.
# No API key required — completely free and open.
OPENALEX_BASE_URL = "https://api.openalex.org"

# Fields to request per Work object via the ?select= parameter.
# Keeping this minimal reduces response size and speeds up pagination.
OPENALEX_SELECT_FIELDS = [
    "id",                       # OpenAlex ID, e.g. "https://openalex.org/W2741809807"
    "doi",                      # DOI string, e.g. "https://doi.org/10.1016/j.bone.2022.01.001"
    "title",                    # Full title
    "abstract_inverted_index",  # Abstract stored as {word: [positions]} — must be reconstructed
    "publication_year",         # Integer year
    "publication_date",         # "YYYY-MM-DD" string
    "authorships",              # List of {author: {display_name}, institutions: [...]}
    "primary_location",         # {source: {display_name}, pdf_url, ...}
    "open_access",              # {is_oa, oa_url, oa_status}
    "cited_by_count",           # Citation count
    "type",                     # "journal-article", "book-chapter", etc.
    "language",                 # ISO 639-1 code, e.g. "en" — set natively by OpenAlex
]

# ── Year & language defaults ──────────────────────────────────────────────────

# Default publication-year range for OpenAlex filter.
# Modern bone science literature starts around 1970; 2026 covers the present.
# OpenAlex format: "YYYY-YYYY" (inclusive on both ends).
DEFAULT_YEAR_RANGE = "1970-2026"

# Default language filter — only fetch English papers.
# OpenAlex natively supports this filter (filter=language:en), so non-English
# papers are excluded at query time, before they ever touch the database.
DEFAULT_LANGUAGE = "en"


# ── Rate limiting & retries ───────────────────────────────────────────────────

# REQUEST_DELAY_SECONDS is the minimum wait between consecutive API calls.
# OpenAlex polite pool allows 10 requests/second — 0.15s gives a safe buffer.
REQUEST_DELAY_SECONDS = 0.15

# MAX_RETRIES — how many times to automatically retry a failed request before
# giving up.  The requests library handles this via urllib3's Retry mechanism.
MAX_RETRIES = 3

# RETRY_BACKOFF_SECONDS — seconds to wait after a 429 response if no
# Retry-After header is present.
RETRY_BACKOFF_SECONDS = 30


# ── Pagination ────────────────────────────────────────────────────────────────

# SEARCH_BATCH_SIZE — number of results to request per API call.
# OpenAlex allows up to 200 results per page (vs S2's 100).
SEARCH_BATCH_SIZE = 200

# MAX_PAPERS_PER_QUERY — upper bound on how many papers to collect per keyword.
# 500 is a good balance: enough for a rich corpus without excessive API time.
# With ~133 keywords at 200/page, each keyword needs ~3 API calls.
MAX_PAPERS_PER_QUERY = 500


# ── Database ──────────────────────────────────────────────────────────────────

# Path to the SQLite file that stores all paper metadata.
# SQLite is a single-file database — no server needed, zero configuration.
PAPERS_DB_PATH = DB_DIR / "papers.db"

# Path to the SQLite file that stores all textbook metadata.
TEXTBOOKS_DB_PATH = DB_DIR / "textbooks.db"

# Path to the SQLite file that stores all text chunks (papers + textbooks).
# Kept separate from papers.db and textbooks.db because the chunks table
# will have hundreds of thousands of rows — isolating it keeps the other
# databases fast and browsable.
CHUNKS_DB_PATH = DB_DIR / "chunks.db"

# Path to the SQLite file that stores the bone knowledge graph (Phase 4).
# Contains two tables: nodes (concepts) and edges (causal relationships).
# Seeded by reasoning/seed.py; grown by reasoning/extractor.py.
ONTOLOGY_DB_PATH = DB_DIR / "ontology.db"

# ── Ollama (Phase 4 extraction & reasoning) ──────────────────────────────────

# Base URL of the running Ollama server.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Model used for triple extraction in reasoning/extractor.py.
# huatuogpt-bone is bone-domain-aware and already available locally.
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "huatuogpt-bone:latest")

# Request timeout in seconds for a single Ollama generate call.
# Triple extraction over a ~400-token chunk typically completes in 5–30 s.
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))

# ── Embedding ────────────────────────────────────────────────────────────────

# SPECTER2 base model — Allen AI's 2023 successor to SPECTER.
# Better retrieval accuracy on scientific text, same 768-dim output.
# Downloaded automatically by HuggingFace on first use (~440 MB).
SPECTER2_BASE_MODEL = "allenai/specter2_base"

# Task-specific adapter for document embedding (proximity).
# SPECTER2 uses separate adapters for documents vs queries:
#   - allenai/specter2        → document embedding (used here, for chunked papers)
#   - allenai/specter2_adhoc_query → query embedding (used at retrieval time)
SPECTER2_ADAPTER = "allenai/specter2"

# Number of chunks to embed in one batch.
# 256 is optimal for MPS (Apple Silicon GPU). Drop to 64 if you hit OOM.
EMBEDDING_BATCH_SIZE = 256

# ── Chunking ──────────────────────────────────────────────────────────────────

# Target chunk size in tokens. SPECTER's hard limit is 512 tokens; we target
# 400 to leave ~100 tokens of headroom so no chunk ever gets silently truncated.
CHUNK_TARGET_TOKENS = 400

# Number of sentences carried from the end of one chunk into the start of the
# next. Sentence-level overlap is cleaner than character overlap — it always
# gives complete, readable context rather than a mid-sentence fragment.
CHUNK_OVERLAP_SENTENCES = 2

# Characters per token approximation for English scientific text.
# Scientific text uses longer technical terms than everyday prose, so the
# chars-per-token ratio is higher (~5) than general English (~4). Using 4
# here is deliberately conservative — it slightly overestimates token count,
# ensuring chunks stay safely under SPECTER's 512-token limit.
CHARS_PER_TOKEN = 4
