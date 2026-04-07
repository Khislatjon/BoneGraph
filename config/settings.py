"""
config/settings.py
==================
Central configuration for the entire BoneLogic project.

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


# ── Unpaywall API ──────────────────────────────────────────────────────────────

# Your email address for the Unpaywall API.
# Unpaywall is a free, legal index of open-access papers.
# Given a DOI it returns a direct PDF URL.  No API key needed — just an email.
# Sign up / info: https://unpaywall.org/products/api
# Rate limit: 100,000 requests/day (very generous for a research project).
# Set this in your .env file as: UNPAYWALL_EMAIL=you@example.com
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "")


# ── Semantic Scholar API ───────────────────────────────────────────────────────

# Your personal API key, read from the .env file.
# With a key the rate limit is 1 request/second.
# Without a key (empty string) it falls back to 1 request/5s — much slower.
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# Base URL for the Semantic Scholar Graph API v1.
# All endpoint paths (e.g. /paper/search) are appended to this.
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"

# PAPER_FIELDS lists which data columns to request for each paper.
# Requesting only what you need keeps responses small and fast.
# Full list of available fields: https://api.semanticscholar.org/graph/v1#tag/Paper-Data
PAPER_FIELDS = [
    "paperId",          # Semantic Scholar's own stable identifier (SHA-like hash)
    "externalIds",      # DOI, ArXiv ID, PubMed ID, etc. — useful for dedup & linking
    "title",            # Full paper title
    "abstract",         # Abstract text — primary input for Phase 1 RAG
    "year",             # Publication year (integer)
    "authors",          # List of {authorId, name} dicts
    "venue",            # Journal or conference name
    "publicationTypes", # E.g. ["JournalArticle"], ["Review"], ["Conference"]
    "publicationDate",  # Full date string "YYYY-MM-DD" where available
    "citationCount",    # How many papers have cited this one — proxy for impact
    "referenceCount",   # How many papers this one cites
    "openAccessPdf",    # Dict with {"url": "..."} if a free legal PDF exists, else null
    "fieldsOfStudy",    # High-level field tags, e.g. ["Medicine", "Biology"]
    "s2FieldsOfStudy",  # More granular S2-specific field tags with confidence scores
]


# ── Rate limiting & retries ───────────────────────────────────────────────────

# REQUEST_DELAY_SECONDS is the minimum wait between consecutive API calls.
# Semantic Scholar enforces exactly 1 request/second for authenticated keys.
# We use 1.1s to add a small safety buffer and avoid edge-case 429 errors.
REQUEST_DELAY_SECONDS = 1.1

# MAX_RETRIES — how many times to automatically retry a failed request before
# giving up.  The requests library handles this via urllib3's Retry mechanism.
MAX_RETRIES = 3

# RETRY_BACKOFF_SECONDS — how long to wait after receiving a 429 (Too Many
# Requests) response.  S2 often sends a Retry-After header; this is the
# fallback if that header is missing.
RETRY_BACKOFF_SECONDS = 60


# ── Pagination ────────────────────────────────────────────────────────────────

# SEARCH_BATCH_SIZE — number of results to request per API call.
# 100 is the hard maximum Semantic Scholar allows per request.
SEARCH_BATCH_SIZE = 100

# MAX_PAPERS_PER_QUERY — upper bound on how many papers to collect per keyword.
# 500 is a good balance: enough for a rich corpus without excessive API time.
# With ~133 keywords this yields ~15,000–30,000 unique papers after deduplication.
# At 1 req/s and batch size 100, each keyword needs ~5 API calls → total ~11 min.
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

# ── Embedding ────────────────────────────────────────────────────────────────

# Local SPECTER model — trained by Allen AI on 146M Semantic Scholar paper
# citations, making it ideal for scientific text retrieval.
# Downloaded automatically by sentence-transformers on first use (~440 MB).
EMBEDDING_MODEL = "allenai-specter"

# Number of chunks to embed in one batch.  Larger batches are faster but use
# more RAM.  64 is a safe default for a machine without a GPU.
EMBEDDING_BATCH_SIZE = 64

# ── Chunking ──────────────────────────────────────────────────────────────────

# Target chunk size in characters. 512 tokens * ~4 chars/token ≈ 2048 chars.
# Keeping chunks at roughly 512 tokens is standard for RAG — large enough to
# carry meaningful context, small enough for embedding models to handle well.
CHUNK_SIZE_CHARS = 2048

# Overlap between consecutive chunks in characters (~50 tokens * 4 chars).
# Overlap ensures that sentences split across chunk boundaries are still
# represented in at least one complete chunk.
CHUNK_OVERLAP_CHARS = 200
