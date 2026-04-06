"""
ingestion/textbooks/scanner.py
================================
Scans the data/raw/textbooks/ folder, extracts basic metadata from each PDF,
and registers every book in the textbooks SQLite database.

How it works
------------
1. Walk the RAW_TEXTBOOKS_DIR tree looking for .pdf files.
2. The immediate parent folder name is used as the source
   (e.g. "MDPI Books", "OpenStax", "NCBI Bookshelf").
3. The filename (without .pdf) is cleaned up to produce a human-readable title.
4. PyMuPDF (fitz) opens each PDF to read the page count.
5. File size is read from the filesystem.
6. All metadata is upserted into the textbooks DB via TextbookStore.

Re-running the scanner is always safe — existing records are updated in place
and no duplicates are created.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from config.settings import RAW_TEXTBOOKS_DIR
from ingestion.textbooks.storage import TextbookStore

logger = logging.getLogger(__name__)


def _clean_title(filename: str) -> str:
    """
    Convert a raw filename into a readable title.

    Examples
    --------
    "Advances_in_Bone_Graft_Materials"  →  "Advances in Bone Graft Materials"
    "Fundamentals-of-Anatomy-and-Physiology-1673303351"  →  "Fundamentals of Anatomy and Physiology"
    "Anatomy_and_Physiology_2e_-_WEB_c9nD9QL"  →  "Anatomy and Physiology 2e"
    """
    # Remove trailing random hash/ID suffixes (digits or alphanumeric, 7+ chars)
    title = re.sub(r'[-_][0-9]{7,}$', '', filename)
    title = re.sub(r'[-_][A-Za-z0-9]{7,}$', '', title)
    # Remove trailing " - WEB" type suffixes
    title = re.sub(r'\s*[-_]\s*WEB.*$', '', title, flags=re.IGNORECASE)
    # Replace underscores and hyphens with spaces
    title = title.replace('_', ' ').replace('-', ' ')
    # Collapse multiple spaces
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def _get_page_count(pdf_path: Path) -> int | None:
    """
    Open the PDF with PyMuPDF and return the number of pages.

    Returns None if the file cannot be opened (corrupt, password-protected, etc.).
    PyMuPDF (imported as 'fitz') is fast — it reads only the PDF cross-reference
    table without loading all page content into memory.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        count = doc.page_count
        doc.close()
        return count
    except Exception as exc:
        logger.warning("Could not read page count for %s: %s", pdf_path.name, exc)
        return None


def scan_textbooks(store: TextbookStore) -> dict[str, int]:
    """
    Walk RAW_TEXTBOOKS_DIR, extract metadata, and register every PDF in the DB.

    Parameters
    ----------
    store : TextbookStore
        An open TextbookStore connection.

    Returns
    -------
    dict with keys:
        "registered" — books successfully added or updated in the DB
        "errors"     — files that could not be processed
    """
    counts = {"registered": 0, "errors": 0}

    pdf_files = sorted(RAW_TEXTBOOKS_DIR.rglob("*.pdf"))
    logger.info("Found %d PDF files in %s", len(pdf_files), RAW_TEXTBOOKS_DIR)

    for pdf_path in pdf_files:
        try:
            # Source = the immediate parent folder name (e.g. "MDPI Books")
            source = pdf_path.parent.name

            # Title = cleaned-up filename without extension
            title = _clean_title(pdf_path.stem)

            # File size in MB, rounded to 2 decimal places
            file_size_mb = round(pdf_path.stat().st_size / (1024 * 1024), 2)

            # Page count via PyMuPDF
            page_count = _get_page_count(pdf_path)

            # Store path relative to RAW_TEXTBOOKS_DIR for portability
            relative_path = str(pdf_path.relative_to(RAW_TEXTBOOKS_DIR))

            store.upsert_textbook(
                file_path    = relative_path,
                title        = title,
                source       = source,
                file_size_mb = file_size_mb,
                page_count   = page_count,
            )

            logger.info(
                "Registered: [%s] %s — %d pages, %.1f MB",
                source, title, page_count or 0, file_size_mb,
            )
            counts["registered"] += 1

        except Exception as exc:
            logger.error("Error processing %s: %s", pdf_path.name, exc)
            counts["errors"] += 1

    return counts
