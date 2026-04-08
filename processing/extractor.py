"""
processing/extractor.py
========================
Extracts plain text from PDF files using PyMuPDF (fitz).

How it works
------------
PyMuPDF opens each PDF and reads text page by page.  Each page's text is
separated by a form-feed character (\f) so downstream chunking code can
split on page boundaries if needed.

The extracted text is written to a .txt file in data/processed/text/.
The folder structure mirrors the source:
  - Papers:    data/processed/text/papers/<paper_id>.txt
  - Textbooks: data/processed/text/textbooks/<source>/<filename>.txt

Why plain .txt files?
---------------------
Storing extracted text as flat files keeps things simple and portable.
Each file is independently readable without any database connection.
The database tracks which files have been extracted (status column) so
we know what still needs processing.

Limitations
-----------
- Scanned PDFs (images of pages with no text layer) will produce empty or
  near-empty text files.  These are detected and flagged as 'scanned'.
- Some PDFs use custom encodings that PyMuPDF cannot decode — these are
  logged as 'failed'.
- Multi-column layouts may produce garbled word order in some papers.
  This is a known limitation of PDF text extraction in general.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Minimum number of characters for a page to be considered non-empty.
# Pages with fewer characters are likely scanned images or cover pages.
MIN_CHARS_PER_PAGE = 50

# Minimum fraction of pages that must have text for the PDF to be considered
# machine-readable (not a scanned document).
MIN_TEXT_PAGE_FRACTION = 0.3

# Regex that matches a references/bibliography section heading on its own line.
# Requires the heading to occupy a line by itself (possibly with whitespace/
# punctuation) so that mid-sentence occurrences of the word "references" are
# not mistakenly treated as section markers.
_REFERENCES_HEADING_RE = re.compile(
    r"^\s*"
    r"(references|bibliography|works cited|literature cited|reference list)"
    r"[\s:]*$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_references(text: str) -> str:
    """
    Remove everything from the references/bibliography heading onward.

    Scans the text for the first line that is solely a references-section
    heading and truncates there.  If no such heading is found the original
    text is returned unchanged.
    """
    match = _REFERENCES_HEADING_RE.search(text)
    if match:
        return text[: match.start()].rstrip()
    return text


def extract_text_from_pdf(pdf_path: Path, out_path: Path) -> dict:
    """
    Extract all text from a PDF and write it to a .txt file.

    Text from each page is joined with a form-feed character (\\f) so the
    file can be split back into pages if needed:
        pages = text.split('\\f')

    Parameters
    ----------
    pdf_path : Path
        Absolute path to the source PDF file.
    out_path : Path
        Absolute path where the extracted .txt file should be written.
        Parent directories are created automatically.

    Returns
    -------
    dict with keys:
        status       : "extracted" | "scanned" | "failed"
        page_count   : total number of pages in the PDF
        char_count   : total characters extracted
        text_pages   : number of pages with meaningful text
    """
    result = {
        "status": "failed",
        "page_count": 0,
        "char_count": 0,
        "text_pages": 0,
    }

    try:
        doc = fitz.open(str(pdf_path))
        result["page_count"] = doc.page_count

        pages_text = []
        text_pages = 0

        for page in doc:
            # get_text() returns the text content of a single page.
            # "text" mode preserves line breaks; "blocks" and "words" modes
            # give more structure but are harder to work with downstream.
            page_text = page.get_text("text")

            if len(page_text.strip()) >= MIN_CHARS_PER_PAGE:
                text_pages += 1

            pages_text.append(page_text)

        doc.close()

        result["text_pages"] = text_pages
        total_pages = result["page_count"]

        # Detect scanned PDFs: if fewer than 30% of pages have text, the PDF
        # is likely a scan without an OCR text layer.
        if total_pages > 0 and (text_pages / total_pages) < MIN_TEXT_PAGE_FRACTION:
            logger.warning(
                "Likely scanned PDF (%d/%d pages have text): %s",
                text_pages, total_pages, pdf_path.name,
            )
            result["status"] = "scanned"
            return result

        # Join pages with form-feed separator, then drop the references section.
        full_text = _strip_references("\f".join(pages_text))
        result["char_count"] = len(full_text)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(full_text, encoding="utf-8", errors="replace")

        result["status"] = "extracted"
        return result

    except Exception as exc:
        logger.error("Failed to extract text from %s: %s", pdf_path.name, exc)
        result["status"] = "failed"
        return result
