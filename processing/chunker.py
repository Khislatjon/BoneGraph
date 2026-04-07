"""
processing/chunker.py
======================
Core text chunking logic used by both paper and textbook chunking pipelines.

What is chunking?
-----------------
Embedding models have a maximum input length (typically 512 tokens).  A full
paper is usually 5,000–15,000 tokens — far too long to embed as a single unit.
Chunking splits the text into overlapping windows that fit within this limit.

Each chunk is a self-contained passage that can be independently embedded and
retrieved.  The overlap ensures that sentences near chunk boundaries appear in
full in at least one chunk, so context is never lost at a split point.

Chunking strategy
-----------------
1. The text is first split on page boundaries (\f separator written by the
   extractor).  This preserves page numbers, which are important for citations.
2. Within each page, text is split into chunks of CHUNK_SIZE_CHARS characters
   with CHUNK_OVERLAP_CHARS characters of overlap between consecutive chunks.
3. Very short pages (e.g. cover pages, reference pages) that are smaller than
   the chunk size are kept as a single chunk.

Each chunk is returned as a dict with:
    chunk_index  : sequential index within the document (0, 1, 2, ...)
    page_number  : page number this chunk starts on (1-indexed)
    text         : the chunk text
    char_count   : length of the text in characters
"""

from __future__ import annotations

from config.settings import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS


def chunk_text(full_text: str) -> list[dict]:
    """
    Split a full extracted text into overlapping chunks.

    Parameters
    ----------
    full_text : str
        The complete extracted text of a document.  Pages are separated
        by the form-feed character \\f (written by extractor.py).

    Returns
    -------
    list of dicts, each with keys:
        chunk_index  : int   — sequential position in the document
        page_number  : int   — page this chunk starts on (1-indexed)
        text         : str   — the chunk text
        char_count   : int   — number of characters in this chunk
    """
    # Split on page boundaries to track page numbers.
    pages = full_text.split("\f")

    chunks = []
    chunk_index = 0

    for page_num, page_text in enumerate(pages, start=1):
        page_text = page_text.strip()

        # Skip blank pages (cover pages, blank separators, etc.)
        if not page_text:
            continue

        # If the page fits within one chunk, keep it as-is.
        if len(page_text) <= CHUNK_SIZE_CHARS:
            chunks.append({
                "chunk_index": chunk_index,
                "page_number": page_num,
                "text": page_text,
                "char_count": len(page_text),
            })
            chunk_index += 1
            continue

        # Slide a window across the page text with overlap.
        # start advances by (CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS) each step
        # so the next chunk begins CHUNK_OVERLAP_CHARS chars before this one ended.
        start = 0
        while start < len(page_text):
            end = start + CHUNK_SIZE_CHARS
            chunk_text_str = page_text[start:end].strip()

            if chunk_text_str:
                chunks.append({
                    "chunk_index": chunk_index,
                    "page_number": page_num,
                    "text": chunk_text_str,
                    "char_count": len(chunk_text_str),
                })
                chunk_index += 1

            # Advance by chunk size minus overlap so next chunk overlaps.
            start += CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS

    return chunks
