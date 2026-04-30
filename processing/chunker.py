"""
processing/chunker.py
======================
Sentence-aware text chunking for the BoneMind RAG pipeline.

Strategy
--------
1. Split the full text on page boundaries (\\f separators written by extractor.py)
   to preserve page numbers for citation tracking.
2. Tokenize each page into sentences using NLTK's Punkt sentence tokenizer.
3. Accumulate sentences until the chunk reaches CHUNK_TARGET_TOKENS (~400 tokens).
   Token count is approximated as len(text) / CHARS_PER_TOKEN.
4. When the target is reached, emit the chunk and start the next one seeded with
   the last CHUNK_OVERLAP_SENTENCES sentences of the previous chunk as context.
5. Chunks always end at a sentence boundary — no mid-sentence cuts.

Why sentence-aware?
-------------------
Character/token sliding windows cut text arbitrarily, splitting sentences mid-way.
This harms both retrieval quality (incomplete context) and embedding quality —
SPECTER was trained on coherent scientific passages, not sentence fragments.

Overlap strategy
----------------
Instead of a fixed character overlap, the last CHUNK_OVERLAP_SENTENCES sentences
of each chunk are carried into the start of the next. This guarantees that content
near a chunk boundary always appears in full in at least one chunk.

Token approximation
-------------------
Exact tokenization via a HuggingFace tokenizer would be ideal but adds latency
at scale (~7,500 papers). We approximate 1 token ≈ CHARS_PER_TOKEN characters,
which is conservative for English scientific text (long technical terms mean more
characters per token than everyday prose). This keeps chunks safely under
SPECTER's 512-token hard limit.
"""

from __future__ import annotations

import logging

import nltk

from config.settings import CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_SENTENCES, CHARS_PER_TOKEN

logger = logging.getLogger(__name__)

# Download the Punkt sentence tokenizer on first use.
# One-time ~13 MB download stored in ~/nltk_data/.
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    logger.info("Downloading NLTK punkt_tab tokenizer...")
    nltk.download("punkt_tab", quiet=True)


def _approx_tokens(text: str) -> int:
    """Approximate token count as character count divided by CHARS_PER_TOKEN."""
    return len(text) // CHARS_PER_TOKEN


def chunk_text(full_text: str) -> list[dict]:
    """
    Split a full extracted text into sentence-aware overlapping chunks.

    Pages are separated by the form-feed character \\f (written by extractor.py).
    Sentences are never split across chunk boundaries.

    Parameters
    ----------
    full_text : str
        The complete extracted text of a document.

    Returns
    -------
    list of dicts, each with keys:
        chunk_index  : int — sequential position in the document (0-based)
        page_number  : int — page number where this chunk starts (1-indexed)
        text         : str — chunk text, always ending at a sentence boundary
        char_count   : int — number of characters in the chunk
    """
    # ── Step 1: collect all sentences tagged with their page number ────────────
    pages = full_text.split("\f")
    all_sentences: list[tuple[str, int]] = []  # (sentence_text, page_number)

    for page_num, page_text in enumerate(pages, start=1):
        page_text = page_text.strip()
        if not page_text:
            continue
        for sent in nltk.sent_tokenize(page_text):
            sent = sent.strip()
            if sent:
                all_sentences.append((sent, page_num))

    if not all_sentences:
        return []

    # ── Step 2: accumulate sentences into token-bounded chunks ─────────────────
    chunks: list[dict] = []
    chunk_index = 0

    # overlap_carry holds the last CHUNK_OVERLAP_SENTENCES sentences from the
    # previous chunk. They seed the next chunk so context at boundaries is preserved.
    overlap_carry: list[tuple[str, int]] = []

    i = 0
    while i < len(all_sentences):

        # Seed this chunk with the overlap carried from the previous one.
        current: list[tuple[str, int]] = list(overlap_carry)
        chunk_start_page = (
            current[0][1] if current else all_sentences[i][1]
        )

        # Accumulate sentences until we hit the token target.
        while i < len(all_sentences):
            sent, page_num = all_sentences[i]
            current.append((sent, page_num))
            i += 1

            if _approx_tokens(" ".join(s for s, _ in current)) >= CHUNK_TARGET_TOKENS:
                break

        if not current:
            break

        text = " ".join(s for s, _ in current)
        chunks.append({
            "chunk_index": chunk_index,
            "page_number": chunk_start_page,
            "text": text,
            "char_count": len(text),
        })
        chunk_index += 1

        # Carry the last N sentences into the next chunk as overlap.
        overlap_carry = current[-CHUNK_OVERLAP_SENTENCES:]

    return chunks
