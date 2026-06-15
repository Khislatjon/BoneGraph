"""
vision/
=======
The Vision tab — a VLM (``llava:13b`` via Ollama) that identifies a bone image
and a correction memory that lets the user teach it.

Deliberately *narrower* than the Reasoning tab: there is **no critic and no
predefined grounding rules** here. The only learning surface is correction
memory — a 👎 plus a free-text note, recalled the next time a *similar image*
shows up (cosine match on an image embedding, so a rotated/re-windowed copy of
the same scan still matches).

correction_store    SQLite store for feedback events + corrections, with
                    embedding-based recall (encoder-agnostic: it stores and
                    matches vectors, it does not compute them)
"""
