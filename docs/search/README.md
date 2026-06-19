# Search tab

The **Search** tab is BoneGraph's semantic search over the corpus — raw
retrieval with no LLM in the loop. Queries are embedded with SPECTER2 and
ranked by cosine similarity against the indexed chunks of the BoneScholar
dataset (54,634 papers + 16 textbooks). Results are returned with source
metadata and relevance scores.

There are no Search-specific deep docs yet — it shares the retrieval engine and
corpus with the Chat tab, documented at the top level:

- [`../architecture.md`](../architecture.md) — system-wide overview
- [`../phase2_rag_pipeline.md`](../phase2_rag_pipeline.md) — chunking, embeddings, retrieval, benchmark (MRR 0.928, Recall@5 1.000)
- [`../papers_ingestion_pipeline.md`](../papers_ingestion_pipeline.md) · [`../textbooks_ingestion_pipeline.md`](../chat/textbooks_ingestion_pipeline.md) — the corpus / dataset

Add Search-tab–specific design notes here as they arise.
