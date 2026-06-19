# Chat tab

The **Chat** tab is BoneGraph's conversational question-answering surface
(internally `ask`). It answers bone-science questions with retrieval-augmented
generation: SPECTER2 retrieval over the corpus + `huatuogpt-bone` streaming,
with inline `[N]` citations and a References section. Multi-turn via a sliding
window; topic-guarded to the bone domain.

There are no Chat-specific deep docs yet — the engine it rests on is shared
with the Search tab and documented at the top level:

- [`../architecture.md`](../architecture.md) — system-wide overview
- [`../phase2_rag_pipeline.md`](../phase2_rag_pipeline.md) — chunking, embeddings, RAG, retrieval benchmark
- [`../papers_ingestion_pipeline.md`](../papers_ingestion_pipeline.md) · [`../textbooks_ingestion_pipeline.md`](textbooks_ingestion_pipeline.md) — the corpus behind it

Add Chat-tab–specific design notes here as they arise.
