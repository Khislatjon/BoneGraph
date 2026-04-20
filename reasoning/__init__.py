"""
reasoning/
==========
Phase 4 — LRM reasoning layer.

Modules
-------
ontology    Node / Edge dataclasses and BoneKnowledgeGraph (NetworkX wrapper)
graph_db    SQLite-backed persistence for ontology.db
seed        Hand-curated seed nodes and edges; bootstrap CLI
extractor   LLM triple extraction from chunks.db       (Step 4.2)
physics     Bone physics engine                         (Step 4.3)
lrm         Core reasoning engine                       (Step 4.4)
novelty     Novelty classifier against corpus           (Step 4.5)
render      Sub-graph → structured LLM prompt context   (Step 4.6)
"""
