"""
reasoning/
==========
Phase 4 — bone knowledge graph + the live Reasoning tab.

Knowledge-graph build
---------------------
ontology    Node / Edge dataclasses and BoneKnowledgeGraph (NetworkX wrapper)
graph_db    SQLite-backed persistence for ontology.db
seed        Hand-curated seed nodes and edges; bootstrap CLI
extractor   LLM triple extraction from chunks.db
visualize_ontology  Render ontology.db to an interactive graph

Reasoning tab (agent + critic + physical grounding + user rules)
----------------------------------------------------------------
physical_grounding  Tier-1 fracture-scoped rules + Tier-2 user-rule compiler
feedback_store      SQLite store for feedback events, corrections, user rules
rule_extractor      llama3.2:3b extraction of a structured rule from feedback
rule_import         Bulk CSV/XLSX user-rule import
kg_context          1-hop knowledge-graph facts fed to the critic
"""
