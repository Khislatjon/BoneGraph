# Vision tab

The **Vision** tab is BoneMind's image-understanding surface (internally
`analyse`). Today it runs a LLaVA-based VLM that produces a structured
description / radiology-style report from an uploaded X-ray, MRI, micro-CT, or
histology image.

**Planned direction** (per the 21 + 28 May supervision meetings): rebuild it to
mirror the Reasoning tab — a VLM agent answering *"what am I looking at?"*,
reviewed by a critic, grounded against vision-specific rules and the
literature / knowledge graph, and corrected through user feedback (the
"Cephalo lesson": don't make the same misidentification twice). A parallel,
deferred research track would fine-tune the VLM on image–caption pairs scraped
from the existing corpus PDFs.

Docs:

- [`phase3_vlm_plan.md`](phase3_vlm_plan.md) — the original VLM integration plan
- [`../reasoning/architecture.md`](../reasoning/architecture.md) — the pipeline this tab will mirror

Add Vision-tab design notes and the rebuild plan here as they take shape.
