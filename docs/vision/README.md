# Vision tab

The **Vision** tab is BoneGraph's image-understanding surface (internally
`analyse`). Today it runs a LLaVA-based VLM that produces a structured
description / radiology-style report from an uploaded X-ray, MRI, micro-CT, or
histology image.

**Direction (scope narrowed, June 2026):** the tab is multi-image chat over a
VLM answering *"what am I looking at?"*, plus **correction memory** — and that
is the *only* feedback surface. **No critic and no grounding rules** here
(unlike the Reasoning tab). The "Cephalo lesson" — don't make the same
misidentification twice — is delivered purely by recall: a 👎 + note is stored
with an **image embedding** (original + a few augmented copies) and surfaced the
next time a *similar image* appears, so a rotated/re-windowed copy still
matches. Fine-tuning the VLM on corpus image–caption pairs remains a separate,
deferred research track.

See the ⚠️ scope banner at the top of [`architecture.md`](architecture.md): the
critic / rules / evidence machinery described there is the original
full-mirror design, now deferred.

Docs:

- [`architecture.md`](architecture.md) — **the authoritative design** (read the scope banner first — correction memory only)
- [`phase0_reuse_map.md`](phase0_reuse_map.md) — audit of what the reasoning pipeline lends to vision
- [`phase3_vlm_plan.md`](phase3_vlm_plan.md) — the original one-shot VLM integration plan
- [`../reasoning/architecture.md`](../reasoning/architecture.md) — the pipeline this tab mirrors

Add Vision-tab design notes and the rebuild plan here as they take shape.
