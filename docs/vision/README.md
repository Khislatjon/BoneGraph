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

**Update (2 July 2026) — add a trained component.** The supervisor review judged
a bare VLM plus correction memory insufficient as a research contribution: the
tab needs a component **trained on real bone images**. The chosen approach is a
*hybrid* — a small classifier trained on frozen BiomedCLIP features (from the
MURA dataset) whose prediction **grounds** the VLM's answer. Correction memory is
kept; the trained classifier is added in front of the VLM. Full design, data,
method, and results: [`training.md`](training.md).

See the ⚠️ scope banner at the top of [`architecture.md`](architecture.md): the
critic / rules / evidence machinery described there is the original
full-mirror design, now deferred.

Docs:

- [`training.md`](training.md) — **the trained classifier**: hybrid design, MURA data, method, compute, and results (for the paper)
- [`roadmap.md`](roadmap.md) — **data & model roadmap**: scope decision, PathChat as reference architecture, the two-path data plan (labeled datasets now vs paper-figure scraping for later fine-tuning), chat/abnormality/eval direction
- [`architecture.md`](architecture.md) — VLM + correction-memory design (read the scope banner first)
- [`phase0_reuse_map.md`](phase0_reuse_map.md) — audit of what the reasoning pipeline lends to vision
- [`phase3_vlm_plan.md`](phase3_vlm_plan.md) — the original one-shot VLM integration plan
- [`../reasoning/architecture.md`](../reasoning/architecture.md) — the pipeline this tab mirrors

Training results (MURA, BiomedCLIP frozen, unseen valid split):
**region** 89.6% linear / **92.6%** MLP (macro-F1 0.92) — strong, the headline
contribution; **abnormal/normal** 71.3% linear / 74.2% MLP (macro-F1 0.71–0.74,
abnormal recall ~0.60) — preliminary/limitation, not a clinical flag. Full
breakdown: [`training.md` §Results](training.md#8-results).
