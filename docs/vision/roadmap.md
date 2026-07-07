# Vision tab — data & model roadmap

Strategic direction for the Vision tab beyond the first trained classifier. Records
the scope decision, the reference architecture (PathChat), the two-path data plan,
and what is deliberately deferred. Written for the paper's discussion/future-work
section and to keep decisions on record.

> Companion to [`training.md`](training.md) (what's built) and
> [`architecture.md`](architecture.md) (the VLM + correction-memory design).

---

## 1. Current scope decision (July 2026)

Narrow and honest for the pre-print:

- **Modalities:** X-ray and clinical CT only. **No micro-CT, no MRI** for now
  (the OOD guard withholds gracefully on those — see [`training.md`](training.md)).
- **Task:** body **region** (which bone) as the grounding/routing signal, plus
  **abnormality** (normal/abnormal) as the clinician-facing signal.
- **Users / framing:** research, education, and **human-in-the-loop** triage /
  second-opinion. **Not a diagnostic tool.** Abnormality is a flag ("this merits
  a closer look"), never a diagnosis. This framing is non-negotiable and must be
  cleared with supervisors before any clinician-facing push.
- **Why region if the clinician already knows it?** Region is *infrastructure* —
  it grounds the VLM (stops it hallucinating the wrong bone), drives the OOD
  guard, and enables per-region abnormality models. The headline output for
  clinicians is abnormality, not region.

## 2. Reference architecture — PathChat

PathChat (Lu et al., *Nature* 2024, `10.1038/s41586-024-07618-3`) is the closest
published analogue: a conversational multimodal AI for pathology. Same
**LLaVA-style architecture** BoneGraph already uses (vision encoder → projector →
LLM). Key differences, which define our roadmap:

| | PathChat | BoneGraph Vision (now) |
|---|---|---|
| Vision encoder | UNI — domain-pretrained on ~100M path images | generic LLaVA encoder; BiomedCLIP for the classifier |
| MLLM | **fine-tuned** on 456k domain instructions | **off-the-shelf** `llava:13b`, prompt-only |
| Domain knowledge | baked into weights | external: a small trained classifier grounds the VLM |
| Guardrails | trained refusal examples | system prompt + OOD guard |

BoneGraph's **hybrid** (off-the-shelf VLM + small grounding classifier + OOD
guard) is a lightweight approximation of PathChat. PathChat is the north star;
replicating it wholesale needs a bone foundation encoder + a large curated
instruction set + ~32 A100s (a multi-year program), so we scale the *recipe*
down rather than copy it.

## 3. The two-path data plan (the key decision)

**Which images to collect depends on which job they do.** Do not conflate these.

### Path A — labeled datasets → the classifier (NOW, for the pre-print)

The region/abnormality classifier learns from **cleanly labeled** datasets.
**Do NOT scrape paper figures for this** — they are weakly labeled (free-text
captions), noisy (plots, multi-panel, blots), low-yield (~1–2 bone images/paper),
and would need a labeling pipeline. Public labeled sets are strictly better and
faster:

| Need | Dataset(s) | Notes |
|---|---|---|
| Upper-limb X-ray region + abnormality | **MURA** | done; abnormality labels already cached |
| Lower-limb X-ray | knee-OA sets, etc. | region expansion |
| Clinical CT region (many bones, spine, pelvis) | **TotalSegmentator, VerSe, CTPelvic1K** | 3D volumes → extract slices |
| Fracture localisation (where) | **GRAZPEDWRI-DX** (wrist, boxes), **VinDr-SpineXR** | later; enables "where", not just "abnormal" |

Access caveat: PhysioNet/OAI need credentialing (days). Start with what's in hand.

### Path B — paper figures + atlases → VLM fine-tuning (LATER, post-pre-print)

Paper-figure scraping becomes the *right* tool only for the **PathChat-style VLM
instruction fine-tune**. That is literally their recipe: turn figure+caption
pairs into image-instruction pairs and fine-tune the MLLM. This is also Gianluca's
"scrape the literature" vision, pointed at images.

- Source: the ~7,690 bone papers (figures + captions) + open **atlases** +
  figshare; micro-CT data lives here too (scattered, specimen-scale).
- Method: extract bone figures → filter → LLM reformats caption↔image into
  instructions (conversation / description / MC / **guardrails**) → **LoRA
  fine-tune `llava`** on the uni GPU.
- Reality check: BoneGraph's corpus yields far fewer usable images than PathChat's
  PubMed-OA scale, so expect a modest instruction set (supplement with atlases).

**Rule of thumb:** classifier training → labeled datasets; VLM fine-tuning →
paper figures + atlases.

## 4. Architecture direction as coverage grows

Not one flat classifier. As modalities/scales are added, the label itself changes
(region for X-ray/CT; **tissue type — cortical vs trabecular — for micro-CT**).
The scalable design is:

1. **Modality/scale router** (generalise today's OOD guard: "what kind of image
   is this?").
2. **Modality-specific heads** behind it (X-ray/CT → region + abnormality;
   micro-CT → tissue type).
3. **VLM** for all natural-language interaction, grounded by whichever head fired.

## 5. Chat handling

The classifier cannot answer free-form questions — that is the VLM's job. The
trained heads supply *persistent grounded facts*; the VLM runs the conversation.
Improvement to make: **inject the trained-model facts (region, abnormality) on
every turn**, not just the first, so follow-ups like "is this abnormal?" are
anchored to the abnormality head rather than the VLM guessing. On OOD images the
prediction is withheld and chat falls back to the bare VLM (honest fallback).

## 6. Evaluation (for the paper)

Follow PathChat's template:
- **Quantitative:** classifier accuracy / AUC on held-out data (region: 89.6%
  linear / 92.6% MLP; abnormality preliminary: 71.3% linear / 74.2% MLP, macro-F1
  0.71–0.74, abnormal recall only ~0.60 — report as a limitation, not a clinical
  flag).
- **Qualitative reader study:** a small bone-expert blinded ranking of grounded
  vs bare-VLM answers. The humerus-called-"femur" case (fixed by grounding) is a
  ready-made example — and this is exactly the Cephalo-style screenshot evidence
  the supervisors asked for.

## 7. Deferred (explicitly not now)

- Micro-CT and MRI support (needs Path B data + tissue-type heads).
- Full VLM fine-tuning (uni GPU; after the pre-print).
- Fracture/lesion **localisation** (needs box/segmentation datasets).
- A bone foundation vision encoder (UNI-scale; out of reach for now).

For the end-July pre-print, the honest v1 is: **upper-limb X-ray region
classifier grounding the VLM + OOD guard that withholds outside scope**, with
abnormality as the next increment. Everything above is post-pre-print.

## References

- Lu et al., *A multimodal generative AI copilot for human pathology*, Nature 2024
  (`10.1038/s41586-024-07618-3`) — PathChat.
- Rajpurkar et al., *MURA*, 2017.
- Zhang et al., *BiomedCLIP*, 2023.
