# Vision Tab — Architecture

**Status: 🟡 Partially built (June 2026).** Multi-image chat ships; correction
memory is in progress (`vision/correction_store.py` ✅).

> ## ⚠️ Scope update — June 2026 (supersedes the full-mirror design below)
>
> The Vision tab is **deliberately narrower** than the Reasoning tab. The only
> learning surface is **correction memory**:
>
> - **No critic** and **no predefined grounding rules** in the Vision tab. (The
>   reasoning tab keeps both; the vision tab does not.)
> - **Correction memory is the whole feedback story**: a 👎 + a free-text note,
>   recalled the next time a *similar image* appears.
> - **Recall is image-embedding cosine match** (was "Option B, a later
>   upgrade" — now the design). The image is embedded (BiomedCLIP / CLIP / the
>   VLM's own tower); the original + a few augmented copies are stored so a
>   rotated / re-windowed copy of the same scan still matches.
>
> Everything below describing the critic loop, the bespoke grounding rule set,
> the correction *extractor*, and the `vision_rules` table is the **original
> fuller design, now deferred** — kept for context, not the current build
> target. The live sections are [Vision memory](#vision-memory-the-only-feedback-surface),
> [Data model](#data-model-datadbvision_feedbackdb), and the
> [File map](#file-map-planned---reused-from-reasoning).

The single authoritative architecture reference for the Vision tab. It was
originally scoped as a deliberate mirror of the
[Reasoning tab](../reasoning/architecture.md) — same agent → grounding → critic
→ feedback loop with the **agent swapped for a vision-language model (VLM)**
answering *"what am I looking at?"* — but has since been narrowed to the scope
banner above.

For the Phase 0 audit that established how much of the reasoning pipeline is
reusable, see [`phase0_reuse_map.md`](phase0_reuse_map.md). For the original
one-shot VLM integration, see [`phase3_vlm_plan.md`](phase3_vlm_plan.md).

---

## The one-line idea

> Same architecture as the Reasoning tab — agent + critic + grounding + evidence
> + feedback rules — but the agent is a **VLM** identifying a bone image, and the
> grounding rules check **image claims** instead of reasoning claims.

This is the "Cephalo lesson" made concrete (supervision, 21 May): a VLM that
*doesn't know what it's looking at* should be able to be corrected by the user,
remember the correction, and not make the same mistake on the next similar
image.

---

## What carries over, and what is new

The Phase 0 audit confirmed the reasoning loop is **text-in / text-out around an
LLM call**. A VLM produces text (a structured description), so most of the loop
is reused unchanged. Three things are genuinely new.

| Reasoning component | In Vision | Why |
|---------------------|-----------|-----|
| Critic loop / revision orchestration | **Reused, adapter** | `question`→`prompt`, `answer`→`description`. Round structure identical. |
| `_enforce_user_rules` | **Reused as-is** | Operates on violation dicts — agent-agnostic. |
| Literature retrieval | **Reused, re-keyed** | Key on the VLM's **identification**, not a user question. |
| KG anchoring | **Reused, re-keyed** | Anchor on the **identified structures**. |
| Physical-grounding *engine* | **Reused as-is** | Same numeric/claim matcher. |
| Physical-grounding *rule set* | **🆕 New (bespoke)** | Image claims, not reasoning claims (see below). |
| Feedback store (events, corrections) | **Reused as-is** | SQLite keyed by turn/text. |
| Correction → memory **recall** | **🆕 New** | Needs an *image* key, not just text (see "Vision memory"). |
| Rule extractor | **🆕 New kind** | Vision corrections are *identification* fixes, not numeric ranges. |
| VLM model call (`/api/analyse`) | **Reused** | `llava:13b` via Ollama is already callable; the *loop* around it is new. |

---

## Design principles

These inherit directly from the Reasoning tab; the differences are called out.

1. **The agent stays (almost) pure.** The VLM sees the **image + the user's
   prompt only** — no retrieval injected into its prompt. Evidence and KG live in
   the critic, exactly as in reasoning. (The one allowed extra input is a
   *recalled past correction* — see principle 5.)
2. **Determinism where it matters.** Image-grounding rules are code, not an LLM.
   User corrections become deterministic, recall-on-match memory.
3. **The critic is the judgement layer.** All evidence-weighing (literature, KG,
   internal-consistency checks) lives in the critic.
   **Vision caveat:** the text critic is **blind to the image** — it reviews the
   *description*, not the pixels. It catches internal inconsistency and
   contradiction-with-evidence; it does **not** catch a purely visual
   misidentification. That is the user's job (principle 5). An image-seeing VLM
   critic is a deferred enhancement, not the MVP.
4. **User corrections are non-negotiable.** A correction the user taught
   overrides the model, enforced deterministically.
5. **Honest reinforcement.** "Remember the correction" = **retrieval of past
   corrections**, not weight updates. Same honest framing as the reasoning tab.
   Fine-tuning the VLM is a separate, deferred research track (V5).
6. **Honest scope.** Identification + description of bone images (X-ray,
   micro-CT, histology). Not clinical diagnosis.

---

## Full pipeline

```
                         IMAGE  +  USER PROMPT ("what is this?")
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ Image guard     │ ── not a bone image ──► refusal
                          │ (size/type +    │     (feedback UI hidden)
                          │  optional VLM)  │
                          └────────┬────────┘
                                   ▼
                          ┌─────────────────┐
                          │ Load user rules │  (vision rules, enabled only)
                          │ + RECALL past   │  ◄── image/identification match
                          │   corrections   │
                          └────────┬────────┘
                                   ▼
                   ┌───────────────────────────────┐
                   │ MODE?                         │
                   └───────┬───────────────┬───────┘
                       quick│           deep│
                           │               │  (evidence fetched AFTER round 1,
                           │               │   keyed on the VLM's identification)
                           ▼               ▼
                   ┌──────────────────────────────────────────┐
                   │ ROUND 1 · Vision agent (VLM, streamed)   │  llava:13b
                   │ structured identification schema:        │
                   │  {identification, modality, morphology,  │
                   │   estimated_scale, features[], confidence}│
                   │  + any recalled correction as context    │
                   └────────────────────┬─────────────────────┘
                                        ▼
                   ┌──────────────────────────────────────────┐
                   │ IMAGE GROUNDING (deterministic)          │
                   │  modality consistency · morphology /     │
                   │  aspect-ratio sanity · scale plausibility│
                   │  built-in + user vision rules            │
                   └────────────────────┬─────────────────────┘
                                        │
                  quick ────────────────┤──────────────── deep
                    │                   │                   │
                    ▼                   │   Fetch critic evidence (A1+A4),
                  DONE                  │   keyed on identification:
            (identification +          │     • literature (RAG)
             grounding badge +         │     • KG edges on identified structures
             feedback bar)             │   — also fetch for the runner-up
                                        │     identification, not just top guess
                                        ▼
                                 ┌──────────────────────────────┐
                                 │ ROUND 2 · Critic (JSON)      │
                                 │ sees: description, grounding │
                                 │ violations, user rules,      │
                                 │ literature, KG  (NOT image)  │
                                 └──────────────┬───────────────┘
                                                ▼
                                    ┌────────────────────────┐
                                    │ _enforce_user_rules()  │  user
                                    └───────────┬────────────┘  override
                                                ▼
                              accept ─┬─ conflicting ─┬─ dispute
                                  │            │           │
                                  ▼            ▼           ▼
                                DONE   ┌──────────────────────────────┐
                                       │ ROUND 3 · VLM re-describes   │
                                       │  dispute → fix the claim     │
                                       │  conflict → present both     │
                                       └──────────────┬───────────────┘
                                                      ▼
                                       IMAGE GROUNDING (round 3)
                                                      ▼
                                       ROUND 4 · Critic re-review
                                       + _enforce_user_rules()
                                                      ▼
                                       accept / conflicting → resolved
                                       dispute → "unresolved" badge
                                                      ▼
                                                    DONE
                                                      │
                                                      ▼
                          ┌──────────────────────────────────────────┐
                          │ UI: image · identification · grounding   │
                          │ badge · critic dialogue (evidence +      │
                          │ verdicts) · 👍/👎 · correction card on 👎│
                          └──────────────────────────────────────────┘
```

**Hard cap:** 2 critic iterations (≤4 VLM/LLM calls). Quick mode = 1 VLM call.

**One deliberate difference from reasoning:** evidence is fetched **after**
Round 1, because the retrieval key is the VLM's *identification*, which doesn't
exist until the agent has run. (In reasoning the key is the user's question, so
it's fetched up front.) Retrieving for the **runner-up identification too**
avoids the failure mode where a misidentified image is "confirmed" against
literature for the wrong structure.

---

## Components (current scope)

| Component | Type | Role |
|-----------|------|------|
| Vision agent | `llava:13b` (VLM) | Produce a structured identification from image + prompt, with any recalled correction prepended. Streamed. |
| Image encoder | BiomedCLIP (`vision/encoder.py`) | 512-d image embeddings (original + augments) for correction recall. |
| Correction memory | SQLite + cosine matcher (`vision/correction_store.py`) | Store a 👎 + note keyed by image embedding; recall it for a similar image. |

Deferred (see scope banner): image guard, image-grounding rules, critic,
correction extractor, and literature / KG evidence. The component table for that
fuller design is preserved in the git history of this doc.

---

## Image grounding — the bespoke rule set

The grounding **engine** is reused unchanged (the same matcher that powers the 8
reasoning rules). What changes is the **rule set**: reasoning rules check
numeric claims in prose (e.g. cortical modulus in GPa); vision rules check the
**structured identification fields** the VLM emits. This is why the structured
schema is decided *first* — the rules are only checkable if the fields exist.

```
┌─────────────────────────────────────────────────────────────┐
│ IMAGE GROUNDING (vision rule set)                          │
│                                                            │
│  Tier 1 · built-in                          source=builtin │
│    • modality consistency — features must match the        │
│      claimed modality (e.g. no "histological staining"     │
│      on an X-ray; no "Hounsfield units" on histology)      │
│    • morphology / aspect-ratio sanity — a vertebral body   │
│      must not be described with long-bone proportions      │
│    • scale plausibility — estimated_scale must be within   │
│      an order of magnitude of the claimed structure        │
│                                                            │
│  Tier 2 · user rules (SQLite)                              │
│    • from feedback (👎 → extractor → confirm)  source=user  │
│    • identification corrections (X→Y) recalled by image    │
└─────────────────────────────────────────────────────────────┘
```

Rule kinds reuse the reasoning compiler where they fit (`range` for
scale/aspect bounds, `forbid_pattern` for modality-inconsistent terms) plus one
**new kind**:

- **identification_fix** — "images previously identified as *X* in this context
  were corrected to *Y*". Stored on a 👎 correction; recalled when a new image's
  identification matches, and surfaced to the agent (principle 1) and enforced by
  the critic (principle 4).

---

## Vision memory (the only feedback surface)

The reasoning tab's "second chat is better" loop, for images — and the *only*
learning surface in the Vision tab (no critic, no rules). A correction must be
recalled for a **similar image**, not just a similar text question, so recall
is a cosine match on an **image embedding**.

```
👍  →  event log (scoreboard; not injected)

👎  →  correction ("that's trabecular bone, not cortical")
        │
        ▼
   embed the image  →  original + a few augmented copies
   (rotations, flips, intensity re-windowings)
        │
        ▼
   corrections table  +  correction_embeddings (one row per variant)
        │
        ▼
   NEXT image → embed → cosine vs every stored vector
        → best score per correction ≥ threshold (~0.9)?
        → surface the prior correction to the VLM as context
        → "the second time, it gets it right"
```

This is honest retrieval-of-corrections, **not** weight updates (principle 5).

**Why embedding recall, and why augmented copies.** The hard requirement is
that a 👎 on one scan is recalled for a *rotated / re-windowed* copy of the same
scan. Vanilla CLIP/BiomedCLIP is **not** rotation-invariant — the embedding
shifts under rotation, so a single stored vector would miss. The cheap, robust
fix is to store the original plus a handful of augmented embeddings per
correction and keep each correction's best-matching variant at recall time.

**Two real risks (storage is not one of them):**

- **Threshold tuning.** Whole-image embeddings of one modality are globally
  similar (all micro-CT slices look alike to CLIP), so a loose threshold
  retrieves the *wrong* correction. Start conservative (~0.9 = near-duplicate)
  and validate against your own rotated copies. Region/patch embeddings are a
  v2 lever if whole-image proves too blunt for local corrections
  ("this region is cortical, not trabecular").
- **Search time, eventually.** Recall is a brute-force NumPy cosine scan —
  sub-millisecond for thousands of vectors. An ANN index (FAISS / hnswlib) is
  only worth it past ~100k vectors, which a single-user tool never reaches.

**Memory cost — fixed-size, never the bottleneck.** Embeddings store *vectors,
not pixels*, so size is independent of image resolution: a 512-d float32 vector
is ~2 KB. A correction with the original + ~4 augments + the note is ~10 KB.
1,000 corrections ≈ 10 MB; 10,000 ≈ 100 MB. It grows linearly at a tiny
constant.

---

## Modes *(deferred — see scope banner)*

The quick/deep split below belongs to the fuller mirror design. In the current
scope there is a single path: recall → VLM. Kept for context.

| Mode | Pipeline | Use |
|------|----------|-----|
| **Deep** (default) | VLM + grounding + critic loop + evidence | The full trustworthy-AI / Cephalo story. |
| **Quick** | VLM + grounding only (no critic, no evidence) | Live demos, fast identification. User rules + badge still apply. |

---

## The critic fork *(deferred — see scope banner)*

The critic is **text-only and blind to the image**. Two options were weighed in
Phase 0:

- **A · text critic (chosen for MVP)** — reviews the description for internal
  consistency and against retrieved evidence. Visual misidentification is caught
  by the **user** (multi-turn correction + memory). Cheapest; matches the
  Cephalo story (user corrects → system remembers).
- **B · VLM critic** — a second VLM that sees the image and can dispute the
  identification itself. More powerful, but doubles VLM cost per turn and the
  critic can be as wrong as the agent. **Deferred** to a later phase.

---

## API surface (current scope — no critic, no rules)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/vision/chat` | POST (multipart) | SSE: VLM identification, with any recalled correction prepended as context. Fields: `image`, `prompt`, `history`. |
| `/api/vision/feedback` | POST | Record 👍/👎. On 👎 with text, embed the image (+ augments) and persist via `save_correction`. |
| `/api/vision/corrections` | GET | List stored corrections (`list_corrections`). |
| `/api/vision/corrections/{id}` | DELETE | Delete a correction + its embeddings (`delete_correction`). |

The recall step (`correction_store.recall`) runs inside `/api/vision/chat`
before the VLM call: embed the incoming image, cosine-match, and if a hit
clears the threshold, prepend its `feedback_text` to the VLM prompt. The legacy
one-shot `/api/analyse` (current LLaVA endpoint) can remain as a fallback.

### `/api/vision/chat` SSE events (current scope)

```
recalled       {corrections:[{correction_id, score, feedback_text}]}  (if memory hit)
token          {content}                                              (VLM stream)
done           {identification}
error          {message}
```

(The deferred fuller design also emitted `round_start`, `image_check`,
`critic_review`, `literature`, and `kg_context` — none of those apply now.)

---

## Data model (`data/db/vision_feedback.db`)

Implemented in `vision/correction_store.py`. **No `vision_rules` table** — the
Vision tab has no rule layer. Embeddings live in their own table (one row per
augmented variant) rather than a single column, so a correction can carry the
original + several augments:

```sql
feedback_events       (id, user_id, turn_id, polarity, created_at)
corrections           (id, user_id, turn_id, prompt, identification,
                       feedback_text, created_at)
correction_embeddings (id, correction_id→corrections.id ON DELETE CASCADE,
                       user_id, variant, dim, vector BLOB, created_at)
```

`vector` is an **L2-normalised float32** blob, so cosine similarity at recall is
a plain dot product. The store is **encoder-agnostic**: it persists and matches
vectors but never loads an encoder — the API layer turns an image into a vector
(BiomedCLIP / CLIP / the VLM's own tower) and hands it in.

---

## File map (✅ = built, 🆕 = planned for current scope)

```
api/main.py                 ✅  /api/vision/{chat,feedback,corrections} — recall in
                                chat, embed-on-👎, list/delete corrections
vision/__init__.py          ✅  package doc
vision/correction_store.py  ✅  SQLite: events, corrections, embedding recall matcher
vision/encoder.py           ✅  BiomedCLIP image embeddings (lazy singleton) + augments
frontend/index.html         ✅  VisionTab, VisionFeedbackBar, RecalledNote banner
```

Dropped from the original mirror design (no longer in scope): `image_grounding.py`,
`correction_extractor.py`, and the critic / re-keyed-evidence wiring in
`api/main.py`.

**Recall flow (live):** `/api/vision/chat` cheap-exits if no corrections exist;
otherwise it embeds the incoming image with BiomedCLIP, cosine-matches via
`correction_store.recall`, emits a `recalled` SSE event, and prepends the
matched note to the VLM instruction. A 👎 + text on any turn re-sends the image
to `/api/vision/feedback`, which embeds the original + 4 augments (rot90/180/270
+ hflip — the rotation-invariance fix) and stores them. Encoder weights download
once (~400 MB) from the HF hub on first use.

---

## Build sequencing (where this maps to the V-tracks)

| Phase | Delivers | V-track |
|-------|----------|---------|
| **0** ✅ | Reuse audit — confirmed loop is model-agnostic | — |
| **1** ✅ | Image upload + multi-turn chat shell + VLM agent (structured schema) | V1 |
| **2** ~~| Critic over description + re-keyed evidence + vision rules~~ | *dropped — see scope banner* |
| **3** ✅ | Correction memory (BiomedCLIP embedding recall) — the Cephalo payoff | V4 |
| **4** | *(deferred, parallel)* fine-tuning audit on corpus image–caption pairs | V5.1 |

Critical path to a demo: **0 → 1 → 3**, all ✅. Phase 4 never gates the demo.
