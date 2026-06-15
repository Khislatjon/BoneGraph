# Vision Tab — Architecture

**Status: 🟡 Designed, not yet built (June 2026).**

The single authoritative architecture reference for the Vision tab. It is a
deliberate mirror of the [Reasoning tab](../reasoning/architecture.md): same
agent → grounding → critic → feedback loop, with the **agent swapped for a
vision-language model (VLM)** answering *"what am I looking at?"*.

This document is the design target. For the Phase 0 audit that established how
much of the reasoning pipeline is reusable, see
[`phase0_reuse_map.md`](phase0_reuse_map.md). For the original one-shot VLM
integration, see [`phase3_vlm_plan.md`](phase3_vlm_plan.md).

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

## Components

| Component | Type | Role |
|-----------|------|------|
| Image guard | size/type + optional VLM | Reject non-bone / non-image uploads before the loop. |
| Vision agent | `llava:13b` (VLM) | Produce a structured identification from image + prompt. Streamed. |
| Image grounding | pure Python | Deterministic checks over the structured fields (not free prose). |
| Critic | `huatuogpt-bone` (JSON) | Review the **description** against violations + evidence. Blind to the image. |
| Correction memory | SQLite + matcher | Recall a prior correction for a similar image/identification. |
| Correction extractor | `llama3.2:3b` (JSON) | Turn a 👎 + correction into a stored identification-fix rule. |
| Evidence: literature | `BoneMindRetriever` | Corpus passages keyed on the identification (A1). |
| Evidence: knowledge graph | `ontology.db` | 1-hop edges around identified structures (A4). |

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

## Vision memory (the reinforcement pillar)

The reasoning tab's "second chat is better" loop, for images. This is the one
place the storage layer needs more than the reasoning store provides: a
correction must be recalled for a **similar image**, not just a similar text
question.

```
👍  →  event log (scoreboard; not injected)

👎  →  correction ("that's trabecular bone, not cortical")
        │
        ▼
   correction_extractor (llama3.2 JSON) → identification_fix proposal
        │
        ▼
   user confirms / edits in the correction card
        │
        ▼
   corrections table  +  recall key  (see below)
        │
        ▼
   NEXT similar image → recall surfaces the prior correction
        → agent sees it as context → grounding + critic enforce it
        → "the second time, it gets it right"
```

**Recall key — design decision (open):**

| Option | Mechanism | Trade-off |
|--------|-----------|-----------|
| **A · text key** (MVP) | Match on the VLM's identification string / context terms | Free — reuses the existing SQLite store. Weaker: only matches images the VLM *describes* similarly. |
| **B · image embedding** | Store an image embedding column; match by cosine similarity | Stronger, true "similar image" recall. Adds an embedding model + a column. |

MVP ships **A**; **B** is a schema-compatible upgrade. Both are honest
retrieval-of-corrections, not weight updates.

---

## Modes

| Mode | Pipeline | Use |
|------|----------|-----|
| **Deep** (default) | VLM + grounding + critic loop + evidence | The full trustworthy-AI / Cephalo story. |
| **Quick** | VLM + grounding only (no critic, no evidence) | Live demos, fast identification. User rules + badge still apply. |

---

## The critic fork (decided for MVP)

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

## API surface (planned, mirrors `/api/reason/*`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/vision/chat` | POST (multipart) | SSE: VLM → grounding → critic loop. Fields: `image`, `prompt`, `history`, `mode`. |
| `/api/vision/feedback` | POST | Record 👍/👎; on 👎 with text, return a proposed correction. |
| `/api/vision/corrections/confirm` | POST | Persist a confirmed identification-fix. |
| `/api/vision/rules` | GET | List vision grounding rules (incl. disabled). |
| `/api/vision/rules/{id}/enabled` | POST | Enable/disable a rule. |
| `/api/vision/rules/{id}` | DELETE | Delete a rule. |

The legacy one-shot `/api/analyse` (current LLaVA endpoint) is superseded by
`/api/vision/chat` but can remain as a "quick analyse" fallback.

### `/api/vision/chat` SSE events (mirror reasoning, image-flavoured)

```
round_start    {phase, round}
identification {schema:{identification,modality,morphology,estimated_scale,
                features,confidence}}                          (after round 1)
recalled       {corrections:[{prior_id, was, corrected_to}]}  (if memory hit)
literature     {passages:[{rank,title,year,snippet,score}]}   (deep only)
kg_context     {facts:[{subject,relation,object,weight}], anchors}(deep only)
token          {content, phase}
image_check    {round, passed, violations:[{rule,name,detail,source}]}
critic_review  {round, verdict, notes, suggested_revision, majority, minority}
done           {identification, rounds, literature, kg, critic_resolved, mode}
error          {message}
```

---

## Data model (planned, `data/db/vision_feedback.db`)

```sql
feedback_events (id, user_id, turn_id, polarity, created_at)
corrections     (id, user_id, turn_id, prompt, identification,
                 feedback_text, recall_key, image_ref, created_at)
vision_rules    (id, user_id, rule_id, name, kind, params(JSON),
                 source_correction_id, origin, enabled, created_at)
```

`kind ∈ {range, forbid_pattern, identification_fix}`. `recall_key` is the
text/identification key for Option A; an `image_embedding BLOB` column is the
Option B upgrade. KG remains read-only in `data/db/ontology.db`.

---

## File map (planned; ⟲ = reused from `reasoning/`)

```
api/main.py                       all /api/vision/* endpoints, VLM call, critic loop
                                  (mirrors the /api/reason/* block)
vision/image_grounding.py    🆕   built-in vision rules + user-rule compiler + check()
vision/correction_store.py   🆕   SQLite: events, corrections, recall matcher
vision/correction_extractor.py 🆕 👎 correction → identification_fix proposal
reasoning/kg_context.py      ⟲    anchor + 1-hop KG edges (re-keyed on identification)
retrieval/retriever.py       ⟲    literature retrieval (re-keyed on identification)
reasoning/physical_grounding.py ⟲ grounding engine reused by image_grounding
frontend/index.html               VisionTab, ImageUpload, ModeToggle, RulesPanel,
                                  ReviewDialogue, FeedbackBar, CorrectionCard
```

---

## Build sequencing (where this maps to the V-tracks)

| Phase | Delivers | V-track |
|-------|----------|---------|
| **0** ✅ | Reuse audit — confirmed loop is model-agnostic | — |
| **1** | Image upload + chat shell + VLM agent (structured schema) | V1 |
| **2** | Critic over description + re-keyed evidence + vision rules | V2 / V3 |
| **3** | Correction memory (recall + enforce) — the Cephalo payoff | V4 |
| **4** | *(deferred, parallel)* fine-tuning audit on corpus image–caption pairs | V5.1 |

Critical path to a demo: **0 → 1 → 3**. Phase 2 hardens it; Phase 4 never gates
the demo.
