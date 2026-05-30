# Reasoning Tab

**Status: 🟢 Live — clean-slate rebuild, May 2026.**

The Reasoning tab is BoneMind's hypothesis-generation and step-by-step
reasoning surface for bone fracture, fragility, and remodelling questions.
It is a complete rewrite of the previous equation-graph reasoner, scoped
narrower and built around three pillars taken from the 21 May supervision
meeting:

1. **Agentic** — a reasoning agent and a critic agent, with bounded revision.
2. **Physical grounding** — deterministic, fracture-scoped rules executed
   on every answer.
3. **Reinforcement** — user feedback grows a personal rule layer that
   applies on every future request.

> The previous reasoning tab (equation-graph reasoner, Proposer/Critic over
> 12 Variables / 7 Relations, Surprises panel) has been retired. See
> [`reasoning/equation_graph.md`](reasoning/equation_graph.md) for the
> archived design.

---

## Scope

Per Gianluca (~13:14, 21 May): the demonstrable, defensible core is
**bone fracture and fracture risk**, driven by

- trauma,
- osteoporosis,
- tumour-related frailty.

The reasoning agent itself will answer any bone-science question (mechanics,
biology, pathology, imaging, biomaterials). The physical-grounding rule
set is intentionally narrower — rules only fire when an answer touches a
fracture-relevant domain.

---

## Pipeline

```
User question
   │
   ▼
Topic guard           ── out of scope ──► refusal (UI hides feedback controls)
   │
   ▼
Load user rules (Tier 2)
   │
   ▼
ROUND 1 · Reasoning agent       (huatuogpt-bone, streamed tokens)
   │
   ▼
Strip thinking block + Physical grounding check
   │
   ▼
ROUND 2 · Critic                (huatuogpt-bone, JSON verdict)
   │
   ├── ACCEPT ───────────────────────────────► display
   │
   └── DISPUTE
        │
        ▼
   ROUND 3 · Agent revises      (sees critic notes + suggestion)
        │
        ▼
   Strip + Physical grounding check
        │
        ▼
   ROUND 4 · Critic re-reviews
        │
        ├── ACCEPT ────────────► display
        │
        └── STILL DISPUTE ─────► display with "unresolved" badge
```

**Hard cap: 2 iterations (≤4 LLM calls).**

Output to the UI includes the final answer, the physical-grounding badge
(pass / fail + violations), a collapsible critic dialogue, the 👍 / 👎
feedback bar, and (on 👎 with text) an editable proposed-rule card.

---

## Phases as built

| Phase | What it added | Key files |
|-------|----------------|-----------|
| **1 — Foundation** | Turn data model, chat shell, multi-turn history (5-pair sliding window), context-window ring, topic guard. | `frontend/index.html::ReasonTab` |
| **2 — Reasoning agent** | Pure-LLM agent behind the `Point N. / Basis.` structured prompt. Thinking-block stripping. | `api/main.py::REASONING_SYSTEM_PROMPT`, `/api/reason/chat` |
| **3 — Physical grounding** | 8 deterministic, domain-gated, fracture-scoped rules. Range-aware (matches "10–20 GPa" as two endpoints), source-tagged violations. | `reasoning/physical_grounding.py` |
| **4 — User feedback → rules** | 👎 + free text → small-LLM extracts a structured rule → user confirms / edits → persisted in SQLite → merged into the grounding check on every future request. | `reasoning/feedback_store.py`, `reasoning/rule_extractor.py`, `/api/reason/feedback`, `/api/reason/rules/*` |
| **5 — Critic agent** | Always-review, conditionally-iterate, hard 2-iteration cap. Critic = same model, sceptical prompt, JSON output. Streaming UX with phase pills (Reasoning / Critic reviewing / Revising / Critic re-checking). Collapsible review-dialogue panel. | `api/main.py::CRITIC_SYSTEM_PROMPT`, `_call_critic`, `_llm_stream` |

---

## Components

### Reasoning agent

Model: `huatuogpt-bone` via Ollama. `temperature: 0.3`. Streamed tokens.

System prompt expects the agent to emit `**Point N.**` / `*Basis.*` blocks —
between 1 and 4 points, "as few as the question actually needs". No
preamble or closing summary.

History is a sliding window of the last 5 turn-pairs, stripped of thinking
blocks before resending so we keep only the visible reasoning. Token
budget is loose — typical turns sit around 5–10 % of the 8192-token
context window, leaving headroom for many follow-ups.

### Topic guard

Two-stage classifier shared with the Ask tab:

1. **Lexical pass.** If the question (or the last three user questions, for
   pronoun carry-over) contains any term in `_BONE_VOCAB`, accept
   immediately. This is the common path.
2. **LLM fallback.** Ambiguous wording escalates to `llama3.2:3b` with the
   `GUARD_PROMPT`. Fails open on network errors.

The vocab set covers tissues, cells, diseases, mechanics, imaging,
anatomy, drugs, and the field name. Adding terms is the safest way to
fix false rejections.

When the guard refuses, the response sets `out_of_scope: true` in the
`done` event. The UI hides the physical-grounding badge, the feedback
bar, and any proposed-rule card on that turn.

### Physical grounding — Tier 1 (built-in rules)

Eight rules in [`reasoning/physical_grounding.py`](../reasoning/physical_grounding.py):

| Rule | Bound |
|------|-------|
| Cortical modulus range | 10–25 GPa |
| Trabecular modulus range | 0.01–3 GPa |
| Cortical apparent density | 1.8–2.0 g/cm³ |
| Trabecular BV/TV | < ~60 % |
| WHO osteoporosis T-score | ≤ −2.5 |
| Wolff's law direction | loading does not weaken bone (outside disuse) |
| Density–strength scaling | power law, not linear / inverse |
| Lytic lesion effect | not negligible |

Each rule has a domain guard — it only inspects sentences containing its
context terms (e.g. `"cortical"`, `"trabecular"`, `"lytic"`). The
`check()` function compiles these together with any Tier-2 rules loaded
for the current user and returns:

```python
{
    "passed": bool,
    "violations": [{"rule", "name", "detail", "source"}],  # source: "builtin" | "user"
    "rule_count": int,
    "user_rule_count": int,
}
```

Number extraction is range-aware: `"10–20 GPa"` matches both endpoints,
and a lookbehind prevents `"-20"` being parsed as a negative inside a
range. Violations are deduped by `(rule_id, detail)`.

### Physical grounding — Tier 2 (user rules)

User-derived rules are stored in `data/db/reasoning_feedback.db` and
compiled at request time into the same `Rule` callable shape as Tier 1.

Supported shapes:

- **range** — `{unit, lo, hi, context_terms, value_terms}`. A numeric
  value in `unit` inside sentences mentioning all required context terms
  triggers if it falls outside `[lo, hi]`.
- **forbid_pattern** — `{context_terms, forbidden_terms, exception_terms,
  explanation}`. Sentences matching forbidden terms near context terms
  trigger, unless they also match an exception term.

### Critic agent

Same model (`huatuogpt-bone`), different persona. Lower variance than
swapping to a second model and avoids loading two large models in Ollama.

Input to the critic on each round:

- the question,
- the agent's answer (thinking-block stripped),
- the list of physical-grounding violations (Tier 1 + Tier 2),
- a one-line-per-rule summary of the user's Tier-2 rules.

Output is one-line JSON:

```json
{"verdict": "accept" | "dispute",
 "notes":   ["≤3 short bullets"],
 "suggested_revision": "<one short instruction>"}
```

Verdict logic:

- **ACCEPT** if the answer is broadly correct. A flagged violation on an
  edge case the agent already qualified is still acceptable — the critic
  says so in notes.
- **DISPUTE** on factual errors, internal contradictions, missing key
  mechanisms, or unaddressed genuine violations.

On DISPUTE, the agent receives a `REVISION_PROMPT_TEMPLATE` containing
the critic's notes and suggested revision, and produces a new answer.
The critic re-reviews once more. Final state — whether or not the second
critic accepts — is surfaced to the user via `critic_resolved` in the
`done` event.

### Feedback loop

`POST /api/reason/feedback`:

- 👍: writes an event to `feedback_events`, returns updated stats.
- 👎 without text: writes the event and a `corrections` row, returns
  `proposed_rule: null`.
- 👎 with text: writes the event, writes the correction, and calls
  `rule_extractor.extract(question, answer, feedback_text)` which uses
  `llama3.2:3b` in JSON mode to propose a structured rule (range,
  forbid_pattern, or `none`). The proposal is returned, not saved — the
  UI shows it as an editable card and the user clicks **Confirm rule**
  to persist via `POST /api/reason/rules/confirm`.

The rules-list endpoints (`GET /api/reason/rules`,
`DELETE /api/reason/rules/{id}`) round out the management surface;
the UI for listing and deleting rules is the obvious next addition.

---

## Two-tier deployment

The shipped baseline is **Tier 1** — the 8 built-in rules in code,
versioned with releases. **Tier 2** is per-install: each user grows their
own SQLite-backed layer through corrections. A clinician's rules never
reach a researcher.

A "propose upstream" promotion workflow is intentionally not yet built;
strong local rules can be hand-promoted into Tier 1 between releases.
For the current single-user demo, Tier 1 + Tier 2 is sufficient.

---

## API surface

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/reason/chat` | POST | SSE stream: agent → grounding → critic loop. |
| `/api/reason/feedback` | POST | Record 👍/👎; on 👎 with text, extract a proposed rule. |
| `/api/reason/rules/confirm` | POST | Persist a confirmed (possibly edited) user rule. |
| `/api/reason/rules` | GET | List the current user's rules. |
| `/api/reason/rules/{id}` | DELETE | Disable a user rule. |

### `/api/reason/chat` SSE events

```
round_start    {phase: "agent" | "critic" | "agent_revise", round: N}
token          {content, phase}
physical_check {round, passed, violations, rule_count, user_rule_count}
critic_review  {round, verdict, notes, suggested_revision}
done           {answer, rounds, prompt_tokens, completion_tokens,
                context_window, out_of_scope, critic_resolved,
                final_physical_check}
error          {message}
```

---

## Data model

Persisted in `data/db/reasoning_feedback.db`:

```sql
feedback_events  (id, user_id, turn_id, polarity, created_at)
corrections      (id, user_id, turn_id, question, answer, feedback_text, created_at)
user_rules       (id, user_id, rule_id, name, kind, params (JSON),
                  source_correction_id, enabled, created_at)
```

`user_id` defaults to `'local'`. The column exists so multi-user
deployment is a schema-compatible change later.

The frontend's per-turn shape:

```js
turn = {
  id, query, answer, out_of_scope,
  physical_check: {passed, violations, rule_count, user_rule_count},
  rounds: [{role, round, content?, verdict?, notes?, ...}],
  critic_resolved: bool,
  timestamp,
}
```

---

## Cost characteristics

| Mode | LLM calls | Typical latency | Tokens vs Phase 2 |
|------|-----------|----------------:|------------------:|
| Phase 2 baseline (no critic) | 1 | 5–10 s | 1× |
| + Physical grounding (Phase 3) | 1 | +<100 ms | 1× |
| + Critic, accept first review | 2 | 10–20 s | ≈ 2× |
| + Critic, one revision needed | 4 | 30–40 s | ≈ 3–4× |
| Hard cap (2 revisions) | 4 | ≤ 40 s | ≤ 4× |

Mean latency roughly doubles versus pure Phase 2. The critic earns the
cost by catching what physical grounding cannot (internal contradictions,
missing mechanisms, ambiguous phrasing) and by surfacing genuine
disagreements rather than hiding them.

---

## Demo flow — "second chat is better than the first"

The headline claim from Gianluca (~23:42):

```
TODAY                                       TOMORROW (or new chat)
─────                                       ─────────────────────
Q.  Cortical bone elastic modulus?           Q.  Cortical bone stiffness
A.  ~80 GPa                                      for an FE model?
● Grounding violation (80 ∉ 10–25 GPa)       A.  ~70 GPa
● Critic dispute → revise                    ● Grounding violation
● User 👎 "cortical is 10–25 GPa"               (Rule R-09 / your rule)
→ Rule R-09 added to Tier 2                  ● Critic resolves on revision
                                             "The system cannot make this
                                              mistake twice."
```

A scripted eval harness running this before/after pattern across N seed
queries and reporting the per-seed delta is the next deliverable from
the meeting. See **What's next**.

---

## What's next

| Item | Why |
|------|-----|
| **Vision tab** | Same reasoning loop on radiographs / micro-CT. Raised by Gianluca on 21 May. |
| **Eval harness** | Scripted same-query before/after correction with measurable delta. Needed to substantiate the paper claim, not just demo it. |
| **Rule management UI** | List / disable / delete user rules. Backend already supports it via the DELETE endpoint. |
| **Promotion workflow** | Hand-curated promotion of strong local rules into the shipped Tier 1 baseline. |
| **Reasoning-mode toggle** | Quick mode (no critic) vs deep mode (with critic) — for time-sensitive demos. |
| **Critic-on-dispute trigger** | Optional: re-run the critic when a user clicks 👎, as a "second opinion" before they write the correction. |

---

## File map

```
api/main.py                       /api/reason/* endpoints
                                  · REASONING_SYSTEM_PROMPT
                                  · CRITIC_SYSTEM_PROMPT
                                  · REVISION_PROMPT_TEMPLATE
                                  · _is_bone_science / _BONE_VOCAB
                                  · _llm_stream / _call_critic

reasoning/physical_grounding.py   Tier 1 rules + Tier 2 compiler
                                  · check(text, user_rules=...)
                                  · _find_values (range-aware)
                                  · _compile_user_rule

reasoning/feedback_store.py       SQLite schema + queries
                                  · save_event / save_correction
                                  · save_user_rule / list_user_rules
                                  · delete_user_rule / stats

reasoning/rule_extractor.py       llama3.2:3b JSON extraction
                                  · extract(question, answer, feedback_text)

frontend/index.html               ReasonTab, ReasonTurn, FeedbackBar,
                                  ProposedRuleCard, ReviewDialogue,
                                  PhysicalCheckBadge
```

Legacy reasoning artefacts (equation-graph reasoner, Proposer/Critic over
12 variables, Surprises panel, physics-grid retrospective) live under
`reasoning/legacy/` and `docs/reasoning/`. They are not used by the
current Reasoning tab.
