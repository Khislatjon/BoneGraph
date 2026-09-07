# Reasoning Tab — Architecture

**Status: 🟢 Live (June 2026).**

The single authoritative architecture reference for the Reasoning tab. For the
phased build log, API field-by-field detail, and demo flow see
[`reasoning_tab.md`](reasoning_tab.md). For the critic evidence channels
see [`evidence_layer.md`](evidence_layer.md); for the feedback/eval harness
see [`feedback_demo.md`](feedback_demo.md).

---

## Design principles

1. **The agent reasons from the question + the user's own rules — never from
   retrieval.** No literature or knowledge-graph evidence is injected into the
   agent's prompt (that lives in the critic), which keeps the tab's identity
   distinct from the Chat tab and the context window lean. The one thing primed
   into the agent is the user's *learned rules*: they go into its system prompt
   so it complies on the **first pass** instead of waiting for the critic to
   catch a violation and force a revision. Retrieval-based evidence still never
   reaches the agent.
2. **Determinism where it matters.** Physical grounding is code, not an LLM —
   same input, same outcome. User corrections become deterministic rules.
3. **The critic is the judgement layer.** All probabilistic, evidence-weighing
   work (literature, knowledge graph, conflict detection) lives in the critic,
   never the agent.
4. **User corrections are non-negotiable.** A rule the user taught overrides the
   model's judgement, enforced deterministically — not left to a prompt.
5. **Honest scope.** Fracture and fracture risk (trauma, osteoporosis,
   tumour-related frailty). The agent answers any bone question; the rules are
   fracture-scoped.

---

## Full pipeline

```
                              USER QUESTION
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ Topic guard     │ ── out of scope ──► refusal
                          │ (vocab + LLM)   │     (feedback UI hidden)
                          └────────┬────────┘
                                   ▼
                          ┌─────────────────┐
                          │ Load user rules │  (enabled only; primes agent + critic)
                          └────────┬────────┘
                                   ▼
                   ┌───────────────────────────────┐
                   │ MODE?                         │
                   └───────┬───────────────┬───────┘
                       quick│           deep│
                           │               │
                           │        ┌──────────────────────────────┐
                           │        │ Fetch critic evidence:       │
                           │        │  • literature (top-5, RAG)   │  A1
                           │        │  • KG 1-hop edges (ontology) │  A4
                           │        └──────────────┬───────────────┘
                           │                       │
                           ▼                       ▼
                   ┌──────────────────────────────────────────┐
                   │ ROUND 1 · Reasoning agent (streamed)     │  LLM + user rules
                   │ huatuogpt-bone · Point/Basis format      │  (no retrieval)
                   │ primed with user rules → right 1st pass  │
                   └────────────────────┬─────────────────────┘
                                        ▼
                   ┌──────────────────────────────────────────┐
                   │ Strip thinking → PHYSICAL GROUNDING      │  deterministic
                   │  built-in (8) + user (feedback+imported) │
                   └────────────────────┬─────────────────────┘
                                        │
                        quick ──────────┤────────── deep
                          │             │             │
                          ▼             │             ▼
                        DONE            │   ┌──────────────────────────────┐
                  (answer + badge       │   │ ROUND 2 · Critic (JSON)      │
                   + feedback bar)      │   │ sees: answer, violations,    │
                                        │   │ user rules, literature, KG   │
                                        │   └──────────────┬───────────────┘
                                        │                  ▼
                                        │      ┌────────────────────────┐
                                        │      │ _enforce_user_rules()  │  user
                                        │      │ any user violation →   │  override
                                        │      │ force verdict=dispute  │
                                        │      └───────────┬────────────┘
                                        │                  │
                          ┌─────────────┼──────────────────┼─────────────────┐
                       accept      conflicting_evidence   dispute            │
                          │             │                  │                 │
                          ▼             ▼                  ▼                 │
                        DONE   ┌──────────────────────────────────────┐      │
                               │ ROUND 3 · Agent revises (streamed)   │      │
                               │  dispute → fix the error             │      │
                               │  conflict → present majority+minority│      │
                               └──────────────────┬───────────────────┘      │
                                                  ▼                          │
                               Strip → PHYSICAL GROUNDING (round 3)          │
                                                  ▼                          │
                               ┌──────────────────────────────────────┐      │
                               │ ROUND 4 · Critic re-review           │      │
                               │ + _enforce_user_rules()              │      │
                               └──────────────────┬───────────────────┘      │
                                                  ▼                          │
                                  accept / conflicting → resolved            │
                                  dispute → "unresolved" badge               │
                                                  ▼                          │
                                                DONE ◄───────────────────────┘
                                                  │
                                                  ▼
                          ┌──────────────────────────────────────────┐
                          │ UI: answer · grounding badge · critic    │
                          │ dialogue (evidence + verdicts) · 👍/👎  │
                          │ · proposed-rule card on 👎               │
                          └──────────────────────────────────────────┘
```

**Hard cap:** 2 critic iterations (≤4 LLM calls). Quick mode = 1 LLM call.

---

## Components

| Component | Type | Role |
|-----------|------|------|
| Topic guard | lexical + `llama3.2:3b` | Keep questions in the bone domain. Shared with Chat tab. |
| Reasoning agent | `huatuogpt-bone` | Produce a 1–4 point reasoning chain. Streamed. Primed with the user's active rules so it complies on the first pass; no retrieval/KG evidence. |
| Physical grounding | pure Python | Deterministic rule check over the answer text. |
| Critic | `huatuogpt-bone` (JSON) | Review the answer against violations + evidence; verdict `accept`/`dispute`/`conflicting_evidence`. On conflict, also tags which `[L#]` back each side → a counted confidence score (two bars). |
| Rule extractor | `llama3.2:3b` (JSON) | Turn a 👎 + free-text correction into a structured rule proposal. |
| Evidence: literature | `BoneGraphRetriever` | Top-`LIT_TOP_K` corpus passages for the critic (A1). `LIT_TOP_K` defaults to 5, configurable via `BONEGRAPH_LIT_TOP_K` (raise to ~10 on a 16k-context box); it also sets the conflict-score denominator. |
| Evidence: knowledge graph | `ontology.db` | 1-hop edges around question concepts for the critic (A4). |

---

## Agent vs critic — what each can access

Two LLM roles, deliberately given different inputs. The agent reasons; the
critic judges. Built from `reason_chat` / `_call_critic` in `api/main.py`.

| Input | Reasoning agent | Critic |
|-------|:---------------:|:------:|
| The question | ✅ | ✅ |
| Conversation history (prior turns) | ✅ | ❌ |
| User's learned rules | ✅ **primed into system prompt** | ✅ as a summary |
| The agent's answer | produces it | ✅ reviews it |
| Physical-grounding violations | ❌ (only via critic notes on revision) | ✅ tagged `builtin`/`user` |
| Retrieved literature `[L#]` | ❌ | ✅ top-`LIT_TOP_K` (default 5, ≤ ~1200 tok) |
| Knowledge-graph facts | ❌ | ✅ 1-hop edges (≤ ~300 tok) |
| Critic notes / suggested revision | ✅ only in Round 3 revision | produces them |
| Majority / minority positions | ✅ only in a conflict revision | produces them |
| Model | `huatuogpt-bone`, streamed | `huatuogpt-bone`, one-shot JSON |
| Output | Point/Basis answer | verdict `accept`/`dispute`/`conflicting_evidence` |

**In one line:** the agent sees *the question + history + the user's own rules*
and nothing probabilistic; the critic sees *everything evidential* (violations,
literature, KG) but **not** the conversation history. User rules are the only
channel that reaches **both** — deliberately, because they are deterministic
ground truth, not evidence to be weighed.

---

## Physical grounding — three rule sources

All merged at request time by `physical_grounding.check(text, user_rules=...)`;
every violation is tagged with its `source`.

```
┌─────────────────────────────────────────────────────────────┐
│ PHYSICAL GROUNDING                                          │
│                                                             │
│  Tier 1 · built-in (8 rules, in code)        source=builtin │
│    cortical/trabecular modulus, density, BV/TV,             │
│    osteoporosis T-score, Wolff direction,                   │
│    density–strength scaling, lytic lesion                   │
│                                                             │
│  Tier 2 · user rules (SQLite)                               │ 
│    • from feedback (👎 → extractor → confirm) source=user   │
│    • imported (CSV/XLSX bulk)               source=user     │
│      origin column distinguishes them                       │
└─────────────────────────────────────────────────────────────┘
```

Rule kinds:
- **range** — a numeric value (notation-tolerant: `GPa`, `g/cm^3`≡`g/cm³`≡`g/cm3`)
  in a sentence matching `context_terms` (+ optional `value_terms`) must lie in
  `[lo, hi]`.
- **forbid_pattern** — a sentence matching any `forbidden_terms` near
  `context_terms` is a violation, unless it also matches `exception_terms`.
- **comparative** — an ordinal/directional claim "A `<comparator>` B" (e.g.
  *trabecular fails before cortical*). Stores the asserted order
  (`first_terms` = A, `second_terms` = B) plus `comparator_terms` naming the
  axis. A sentence on that axis that states the **reverse** order (B before A)
  is a violation. Negated/contrast sentences (e.g. *"cortical does **not** fail
  before trabecular"*) are skipped to avoid mis-flagging a correct rebuttal.
  Heuristic by design — a flag for the critic to weigh, not a truth oracle.

User rules can be enabled/disabled/deleted in the "Your rules" panel; disabled
rules stay stored but drop out of the check.

---

## Feedback loop (the reinforcement pillar)

```
👍  →  event log (scoreboard; not injected into prompts)

👎  →  free-text correction
        │
        ▼
   rule_extractor (llama3.2 JSON)  →  proposed rule
                                      (range | forbid_pattern | comparative | none)
        │
        ▼
   user confirms / edits in the proposed-rule card
        │
        ▼
   user_rules table (Tier 2)
        │
        ▼
   applied on EVERY future request, two ways:
     • primed into the agent's prompt → usually correct on the FIRST pass
     • checked deterministically by grounding → if the agent ignores the prime,
       a user violation forces the critic to dispute → agent revises
   → "second chat is better"
```

Bulk path (A5): a CSV/XLSX of rules → `rule_import` (parse → validate → dedup →
cap 50) → same `user_rules` table, tagged `origin:"imported"`.

---

## Modes (B3)

| Mode | Pipeline | Latency | Use |
|------|----------|---------|-----|
| **Deep** (default) | agent + grounding + critic loop + evidence | ~10–40 s | Serious questions; the full trustworthy-AI story. |
| **Quick** | agent + grounding only (no critic, no evidence fetch) | ~5–10 s | Live demos, casual questions. User rules + badge still apply. |

---

## Two-tier deployment

- **Tier 1** ships with the app: the 8 built-in rules, versioned with releases.
- **Tier 2** is per-install: each user grows their own rules (feedback +
  imports) in local SQLite. A clinician's rules never reach a researcher.

The `user_id` column defaults to `'local'`; multi-user is a schema-compatible
change later. A "promote to Tier 1" workflow is intentionally deferred — strong
local rules can be hand-promoted between releases.

---

## API surface

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/reason/chat` | POST | SSE: agent → grounding → critic loop. Fields: `question`, `history`, `all_questions`, `mode`. |
| `/api/reason/feedback` | POST | Record 👍/👎; on 👎 with text, return a proposed rule. |
| `/api/reason/rules/confirm` | POST | Persist a confirmed rule. |
| `/api/reason/rules` | GET | List all rules (incl. disabled) for the panel. |
| `/api/reason/rules/{id}/enabled` | POST | Enable/disable a rule. |
| `/api/reason/rules/{id}` | DELETE | Delete a rule. |
| `/api/reason/rules/import` | POST | Bulk import from CSV/XLSX. |
| `/api/reason/rules/template` | GET | Download the import template. |

### `/api/reason/chat` SSE events

```
round_start    {phase, round}
literature     {passages:[{rank,title,year,snippet,score}]}      (deep only)
kg_context     {facts:[{subject,relation,object,weight}], anchors}(deep only)
token          {content, phase}
physical_check {round, passed, violations:[{rule,name,detail,source}], rule_count, user_rule_count}
critic_review  {round, verdict, notes, suggested_revision, majority, minority, user_override,
                score}   ← score present only on conflicting_evidence; null otherwise:
                {majority_pct, minority_pct, n_supporting, n_passages,
                 majority_support:[L#], minority_support:[L#]}
done           {answer, rounds, literature, kg, prompt_tokens, completion_tokens,
                context_window, out_of_scope, critic_resolved, final_physical_check, mode}
error          {message}
```

---

## Data model (`data/db/reasoning_feedback.db`)

```sql
feedback_events (id, user_id, turn_id, polarity, created_at)
corrections     (id, user_id, turn_id, question, answer, feedback_text, created_at)
user_rules      (id, user_id, rule_id, name, kind, params(JSON),
                 source_correction_id, origin, enabled, created_at)
```

`origin ∈ {feedback, imported}`. Knowledge graph lives separately in
`data/db/ontology.db` (read-only for this tab).

---

## File map

```
api/main.py                     all /api/reason/* endpoints, prompts, critic loop,
                                _enforce_user_rules, evidence fetch, mode handling
reasoning/physical_grounding.py built-in rules + user-rule compiler + check()
reasoning/feedback_store.py     SQLite: events, corrections, user_rules (+ migration)
reasoning/rule_extractor.py     👎 correction → proposed rule (llama3.2 JSON)
reasoning/rule_import.py        CSV/XLSX bulk import: parse, validate, dedup, template
reasoning/kg_context.py         anchor + 1-hop KG edges for the critic
frontend/index.html             ReasonTab, ModeToggle, RulesPanel, ReviewDialogue,
                                FeedbackBar, ProposedRuleCard, PhysicalCheckBadge
eval/evidence/evidence_audit.py          Phase 0.1 audit of literature + KG quality
eval/feedback/feedback_demo.py           B1 "second chat is better" demo harness
```
