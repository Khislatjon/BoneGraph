# Feedback-Loop Demo (B1)

**Status: 🟢 Built and live-verified (June 2026).**

A harness that substantiates Gianluca's headline claim (~23:42, 21 May): *"the
second chat is better than the first."* It demonstrates, with a measurable
delta, that a user correction changes the system's behaviour on a repeat of the
same question.

> Code: [`eval/feedback/feedback_demo.py`](../../eval/feedback/feedback_demo.py)
> Feedback mechanism: [`reasoning_tab.md`](reasoning_tab.md) §"Feedback loop"

---

## Two-layer design

The harness deliberately separates a **reproducible metric** from a
**variable demonstration**:

### Part A — deterministic rule effect (no LLM, reproducible)

For a known answer that the **built-in** rules let through, show that adding
the user's correction (a Tier-2 rule) makes the physical-grounding check catch
it. Pure `physical_grounding.check()` calls — same input, same output, every
run. This is the metric you can quote.

```
built-in only : 0 violations
+ user rule   : 1 violation (from the user rule)
→ correction demonstrably changes the grounding outcome
```

### Part B — live pipeline before/after (real, may vary)

Run the **same query** through `/api/reason/chat` before and after injecting
the user rule, and capture the grounding + critic picture each time. This is
the actual trace a user/reviewer sees. The harness is non-destructive: it
injects its test rules via the HTTP API, captures their ids, and deletes them
afterwards.

---

## Result (June 2026, live-verified)

Part A: **2/2** corrections deterministically change the grounding outcome.

Part B, both cases — the full loop:

| Case | BEFORE correction | AFTER correction |
|------|-------------------|------------------|
| Cortical modulus (rule: 15–20 GPa) | "12–15 GPa", critic `accept`, 0 violations | round-1 **catches** → critic `dispute` → revised to **17 GPa** → final 0 violations |
| Cortical density (rule: 1.85–1.95 g/cm³) | "1.8–2.0 g/cm³", `accept`, 0 violations | round-1 **catches both endpoints** → `dispute` → revised to **1.9 g/cm³** → final 0 violations |

In both cases the second answer lands inside the user's range. The second chat
does not just *flag* the mistake — it *answers* better.

---

## Two bugs this harness surfaced (and fixed)

B1 did its job — running it exposed two real defects in the feedback loop:

### Bug 1 — the critic ignored user rules

On the first live run, the user rule fired (round-1 caught the violation) but
the critic returned `accept`, so no revision happened: the mistake was flagged
but not corrected. Fixed with **two layers** (belt and suspenders):

- **Deterministic override** — `_enforce_user_rules()` in `api/main.py`: if any
  `source:"user"` violation is present after a round, the verdict is forced to
  `dispute` regardless of the critic model's output. The user explicitly taught
  the rule; a model cannot silently overrule it.
- **Prompt** — user violations are tagged `[USER RULE — must dispute]` in the
  critic's input, and the critic prompt states user rules are non-negotiable.
  This keeps the critic's *notes* coherent with the forced verdict.

The UI shows a "forced by your rule" chip when the override fires.

### Bug 2 — unit-notation mismatch

The density rule (`g/cm^3`) silently missed the model's output (`g/cm³`,
superscript) because the unit strings did not match. Any unit with an exponent
was affected — a user could add a valid rule that never fired. Fixed with
`_unit_to_pattern()` in `physical_grounding.py`: `g/cm^3` ≡ `g/cm3` ≡ `g/cm³`,
spacing-tolerant, and works on range forms ("1.8 to 2.0 g/cm³"). Verified all
notations fire.

---

## Running it

```bash
.venv/bin/python -m eval.feedback.feedback_demo                 # Part A + B (needs server)
.venv/bin/python -m eval.feedback.feedback_demo --part-a-only   # deterministic only, no server
.venv/bin/python -m eval.feedback.feedback_demo --json out.json # save full results
```

Note: after editing `reasoning/physical_grounding.py` or the pipeline, restart
the server before Part B — the running process caches the imported module.

---

## Why this matters for the paper

Part A gives a clean, reproducible claim — *"a user correction provably changes
the grounding outcome"* — that does not depend on stochastic model output. Part
B shows the end-to-end behaviour. Together they evidence the reinforcement
pillar (corrections → deterministic rules → caught and corrected on repeat)
rather than asserting it.
