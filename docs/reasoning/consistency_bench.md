# Consistency Bench (A3)

**Status: 🟢 Built and validated (May 2026).**

A semi-automated robustness check for the Reasoning tab, implementing
Gianluca's 28 May test: *"ask the opposite question, because if one is true,
the other must be false."*

> Code: [`eval/reasoning_consistency.py`](../../eval/reasoning_consistency.py)
> Pipeline doc: [`../reasoning_tab.md`](../reasoning_tab.md)

---

## What it measures

For each seed pair `(claim_question, opposite_question)`:

1. Run **both** questions through the live `/api/reason/chat` pipeline
   (agent + physical grounding + critic loop — exactly what a user sees).
2. Extract the system's **stance** on each answer: `affirm` / `deny` /
   `uncertain`, via an LLM judge.
3. Score two metrics:
   - **Consistency** — the two stances must be *opposite* (affirm one, deny
     its negation). Affirming both, or denying both, is the logical failure
     Gianluca wants caught. **Headline metric.**
   - **Correctness** — among consistent pairs, the affirmed side must match
     the seed's ground-truth `supported` field.

The six seed pairs are fracture-domain and deliberately clean (one side
clearly true): stiffness, osteoporosis failure order, cortical porosity vs
strength, Wolff's-law loading, lytic-lesion effect, density vs strength.
Extend `SEED_PAIRS` to grow the bench.

---

## Result (May 2026)

```
Consistency: 5/6 = 83%
Correctness: 5/5 = 100%   (every consistent pair affirmed the correct side)
```

The one non-consistent pair (`density_strength`) is **not a model failure**:
the agent hedged "higher apparent density does not *guarantee* higher
strength," which is scientifically defensible (BMD ≠ bone quality). The
honest read is that the reasoning pipeline is highly self-consistent on
opposite questions.

---

## ⚠️ The stance-judge reliability finding

This is the important methodological lesson, and it must be stated whenever
the consistency number is quoted.

**The first run reported 17% consistency — which was false.** The reasoning
model was answering *correctly*; the **stance judge** was broken. Example:

```
Q (opposite): "Is trabecular bone stiffer than cortical bone?"
A:            "Trabecular bone is generally less stiff than cortical bone…"
```

The agent correctly rejected the false premise, but the first judge
(llama3.2:3b, "does the answer agree with claim X?" framing) labelled it
`affirm` — it was detecting topic-sentiment ("the answer talks
affirmatively about trabecular bone"), not stance-on-the-claim.

### Bake-off

Judges were scored against hand-labelled ground truth on the 12 saved
answers:

| Judge | Framing | Accuracy |
|-------|---------|----------|
| llama3.2:3b | "compare answer to extracted claim" | broken (0–17%) |
| llama3.2:3b | "answer-as-author" | 75% |
| **huatuogpt-bone (8B)** | **"answer-as-author"** | **83% — adopted** |

The **answer-as-author** framing wins: instead of asking the judge to compare
an answer against an extracted claim (which fails when claim and answer share
entities but differ in direction), it asks the judge to *role-play answering
the original yes/no question using only the answer text*. `yes → affirm`,
`no → deny`, `unclear → uncertain`.

### Why this matters

Automated stance detection on **adversarially-similar comparative claims** is
genuinely hard — even the 8B judge misses the occasional case (e.g. when the
agent gives the *same correct answer* to both "does trabecular fail first?"
and "does cortical fail first?", the judge can mislabel one). So:

- The harness uses the 8B judge with the answer-as-author framing.
- It **prints the judge's one-clause reason** for every stance.
- The report carries an explicit caveat: the auto-judge is **~83% reliable**
  on these claims — **spot-check and correct mislabels by hand**.

A3 is therefore best presented as a **semi-automated bench**: it runs the
pipeline, drafts stances + reasons, and a human confirms the final labels.
For a 6–12 pair bench this is cheap and correct — and far more defensible in
a paper than quoting a single auto-judged percentage.

---

## Running it

Requires the BoneMind server running (`python serve.py`) and Ollama up.

```bash
.venv/bin/python -m eval.reasoning_consistency
.venv/bin/python -m eval.reasoning_consistency --json results.json   # save full answers + reasons
.venv/bin/python -m eval.reasoning_consistency --base http://localhost:8000
```

Output: a per-pair stance table, the two percentages, and the judge's
reasoning for each stance (for spot-checking). The `--json` dump keeps full
answers for a paper appendix.

Note: the harness must run under `.venv/bin/python` — the system Python lacks
the `adapters` package the retriever needs (loaded indirectly via the
pipeline, not the harness itself, but the server already handles that; the
harness only needs `requests`).

---

## Design choices

- **Stance is extracted per-answer**, not by judging the pair jointly —
  stops a lazy judge from rubber-stamping.
- **It hits the real HTTP pipeline**, so consistency reflects exactly what a
  user experiences (critic, revision, and all). Cost: 12 full pipeline runs
  (~minutes), acceptable for an offline eval.
- **The seed pairs are load-bearing.** They are clean opposites where one
  side is clearly true — not leading or ambiguous pairs. The
  `osteoporosis_failure_order` ground truth encodes the "trabecular fails
  first" position, which the 28 May meeting itself noted has nuance; treat
  its label as the majority view, not absolute.

---

## Relationship to the evidence layer

When a question is genuinely contested, the conflict-aware critic (see
[`evidence_layer.md`](evidence_layer.md)) may return `conflicting_evidence`
and the agent presents both positions. The stance judge will read such an
answer as `uncertain` — which is correct behaviour, and will show as a
non-consistent pair under the current binary scoring. If that occurs on a
genuinely contested pair, it is a feature, not a failure; note it in the
write-up rather than counting it against consistency.
