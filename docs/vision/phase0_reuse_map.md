# Vision Phase 0 — Reuse Map

**The gate question:** is the Reasoning tab's loop genuinely model-agnostic, so
"vision = swap the agent" is cheap — or is it secretly coupled to *text*?

**Answer: the loop is text-in / text-out around an LLM call.** A VLM produces
text (a structured description), so most of the loop is reused. The audit found
**three genuinely new pieces** and **one design fork**. The cheapness thesis for
V1–V4 holds.

This is the evidence behind [`architecture.md`](architecture.md).

---

## Component-by-component verdict

| Component | Location | Coupling | Verdict |
|-----------|----------|----------|---------|
| `_enforce_user_rules` | [api/main.py:747](../../api/main.py) | Operates on violation dicts; zero text assumptions | **Lift as-is** |
| Critic call | `_call_critic` [api/main.py:674](../../api/main.py) | `question` + `answer` strings → JSON verdict. Never sees the image | **Lift + adapter** (rename → prompt/description). *Blind to the image — see fork.* |
| Critic loop / revision | `reason_chat` [api/main.py:803](../../api/main.py) | SSE round orchestration; agent-agnostic | **Lift + adapter** |
| Literature fetch | `_fetch_literature` [api/main.py:629](../../api/main.py) | `retriever.query(question)` — keyed on a text string | **Lift, change the key** → VLM identification |
| KG anchoring | `kg_facts` / `anchor_nodes` [reasoning/kg_context.py:74](../../reasoning/kg_context.py) | Anchors nodes by matching terms in `question` | **Lift, change the key** → identified structures |
| Grounding engine | `check(text, user_rules)` [reasoning/physical_grounding.py](../../reasoning/physical_grounding.py) | Pure functions matching numeric/directional claims | **Lift as-is** |
| Grounding rule **set** | physical_grounding.py | The 8 rules are fracture-prose specific | **🆕 Bespoke vision rules** |
| Feedback store | [reasoning/feedback_store.py](../../reasoning/feedback_store.py) | SQLite keyed by `turn_id` / text. Agent-agnostic | **Lift for storage** — *no image key (new piece #1)* |
| Rule extractor | `extract(q,a,fb)` [reasoning/rule_extractor.py:85](../../reasoning/rule_extractor.py) | Emits `range` / `forbid_pattern` — text-claim shaped | **Adapter / partial rewrite** (new kind) |
| VLM model call | `/api/analyse` [api/main.py:1193](../../api/main.py) | `llava:13b` via Ollama, **confirmed callable**, returns structured JSON | **Reuse the call + parse; the loop is new** |

---

## The three new pieces

1. **Image-keyed recall.** `feedback_store` keys corrections on `turn_id` / text.
   "Recall the correction on a *similar image*" needs an image-similarity match —
   either match on the VLM's identification text (cheap, Option A) or add an
   image-embedding column (stronger, Option B). This is the Cephalo payoff, so
   the key choice is deliberate. See architecture → *Vision memory*.
2. **The critic is blind to the image.** It reviews the VLM's *text description* —
   it catches "you said cortical but cited trabecular density numbers"; it can
   **not** catch "this is actually trabecular bone." Visual misID is the user's
   job (correction + memory).
3. **An `identification_fix` rule kind.** Vision corrections are "X→Y
   identification" fixes, not the numeric ranges the reasoning extractor emits.

---

## The design fork (resolved in architecture.md)

**Critic A/B** — text-only critic (cheap, user catches visual misID) vs. a
second image-seeing VLM critic (powerful, doubles VLM cost). **MVP = A.** The
user-in-the-loop *is* the correction mechanism Gianluca described; a blind
text-critic + user correction + memory tells the full story without betting on a
VLM critiquing a VLM. B is deferred.

---

## Two corrections to the earlier plan summary

- The running model is **`llava:13b`**, not "LLaVA 1.6" — worth fixing wherever
  the plan says 1.6.
- `/api/analyse` is a **one-shot** call (no history, streaming, critic, or
  feedback) — it's a starting model + JSON schema, **not** the pipeline. The
  orchestration around it is new code (mirrors `reason_chat`).

---

## Conclusion

V1–V4 reuse the loop; the only real new code is (1) image-keyed recall, (2) the
bespoke vision rule set, and (3) the `identification_fix` rule kind — all
bounded. The estimate stands: a working "upload → identify → critic → correct →
remember" vision tab is roughly one A-task of effort, not a rewrite.
