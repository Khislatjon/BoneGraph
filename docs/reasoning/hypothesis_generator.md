# Hypothesis Generator — What Makes It Different

A common question: *I have a hypothesis generator. What is its hypothesis
based on? Any AI — Claude Chat or ChatGPT — can also give me a hypothesis.
What's the difference?*

BoneMind's hypothesis generator is grounded in four things a generic LLM
does not have access to.

---

## 1. A bone-mechanics equation graph

The Proposer agent doesn't free-associate — it calls tools backed by
[`reasoning/relation.py`](../../reasoning/relation.py): 12 Variables
(E, ρ, ΔK, σ, microstrain, …) and 7 Relations (Currey, Paris–Vashishth,
beam bending, Frost mechanostat, etc.) with explicit units and validity
ranges.

When the agent claims "E ≈ X GPa at ρ = Y", that number comes from
evaluating Currey's law inside the reasoner — not from token statistics.
A plain Claude/ChatGPT chat can recite Currey's exponent, but it cannot
reliably stay numerically consistent across a multi-step chain, and
nothing stops it from inventing a wrong coefficient.

## 2. A Critic agent that re-runs the same equations

Before a hypothesis is shown to the user, the Critic agent
([`reasoning/critic_agent.py`](../../reasoning/critic_agent.py)) checks
it against the equation graph and can reject the hypothesis or request
a revision. ChatGPT alone has no second pass — and even if you prompt
it to "self-critique," the critique uses the same flawed reasoning that
produced the answer.

## 3. Your corpus, via the novelty classifier

[`reasoning/novelty.py`](../../reasoning/novelty.py) embeds the
hypothesis with SPECTER2 and compares it to your 248,629 corpus chunks.
A claim is labelled **GROUNDED**, **SPECULATIVE**, or **NOVEL** based on
actual cosine similarity to bone-science papers *you have*. A generic
chat assistant cannot tell you whether a claim is already published in
your specific reading list — it can only guess from its training cutoff.

## 4. The typed ontology

[`data/db/ontology.db`](../../data/db/ontology.db) (1,597 nodes) tells
the system that `osteocyte_lacuna` is a `structure`, `fracture_toughness`
is a `property`, `bone_remodelling` is a `process`, and so on. This is
what lets the semantic anchor
([`reasoning/semantic_anchor.py`](../../reasoning/semantic_anchor.py))
map a free-text query to the right Variable rather than the most
lexically similar word.

---

## The practical difference

| | Generic Claude / ChatGPT | BoneMind hypothesis generator |
|---|---|---|
| Numerical claims | Plausible-looking, unverified | Computed from registered laws with units |
| Multi-step coherence | Drifts across turns | Same equations on every step |
| "Is this novel?" | Vibes | SPECTER2 vs. your corpus |
| Domain anchoring | Whatever's in pretraining | Bone-specific ontology + Variables |
| Failure mode | Confident hallucination | Out-of-range value → Critic rejects |

---

## In one sentence

ChatGPT gives you *a* hypothesis; BoneMind gives you a hypothesis that
has been forced through bone-physics equations and checked against the
papers you actually own. The value is not the prose — it's the
constraint layer behind the prose.

---

## Related docs

- [`equation_graph.md`](equation_graph.md) — full reasoner design and API
- [`graph_concept_reclassification.md`](graph_concept_reclassification.md) — how the typed ontology was built
- [`physics_grid.md`](physics_grid.md) — deprecated retrospective on the earlier reasoner
