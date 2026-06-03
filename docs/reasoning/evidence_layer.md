# Evidence Layer (Reasoning tab)

**Status: 🟡 Planned — audited, not yet wired in (May 2026).**

This document covers the **evidence layer** proposed for the Reasoning tab
after the 28 May supervision meeting: giving the **critic agent** access to
external evidence (published literature and the knowledge graph) so it can
dispute the reasoning agent with citations rather than with its own
unanchored opinion.

> See [`../reasoning_tab.md`](../reasoning_tab.md) for the live pipeline.
> The evidence layer is an addition to that pipeline, not a replacement.

---

## Why it exists

The critic is currently an LLM judging another LLM using only its own
parametric knowledge — the same knowledge that produced the answer it is
reviewing. That has three failure modes:

1. **Shared hallucination** — if the reasoning agent believes something
   false, the critic (same model family) likely believes it too, and
   rubber-stamps the error.
2. **No grounds to dispute** — without evidence the critic's objections are
   stylistic ("be more specific") rather than substantive ("the literature
   says otherwise").
3. **Cannot detect conflict** — the `conflicting_evidence` verdict is
   impossible without retrieving multiple sources; a single model has
   already blended disagreeing literature into one averaged answer.

The fix: arm the **critic** (not the agent) with two evidence channels.
The agent stays a pure reasoner — that is the tab's identity, and keeping
evidence out of the agent prompt keeps the chat context window lean.

---

## Two channels

| Channel | Source | Risk | Decision |
|---------|--------|------|----------|
| **Literature** | `BoneMindRetriever` (the Ask tab's RAG over the SPECTER2 corpus) | Low — proven, used daily | ✅ proceed (task A1) |
| **Knowledge graph** | `data/db/ontology.db` (1,597 nodes / ~1,584 usable edges) | Medium — machine-extracted, pruned, unverified for fracture | ✅ proceed with care (task A4) |

Both feed only the critic's one-shot prompt. They are **not** added to chat
history and **not** shown in the context-window ring. Cost is concentrated
in the critic call, bounded by a per-channel token cap
(literature ≤ ~1200 tokens, KG ≤ ~300 tokens), and discarded after the
verdict.

---

## Phase 0.1 audit findings

Before building, the evidence sources were audited on five fracture-domain
probes (see [`eval/evidence_audit.py`](../../eval/evidence_audit.py),
re-runnable under `.venv/bin/python`).

### Literature — strong

All five probes returned on-topic passages at high cosine similarity
(0.78–0.84). The retriever surfaces exactly the papers a critic would cite
(e.g. *"Fracture risk… accounted for by cortical porosity"* at 0.843 for the
cortical-porosity query). **No tuning needed before wiring in.**

### Knowledge graph — usable, with one caveat

Four of five concept-pair shortcuts were clean and ≤3 hops
(`osteoporosis → fragility_fracture` in one `leads_to` hop;
`cortical_bone → bone_strength` via `determines`; etc.). One-hop
neighbourhoods are excellent (`osteoporosis → fractures`, weight 12).

The graph is fragmented (356 undirected components), so some pairs have no
path — handled gracefully. Some nodes are sinks (e.g. `fragility_fracture`
has no outgoing edges): fine as a path target, useless as a source.

The caveat is **composed paths** — see below.

---

## ⚠️ Composed paths — the key design constraint

A **single edge** is one fact the graph is confident about:

```
porosity --[increases]--> bone resorption          ✓ true on its own
```

A **composed path** stitches several edges into one chain and reads them as
a single "therefore" argument:

```
porosity → resorption → cortical thickness → bone strength
```

**Each edge can be individually correct while the composed chain implies
something false** — because direction/sign does not compose cleanly. The
real path the audit produced:

```
porosity --[increases]--> resorption --[decreases]--> cortical thickness --[increases]--> bone strength
```

Read end-to-end this implies *"porosity → … → increases bone strength"*,
which is **wrong** (porosity reduces strength). Yet every hop is locally
true. The error is manufactured by chaining; the graph never claimed the
end-to-end relationship.

### Rule for the implementation

> **Feed the critic raw labelled edges, never a pre-composed multi-hop
> "therefore" chain.**

Concretely:

- **Prefer 1-hop neighbourhoods.** Give the critic everything directly
  touching a concept, as discrete facts, and let it do the inference:
  ```
  • porosity increases bone resorption
  • bone resorption decreases cortical thickness
  • cortical thickness increases bone strength
  ```
- **If a path between two named concepts is shown, present it as a list of
  independent edges, not a narrative.** Do not concatenate the relations
  into a single causal sentence.
- **Let the LLM compose; do not pre-bake the composition.** Individual edges
  are evidence; long chains of them are inference the graph never made.

This is why 1-hop neighbourhoods are the safer default and multi-hop
shortcuts are used sparingly.

---

## Conflict-aware critic (depends on this layer)

Once the critic can see multiple retrieved passages, it gains a third
verdict alongside `accept` / `dispute`:

- **`conflicting_evidence`** — gated: the critic may only use it if it can
  cite **both** a majority passage and a minority passage. On this verdict
  the agent revises to add a "the bulk of literature points to X; a minority
  position is Y" caveat instead of picking a side.

This implements James's point (28 May) that two camps in the literature can
both be well-supported, and Gianluca's guidance to "give the most likelihood
based on the bulk of literature" while letting the user re-weight later.

---

## Status & next steps

| Task | Description | State |
|------|-------------|-------|
| 0.1 | Evidence audit (literature + KG) | ✅ done — both pass |
| A1 | Literature → critic prompt | 🟢 built (`_fetch_literature`, ≤3 passages, ~1200-token cap) |
| A2 | `conflicting_evidence` verdict + caveat revision | 🟢 built (gated on citing both sides) |
| A4 | KG 1-hop / raw-edge context → critic | 🟢 built (`reasoning/kg_context.py`) |

The agent remains pure-LLM throughout. The evidence layer touches only the
critic.

### A4 as built

`reasoning/kg_context.py` exposes `kg_facts(question)`:

1. **Anchor** — find `ontology.db` nodes whose label appears in the question
   (word-boundary match; generic labels like "bone"/"tissue" are skipped;
   longest/most-specific labels win; capped at 4 anchors).
2. **Gather** — collect 1-hop edges (both directions) around each anchor,
   highest-weight first, deduped, capped at 12 facts / ~1200 chars.
3. **Format** — render as a flat list of independent edges
   (`subject [relation] object`). **No path narratives** — honouring the
   composed-path rule above. The critic prompt repeats the instruction not to
   chain edges into a causal argument.

The critic sees the facts under a header that explicitly says "independent
edges — do NOT chain them into a causal path". Both evidence channels
(literature + KG) are fetched once per request and reused across both critic
rounds; the agent never sees either. Emitted to the UI as a `kg_context` SSE
event and shown in the review-dialogue panel under "Knowledge-graph facts the
critic saw".
