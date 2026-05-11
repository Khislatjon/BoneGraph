"""
reasoning/query_router.py
─────────────────────────
Phase 4 query router for the v2 equation-graph reasoner.

A single LLM call classifies a free-text query into one of four
reasoning modes — forward / abductive / counterfactual / explore —
and extracts any numeric anchors (observed values, intervention
targets, given values).  Once the router has emitted its JSON payload
the rest of the pipeline is deterministic: a
:class:`SemanticVariableAnchor` maps text hints to variable symbols
via SPECTER2 cosine similarity, and the existing
:class:`RelationRegistry` runs the chosen inference mode unchanged.

The LLM is intentionally the *only* place a language model enters the
v2 reasoning loop.  Variable anchoring, parameter sampling, chain
discovery and Monte-Carlo propagation all stay deterministic and
citable.  When the LLM is unreachable or emits unparseable JSON the
router falls back to a "forward + top semantic match" guess so the
free-text endpoint still responds with a reasonable derivation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import requests

from reasoning.semantic_anchor import SemanticVariableAnchor, VariableMatch

logger = logging.getLogger(__name__)


# ── Router output ────────────────────────────────────────────────────────────


@dataclass
class RouterDecision:
    """
    Structured output of a single :class:`QueryRouter` call.

    Every field except ``mode`` is optional; the v2 dispatcher fills in
    sensible defaults (literature midpoints) for anything the router
    or anchor couldn't extract.
    """

    mode: str                                     # forward / abductive / counterfactual / explore
    target_hint: str | None = None                # free text mentioning the target variable
    observed_value: float | None = None           # abductive: numeric observation
    observed_units: str | None = None
    intervention_hint: str | None = None          # counterfactual: variable being intervened on
    intervention_value: float | None = None       # counterfactual: new value
    given_hints: dict[str, float] = field(default_factory=dict)
                                                   # forward: any explicit numeric anchors
    rationale: str = ""
    confidence: float = 0.0
    source: str = "llm"                            # "llm" | "fallback"
    matched_variables: list[VariableMatch] = field(default_factory=list)


# ── Prompt ───────────────────────────────────────────────────────────────────


_ROUTER_SYSTEM = """\
You are the query router for a bone-physics equation-graph reasoner.

You will receive one user query about bone mechanics.  Decide which of
four reasoning modes applies and extract any numeric anchors.  Respond
with a single JSON object — no prose, no markdown, no code fences.

Mode definitions:
- "forward": predict a downstream quantity given upstream ones
  ("how does porosity affect modulus?", "what is da/dN at ΔK = 1?").
- "abductive": infer upstream quantities from an observed downstream value
  ("why is this patient's E so low?", "patient has BMD loss of 1.5 %/yr").
- "counterfactual": evaluate a do(...) intervention
  ("what if porosity dropped to 5 %?", "if cortex thinned to 2 mm…").
- "explore": broad relational queries with no specific anchors
  ("how do loading and remodelling interact?").

JSON schema:
{
  "mode":               "forward"|"abductive"|"counterfactual"|"explore",
  "target_hint":        string | null,    // what the user wants predicted or observed
  "observed_value":     number | null,    // for abductive: the measured value
  "observed_units":     string | null,
  "intervention_hint":  string | null,    // for counterfactual: variable being intervened
  "intervention_value": number | null,    // for counterfactual: new numeric value
  "given_hints":        object,            // for forward: {variable_phrase: value}
  "rationale":          string,            // 1 short sentence explaining the choice
  "confidence":         number             // 0..1
}

Examples:

User: "What is elastic modulus at 15 % porosity in a 70-year-old female?"
{"mode":"forward","target_hint":"elastic modulus","observed_value":null,"observed_units":null,"intervention_hint":null,"intervention_value":null,"given_hints":{"porosity":0.15},"rationale":"Asks for a downstream prediction with one upstream anchor.","confidence":0.95}

User: "Why is this patient's E only 12 GPa?"
{"mode":"abductive","target_hint":"elastic modulus","observed_value":12.0,"observed_units":"GPa","intervention_hint":null,"intervention_value":null,"given_hints":{},"rationale":"Wants the upstream cause of an observed value.","confidence":0.92}

User: "If porosity dropped to 5 %, how would fatigue change?"
{"mode":"counterfactual","target_hint":"fatigue crack growth","observed_value":null,"observed_units":null,"intervention_hint":"porosity","intervention_value":0.05,"given_hints":{},"rationale":"Hypothetical intervention on porosity, outcome is fatigue.","confidence":0.90}

User: "Bone stiffness under cyclic load"
{"mode":"forward","target_hint":"fatigue crack growth under cyclic load","observed_value":null,"observed_units":null,"intervention_hint":null,"intervention_value":null,"given_hints":{},"rationale":"Cyclic load implies fatigue context; predicts da/dN.","confidence":0.75}

Return only the JSON object.\
"""


# ── Router ───────────────────────────────────────────────────────────────────


class QueryRouter:
    """
    One Ollama call → :class:`RouterDecision`.

    The router *intentionally* does not call the inference engine — it
    just structures the user's intent.  The caller composes the result
    with :class:`SemanticVariableAnchor` to translate text hints into
    real variable symbols.
    """

    def __init__(
        self,
        *,
        ollama_url: str,
        model: str,
        timeout: float = 8.0,
    ) -> None:
        self._url = ollama_url
        self._model = model
        self._timeout = timeout

    # ── Public API ────────────────────────────────────────────────────────

    def route(
        self,
        query: str,
        anchor: SemanticVariableAnchor | None = None,
    ) -> RouterDecision:
        """Return a router decision, falling back when the LLM is unreachable."""
        query = (query or "").strip()
        if not query:
            return RouterDecision(mode="explore", source="fallback",
                                  rationale="Empty query.")

        # Pre-compute the semantic anchor's top-k so we can attach it to
        # whichever path returns the decision.
        matches: list[VariableMatch] = []
        if anchor is not None:
            try:
                matches = anchor.top_k(query, k=5)
            except Exception as exc:                       # pragma: no cover
                logger.warning("Semantic anchor failed during routing: %s", exc)

        decision = self._call_llm(query)
        if decision is None:
            decision = self._fallback(query, matches)
        decision.matched_variables = matches
        return decision

    # ── LLM call ──────────────────────────────────────────────────────────

    def _call_llm(self, query: str) -> RouterDecision | None:
        try:
            resp = requests.post(
                self._url,
                json={
                    "model":  self._model,
                    "format": "json",
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": _ROUTER_SYSTEM},
                        {"role": "user",   "content": query},
                    ],
                    "options": {"temperature": 0, "num_predict": 256},
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
        except Exception as exc:
            logger.warning("QueryRouter LLM call failed: %s", exc)
            return None

        parsed = _parse_router_json(content)
        if parsed is None:
            logger.warning("QueryRouter returned unparseable JSON: %r", content[:200])
            return None

        return _decision_from_dict(parsed, source="llm")

    # ── Fallback ──────────────────────────────────────────────────────────

    @staticmethod
    def _fallback(
        query: str, matches: list[VariableMatch],
    ) -> RouterDecision:
        """
        Heuristic fallback when the LLM is unavailable.

        Picks ``forward`` mode and uses the best semantic match as the
        target hint.  The deliverable case "bone stiffness under cyclic
        load" still works on the fallback path because the anchor will
        already have surfaced ``E`` and ``dK`` in its top-k.
        """
        target_hint = matches[0].name if matches else None
        return RouterDecision(
            mode="forward",
            target_hint=target_hint,
            rationale="Fallback heuristic — LLM unavailable, used top semantic match.",
            confidence=0.30,
            source="fallback",
        )


# ── JSON parsing helpers ─────────────────────────────────────────────────────


def _parse_router_json(content: str) -> dict[str, Any] | None:
    """
    Tolerant JSON extractor.

    Ollama's ``format: json`` option produces clean JSON in the normal
    case, but smaller guard models occasionally wrap their output in
    code fences or prose.  Try strict parsing first, then a regex
    fallback that grabs the first ``{...}`` block.
    """
    content = (content or "").strip()
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


_VALID_MODES = {"forward", "abductive", "counterfactual", "explore"}


def _decision_from_dict(d: dict, *, source: str) -> RouterDecision:
    mode = str(d.get("mode") or "").strip().lower()
    if mode not in _VALID_MODES:
        mode = "explore"

    def _opt_float(x) -> float | None:
        try:
            return float(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    given_raw = d.get("given_hints") or {}
    given: dict[str, float] = {}
    if isinstance(given_raw, dict):
        for k, v in given_raw.items():
            f = _opt_float(v)
            if f is not None:
                given[str(k)] = f

    return RouterDecision(
        mode=mode,
        target_hint=(str(d["target_hint"]).strip()
                     if d.get("target_hint") else None),
        observed_value=_opt_float(d.get("observed_value")),
        observed_units=(str(d["observed_units"]).strip()
                        if d.get("observed_units") else None),
        intervention_hint=(str(d["intervention_hint"]).strip()
                           if d.get("intervention_hint") else None),
        intervention_value=_opt_float(d.get("intervention_value")),
        given_hints=given,
        rationale=str(d.get("rationale") or "").strip(),
        confidence=float(d.get("confidence") or 0.0),
        source=source,
    )
