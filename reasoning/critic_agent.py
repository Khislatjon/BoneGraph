"""
reasoning/critic_agent.py
─────────────────────────
Phase 8 — Critic agent.

Two-step ReAct loop: one corpus_search call to gather evidence, then a
finish verdict.  The critic genuinely benefits from corpus contact —
it needs to know whether a physics-predicted relationship is already
well-documented in the literature or is a potential discovery.

Verdicts
--------
interesting       Physics predicts something concrete; corpus shows this
                  is an active research topic — discrepancy or confirmation
                  is detectable and worth reporting.
trivial           Well-established, unsurprising result.
out-of-domain     Outside the model's validity range or off-topic.
needs-more-data   Direction is interesting but prediction uncertainty
                  is too high to make a strong claim.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import requests

from reasoning.agent_tools import ToolDispatcher

logger = logging.getLogger(__name__)

VALID_VERDICTS = frozenset({"interesting", "trivial", "out-of-domain", "needs-more-data"})


@dataclass
class CriticStep:
    thought: str
    action: str
    action_input: dict
    observation: str | None = None


@dataclass
class CriticResult:
    verdict: str              # one of VALID_VERDICTS
    reason: str               # 1-2 sentence explanation
    confidence: float         # 0..1
    corpus_passages: list[str]
    source: str = "agent"     # "agent" | "fallback"
    trace: list[CriticStep] = field(default_factory=list)


_CRITIC_SYSTEM = """\
You are a critic for bone-physics hypotheses.  Given a proposed hypothesis and its
physics prediction, decide whether it is scientifically interesting or noise.

You have one tool:
  corpus_search: {"query": "..."} → search 248,629 bone-science passages

Verdict options:
  "interesting"     — physics predicts something concrete AND corpus shows active research here
  "trivial"         — well-established, unsurprising
  "out-of-domain"   — outside model validity or not bone-relevant
  "needs-more-data" — direction may be real but uncertainty is too high

Rules:
- Call corpus_search exactly once, then finish.
- Base the verdict on both the physics result AND the corpus hits.
- Output only JSON — no prose, no code fences.

Turn 1 format:
{"thought":"...","action":"corpus_search","action_input":{"query":"..."}}

Turn 2 format:
{"thought":"...","action":"finish","answer":{"verdict":"...","reason":"1-2 sentences","confidence":0.0-1.0,"corpus_passages":["snippet 1","snippet 2"]}}
"""


class CriticAgent:
    """
    Two-step ReAct critic.

    Parameters
    ----------
    dispatcher : ToolDispatcher
        Wired to the live registry and retriever.
    ollama_url, model, timeout : Ollama config.
    """

    def __init__(
        self,
        *,
        dispatcher: ToolDispatcher,
        ollama_url: str,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        self._dispatcher = dispatcher
        self._url = ollama_url
        self._model = model
        self._timeout = timeout

    def critique(
        self,
        hypothesis: dict,
        physics: dict,
    ) -> CriticResult:
        """
        Evaluate a proposed hypothesis.

        Parameters
        ----------
        hypothesis : dict
            Keys: target, sweep_var, given, sweep_values, rationale.
        physics : dict
            Keys: direction, relative_change, physics_confidence,
                  magnitude, summary, citations.
        """
        context = _build_context(hypothesis, physics)
        messages = [
            {"role": "system", "content": _CRITIC_SYSTEM},
            {"role": "user",   "content": context},
        ]
        trace: list[CriticStep] = []

        for _ in range(3):           # at most 3 turns: search + finish (+ one retry)
            content = self._call_llm(messages)
            if content is None:
                return _fallback(trace)

            parsed = _parse_json(content)
            if parsed is None:
                logger.warning("CriticAgent unparseable JSON: %r", content[:200])
                return _fallback(trace)

            thought = str(parsed.get("thought") or "")
            action  = str(parsed.get("action")  or "").strip().lower()

            if action == "finish":
                answer = parsed.get("answer") or {}
                step = CriticStep(thought=thought, action=action, action_input=answer)
                trace.append(step)
                return _build_result(answer, trace)

            action_input = parsed.get("action_input") or {}
            if not isinstance(action_input, dict):
                action_input = {}

            observation = self._dispatcher.call(action, action_input)
            step = CriticStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            )
            trace.append(step)

            messages.append({"role": "assistant", "content": json.dumps({
                "thought": thought, "action": action, "action_input": action_input,
            })})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation[:1400]}",
            })

        return _fallback(trace)

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, messages: list[dict]) -> str | None:
        try:
            resp = requests.post(
                self._url,
                json={
                    "model":    self._model,
                    "format":   "json",
                    "stream":   False,
                    "messages": messages,
                    "options":  {"temperature": 0.2, "num_predict": 450},
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")
        except Exception as exc:
            logger.warning(
                "CriticAgent LLM call failed (%s): %s",
                type(exc).__name__, exc,
            )
            return None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_context(hypothesis: dict, physics: dict) -> str:
    rel_pct = ""
    rc = physics.get("relative_change")
    if rc is not None:
        try:
            rel_pct = f"{float(rc) * 100:+.0f}%"
        except (TypeError, ValueError):
            pass

    lines = [
        f"Hypothesis: sweeping {hypothesis.get('sweep_var','?')} predicts {hypothesis.get('target','?')}.",
        f"Rationale: {hypothesis.get('rationale','')}",
        f"Physics: {physics.get('summary', '')}",
        f"Direction: {physics.get('direction','?')}",
        f"Relative change: {rel_pct}",
        f"Physics confidence: {physics.get('physics_confidence', 0):.2f}",
        f"Citations: {', '.join(physics.get('citations', [])[:3])}",
        "",
        "Search the corpus and evaluate this hypothesis.",
    ]
    return "\n".join(lines)


def _parse_json(content: str) -> dict | None:
    content = (content or "").strip()
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if m is None:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _build_result(answer: dict, trace: list[CriticStep]) -> CriticResult:
    verdict = str(answer.get("verdict") or "needs-more-data").strip().lower()
    if verdict not in VALID_VERDICTS:
        verdict = "needs-more-data"

    raw_passages = answer.get("corpus_passages") or []
    passages = [str(p)[:300] for p in raw_passages[:3] if p]

    return CriticResult(
        verdict=verdict,
        reason=str(answer.get("reason") or ""),
        confidence=float(answer.get("confidence") or 0.5),
        corpus_passages=passages,
        source="agent",
        trace=trace,
    )


def _fallback(trace: list[CriticStep]) -> CriticResult:
    return CriticResult(
        verdict="needs-more-data",
        reason="Critic agent unavailable; could not evaluate.",
        confidence=0.0,
        corpus_passages=[],
        source="fallback",
        trace=trace,
    )
