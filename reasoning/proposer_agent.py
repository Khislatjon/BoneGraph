"""
reasoning/proposer_agent.py
───────────────────────────
Phase 7 — Hypothesis Proposer agent.

Single Ollama call with the full variable graph embedded in the prompt.
Proposes a (target, sweep_var, given, sweep_values, rationale) tuple
that is worth evaluating — replacing the hand-picked sweep table in
explorer.py with an LLM-guided search.

Kept simple by design: the variable graph has only 12 nodes + 7 edges,
so embedding the entire graph in the system prompt is cheap and avoids
the latency of a multi-turn ReAct loop.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import requests

from reasoning.relation import RelationRegistry

logger = logging.getLogger(__name__)


@dataclass
class ProposerResult:
    target: str
    sweep_var: str
    given: dict[str, float]
    sweep_values: list[float]
    rationale: str
    predicted_direction: str    # "up" | "down" | "mixed" | "unknown"
    source: str = "agent"       # "agent" | "fallback"
    llm_raw: str = ""           # raw LLM output, for debugging


_RELATIONS_BLOCK = """\
Relations (inputs → output):
  density_from_porosity : phi → rho
  currey_modulus        : rho → E
  vashishth_paris       : rho, dK → da_dN
  cortical_inertia      : R, t → I_section
  beam_bending          : M, R, I_section → sigma
  hookes_law            : sigma, E → eps
  frost_mechanostat     : eps → dBMD_dt

Required root inputs to reach each target (these are what `given` must pin,
minus whatever you choose as sweep_var):
  rho      : {phi}
  E        : {phi}
  da_dN    : {phi, dK}
  I_section: {R, t}
  sigma    : {M, R, t}
  eps      : {M, R, t, phi}
  dBMD_dt  : {M, R, t, phi}
"""


def _format_variable_block(registry: RelationRegistry, bone_type: str | None) -> str:
    """Render the Variable list with ranges narrowed to ``bone_type``."""
    header = "Variables (symbol → name, unit, sweep range"
    header += f" for bone_type={bone_type!r})" if bone_type else ")"
    lines = [header + ":"]
    for v in registry.iter_variables():
        lo, hi = v.range_for(bone_type)
        unit = v.unit or "—"
        lines.append(f"  {v.symbol:<9} → {v.name}, {unit}, [{lo:g}, {hi:g}]")
    return "\n".join(lines)


def _format_valid_pairs(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return ""
    lines = [f"  {sweep} → {target}" for sweep, target in pairs]
    return "Valid (sweep_var → target) pairs — pick ONE of these:\n" + "\n".join(lines) + "\n"


def _build_system_prompt(
    scratchpad: list[dict] | None,
    valid_pairs: list[tuple[str, str]],
    variable_block: str,
    bone_type: str | None,
) -> str:
    already = ""
    if scratchpad:
        items = [
            f"{h.get('sweep_var', '?')}→{h.get('target', '?')}"
            for h in scratchpad[-8:]
        ]
        already = f"\nAlready explored (avoid repeating): {', '.join(items)}\n"

    pair_menu = _format_valid_pairs(valid_pairs)
    bone_line = (
        f"Active tissue regime: bone_type = {bone_type!r}. "
        "All sweep_values MUST stay inside the per-variable sweep range above; "
        "going outside extrapolates the equations past their fitted regime.\n\n"
        if bone_type else ""
    )

    return f"""\
You are a hypothesis proposer for a bone-physics equation-graph reasoner.
{variable_block}

{_RELATIONS_BLOCK}
{pair_menu}{already}
{bone_line}Propose ONE (target, sweep_var, given, sweep_values, rationale) hypothesis where:
- (sweep_var, target) MUST be one of the valid pairs listed above. Any other
  combination has no derivation chain and will be rejected.
- given pins all other upstream variables needed to complete the chain to
  realistic values (see "Required root inputs" above).
- sweep_values: exactly 3 floats within the sweep_var's range (low, mid, high).
- predicted_direction: one of "up" | "down" | "mixed", stating what you predict
  will happen to TARGET as SWEEP_VAR increases. Decide this carefully by
  counting the sign flips along the chain — e.g. higher phi → lower rho →
  lower E → lower toughness → HIGHER da_dN (so predicted_direction = "up",
  not "down"). The engine will compare your prediction to its computed
  result, so an incorrect direction will be flagged.
- rationale: 2 sentences that MUST contain all three of:
  (a) a directional prediction with rough magnitude — e.g. "expect target to
      fall by 50–90% as sweep_var doubles";
  (b) the name of the Relation (or short chain) driving the prediction —
      e.g. "via Currey's law" or "via beam_bending ∘ cortical_inertia";
  (c) the regime the sweep covers — e.g. "spans healthy cortical to
      osteoporotic", "femoral midshaft thickening", "physiological loading".
  DO NOT use the words "interesting", "intriguing", "explore", "reveal",
  "structural integrity", or "mechanical properties" — they add no information.

Prefer multi-hop chains over direct pairs (e.g. M→dBMD_dt is more informative
than phi→E alone).  Explore parts of the graph the "already explored" list misses.

Output a single JSON object — no prose, no code fences:
{{"target":"...","sweep_var":"...","given":{{...}},"sweep_values":[v1,v2,v3],"predicted_direction":"up|down|mixed","rationale":"..."}}"""


class ProposerAgent:
    """
    Single-call hypothesis proposer.

    Parameters
    ----------
    ollama_url : str
    model : str
        Ollama model name.  llama3.2:3b works well; huatuogpt-bone gives
        richer bone-science rationales.
    timeout : float
        Seconds to wait for Ollama.
    temperature : float
        Use >0 so successive calls produce distinct proposals.
    """

    def __init__(
        self,
        *,
        registry: RelationRegistry,
        ollama_url: str,
        model: str,
        timeout: float = 30.0,
        temperature: float = 0.8,
    ) -> None:
        self._registry = registry
        self._url = ollama_url
        self._model = model
        self._timeout = timeout
        self._temperature = temperature
        # Computed once — the registry is fixed for the life of the agent.
        self._valid_pairs = registry.valid_sweep_pairs()

    def propose(
        self,
        scratchpad: list[dict] | None = None,
        bone_type: str | None = None,
    ) -> ProposerResult | None:
        """
        Propose one hypothesis.

        Returns None when Ollama is unreachable or the response is
        unparseable.  The caller should handle None gracefully.
        """
        variable_block = _format_variable_block(self._registry, bone_type)
        system = _build_system_prompt(
            scratchpad, self._valid_pairs, variable_block, bone_type,
        )
        user   = "Propose one interesting hypothesis."

        # One retry on parse failure — matches the Critic's tolerance.
        # The temperature is non-zero so a second sample is independent.
        for attempt in range(2):
            raw = self._call_llm(system, user)
            if raw is None:
                return None
            parsed = _parse_json(raw)
            if parsed is not None:
                return _build_result(parsed, raw)
            logger.warning(
                "ProposerAgent unparseable JSON (attempt %d): %r",
                attempt + 1, raw[:200],
            )
        return None

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, system: str, user: str) -> str | None:
        try:
            resp = requests.post(
                self._url,
                json={
                    "model":   self._model,
                    "format":  "json",
                    "stream":  False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "options": {
                        "temperature": self._temperature,
                        "num_predict": 350,
                    },
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")
        except Exception as exc:
            logger.warning("ProposerAgent LLM call failed: %s", exc)
            return None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_json(content: str) -> dict | None:
    """Tolerant JSON extractor — same pattern as QueryRouter."""
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


def _build_result(d: dict, raw: str) -> ProposerResult | None:
    target    = str(d.get("target")    or "").strip()
    sweep_var = str(d.get("sweep_var") or "").strip()
    rationale = str(d.get("rationale") or "")

    if not target or not sweep_var:
        logger.warning("ProposerAgent missing target/sweep_var: %s", d)
        return None

    given_raw = d.get("given") or {}
    given: dict[str, float] = {}
    if isinstance(given_raw, dict):
        for k, v in given_raw.items():
            try:
                given[str(k)] = float(v)
            except (TypeError, ValueError):
                pass

    sv_raw = d.get("sweep_values") or []
    sweep_values: list[float] = []
    for v in sv_raw:
        try:
            sweep_values.append(float(v))
        except (TypeError, ValueError):
            pass
    if len(sweep_values) < 2:
        # Emergency fallback — unlikely with a well-prompted model.
        sweep_values = [0.05, 0.15, 0.30]

    pred = str(d.get("predicted_direction") or "").strip().lower()
    if pred not in {"up", "down", "mixed"}:
        pred = "unknown"

    return ProposerResult(
        target=target,
        sweep_var=sweep_var,
        given=given,
        sweep_values=sweep_values,
        rationale=rationale,
        predicted_direction=pred,
        source="agent",
        llm_raw=raw,
    )
