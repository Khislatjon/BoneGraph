"""
reasoning/physical_grounding.py
================================

Lightweight physical-grounding filter for the Reasoning tab.

Scope follows the supervisor meeting (2026-05-21): fracture and fracture risk,
driven by trauma, osteoporosis, and tumour-related frailty. Rules encode
order-of-magnitude bounds and sign/direction constraints from bone mechanics
literature. They are intentionally permissive — a rule only fires when the
agent's text contains a numeric claim or directional claim in the rule's
domain. Most reasoning steps pass through untouched.

Rules are pure functions over the agent's final text. No LLM calls.

Usage
-----

    from reasoning.physical_grounding import check

    result = check(agent_text)
    # result = {"passed": bool, "violations": [{"rule": ..., "detail": ...}], "applied": [rule_ids]}
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


# ── Number extraction helpers ─────────────────────────────────────────────────

_NUMBER = r"(-?\d+(?:\.\d+)?)"


def _find_values(text: str, unit_pattern: str, context_terms: list[str]) -> list[tuple[float, str]]:
    """Find numeric claims like "12.3 GPa" near any of the context terms.

    Returns (value, snippet) pairs. A claim is "near" a context term if both
    appear within the same sentence.
    """
    out: list[tuple[float, str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        low = sentence.lower()
        if not any(term in low for term in context_terms):
            continue
        for m in re.finditer(rf"{_NUMBER}\s*{unit_pattern}", sentence, flags=re.IGNORECASE):
            try:
                out.append((float(m.group(1)), sentence.strip()))
            except ValueError:
                continue
    return out


def _contains_any(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(t in low for t in terms)


# ── Rule definitions ──────────────────────────────────────────────────────────

@dataclass
class Rule:
    id: str
    name: str
    check: Callable[[str], list[str]]  # returns list of violation detail strings (empty = pass)


def _rule_cortical_modulus(text: str) -> list[str]:
    """Cortical bone longitudinal elastic modulus is ~10–25 GPa."""
    LO, HI = 5.0, 30.0
    hits = _find_values(text, r"GPa", ["cortical"])
    bad = []
    for val, snippet in hits:
        # Only flag values that look like a modulus claim (avoid e.g. stress in GPa).
        if not re.search(r"modulus|stiffness|young", snippet, re.IGNORECASE):
            continue
        if val < LO or val > HI:
            bad.append(f"Cortical bone modulus claimed at {val} GPa — outside the typical 10–25 GPa range.")
    return bad


def _rule_trabecular_modulus(text: str) -> list[str]:
    """Trabecular (cancellous) apparent modulus is ~0.01–3 GPa."""
    LO, HI = 0.005, 5.0
    hits = _find_values(text, r"GPa", ["trabecular", "cancellous"])
    bad = []
    for val, snippet in hits:
        if not re.search(r"modulus|stiffness|young", snippet, re.IGNORECASE):
            continue
        if val < LO or val > HI:
            bad.append(f"Trabecular modulus claimed at {val} GPa — outside the typical 0.01–3 GPa range.")
    # Also catch MPa claims that imply GPa magnitudes.
    hits_mpa = _find_values(text, r"MPa", ["trabecular", "cancellous"])
    for val, snippet in hits_mpa:
        if not re.search(r"modulus|stiffness|young", snippet, re.IGNORECASE):
            continue
        gpa = val / 1000.0
        if gpa > HI:
            bad.append(f"Trabecular modulus claimed at {val} MPa (~{gpa:.2f} GPa) — exceeds typical 3 GPa upper bound.")
    return bad


def _rule_cortical_density(text: str) -> list[str]:
    """Cortical apparent density is ~1.8–2.0 g/cm³ (true mineral density up to ~2.1)."""
    hits = _find_values(text, r"g\s*/\s*cm[³3]|g\s*cm[⁻−-]3", ["cortical"])
    bad = []
    for val, _ in hits:
        if val < 1.4 or val > 2.2:
            bad.append(f"Cortical bone density claimed at {val} g/cm³ — outside the typical 1.8–2.0 g/cm³ range.")
    return bad


def _rule_trabecular_bvtv(text: str) -> list[str]:
    """Trabecular BV/TV typically 5–40%."""
    bad = []
    if not _contains_any(text, ["bv/tv", "bone volume fraction", "volume fraction"]):
        return bad
    for m in re.finditer(rf"{_NUMBER}\s*%", text):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if val > 60.0:
            bad.append(f"BV/TV claimed at {val}% — exceeds physiological trabecular range (typically 5–40%).")
    return bad


def _rule_osteoporosis_tscore(text: str) -> list[str]:
    """WHO osteoporosis threshold is T-score ≤ -2.5."""
    bad = []
    if not _contains_any(text, ["osteoporosis", "t-score", "t score"]):
        return bad
    # Look for any explicit threshold claim
    for m in re.finditer(rf"t[- ]?score\s*(?:of|=|<=|≤|<|>=|≥|>)?\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        # Only flag if it's being framed as the osteoporosis cutoff.
        window = text[max(0, m.start() - 80): m.end() + 80].lower()
        if "osteoporosis" in window and ("threshold" in window or "cutoff" in window or "diagnos" in window or "defin" in window):
            if abs(val - (-2.5)) > 0.3:
                bad.append(f"Osteoporosis diagnostic threshold stated as T-score {val} — WHO criterion is ≤ -2.5.")
    return bad


def _rule_wolff_direction(text: str) -> list[str]:
    """Bone adapts to mechanical load by gaining mass / stiffness, not losing it.
    Flag claims that mechanical loading *weakens* healthy bone.
    """
    bad = []
    low = text.lower()
    if "wolff" not in low and "mechanostat" not in low and "loading" not in low and "load" not in low:
        return bad
    # Look for sentences that pair "loading"/"load" with "weaken"/"loses"/"decreases mass"
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        s = sentence.lower()
        has_load = re.search(r"\b(loading|mechanical load|exercise|weight[- ]bearing)\b", s)
        has_weaken = re.search(r"\b(weaken|loses mass|decreases? (?:bone )?mass|reduces? (?:bone )?density|atroph)\b", s)
        has_disuse = re.search(r"\b(disuse|unload|microgravit|bed[- ]rest|immobilis)", s)
        if has_load and has_weaken and not has_disuse:
            bad.append(f"Claim that mechanical loading weakens bone contradicts Wolff's law / mechanostat theory (sentence: \"{sentence.strip()}\").")
    return bad


def _rule_density_strength_powerlaw(text: str) -> list[str]:
    """Trabecular strength scales with apparent density as a power law (exponent ~2).
    Flag claims of linear or inverse scaling.
    """
    bad = []
    low = text.lower()
    if not any(t in low for t in ["density", "bv/tv"]):
        return bad
    if not any(t in low for t in ["strength", "modulus", "stiffness"]):
        return bad
    # Linear claim
    if re.search(r"\b(linear(ly)? (?:proportional|scal\w+|related|with)|directly proportional)\b.*\b(density|bv/tv)\b", low):
        # only flag for trabecular context to avoid noise
        if "trabecular" in low or "cancellous" in low:
            bad.append("Trabecular strength scales nonlinearly (power-law, exponent ≈ 2) with apparent density, not linearly.")
    # Inverse claim
    if re.search(r"\b(strength|modulus)\b.*\b(decreases|inverse\w*)\b.*\bdensity\b", low):
        bad.append("Strength/modulus increases with apparent density; inverse relationship contradicts established power-law scaling.")
    return bad


def _rule_lytic_lesion_effect(text: str) -> list[str]:
    """Lytic lesions reduce vertebral failure load even at modest sizes.
    Flag claims that small lytic lesions have no mechanical effect.
    """
    bad = []
    low = text.lower()
    if not any(t in low for t in ["lytic", "metastas", "tumour", "tumor"]):
        return bad
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        s = sentence.lower()
        if "lytic" in s or "lesion" in s:
            if re.search(r"\b(no (mechanical )?effect|does not affect|negligible (?:mechanical )?(?:effect|impact))\b", s):
                bad.append(f"Claim that lytic lesions have no mechanical effect contradicts evidence that even modest lesions reduce vertebral failure load (sentence: \"{sentence.strip()}\").")
    return bad


# ── Rule registry ─────────────────────────────────────────────────────────────

RULES: list[Rule] = [
    Rule("cortical_modulus_range",     "Cortical modulus range",       _rule_cortical_modulus),
    Rule("trabecular_modulus_range",   "Trabecular modulus range",     _rule_trabecular_modulus),
    Rule("cortical_density_range",     "Cortical density range",       _rule_cortical_density),
    Rule("trabecular_bvtv_range",      "Trabecular BV/TV range",       _rule_trabecular_bvtv),
    Rule("osteoporosis_tscore",        "Osteoporosis T-score (WHO)",   _rule_osteoporosis_tscore),
    Rule("wolff_loading_direction",    "Wolff's law direction",        _rule_wolff_direction),
    Rule("density_strength_powerlaw",  "Density–strength scaling",     _rule_density_strength_powerlaw),
    Rule("lytic_lesion_effect",        "Lytic lesion mechanical effect", _rule_lytic_lesion_effect),
]


# ── Public API ────────────────────────────────────────────────────────────────

def check(text: str) -> dict:
    """Run all rules over the agent's text.

    Returns
    -------
    dict with keys:
      passed (bool): True if no rule produced a violation.
      violations (list): [{"rule": rule_id, "name": rule_name, "detail": str}, ...]
      applied (list): ids of rules that actually inspected text in their domain.
                      (Currently every rule runs; "applied" reports which ones
                      reached past their early-exit guard.)
    """
    violations: list[dict] = []
    for rule in RULES:
        try:
            details = rule.check(text) or []
        except Exception as e:  # never let a buggy rule kill the request
            details = [f"(rule {rule.id} errored: {e})"]
        for detail in details:
            violations.append({"rule": rule.id, "name": rule.name, "detail": detail})
    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "rule_count": len(RULES),
    }
