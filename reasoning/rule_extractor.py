"""
reasoning/rule_extractor.py
============================

Turn a user correction into a structured rule that physical_grounding can
execute, using a small LLM in JSON mode. If the correction does not look
rule-shaped (e.g. "you missed the role of osteocytes"), the extractor returns
None and the caller can fall back to free-form soft memory.

Two rule shapes are supported initially:

  range          — a numeric value with a unit must lie within [lo, hi]
                   inside sentences mentioning the domain context terms.
                   { "kind": "range",
                     "name": "...",
                     "params": { "unit": "GPa",
                                 "lo": 8, "hi": 22,
                                 "context_terms": ["cortical"],
                                 "value_terms":   ["modulus","stiffness","young"] } }

  forbid_pattern — sentences matching `forbidden_terms` near `context_terms`
                   are violations unless they also match `exception_terms`.
                   { "kind": "forbid_pattern",
                     "name": "...",
                     "params": { "context_terms":   ["bone","loading"],
                                 "forbidden_terms": ["weaken","loses mass"],
                                 "exception_terms": ["disuse","unloading"],
                                 "explanation":     "Loading strengthens, doesn't weaken." } }

The extractor prompt asks the model to either return one of those shapes or
return {"kind": "none"} for non-rule-shaped corrections.
"""

from __future__ import annotations

import json
import re

import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
EXTRACTOR_MODEL = "llama3.2:3b"   # small/fast; same as the topic guard


EXTRACTOR_PROMPT = """You convert a user correction of a bone-science answer into a structured rule, OR you decide the correction is too open-ended to express as a rule.

You will receive:
- the original question
- the model's (incorrect) answer
- the user's correction text

You must reply with ONE JSON object on a single line. No prose, no markdown fences. One of these shapes:

(A) Numeric range correction — when the user is asserting that a quantity must lie in a range:
{"kind":"range","name":"<short label>","params":{"unit":"<GPa|MPa|g/cm^3|%|...>","lo":<number>,"hi":<number>,"context_terms":["<lowercase term>", "..."],"value_terms":["<lowercase term>","..."]}}

(B) Forbidden directional / definitional claim — when the user is asserting a sign or definition error:
{"kind":"forbid_pattern","name":"<short label>","params":{"context_terms":["..."],"forbidden_terms":["..."],"exception_terms":["..."],"explanation":"<one sentence>"}}

(C) No extractable rule:
{"kind":"none","reason":"<short reason>"}

Guidelines:
- "context_terms" are lowercase substrings that must appear in a sentence for the rule to look at it (e.g. "cortical", "trabecular", "osteoporosis", "lytic"). Pick 1–3 specific terms.
- "value_terms" narrow numeric range rules to the right kind of claim ("modulus", "density", "stiffness", "porosity"). Pick 1–3.
- "forbidden_terms" / "exception_terms" are lowercase substrings or short stems. Use a few variants ("weaken", "weakens", "loses mass").
- Stay conservative on bounds: prefer the range the user explicitly stated.
- Use units exactly as the user wrote them (case-insensitive matching is handled downstream).
- If the correction is vague ("this is wrong", "you missed something"), reply with kind:"none".

Examples:

CORRECTION: "Cortical bone Young's modulus is 10–25 GPa, not 80."
→ {"kind":"range","name":"Cortical modulus 10–25 GPa","params":{"unit":"GPa","lo":10,"hi":25,"context_terms":["cortical"],"value_terms":["modulus","young","stiffness"]}}

CORRECTION: "Mechanical loading strengthens bone, it doesn't weaken it — except in disuse."
→ {"kind":"forbid_pattern","name":"Loading does not weaken bone","params":{"context_terms":["bone","loading","weight-bearing"],"forbidden_terms":["weaken","weakens","loses mass","reduces bone density"],"exception_terms":["disuse","unloading","microgravity","bed rest"],"explanation":"Mechanical loading strengthens bone per Wolff's law unless the context is disuse."}}

CORRECTION: "you didn't mention osteocyte mechanosensing"
→ {"kind":"none","reason":"correction is a content gap, not a rule constraint"}
"""


def extract(question: str, answer: str, feedback_text: str) -> dict:
    """Call the small extractor model and return a proposal dict.

    Returns one of:
      {"kind": "range",          "name": ..., "params": {...}}
      {"kind": "forbid_pattern", "name": ..., "params": {...}}
      {"kind": "none",           "reason": ...}

    Never raises — on any failure returns {"kind":"none","reason":"extractor_error"}.
    """
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"CORRECTION:\n{feedback_text}\n"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": EXTRACTOR_MODEL,
                "messages": [
                    {"role": "system", "content": EXTRACTOR_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_predict": 400},
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
    except Exception as e:
        return {"kind": "none", "reason": f"extractor_error: {e}"}

    parsed = _safe_parse(raw)
    if not parsed:
        return {"kind": "none", "reason": "extractor returned unparseable JSON"}

    return _validate(parsed)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_parse(raw: str) -> dict | None:
    # Strip code fences if the model snuck them in
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except Exception:
        # last-ditch: pull the first {...} blob
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _validate(p: dict) -> dict:
    kind = p.get("kind")
    if kind == "range":
        params = p.get("params") or {}
        try:
            params["lo"] = float(params["lo"])
            params["hi"] = float(params["hi"])
        except Exception:
            return {"kind": "none", "reason": "range bounds not numeric"}
        if params["lo"] > params["hi"]:
            params["lo"], params["hi"] = params["hi"], params["lo"]
        params["unit"] = str(params.get("unit", "")).strip()
        params["context_terms"] = _str_list(params.get("context_terms"))
        params["value_terms"]   = _str_list(params.get("value_terms"))
        if not params["unit"] or not params["context_terms"]:
            return {"kind": "none", "reason": "range rule missing unit or context"}
        return {"kind": "range", "name": str(p.get("name") or "User range rule"), "params": params}

    if kind == "forbid_pattern":
        params = p.get("params") or {}
        params["context_terms"]   = _str_list(params.get("context_terms"))
        params["forbidden_terms"] = _str_list(params.get("forbidden_terms"))
        params["exception_terms"] = _str_list(params.get("exception_terms"))
        params["explanation"]     = str(params.get("explanation") or "").strip()
        if not params["context_terms"] or not params["forbidden_terms"]:
            return {"kind": "none", "reason": "forbid rule missing context or forbidden terms"}
        return {"kind": "forbid_pattern", "name": str(p.get("name") or "User forbid rule"), "params": params}

    return {"kind": "none", "reason": p.get("reason") or "not a rule"}


def _str_list(x) -> list[str]:
    if not x:
        return []
    if isinstance(x, str):
        x = [x]
    return [str(t).strip().lower() for t in x if str(t).strip()]
