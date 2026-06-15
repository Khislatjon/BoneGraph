"""
reasoning/rule_extractor.py
============================

Turn a user correction into a structured rule that physical_grounding can
execute, using a small LLM in JSON mode. If the correction does not look
rule-shaped (e.g. "you missed the role of osteocytes"), the extractor returns
None and the caller can fall back to free-form soft memory.

Three rule shapes are supported:

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

  comparative    — an ordinal / directional claim "A <comparator> B" (e.g.
                   "trabecular fails before cortical", "cortical is stiffer than
                   trabecular"). The rule stores the asserted order (first_terms =
                   A, second_terms = B) plus the comparator words that establish
                   that axis. It is violated when a sentence on that axis asserts
                   the *reverse* order (B before A). This covers logical, non-
                   numeric corrections that are neither ranges nor forbidden words.
                   { "kind": "comparative",
                     "name": "...",
                     "params": { "first_terms":      ["trabecular","cancellous"],
                                 "second_terms":     ["cortical"],
                                 "comparator_terms": ["fails before","fails first","weaker","lower strength"],
                                 "explanation":      "Trabecular fails before cortical (higher surface area)." } }

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

CRITICAL — THE CORRECTION IS THE SOURCE OF TRUTH. The question and the answer are the thing being CORRECTED, so they often assert the OPPOSITE of what is true. The question may even be leading (e.g. "Why does A beat B?" when in fact B beats A). Build the rule ONLY from the correction text. Never let the question's or answer's wording decide a direction, an order, or a numeric bound. If the correction says "B fails before A", the rule must say B-before-A even though the question said the reverse.

You must reply with ONE JSON object on a single line. No prose, no markdown fences. One of these shapes:

(A) Numeric range correction — when the user is asserting that a quantity must lie in a range:
{"kind":"range","name":"<short label>","params":{"unit":"<GPa|MPa|g/cm^3|%|...>","lo":<number>,"hi":<number>,"context_terms":["<lowercase term>", "..."],"value_terms":["<lowercase term>","..."]}}

(B) Forbidden directional / definitional claim — when the user is asserting a sign or definition error:
{"kind":"forbid_pattern","name":"<short label>","params":{"context_terms":["..."],"forbidden_terms":["..."],"exception_terms":["..."],"explanation":"<one sentence>"}}

(C) Comparative / ordinal claim — when the user is asserting that one thing comes before / exceeds / outranks another along some axis (failure order, stiffness, strength, density, risk, …) WITHOUT giving a number:
{"kind":"comparative","name":"<short label>","params":{"first_terms":["<the A side>","..."],"second_terms":["<the B side>","..."],"comparator_terms":["<axis word>","..."],"explanation":"<one sentence restating the claim>"}}

(D) No extractable rule:
{"kind":"none","reason":"<short reason>"}

Guidelines:
- "context_terms" are lowercase substrings that must appear in a sentence for the rule to look at it (e.g. "cortical", "trabecular", "osteoporosis", "lytic"). Pick 1–3 specific terms.
- "value_terms" narrow numeric range rules to the right kind of claim ("modulus", "density", "stiffness", "porosity"). Pick 1–3.
- "forbidden_terms" / "exception_terms" are lowercase substrings or short stems. Use a few variants ("weaken", "weakens", "loses mass").
- For a COMPARATIVE claim, read the CORRECTION as "A <comparator> B" and put A (the thing the CORRECTION says comes first / wins) in "first_terms" and B in "second_terms" (include synonyms, e.g. trabecular/cancellous). IGNORE the order the question/answer used — only the correction decides which side is A. "comparator_terms" are the words that name the axis being compared — failure order ("before","first","earlier","fails first"), magnitude ("stiffer","stronger","weaker","higher","lower"), etc. Pick the axis the user is actually talking about; do NOT mix axes (stiffness words for a failure-order claim cause false alarms).
- The "explanation" must paraphrase the CORRECTION, never the answer. If the answer claimed the opposite, the explanation must state the correction's position.
- Prefer "comparative" over "none" whenever the correction is a clear A-beats-B / A-before-B statement with no number. Prefer "range" when a number+unit is involved.
- Stay conservative on bounds: prefer the range the user explicitly stated.
- Use units exactly as the user wrote them (case-insensitive matching is handled downstream).
- If the correction is vague ("this is wrong", "you missed something"), reply with kind:"none".

Examples:

CORRECTION: "Cortical bone Young's modulus is 10–25 GPa, not 80."
→ {"kind":"range","name":"Cortical modulus 10–25 GPa","params":{"unit":"GPa","lo":10,"hi":25,"context_terms":["cortical"],"value_terms":["modulus","young","stiffness"]}}

CORRECTION: "Mechanical loading strengthens bone, it doesn't weaken it — except in disuse."
→ {"kind":"forbid_pattern","name":"Loading does not weaken bone","params":{"context_terms":["bone","loading","weight-bearing"],"forbidden_terms":["weaken","weakens","loses mass","reduces bone density"],"exception_terms":["disuse","unloading","microgravity","bed rest"],"explanation":"Mechanical loading strengthens bone per Wolff's law unless the context is disuse."}}

CORRECTION: "Trabecular bone typically fails before cortical due to its higher surface area."
→ {"kind":"comparative","name":"Trabecular fails before cortical","params":{"first_terms":["trabecular","cancellous"],"second_terms":["cortical","compact"],"comparator_terms":["fails before","fails first","fail first","yields first","weaker","lower strength"],"explanation":"Trabecular bone typically fails before cortical because of its higher surface area."}}

LEADING QUESTION — the question and answer assert the WRONG direction; the correction reverses it. Follow the correction, not the question.
QUESTION: "Why does cortical bone fail before trabecular bone in osteoporosis?"
ANSWER: "Cortical bone fails before trabecular bone because its high density makes it more sensitive to stress."
CORRECTION: "No — trabecular bone fails before cortical, because of its higher surface area."
→ {"kind":"comparative","name":"Trabecular fails before cortical","params":{"first_terms":["trabecular","cancellous"],"second_terms":["cortical","compact"],"comparator_terms":["fails before","fails first","fail first","yields first","weaker","lower strength"],"explanation":"Trabecular bone fails before cortical because of its higher surface area, contrary to the question's premise."}}

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
    # The extractor is a small model, but on a shared GPU (the Jetson demo box)
    # the larger reasoning/vision models are already resident, so the first call
    # has to cold-load llama3.2:3b — which can exceed a tight timeout and surface
    # as a spurious "couldn't extract a rule". We therefore (a) give it a generous
    # timeout, (b) keep it warm between corrections via keep_alive, and (c) retry
    # once on a transport error (the retry usually hits a now-warm model).
    payload = {
        "model": EXTRACTOR_MODEL,
        "messages": [
            {"role": "system", "content": EXTRACTOR_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "stream": False,
        "format": "json",
        "keep_alive": "10m",
        "options": {"temperature": 0, "num_predict": 400},
    }
    raw = None
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=90)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip()
            break
        except Exception as e:
            last_err = e
    if raw is None:
        return {"kind": "none", "reason": f"extractor_error: {last_err}"}

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

    if kind == "comparative":
        params = p.get("params") or {}
        params["first_terms"]      = _str_list(params.get("first_terms"))
        params["second_terms"]     = _str_list(params.get("second_terms"))
        params["comparator_terms"] = _str_list(params.get("comparator_terms"))
        params["explanation"]      = str(params.get("explanation") or "").strip()
        if not params["first_terms"] or not params["second_terms"] or not params["comparator_terms"]:
            return {"kind": "none", "reason": "comparative rule missing first/second/comparator terms"}
        return {"kind": "comparative", "name": str(p.get("name") or "User comparative rule"), "params": params}

    return {"kind": "none", "reason": p.get("reason") or "not a rule"}


def _str_list(x) -> list[str]:
    if not x:
        return []
    if isinstance(x, str):
        x = [x]
    return [str(t).strip().lower() for t in x if str(t).strip()]
