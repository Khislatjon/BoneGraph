"""
eval/reasoning_consistency.py
=============================

A3 — opposite-question consistency bench for the Reasoning tab.

Gianluca's robustness test (28 May): "Ask for each one of them the opposite
question, because if one is true, the other one must be false."

For each seed pair (claim_question, opposite_question):
  1. Run BOTH through the live pipeline (/api/reason/chat) — agent + grounding
     + critic loop, exactly as a user would experience it.
  2. Extract the system's STANCE on each answer: affirm | deny | uncertain
     (via a small LLM judge in JSON mode).
  3. Score:
       CONSISTENCY — the two stances must be opposite (one affirm, one deny).
                     Affirming both, or denying both, is a logical failure.
       CORRECTNESS — the affirmed side must match the seed's ground truth.

Reports a per-pair table plus overall consistency % and correctness %.
Consistency is the headline metric for the showcase / paper.

Requirements:
  - The BoneGraph server must be running (python serve.py) at --base.
  - Ollama must be up (the pipeline + the stance judge both use it).

Run:
    .venv/bin/python -m eval.reasoning_consistency
    .venv/bin/python -m eval.reasoning_consistency --base http://localhost:8000
    .venv/bin/python -m eval.reasoning_consistency --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys

import requests

JUDGE_URL = "http://localhost:11434/api/chat"
# The 8B bone model is a markedly better stance judge than llama3.2:3b on these
# adversarially-similar comparative claims (83% vs 75% vs hand-labelled truth in
# the May 2026 bake-off). Stance detection here is genuinely hard, so the
# harness also prints the judge's reasoning for human spot-checking.
JUDGE_MODEL = "huatuogpt-bone:latest"


# ── Seed pairs ────────────────────────────────────────────────────────────────
# Each pair states a fracture-domain claim and its logical opposite. `supported`
# names which question the system *should* affirm, given the bulk of literature.
# Pairs are deliberately clean (one side clearly true) — not leading or
# ambiguous. Edit / extend this list to grow the bench.

SEED_PAIRS = [
    {
        "id": "stiffness",
        "claim_question":    "Is cortical bone stiffer (higher elastic modulus) than trabecular bone?",
        "opposite_question": "Is trabecular bone stiffer (higher elastic modulus) than cortical bone?",
        "supported": "claim",
    },
    {
        "id": "osteoporosis_failure_order",
        "claim_question":    "In osteoporosis, does trabecular bone typically deteriorate before cortical bone?",
        "opposite_question": "In osteoporosis, does cortical bone typically deteriorate before trabecular bone?",
        "supported": "claim",
    },
    {
        "id": "cortical_porosity_strength",
        "claim_question":    "Does increased cortical porosity reduce whole-bone strength?",
        "opposite_question": "Does increased cortical porosity increase whole-bone strength?",
        "supported": "claim",
    },
    {
        "id": "wolff_loading",
        "claim_question":    "Does sustained increased mechanical loading increase bone strength?",
        "opposite_question": "Does sustained increased mechanical loading decrease bone strength?",
        "supported": "claim",
    },
    {
        "id": "lytic_lesion",
        "claim_question":    "Does a lytic metastatic lesion reduce vertebral failure load?",
        "opposite_question": "Does a lytic metastatic lesion increase vertebral failure load?",
        "supported": "claim",
    },
    {
        "id": "density_strength",
        "claim_question":    "Does higher apparent bone density increase bone strength?",
        "opposite_question": "Does lower apparent bone density increase bone strength?",
        "supported": "claim",
    },
]


# ── Pipeline call ─────────────────────────────────────────────────────────────

def ask_pipeline(base: str, question: str) -> str:
    """POST to /api/reason/chat, parse the SSE stream, return the final answer."""
    resp = requests.post(
        f"{base}/api/reason/chat",
        data={"question": question, "history": "[]", "all_questions": "[]"},
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()
    final = ""
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line.startswith("data: "):
            continue
        ev = json.loads(line[6:])
        if ev.get("type") == "done":
            final = ev.get("answer", "")
        elif ev.get("type") == "error":
            return f"(pipeline error: {ev.get('message')})"
    # strip any thinking block the same way the UI does
    import re
    m = re.search(r"final response\s*", final, flags=re.IGNORECASE)
    return final[m.end():].lstrip() if m else final


# ── Stance judge ──────────────────────────────────────────────────────────────

# "Answer-as-author" framing: rather than asking the judge to compare an answer
# to an extracted claim (which a small model gets wrong when claim and answer
# share entities but differ in direction), we ask it to role-play answering the
# ORIGINAL yes/no question using only the answer text. This scored best in the
# bake-off. yes→affirm, no→deny, unclear→uncertain.
JUDGE_PROMPT = """You are given a QUESTION and an ANSWER written by an expert. Based ONLY on the answer, how would that expert answer the original yes/no question?

Reply with ONE JSON object, no prose, no fences:
{"reason":"<one short clause>","stance":"yes"|"no"|"unclear"}

  "yes"     = the answer's conclusion is YES to the question
  "no"      = the answer's conclusion is NO to the question
  "unclear" = the answer is non-committal, says it depends, or presents both sides

Judge the answer's CONCLUSION, not its surface wording. The answer may discuss the
same entities as the question but reach the opposite conclusion."""

_STANCE_MAP = {"yes": "affirm", "no": "deny", "unclear": "uncertain"}


def judge_stance(question: str, answer: str) -> tuple[str, str]:
    """Return (stance, reason). stance ∈ {affirm, deny, uncertain}.
    reason is the judge's one-clause justification, for human spot-checking."""
    user = f"QUESTION: {question}\n\nANSWER:\n{answer}"
    try:
        resp = requests.post(
            JUDGE_URL,
            json={
                "model": JUDGE_MODEL,
                "messages": [
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_predict": 150},
            },
            timeout=120,
        )
        resp.raise_for_status()
        parsed = json.loads(resp.json()["message"]["content"])
        raw_stance = str(parsed.get("stance", "unclear")).strip().lower()
        reason = str(parsed.get("reason", "")).strip()
        return _STANCE_MAP.get(raw_stance, "uncertain"), reason
    except Exception as e:
        return "uncertain", f"(judge error: {e})"


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_pair(claim_stance: str, opp_stance: str, supported: str) -> dict:
    # Consistent = opposite stances on opposite questions.
    consistent = {claim_stance, opp_stance} == {"affirm", "deny"}

    # Correct = the affirmed side matches ground truth (only meaningful if consistent).
    correct = None
    if consistent:
        affirmed_side = "claim" if claim_stance == "affirm" else "opposite"
        correct = (affirmed_side == supported)

    return {"consistent": consistent, "correct": correct}


# ── Runner ────────────────────────────────────────────────────────────────────

def run(base: str, pairs: list[dict]) -> list[dict]:
    results = []
    for i, pair in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {pair['id']} …", flush=True)
        print("    · claim    …", end="", flush=True)
        a_claim = ask_pipeline(base, pair["claim_question"])
        s_claim, r_claim = judge_stance(pair["claim_question"], a_claim)
        print(f" {s_claim}")
        print("    · opposite …", end="", flush=True)
        a_opp = ask_pipeline(base, pair["opposite_question"])
        s_opp, r_opp = judge_stance(pair["opposite_question"], a_opp)
        print(f" {s_opp}")
        score = score_pair(s_claim, s_opp, pair["supported"])
        results.append({
            "id": pair["id"],
            "claim_question": pair["claim_question"],
            "opposite_question": pair["opposite_question"],
            "supported": pair["supported"],
            "claim_stance": s_claim,
            "opposite_stance": s_opp,
            "claim_reason": r_claim,
            "opposite_reason": r_opp,
            "claim_answer": a_claim,
            "opposite_answer": a_opp,
            **score,
        })
    return results


def report(results: list[dict]) -> None:
    n = len(results)
    n_consistent = sum(1 for r in results if r["consistent"])
    judged = [r for r in results if r["correct"] is not None]
    n_correct = sum(1 for r in judged if r["correct"])

    print("\n" + "=" * 74)
    print("CONSISTENCY BENCH RESULTS")
    print("=" * 74)
    print(f"{'pair':<28} {'claim':<9} {'opposite':<9} {'consistent':<11} {'correct'}")
    print("-" * 74)
    for r in results:
        cons = "✓" if r["consistent"] else "✗"
        corr = "—" if r["correct"] is None else ("✓" if r["correct"] else "✗")
        print(f"{r['id']:<28} {r['claim_stance']:<9} {r['opposite_stance']:<9} {cons:<11} {corr}")
    print("-" * 74)
    print(f"Consistency: {n_consistent}/{n} = {100*n_consistent/n:.0f}%   "
          f"(opposite stances on opposite questions)")
    if judged:
        print(f"Correctness: {n_correct}/{len(judged)} = {100*n_correct/len(judged):.0f}%   "
              f"(affirmed side matches ground truth, among consistent pairs)")
    print("=" * 74)
    print("Judge reasoning (spot-check these — the auto-judge is ~83% reliable on")
    print("these adversarial comparative claims; correct any mislabels by hand):")
    for r in results:
        print(f"  · {r['id']}")
        print(f"      claim    [{r['claim_stance']}]: {r.get('claim_reason','')}")
        print(f"      opposite [{r['opposite_stance']}]: {r.get('opposite_reason','')}")
    print("=" * 74)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000",
                    help="BoneGraph server base URL (default: http://localhost:8000)")
    ap.add_argument("--json", help="Write full results (incl. answers) to this JSON file.")
    args = ap.parse_args()

    try:
        requests.get(f"{args.base}/api/stats", timeout=5)
    except Exception:
        print(f"ERROR: cannot reach {args.base}. Start the server: python serve.py")
        return 1

    results = run(args.base, SEED_PAIRS)
    report(results)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
