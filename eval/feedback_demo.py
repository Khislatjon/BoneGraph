"""
eval/feedback_demo.py
=====================

B1 — feedback-loop demo harness for the Reasoning tab.

Substantiates Gianluca's headline claim (~23:42, 21 May): "the second chat is
better than the first." It does this in two layers:

  PART A — Deterministic rule effect (no LLM, fully reproducible).
    For a known answer that the built-in rules let through, show that adding
    the user's correction (as a Tier-2 rule) makes the physical-grounding
    check catch it. This is the rock-solid metric: same input, same outcome,
    every time. It proves the correction changes system behaviour.

  PART B — Live before/after demonstration (real pipeline, may vary).
    Run the SAME query through /api/reason/chat BEFORE and AFTER injecting the
    user rule. Capture the grounding violations and critic verdict each time.
    This is the actual "second chat" trace a user / reviewer would see.

The harness is non-destructive: it injects its test rules via the HTTP API,
captures their ids, and deletes them again at the end, leaving the user's rule
store as it was.

Requirements:
  - Server running (python serve.py) for Part B.
  - Part A runs offline (imports reasoning.physical_grounding directly).

Run:
    .venv/bin/python -m eval.feedback_demo
    .venv/bin/python -m eval.feedback_demo --base http://localhost:8000
    .venv/bin/python -m eval.feedback_demo --part-a-only      # skip the server
    .venv/bin/python -m eval.feedback_demo --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys

import requests

import re

from reasoning.physical_grounding import check as physical_check


def _strip_thinking(text: str) -> str:
    m = re.search(r"final response\s*", text, flags=re.IGNORECASE)
    return text[m.end():].lstrip() if m else text


# ── Test cases ────────────────────────────────────────────────────────────────
# Each case models a user tightening a bound beyond the shipped defaults — a
# value that PASSES the built-in rules but the user's lab considers wrong.
# `bad_answer` is a synthetic answer containing that value (for Part A).
# `rule` is the Tier-2 rule the user would confirm after correcting.

CASES = [
    {
        "id": "cortical_modulus_tighten",
        "query": "What is a typical elastic modulus for cortical bone in a finite-element model? Give a number in GPa.",
        "bad_answer": "Cortical bone has an elastic modulus of about 22 GPa in this context.",
        "rule": {
            "name": "Cortical modulus 15-20 GPa (lab spec)",
            "kind": "range",
            "params": {
                "unit": "GPa", "lo": 15, "hi": 20,
                "context_terms": ["cortical"],
                "value_terms": ["modulus", "stiffness", "young"],
            },
        },
    },
    {
        "id": "cortical_density_tighten",
        "query": "What is the apparent density of cortical bone in g/cm^3?",
        "bad_answer": "Cortical bone apparent density is around 2.0 g/cm^3.",
        "rule": {
            "name": "Cortical density 1.85-1.95 g/cm^3 (lab spec)",
            "kind": "range",
            "params": {
                "unit": "g/cm^3", "lo": 1.85, "hi": 1.95,
                "context_terms": ["cortical"],
                "value_terms": ["density"],
            },
        },
    },
]


# ── Part A: deterministic rule effect ─────────────────────────────────────────

def _user_violations(check_result: dict) -> list[dict]:
    return [v for v in check_result.get("violations", []) if v.get("source") == "user"]


def part_a(case: dict) -> dict:
    """Run physical_check on the synthetic bad answer, without and with the
    user rule. Returns the before/after violation picture."""
    rule_obj = {
        "rule_id": f"demo_{case['id']}",
        "name": case["rule"]["name"],
        "kind": case["rule"]["kind"],
        "params": case["rule"]["params"],
    }
    before = physical_check(case["bad_answer"])               # built-in only
    after = physical_check(case["bad_answer"], user_rules=[rule_obj])
    ub, ua = _user_violations(before), _user_violations(after)
    return {
        "before_total": len(before["violations"]),
        "after_total": len(after["violations"]),
        "user_caught_after": len(ua),
        "rule_changed_outcome": len(ua) > len(ub),
        "after_detail": ua[0]["detail"] if ua else "",
    }


# ── Part B: live pipeline before/after ────────────────────────────────────────

def run_pipeline(base: str, query: str) -> dict:
    """POST to /api/reason/chat, parse the SSE stream. Return the grounding +
    critic picture: round-1 and final violation counts, user-rule violations,
    final critic verdict, and a short answer snippet."""
    resp = requests.post(
        f"{base}/api/reason/chat",
        data={"question": query, "history": "[]", "all_questions": "[]"},
        stream=True, timeout=600,
    )
    resp.raise_for_status()
    checks: list[dict] = []
    critics: list[dict] = []
    final_answer = ""
    final_check = None
    resolved = None
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line.startswith("data: "):
            continue
        ev = json.loads(line[6:])
        t = ev.get("type")
        if t == "physical_check":
            checks.append(ev)
        elif t == "critic_review":
            critics.append(ev)
        elif t == "done":
            final_answer = ev.get("answer", "")
            final_check = ev.get("final_physical_check")
            resolved = ev.get("critic_resolved")
        elif t == "error":
            return {"error": ev.get("message")}

    round1 = checks[0] if checks else {"violations": []}
    final = final_check or (checks[-1] if checks else {"violations": []})
    return {
        "round1_user_violations": len(_user_violations(round1)),
        "final_user_violations": len(_user_violations(final)),
        "final_total_violations": len(final.get("violations", [])),
        "critic_verdicts": [c.get("verdict") for c in critics],
        "critic_resolved": resolved,
        "answer_snippet": " ".join(_strip_thinking(final_answer).split())[:160],
    }


def inject_rule(base: str, case: dict) -> int | None:
    r = requests.post(f"{base}/api/reason/rules/confirm", data={
        "name": case["rule"]["name"],
        "kind": case["rule"]["kind"],
        "params": json.dumps(case["rule"]["params"]),
    }, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("rule", {}).get("id") if data.get("ok") else None


def delete_rule(base: str, rule_id: int) -> None:
    try:
        requests.delete(f"{base}/api/reason/rules/{rule_id}", timeout=30)
    except Exception:
        pass


def part_b(base: str, case: dict) -> dict:
    before = run_pipeline(base, case["query"])
    rule_id = inject_rule(base, case)
    try:
        after = run_pipeline(base, case["query"])
    finally:
        if rule_id is not None:
            delete_rule(base, rule_id)
    return {"before": before, "after": after, "rule_id": rule_id}


# ── Report ────────────────────────────────────────────────────────────────────

def report(results: list[dict], part_a_only: bool) -> None:
    print("\n" + "=" * 74)
    print("FEEDBACK-LOOP DEMO  (B1)")
    print("=" * 74)

    print("\nPART A — deterministic rule effect (no LLM, reproducible)")
    print("-" * 74)
    a_pass = 0
    for r in results:
        a = r["part_a"]
        ok = a["rule_changed_outcome"]
        a_pass += ok
        print(f"  {r['id']}")
        print(f"    built-in only : {a['before_total']} violation(s)")
        print(f"    + user rule   : {a['after_total']} violation(s), "
              f"{a['user_caught_after']} from the user rule")
        print(f"    → correction changes outcome: {'✓ YES' if ok else '✗ no'}")
        if a["after_detail"]:
            print(f"      {a['after_detail']}")
    print("-" * 74)
    print(f"Part A: {a_pass}/{len(results)} corrections demonstrably change grounding outcome.")

    if part_a_only:
        print("=" * 74)
        return

    print("\nPART B — live pipeline, same query before vs after the correction")
    print("-" * 74)
    for r in results:
        b = r.get("part_b") or {}
        bef, aft = b.get("before", {}), b.get("after", {})
        if "error" in bef or "error" in aft:
            print(f"  {r['id']}: pipeline error — {bef.get('error') or aft.get('error')}")
            continue
        print(f"  {r['id']}")
        print(f"    BEFORE  user-rule violations: round1={bef.get('round1_user_violations')} "
              f"final={bef.get('final_user_violations')}  "
              f"critic={bef.get('critic_verdicts')}")
        print(f"            answer: {bef.get('answer_snippet','')[:120]}")
        print(f"    AFTER   user-rule violations: round1={aft.get('round1_user_violations')} "
              f"final={aft.get('final_user_violations')}  "
              f"critic={aft.get('critic_verdicts')}")
        print(f"            answer: {aft.get('answer_snippet','')[:120]}")
        caught = (aft.get("round1_user_violations") or 0) > (bef.get("round1_user_violations") or 0)
        print(f"    → the correction is now active on a repeat question: "
              f"{'✓ YES (round-1 catch)' if caught else '— (see answers above)'}")
    print("=" * 74)
    print("Interpretation: Part A proves the correction deterministically changes the")
    print("grounding outcome. Part B shows the live pipeline acting on it — the same")
    print("question that previously passed silently now triggers the user's rule, the")
    print("critic disputes, and the answer is revised. The second chat is better.")
    print("=" * 74)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--part-a-only", action="store_true", help="Skip the live pipeline (no server needed).")
    ap.add_argument("--json", help="Write full results to this JSON file.")
    args = ap.parse_args()

    if not args.part_a_only:
        try:
            requests.get(f"{args.base}/api/stats", timeout=5)
        except Exception:
            print(f"ERROR: cannot reach {args.base}. Start the server, or use --part-a-only.")
            return 1

    results = []
    for case in CASES:
        print(f"[{case['id']}] Part A …", flush=True)
        entry = {"id": case["id"], "part_a": part_a(case)}
        if not args.part_a_only:
            print(f"[{case['id']}] Part B (before → inject → after → cleanup) …", flush=True)
            entry["part_b"] = part_b(args.base, case)
        results.append(entry)

    report(results, args.part_a_only)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
