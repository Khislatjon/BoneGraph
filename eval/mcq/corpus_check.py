"""
eval/mcq/corpus_check.py
========================
First-pass corpus check for drafted MCQ items.

Each item's question is run through BoneGraphRetriever, and the keyed answer is
scored against the retrieved passages alongside every distractor. Two different
adjudicators are used, because they have very different strength:

  NUMERIC  — for options carrying a value and a recognised unit ("17-20 GPa",
             "8-15%"). Reuses reasoning.physical_grounding._find_values, which
             extracts numbers only when the unit sits adjacent to them AND the
             sentence mentions a question context term. Each option's range then
             競 votes: the option whose range captures the most corpus-reported
             values wins. This is a real signal.

  OVERLAP  — for everything else. Bare content-term overlap between option and
             passages. This is WEAK and known to fail on two common patterns:
               * mirror-image distractors ("A causes B" vs "B causes A") score
                 identically, because the word sets are the same;
               * common domain words inflate whichever option happens to use
                 them.
             Overlap results are reported as "needs_reading", never as support.
             Adjudicating them requires a model or a human reading the passages.

Verdicts
--------
  supported     numeric adjudicator: keyed answer captures the most corpus values
  tie           numeric adjudicator: a distractor is equally supported — not support
  flagged       numeric adjudicator: a DISTRACTOR captures more — look at this
  silent        retrieval off-topic, or no corpus values found for the unit
  needs_reading non-numeric item; overlap cannot decide it

Silent items are KEPT. Dropping them would select the benchmark for
retrievability and hand the RAG arm a win by construction.

Run:
    .venv/bin/python -m eval.mcq.corpus_check
    .venv/bin/python -m eval.mcq.corpus_check --items eval/mcq/items_draft_v1.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from retrieval.retriever import BoneGraphRetriever
from reasoning.physical_grounding import _find_values, _unit_to_pattern

STOP = {
    "about","the","a","an","of","in","on","and","or","is","are","it","its","with","that","which",
    "by","to","from","as","at","for","than","then","only","both","they","their","this","these",
    "greatly","roughly","around","order","magnitude","approximately","principally","most","more",
    "less","higher","lower","increases","decreases","unchanged","effect","between","using","used",
    "per","one","two","hundred","five","ten","essentially","commonly","typically","within","while",
    "what","how","does","which","human","healthy","adult","bone","compared","compare","measured",
}

# Unit spellings as written in options -> notation-tolerant regex for the corpus.
UNIT_ALIASES: list[tuple[str, str]] = [
    ("gpa",          r"gpa"),
    ("mpa",          r"mpa"),
    ("microstrain",  r"(?:microstrain|µ\s*strain|micro-strain)"),
    ("micrometres",  r"(?:µm|um|micrometre?s?|micrometer?s?|microns?)"),
    ("micrometre",   r"(?:µm|um|micrometre?s?|micrometer?s?|microns?)"),
    ("millimetre",   r"(?:mm|millimetre?s?|millimeter?s?)"),
    ("nanometre",    r"(?:nm|nanometre?s?|nanometer?s?)"),
    ("percent",      r"%"),
    ("%",            r"%"),
    ("g/cm3",        None),   # built via _unit_to_pattern
]

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_RANGE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:[-–—]|to)\s*(-?\d+(?:\.\d+)?)")


def terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 4 and w not in STOP}


def context_terms(question: str) -> list[str]:
    ts = sorted(terms(question))
    return ts or ["bone"]


def numeric_claim(option: str) -> tuple[float, float, str] | None:
    """Return (lo, hi, unit_pattern) if the option states a value with a unit."""
    low = option.lower()
    unit_pat = None
    for spelling, pat in UNIT_ALIASES:
        if spelling in low:
            unit_pat = pat if pat is not None else _unit_to_pattern(spelling)
            break
    if unit_pat is None:
        return None
    m = _RANGE.search(option)
    if m:
        lo, hi = sorted((float(m.group(1)), float(m.group(2))))
        return lo * 0.8, hi * 1.25, unit_pat
    nums = [float(n) for n in _NUM.findall(option)]
    if not nums:
        return None
    v = nums[0]
    return v * 0.85, v * 1.15, unit_pat


def overlap_score(option: str, blob: str) -> float:
    ts = terms(option)
    return round(sum(1 for t in ts if t in blob) / len(ts), 3) if ts else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="eval/mcq/items_draft_v1.json")
    ap.add_argument("--out", default="eval/mcq/corpus_check_v1.json")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-score", type=float, default=0.55)
    args = ap.parse_args()

    items = json.loads(Path(args.items).read_text())
    print(f"Loading retriever for {len(items)} items...")
    r = BoneGraphRetriever()
    r.load()
    print()

    out, tally = [], {"supported": 0, "tie": 0, "flagged": 0, "silent": 0, "needs_reading": 0}

    for it in items:
        res = r.query(it["question"], top_k=args.top_k)
        top = res[0]["score"] if res else 0.0
        text = "\n".join(x["text"] for x in res)
        blob = text.lower()

        claims = {o: numeric_claim(o) for o in it["options"]}
        is_numeric = claims[it["answer"]] is not None and sum(c is not None for c in claims.values()) >= 2

        detail: dict = {}
        if is_numeric:
            method = "numeric"
            ctx = context_terms(it["question"])
            unit_pat = claims[it["answer"]][2]
            found = [v for v, _ in _find_values(text, unit_pat, ctx)]
            votes = {}
            for o, c in claims.items():
                votes[o] = sum(1 for v in found if c and c[0] <= v <= c[1]) if c else 0
            a_votes = votes[it["answer"]]
            best_d = max((v for o, v in votes.items() if o != it["answer"]), default=0)
            detail = {"corpus_values_found": len(found),
                      "sample_values": sorted(found)[:12],
                      "option_votes": votes}
            if top < args.min_score or not found or max(votes.values()) == 0:
                verdict = "silent"          # nothing votes -> corpus is mute, not in conflict
            elif a_votes > best_d:
                verdict = "supported"
            elif a_votes == best_d:
                verdict = "tie"             # key and a distractor equally supported
            else:
                verdict = "flagged"
        else:
            method = "overlap"
            scores = {o: overlap_score(o, blob) for o in it["options"]}
            detail = {"option_overlap": scores}
            verdict = "silent" if top < args.min_score else "needs_reading"

        tally[verdict] += 1
        out.append({**it, "verdict": verdict, "method": method,
                    "top_score": round(top, 3), **detail,
                    "passages": [{"rank": x["rank"], "score": round(x["score"], 3),
                                  "title": x["title"], "year": x.get("year"),
                                  "text": x["text"][:600]} for x in res[:3]]})
        extra = f"votes={detail.get('option_votes', {}).get(it['answer'], '-')}/{detail.get('corpus_values_found', '-')}" if method == "numeric" else ""
        print(f"  {it['id']}  {it['domain']:<15} {method:<8} {verdict:<14} top={top:.3f} {extra}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    n = len(items)
    print("\n" + "=" * 62)
    for k, v in tally.items():
        print(f"  {k:<15} {v:>3}/{n}  ({v/n:.0%})")
    print("=" * 62)
    flagged = [o["id"] for o in out if o["verdict"] == "flagged"]
    if flagged:
        print(f"\nNumeric disagreement, check these: {', '.join(flagged)}")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
