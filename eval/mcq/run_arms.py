"""
eval/mcq/run_arms.py
====================
Run MCQ items through one experimental arm and grade deterministically.

Arms
----
  bare   huatuogpt-bone with the MCQ instruction only. No retrieval, no graph,
         no critic. This is the parametric-knowledge baseline.
  (rag / graph / full to follow — the arm only changes what evidence is put in
   front of the model; the instruction and grading stay identical, so any
   difference between arms is attributable to the evidence.)

Design notes
------------
* Options are shuffled per item with a fixed seed, so the correct answer is not
  always in the same position, and the run is reproducible. The shuffled order
  and the resulting correct letter are recorded per item.
* temperature=0 for reproducibility. Repeat runs with --seed to measure spread.
* Grading is deterministic letter extraction. Every raw generation is stored so
  refusals and hedges can be diagnosed rather than silently scored wrong.

Run
---
    .venv/bin/python -m eval.mcq.run_arms --arm bare
    .venv/bin/python -m eval.mcq.run_arms --arm bare --limit 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "huatuogpt-bone"
LETTERS = ["A", "B", "C", "D"]

DEFAULT_PROMPT = Path(__file__).with_name("prompt_v1.txt")


def load_prompt(path: Path) -> str:
    """The system prompt is held in a file, identical for every arm.

    Arms differ ONLY in what evidence is appended to the USER message. Never
    add an evidence instruction here for the grounded arms — that would make
    the prompt a second variable and destroy attribution of any difference.
    """
    return path.read_text().strip()


def strip_thinking(text: str) -> str:
    m = re.search(r"##\s*final response\s*", text, flags=re.IGNORECASE)
    return text[m.end():].lstrip() if m else text


def extract_letter(raw: str) -> tuple[str | None, str]:
    """Return (letter, how_it_was_found)."""
    body = strip_thinking(raw)
    for pat, how in [
        (r"\\boxed\{\s*([A-D])\s*\}", "boxed"),
        (r"\b(?:answer|option)\s*(?:is|:)\s*\(?([A-D])\)?\b", "stated"),
        (r"^\s*\(?([A-D])\)?\s*[.):]", "leading"),
    ]:
        m = re.search(pat, body, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper(), how
    tail = re.findall(r"\b([A-D])\b", body)
    if tail:
        return tail[-1].upper(), "last_letter"
    return None, "unparsed"


def build_prompt(item: dict, order: list[str]) -> str:
    lines = [item["question"], ""]
    for L, opt in zip(LETTERS, order):
        lines.append(f"{L}. {opt}")
    lines += ["", "Give your final answer as \\boxed{LETTER}."]
    return "\n".join(lines)


def call_model(system_prompt: str, prompt: str, timeout: int, num_predict: int) -> tuple[str, float]:
    t0 = time.time()
    r = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": 8192},
    }, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"], time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="bare", choices=["bare"])
    ap.add_argument("--items", default="eval/mcq/items_draft_v1.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--num-predict", type=int, default=1200)
    ap.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    args = ap.parse_args()

    prompt_path = Path(args.prompt)
    system_prompt = load_prompt(prompt_path)
    prompt_sha = hashlib.sha256(system_prompt.encode()).hexdigest()[:12]

    out_path = Path(args.out or f"eval/mcq/run_{args.arm}_seed{args.seed}.json")
    items = json.loads(Path(args.items).read_text())
    if args.limit:
        items = items[:args.limit]

    print(f"arm={args.arm}  model={OLLAMA_MODEL}  items={len(items)}  seed={args.seed}")
    print(f"prompt={prompt_path.name}  sha256[:12]={prompt_sha}\n")
    results, correct = [], 0

    for n, it in enumerate(items, 1):
        rng = random.Random(f"{args.seed}-{it['id']}")
        order = it["options"][:]
        rng.shuffle(order)
        gold = LETTERS[order.index(it["answer"])]

        prompt = build_prompt(it, order)
        try:
            raw, secs = call_model(system_prompt, prompt, args.timeout, args.num_predict)
            err = None
        except Exception as e:
            raw, secs, err = "", 0.0, f"{type(e).__name__}: {e}"

        letter, how = extract_letter(raw) if raw else (None, "error")
        ok = (letter == gold)
        correct += ok

        results.append({
            "id": it["id"], "domain": it["domain"], "type": it["type"],
            "gold_letter": gold, "predicted": letter, "parse": how,
            "correct": ok, "seconds": round(secs, 1), "error": err,
            "shuffled_options": order, "raw": raw,
        })
        flag = "ok " if ok else "XX "
        print(f"  [{n:>2}/{len(items)}] {flag}{it['id']}  {it['domain']:<15} "
              f"gold={gold} pred={letter or '-'} ({how}) {secs:>5.1f}s")

    n = len(results)
    acc = correct / n if n else 0.0
    print("\n" + "=" * 60)
    print(f"  arm            : {args.arm}")
    print(f"  accuracy       : {acc:.3f}  ({correct}/{n})")
    print(f"  chance         : 0.250")
    print(f"  mean latency   : {sum(r['seconds'] for r in results)/n:.1f}s")
    print(f"  parse failures : {sum(1 for r in results if r['parse'] in ('unparsed','error'))}")
    print("=" * 60)

    print("\n  per domain:")
    bydom = defaultdict(list)
    for r in results:
        bydom[r["domain"]].append(r["correct"])
    for d in sorted(bydom):
        v = bydom[d]
        print(f"    {d:<16} {sum(v)}/{len(v)}  ({sum(v)/len(v):.2f})")

    print("\n  parse method :", dict(Counter(r["parse"] for r in results)))
    print("  predicted    :", dict(Counter(r["predicted"] for r in results)))
    print("  gold spread  :", dict(Counter(r["gold_letter"] for r in results)))

    out_path.write_text(json.dumps(
        {"arm": args.arm, "model": OLLAMA_MODEL, "seed": args.seed,
         "prompt_file": str(prompt_path), "prompt_sha256_12": prompt_sha,
         "system_prompt": system_prompt,
         "n": n, "correct": correct, "accuracy": round(acc, 4),
         "results": results}, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
