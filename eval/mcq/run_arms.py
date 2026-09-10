"""
eval/mcq/run_arms.py
====================
Run MCQ items through one experimental arm and grade deterministically.

Arms
----
  bare    huatuogpt-bone with the MCQ instruction only. No retrieval, no graph.
  rag     + top-5 corpus passages (same LIT_TOP_K / LIT_CHAR_BUDGET as the API)
  graph   + up to 12 one-hop knowledge-graph edges (same as the critic gets)
  full    + both

The arm changes ONLY what evidence is appended to the USER message. The system
prompt and the grading path are byte-identical across arms, so any difference
between arms is attributable to the evidence and nothing else.

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


LIT_TOP_K = 5
LIT_CHAR_BUDGET = 4200

_retriever = None


def _get_retriever():
    """Loaded lazily — the bare arm must never pay the 588 MB / SPECTER2 cost."""
    global _retriever
    if _retriever is None:
        from retrieval.retriever import BoneGraphRetriever
        _retriever = BoneGraphRetriever()
        _retriever.load()
    return _retriever


def fetch_literature(question: str) -> list[dict]:
    """Mirrors api/main.py::_fetch_literature — same top-k, dedup and budget.

    The retriever load is deliberately OUTSIDE the try. A load failure must
    crash the run, not silently yield zero passages — that produced a complete
    but meaningless "rag" result once already.
    """
    r = _get_retriever()
    try:
        results = r.query(question, top_k=LIT_TOP_K * 2)
    except Exception:
        return []
    seen, out, budget = set(), [], LIT_CHAR_BUDGET
    per = max(400, LIT_CHAR_BUDGET // max(1, LIT_TOP_K))
    for r in results:
        title = (r.get("title") or "").strip()
        key = title.lower()
        if key and key in seen:
            continue
        seen.add(key)
        snippet = (r.get("text") or "").strip().replace("\n", " ")[:per]
        if budget - len(snippet) < 0:
            break
        budget -= len(snippet)
        out.append({"rank": len(out) + 1, "title": title or "(untitled)",
                    "year": r.get("year"), "snippet": snippet,
                    "score": round(r.get("score") or 0.0, 3)})
        if len(out) >= LIT_TOP_K:
            break
    return out


def build_evidence(arm: str, question: str) -> tuple[str, dict]:
    """Return (evidence_block, meta). Retrieval uses the QUESTION ONLY — never
    the options, or the correct answer's phrasing would steer what comes back."""
    blocks, meta = [], {}
    if arm in ("rag", "full"):
        passages = fetch_literature(question)
        meta["n_passages"] = len(passages)
        meta["passage_titles"] = [p["title"][:70] for p in passages]
        meta["top_passage_score"] = passages[0]["score"] if passages else None
        if passages:
            lines = [f"[L{p['rank']}] {p['title']}" + (f" ({p['year']})" if p.get("year") else "")
                     + f"\n    {p['snippet']}" for p in passages]
            blocks.append("Evidence from the literature:\n" + "\n".join(lines))
        else:
            blocks.append("Evidence from the literature:\n(no literature retrieved)")
    if arm in ("graph", "full"):
        from reasoning import kg_context
        kg = kg_context.kg_facts(question)
        meta["n_kg_facts"] = len(kg.get("facts") or [])
        meta["kg_anchors"] = kg.get("anchors") or []
        blocks.append("Knowledge-graph facts:\n" + kg_context.format_facts(kg))
    return ("\n\n".join(blocks), meta)


def build_prompt(item: dict, order: list[str], evidence: str = "") -> str:
    lines = [item["question"], ""]
    for L, opt in zip(LETTERS, order):
        lines.append(f"{L}. {opt}")
    if evidence:
        lines += ["", evidence]
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
    ap.add_argument("--arm", default="bare", choices=["bare", "rag", "graph", "full"])
    ap.add_argument("--items", default="eval/mcq/items_reason.json")
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

        evidence, ev_meta = build_evidence(args.arm, it["question"])
        if n == 1 and args.arm in ("rag", "full") and not ev_meta.get("n_passages"):
            sys.exit(f"ABORT: arm '{args.arm}' retrieved 0 passages on the first "
                     f"item. Evidence is not reaching the model — check LD_PRELOAD "
                     f"(see run_api.sh) before trusting any result.")
        prompt = build_prompt(it, order, evidence)
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
            "prompt_chars": len(prompt), "evidence": ev_meta,
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
    print(f"  mean prompt    : {sum(r['prompt_chars'] for r in results)//n} chars "
          f"(~{sum(r['prompt_chars'] for r in results)//n//4} tok, ctx 8192)")
    if args.arm in ("rag", "full"):
        print(f"  mean passages  : {sum(r['evidence'].get('n_passages',0) for r in results)/n:.1f}")
    if args.arm in ("graph", "full"):
        print(f"  mean kg facts  : {sum(r['evidence'].get('n_kg_facts',0) for r in results)/n:.1f}")
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
