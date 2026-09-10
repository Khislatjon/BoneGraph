"""
eval/mcq/run_oracle.py
======================
Oracle arm: the same scorer, the same items, but the evidence is the passage
that actually contains the answer — fetched by chunk id, bypassing retrieval.

Why this arm exists
-------------------
A pre-flight on items_reason_v1.json found that the retriever surfaces the
deciding quantity for 0 of 10 items, while still returning passages at cosine
0.76-0.81. So a `full` arm that scores no better than `bare` on these items is
ambiguous: it could mean grounding does not help, or it could mean grounding was
never supplied. Those are very different claims.

This arm separates them. It pastes the known-correct chunk in place of the
retrieved ones, changing nothing else — same prompt_v1.txt, same seeded shuffle,
same extract_letter, same evidence header as `rag`/`full`.

    oracle >> full   →  the grounding mechanism works; retrieval is the failure.
    oracle ~= full   →  the model cannot use the evidence even when handed it.

Run
---
    .venv/bin/python -m eval.mcq.run_oracle \
        --items eval/mcq/items_reason.json \
        --out   eval/mcq/results/run_oracle_p1_seed17.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.mcq.run_arms import (  # noqa: E402
    DEFAULT_PROMPT, LETTERS,
    build_prompt, call_model, extract_letter, load_prompt,
)

CHUNKS_DB = "data/db/chunks.db"

# Deliberately NOT run_arms' per-passage budget (LIT_CHAR_BUDGET // LIT_TOP_K
# = 840). Corpus chunks run to ~1800 characters and the deciding quantity is
# often in the last third — the first version of this arm truncated the answer
# off 7 of 10 items and scored them anyway. An oracle that does not contain the
# answer is just a slower `bare`, so the window must fit a whole chunk.
ORACLE_PER_CHUNK = 2400


def fetch_chunks(db: str, ids: list[int]) -> list[dict]:
    """The oracle passages, formatted exactly as run_arms formats retrieved ones
    so the only difference between arms is WHICH passages arrive, not how they
    look to the model."""
    if not ids:
        return []
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    q = f"SELECT id, source_id, text FROM chunks WHERE id IN ({','.join('?'*len(ids))})"
    rows = {r["id"]: r for r in con.execute(q, ids)}
    con.close()
    per = ORACLE_PER_CHUNK
    out = []
    for cid in ids:                      # preserve the item's stated order
        r = rows.get(cid)
        if r is None:
            continue
        out.append({"rank": len(out) + 1, "chunk_id": cid,
                    "title": f"corpus chunk {cid}",
                    "snippet": " ".join((r["text"] or "").split())[:per]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="eval/mcq/items_reason.json")
    ap.add_argument("--out", default="eval/mcq/results/run_oracle_seed17.json")
    ap.add_argument("--db", default=CHUNKS_DB)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--num-predict", type=int, default=1200)
    ap.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    args = ap.parse_args()

    system_prompt = load_prompt(Path(args.prompt))
    prompt_sha = hashlib.sha256(system_prompt.encode()).hexdigest()[:12]
    items = json.loads(Path(args.items).read_text())
    if args.limit:
        items = items[:args.limit]

    print(f"arm=oracle  items={len(items)}  seed={args.seed}  sha={prompt_sha}\n")
    results, correct = [], 0

    for n, it in enumerate(items, 1):
        rng = random.Random(f"{args.seed}-{it['id']}")
        order = it["options"][:]
        rng.shuffle(order)
        gold = LETTERS[order.index(it["answer"])]

        passages = fetch_chunks(args.db, it.get("source_chunks") or [])
        if n == 1 and not passages:
            sys.exit("ABORT: no oracle chunks resolved for item 1 — check "
                     "source_chunks ids and the chunks.db path.")
        lines = [f"[L{p['rank']}] {p['title']}\n    {p['snippet']}" for p in passages]
        evidence = "Evidence from the literature:\n" + "\n".join(lines)

        # The guard this arm was missing. `answer_needle` is the quantity the
        # item cannot be answered without; if it is not in what we actually
        # paste, this is not an oracle observation and must not be scored as one.
        needle = it.get("answer_needle")
        needle_ok = bool(needle) and bool(re.search(needle, evidence, re.IGNORECASE))
        if needle and not needle_ok:
            if n == 1:
                sys.exit(f"ABORT: oracle evidence for {it['id']} does not contain "
                         f"its answer_needle ({needle!r}). The paste window is "
                         f"cutting the answer off — widen ORACLE_PER_CHUNK.")
            print(f"  !! {it['id']}: answer_needle missing from pasted evidence "
                  f"— scored, but flagged not_oracle")

        prompt = build_prompt(it, order, evidence)

        try:
            raw, secs = call_model(system_prompt, prompt, args.timeout, args.num_predict)
            err = None
        except Exception as e:
            raw, secs, err = "", 0.0, f"{type(e).__name__}: {e}"
        letter, how = extract_letter(raw) if raw else (None, "error")
        ok = (letter == gold)
        correct += ok

        results.append({"id": it["id"], "domain": it["domain"], "type": it["type"],
                        "gold_letter": gold, "predicted": letter, "parse": how,
                        "correct": ok, "seconds": round(secs, 1), "error": err,
                        "prompt_chars": len(prompt),
                        "evidence": {"n_passages": len(passages),
                                     "chunk_ids": [p["chunk_id"] for p in passages],
                                     "needle_present": needle_ok},
                        "shuffled_options": order, "raw": raw})
        print(f"  [{n:>2}/{len(items)}] {'ok ' if ok else 'XX '}{it['id']}  "
              f"gold={gold} pred={letter or '-'} ({how})  "
              f"chunks={[p['chunk_id'] for p in passages]}  {secs:>5.1f}s")

    n = len(results)
    print("\n" + "=" * 58)
    print(f"  arm       : oracle")
    print(f"  accuracy  : {correct/n:.3f}  ({correct}/{n})")
    print(f"  chance    : 0.250")
    print(f"  mean chars: {sum(r['prompt_chars'] for r in results)//n}")
    print("=" * 58)
    print("\n  parse:", dict(Counter(r["parse"] for r in results)))
    true_or = [r for r in results if r["evidence"].get("needle_present")]
    if len(true_or) != n:
        print(f"  !! only {len(true_or)}/{n} items were TRUE oracle observations "
              f"(answer present in pasted evidence). Headline accuracy above "
              f"mixes them with items where it was not.")
    if true_or:
        print(f"  true-oracle subset: {sum(r['correct'] for r in true_or)}/{len(true_or)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"arm": "oracle", "seed": args.seed, "items_file": args.items,
         "prompt_file": args.prompt, "prompt_sha256_12": prompt_sha,
         "system_prompt": system_prompt, "n": n, "correct": correct,
         "accuracy": round(correct / n, 4), "results": results}, indent=2))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
