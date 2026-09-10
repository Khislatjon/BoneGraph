"""
eval/mcq/run_bm25rerank.py
=====================
`bm25rerank` — bm25's lexical retrieval, but deep + reranked.

Why, from the failure analysis of bm25 (26/50) against oracle (41/50):

  PROBLEM 1 — the needle is absent on 19/50. But BM25 does find the source
  chunk; it is simply deeper than bm25 looks:
        rank 15-40    2 items
        rank 41-150   9 items      <- the big bucket
        rank 151-1000 6 items
    So pull a pool of 150 and let a reranker pick from it. That reaches 11.

  PROBLEM 2 — dilution: on the 31 items where the needle DID arrive, bm25
  scored 61% against oracle's 82%. Oracle pastes 1-2 chunks; bm25 pastes 14
  (~24k chars). Keeping 4 closes that.

An earlier version of this arm netted zero, because under prompt_v1 the model
barely used evidence at all (48% with the needle vs 42% without). Under
prompt_v2 that gap is 61% vs 37%, so precision now converts.

Two details that matter:

* PREVIEWS ARE QUERY-CENTRED, not chunk heads. Showing the reranker the first N
  characters would reproduce the original truncation bug one level up — the
  deciding quantity is usually mid-chunk. Each preview is built from sentences
  containing both a query term and a numeric value.

* NO WORKED EXAMPLE IN THE RERANK PROMPT. An earlier version ended with
  "Example: 7, 22, 3, 15" and the model returned exactly 7, 22, 3, 15 on one
  item — copying the illustration instead of ranking.

* THE RERANKER NEVER SEES THE OPTIONS — question only, so it selects for
  evidence, not for an answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.mcq.run_arms import (  # noqa: E402
    DEFAULT_PROMPT, LETTERS, OLLAMA_MODEL, OLLAMA_URL,
    build_prompt, call_model, extract_letter, load_prompt,
)
from eval.mcq.run_bm25 import CHUNKS_DB, lexical_search, _STOP  # noqa: E402

POOL = 150         # BM25 candidates (bm25 pasted the top 14 blind)
KEEP = 4           # passages pasted after reranking
PER_CHUNK = 1700   # whole median chunk
PREVIEW = 130      # chars per candidate in the rerank call; 150 x 130 ~= 4.9k tok

_NUM = re.compile(
    r"\d+(?:\.\d+)?\s*(?:GPa|MPa|kPa|µm|μm|um|nm|mm|cm|N\b|kN|%|g\s*/\s*cm|"
    r"per\s*mm|days?|weeks?|months?|years?|microstrain)", re.I)


def fetch_texts(ids: list[int]) -> dict[int, str]:
    if not ids:
        return {}
    con = sqlite3.connect(f"file:{CHUNKS_DB}?mode=ro", uri=True)
    q = f"SELECT id, text FROM chunks WHERE id IN ({','.join('?'*len(ids))})"
    out = {i: " ".join((t or "").split()) for i, t in con.execute(q, ids)}
    con.close()
    return out


def preview(text: str, question: str, width: int = PREVIEW) -> str:
    terms = {w for w in re.findall(r"[a-z][a-z0-9\-]{3,}", question.lower())
             if w not in _STOP}
    best, out = [], ""
    for s in re.split(r"(?<=[.;])\s+", text):
        low = s.lower()
        best.append((2 * len(_NUM.findall(s)) + sum(1 for w in terms if w in low), s))
    best.sort(key=lambda x: -x[0])
    for score, s in best:
        if score <= 0:
            break
        if len(out) + len(s) > width:
            out += s[:max(0, width - len(out))]
            break
        out += s + " "
    return (out or text[:width]).strip()


RERANK_PROMPT = """You are selecting evidence for a bone-science question that must be answered by calculating with reported numerical values.

QUESTION: {q}

Below are numbered candidate passages. Choose the {k} MOST LIKELY to contain the numerical values needed to answer. Prefer passages stating concrete quantities with units. Ignore passages that are merely on the same topic but report no relevant numbers.

{cands}

Reply with ONLY the {k} passage numbers, comma-separated, best first. No other text."""


def llm_rerank(question, cands, k, timeout):
    listing = "\n".join(f"[{i}] {preview(t, question)}" for i, (_, t) in enumerate(cands, 1))
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": RERANK_PROMPT.format(
                q=question, k=k, cands=listing)}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 200, "num_ctx": 8192},
        }, timeout=timeout)
        r.raise_for_status()
        raw = r.json()["message"]["content"]
    except Exception as e:
        return [c for c, _ in cands[:k]], f"(rerank failed: {e})"
    m = re.search(r"final response\s*", raw, flags=re.I)
    body = raw[m.end():] if m else raw
    picks, seen = [], set()
    for tok in re.findall(r"\b(\d{1,3})\b", body):
        i = int(tok)
        if 1 <= i <= len(cands) and i not in seen:
            seen.add(i)
            picks.append(cands[i - 1][0])
        if len(picks) >= k:
            break
    if not picks:
        return [c for c, _ in cands[:k]], raw
    for cid, _ in cands:
        if len(picks) >= k:
            break
        if cid not in picks:
            picks.append(cid)
    return picks[:k], raw


def build_evidence(question, timeout):
    ids = lexical_search(question, POOL)
    texts = fetch_texts(ids)
    cands = [(c, texts[c]) for c in ids if texts.get(c)]
    if not cands:
        return "Evidence from the literature:\n(no literature retrieved)", {
            "n_passages": 0, "chunk_ids": [], "n_candidates": 0}
    picked, raw = llm_rerank(question, cands, KEEP, timeout)
    bm25 = {c: i + 1 for i, (c, _) in enumerate(cands)}
    lines = [f"[L{n}] corpus chunk {c}\n    {texts[c][:PER_CHUNK]}"
             for n, c in enumerate(picked, 1)]
    blocks = ["Evidence from the literature:\n" + "\n".join(lines)]
    from reasoning import kg_context
    try:
        kg = kg_context.kg_facts(question)
    except Exception:
        kg = {"facts": [], "anchors": []}
    blocks.append("Knowledge-graph facts:\n" + kg_context.format_facts(kg))
    return "\n\n".join(blocks), {
        "n_passages": len(picked), "chunk_ids": picked, "n_candidates": len(cands),
        "n_kg_facts": len(kg.get("facts") or []),
        "bm25_ranks_of_picked": [bm25.get(c) for c in picked],
        "rerank_raw": raw[-160:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="eval/mcq/items_reason.json")
    ap.add_argument("--out", default="eval/mcq/run_bm25rerank_seed17.json")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--num-predict", type=int, default=1200)
    ap.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    args = ap.parse_args()

    system_prompt = load_prompt(Path(args.prompt))
    sha = hashlib.sha256(system_prompt.encode()).hexdigest()[:12]
    items = json.loads(Path(args.items).read_text())
    if args.limit:
        items = items[:args.limit]
    print(f"arm=bm25rerank  items={len(items)}  seed={args.seed}  prompt={Path(args.prompt).name} sha={sha}")
    print(f"  BM25 top{POOL} -> LLM rerank -> keep {KEEP}, {PER_CHUNK} chars/chunk\n")
    results, correct = [], 0

    for n, it in enumerate(items, 1):
        rng = random.Random(f"{args.seed}-{it['id']}")
        order = it["options"][:]
        rng.shuffle(order)
        gold = LETTERS[order.index(it["answer"])]
        t0 = time.time()
        evidence, meta = build_evidence(it["question"], args.timeout)
        if n == 1 and not meta["n_passages"]:
            sys.exit("ABORT: 0 passages on item 1 — check data/db/chunks_fts.db.")
        prompt = build_prompt(it, order, evidence)
        try:
            raw, _ = call_model(system_prompt, prompt, args.timeout, args.num_predict)
            err = None
        except Exception as e:
            raw, err = "", f"{type(e).__name__}: {e}"
        letter, how = extract_letter(raw) if raw else (None, "error")
        ok = (letter == gold)
        correct += ok
        needle = it.get("answer_needle")
        nk = bool(needle) and bool(re.search(needle, evidence, re.I))
        sc = bool(set(it.get("source_chunks") or []) & set(meta["chunk_ids"]))
        results.append({"id": it["id"], "domain": it["domain"], "type": it["type"],
                        "gold_letter": gold, "predicted": letter, "parse": how,
                        "correct": ok, "seconds": round(time.time()-t0, 1), "error": err,
                        "prompt_chars": len(prompt), "evidence": meta,
                        "needle_present": nk, "source_chunk_retrieved": sc,
                        "shuffled_options": order, "raw": raw})
        print(f"  [{n:>2}/{len(items)}] {'ok ' if ok else 'XX '}{it['id']}  gold={gold} "
              f"pred={letter or '-'}  needle={'Y' if nk else 'n'} src={'Y' if sc else 'n'} "
              f"ranks={meta['bm25_ranks_of_picked']} {results[-1]['seconds']:>5.1f}s")

    n = len(results)
    print("\n" + "="*64)
    print(f"  arm            : bm25rerank (BM25 top{POOL} + rerank -> {KEEP})")
    print(f"  prompt         : {Path(args.prompt).name}")
    print(f"  accuracy       : {correct/n:.3f}  ({correct}/{n})")
    print(f"  needle present : {sum(r['needle_present'] for r in results)}/{n}")
    print(f"  source chunk   : {sum(r['source_chunk_retrieved'] for r in results)}/{n}")
    print(f"  mean prompt    : {sum(r['prompt_chars'] for r in results)//n} chars")
    print("="*64)
    Path(args.out).write_text(json.dumps(
        {"arm": "bm25rerank", "seed": args.seed, "items_file": args.items,
         "prompt_file": args.prompt, "prompt_sha256_12": sha,
         "system_prompt": system_prompt,
         "config": {"pool": POOL, "keep": KEEP, "per_chunk": PER_CHUNK, "preview": PREVIEW},
         "n": n, "correct": correct, "accuracy": round(correct/n, 4),
         "results": results}, indent=2))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
