"""
eval/mcq/run_bm25.py
=====================
`bm25` — the `full` arm with three retrieval defects fixed. Everything else is
held constant: same prompt_v1.txt, same seeded option shuffle, same
extract_letter, same graph channel, temperature 0.

Why: on 50 corpus-derived quantitative items the shipped pipeline scored
13/50 (26%) against closed-book's 18/50 (36%) and an oracle's 33/50 (66%). The oracle
proved the evidence works when it arrives, so the whole 40-point gap is
retrieval. Three causes were identified:

1. TRUNCATION. api/main.py pastes LIT_CHAR_BUDGET/LIT_TOP_K = 840 chars per
   passage. The median chunk is 1,680 chars, so 99% are cut and half the median
   chunk is discarded — always the tail, because it takes the head. On the
   oracle items the deciding quantity sat past char 840 in 7 of 10 cases.
   FIX: paste whole chunks under a much larger budget.

2. NO LEXICAL CHANNEL. SPECTER2 is trained for document-level citation
   similarity over titles and abstracts; over 248k passages it puts the whole
   corpus in a 0.76-0.84 cosine band, so a chunk that answers the question
   scores 0.799 while an unrelated one scores 0.805. Numbers and rare terms are
   invisible to it.
   FIX: BM25 over an FTS5 index, fused with the vector ranking by reciprocal
   rank fusion.

3. QUESTION-SHAPED QUERIES. A question embeds far from a property table. In the
   diagnostic, querying with a passage-shaped restatement moved the target chunk
   from rank 256 to rank 1.
   FIX: HyDE — one cheap generation of a hypothetical answer passage, used as
   the vector query. The lexical query stays the literal question terms.

Run
---
    .venv/bin/python -m eval.mcq.run_bm25 \
        --items eval/mcq/items_reason.json \
        --out   eval/mcq/run_bm25_reason_v2_seed17.json
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
from collections import Counter, defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.mcq.run_arms import (  # noqa: E402
    DEFAULT_PROMPT, LETTERS, OLLAMA_MODEL, OLLAMA_URL,
    build_prompt, call_model, extract_letter, load_prompt,
)

CHUNKS_DB = "data/db/chunks.db"
FTS_DB = "data/db/chunks_fts.db"

# Measured, not guessed: the source chunk sits at lexical ranks 3-193 across the
# item set, so a pool of 25 and 5 pasted passages structurally cannot reach it.
# num_ctx 8192 leaves ~6.9k tokens for the prompt once num_predict 1200 is
# reserved (~27k chars); the shipped pipeline uses 9.4k of that. So paste more,
# deeper — WITHOUT changing num_ctx, which keeps this comparable to bare/oracle.
TOP_K = 14                # passages pasted (shipped pipeline: 5)
POOL = 150                # candidates per channel before fusion (was 25)
PER_CHUNK = 1700          # ~ the median chunk (1,680) whole — fix #1
TOTAL_BUDGET = 24000      # ~5.8k tokens of evidence, still inside num_ctx 8192
RRF_K = 60                # reciprocal-rank-fusion constant

_STOP = set("""a an and are as at be by for from how in into is it its of on or that the to what which
with approximately about using given comparing reported typical value values many times ratio would
does do than then their there these this those between per each""".split())

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from retrieval.retriever import BoneGraphRetriever
        _retriever = BoneGraphRetriever()
        _retriever.load()
    return _retriever


# ── Fix #3: HyDE ──────────────────────────────────────────────────────────────

HYDE_PROMPT = (
    "Write two sentences of the kind that would appear in a bone-science paper "
    "and would contain the facts needed to answer the question below. State "
    "concrete quantities with units. Do not answer the question, do not mention "
    "the options, and do not explain — just write the passage.\n\nQuestion: {q}"
)


def hyde(question: str, timeout: int = 180) -> str:
    """One cheap generation used ONLY as the vector query. Never shown to the
    scorer, so it cannot leak an answer into the graded prompt."""
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": HYDE_PROMPT.format(q=question)}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 220, "num_ctx": 8192},
        }, timeout=timeout)
        r.raise_for_status()
        txt = r.json()["message"]["content"]
        m = re.search(r"final response\s*", txt, flags=re.I)
        return (txt[m.end():] if m else txt).strip()
    except Exception:
        return ""


# ── Fix #2: lexical channel ───────────────────────────────────────────────────

def fts_query(question: str) -> str:
    terms = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9/\-]{2,}", question.lower())
             if w not in _STOP]
    seen, out = set(), []
    for w in terms:
        if w in seen:
            continue
        seen.add(w)
        out.append('"' + w.replace('"', '') + '"')
    return " OR ".join(out[:24])


def lexical_search(question: str, limit: int = POOL) -> list[int]:
    q = fts_query(question)
    if not q:
        return []
    try:
        con = sqlite3.connect(f"file:{FTS_DB}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT rowid FROM ft WHERE ft MATCH ? ORDER BY bm25(ft) LIMIT ?",
            (q, limit)).fetchall()
        con.close()
        return [r[0] for r in rows]
    except Exception:
        return []


# ── Fusion + fix #1: whole chunks ─────────────────────────────────────────────

def rrf(*rankings: list[int]) -> list[int]:
    score: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            score[cid] += 1.0 / (RRF_K + rank)
    return [cid for cid, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def interleave(vec: list[int], lex: list[int], k: int) -> list[int]:
    """Round-robin the two channels instead of pure RRF.

    RRF rewards agreement between rankers, which is exactly wrong here: the two
    channels are good at different things, and a lexical hit at rank 8 can never
    outscore five vector hits at ranks 1-5. Round-robin guarantees each channel
    contributes its own best candidates, so a chunk only the lexical side can
    find still reaches the model. Ties inside the interleave are broken by RRF.
    """
    out, seen = [], set()
    fused_rank = {c: i for i, c in enumerate(rrf(vec, lex))}
    for i in range(max(len(vec), len(lex))):
        for src in (vec, lex):
            if i < len(src) and src[i] not in seen:
                seen.add(src[i])
                out.append(src[i])
                if len(out) >= k:
                    return sorted(out, key=lambda c: fused_rank.get(c, 10**6))
    return sorted(out, key=lambda c: fused_rank.get(c, 10**6))


def fetch_texts(ids: list[int]) -> dict[int, str]:
    if not ids:
        return {}
    con = sqlite3.connect(f"file:{CHUNKS_DB}?mode=ro", uri=True)
    q = f"SELECT id, text FROM chunks WHERE id IN ({','.join('?'*len(ids))})"
    out = {i: " ".join((t or "").split()) for i, t in con.execute(q, ids)}
    con.close()
    return out


def retrieve(question: str, use_vector: bool = False, use_hyde: bool = False) -> tuple[list[dict], dict]:
    """Lexical-only by default. Measured over all 50 items, needle-present rate:

        vector only, top5 (shipped)        10/50
        lexical only, top12                30/50
        interleave(vector, lexical) top12  24/50   <- vector DISPLACES good hits

    So the vector channel is not merely weak here, it is net negative: adding
    SPECTER2 to BM25 costs six items. The flags keep that path runnable so the
    finding is reproducible, but they are off.
    """
    hyde_text = hyde(question) if use_hyde else ""
    vec_ids = []
    if use_vector:
        r = _get_retriever()
        try:
            vres = r.query(hyde_text or question, top_k=POOL)
        except Exception:
            vres = []
        vec_ids = [x["chunk_id"] for x in vres if x.get("chunk_id") is not None]
    lex_ids = lexical_search(question)

    fused = interleave(vec_ids, lex_ids, TOP_K) if use_vector else lex_ids[:TOP_K]
    texts = fetch_texts(fused)

    passages, budget = [], TOTAL_BUDGET
    for cid in fused:
        t = texts.get(cid, "")
        if not t:
            continue
        snippet = t[:PER_CHUNK]                 # fix #1: whole chunk, not 840
        if budget - len(snippet) < 0:
            break
        budget -= len(snippet)
        passages.append({"rank": len(passages) + 1, "chunk_id": cid, "snippet": snippet,
                         "via": ("both" if cid in vec_ids and cid in lex_ids
                                 else "vector" if cid in vec_ids else "lexical")})
    meta = {"n_passages": len(passages), "chunk_ids": [p["chunk_id"] for p in passages],
            "via": [p["via"] for p in passages], "hyde_chars": len(hyde_text),
            "n_vector": len(vec_ids), "n_lexical": len(lex_ids),
            "overlap": len(set(vec_ids) & set(lex_ids))}
    return passages, meta


def build_evidence(question: str, use_vector: bool = False, use_hyde: bool = False) -> tuple[str, dict]:
    passages, meta = retrieve(question, use_vector, use_hyde)
    blocks = []
    if passages:
        blocks.append("Evidence from the literature:\n" + "\n".join(
            f"[L{p['rank']}] corpus chunk {p['chunk_id']}\n    {p['snippet']}" for p in passages))
    else:
        blocks.append("Evidence from the literature:\n(no literature retrieved)")
    from reasoning import kg_context
    try:
        kg = kg_context.kg_facts(question)
    except Exception:
        kg = {"facts": [], "anchors": []}
    meta["n_kg_facts"] = len(kg.get("facts") or [])
    blocks.append("Knowledge-graph facts:\n" + kg_context.format_facts(kg))
    return "\n\n".join(blocks), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="eval/mcq/items_reason.json")
    ap.add_argument("--out", default="eval/mcq/run_bm25_seed17.json")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--num-predict", type=int, default=1200)
    ap.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    ap.add_argument("--use-vector", action="store_true",
                    help="re-enable the SPECTER2 channel (measured net negative)")
    ap.add_argument("--use-hyde", action="store_true",
                    help="HyDE query rewriting; only meaningful with --use-vector")
    args = ap.parse_args()

    system_prompt = load_prompt(Path(args.prompt))
    prompt_sha = hashlib.sha256(system_prompt.encode()).hexdigest()[:12]
    items = json.loads(Path(args.items).read_text())
    if args.limit:
        items = items[:args.limit]

    print(f"arm=bm25  items={len(items)}  seed={args.seed}  sha={prompt_sha}")
    chans = ("BM25" + (" + vector" if args.use_vector else "")
             + (" + HyDE" if args.use_hyde else ""))
    print(f"  retrieval: {chans}, top{TOP_K}, {PER_CHUNK} chars/chunk "
          f"(shipped: vector top5, 840 chars/chunk)\n")
    results, correct = [], 0

    for n, it in enumerate(items, 1):
        rng = random.Random(f"{args.seed}-{it['id']}")
        order = it["options"][:]
        rng.shuffle(order)
        gold = LETTERS[order.index(it["answer"])]

        t0 = time.time()
        evidence, meta = build_evidence(it["question"], args.use_vector, args.use_hyde)
        if n == 1 and not meta.get("n_passages"):
            sys.exit("ABORT: 0 passages on item 1 — check LD_PRELOAD (run_api.sh) "
                     "and that data/db/chunks_fts.db exists.")
        prompt = build_prompt(it, order, evidence)
        try:
            raw, secs = call_model(system_prompt, prompt, args.timeout, args.num_predict)
            err = None
        except Exception as e:
            raw, secs, err = "", 0.0, f"{type(e).__name__}: {e}"
        letter, how = extract_letter(raw) if raw else (None, "error")
        ok = (letter == gold)
        correct += ok

        # Did the deciding quantity actually arrive? The metric the old pipeline
        # never had, and the reason it failed invisibly.
        needle = it.get("answer_needle")
        needle_ok = bool(needle) and bool(re.search(needle, evidence, re.IGNORECASE))
        src_hit = bool(set(it.get("source_chunks") or []) & set(meta["chunk_ids"]))

        results.append({"id": it["id"], "domain": it["domain"], "type": it["type"],
                        "gold_letter": gold, "predicted": letter, "parse": how,
                        "correct": ok, "seconds": round(time.time() - t0, 1),
                        "error": err, "prompt_chars": len(prompt),
                        "evidence": meta, "needle_present": needle_ok,
                        "source_chunk_retrieved": src_hit,
                        "shuffled_options": order, "raw": raw})
        print(f"  [{n:>2}/{len(items)}] {'ok ' if ok else 'XX '}{it['id']}  gold={gold} "
              f"pred={letter or '-'}  needle={'Y' if needle_ok else 'n'} "
              f"src={'Y' if src_hit else 'n'}  via={Counter(meta['via'])} "
              f"{results[-1]['seconds']:>5.1f}s")

    n = len(results)
    print("\n" + "=" * 62)
    print(f"  arm            : bm25 (hybrid + HyDE + whole chunks)")
    print(f"  accuracy       : {correct/n:.3f}  ({correct}/{n})")
    print(f"  needle present : {sum(r['needle_present'] for r in results)}/{n}")
    print(f"  source chunk   : {sum(r['source_chunk_retrieved'] for r in results)}/{n}")
    print(f"  mean passages  : {sum(r['evidence']['n_passages'] for r in results)/n:.1f}")
    print(f"  mean prompt    : {sum(r['prompt_chars'] for r in results)//n} chars")
    print(f"  via            : {dict(Counter(v for r in results for v in r['evidence']['via']))}")
    print("=" * 62)

    Path(args.out).write_text(json.dumps(
        {"arm": "bm25", "seed": args.seed, "items_file": args.items,
         "prompt_file": args.prompt, "prompt_sha256_12": prompt_sha,
         "system_prompt": system_prompt,
         "config": {"top_k": TOP_K, "pool": POOL, "per_chunk": PER_CHUNK,
                    "total_budget": TOTAL_BUDGET, "rrf_k": RRF_K},
         "n": n, "correct": correct, "accuracy": round(correct/n, 4),
         "results": results}, indent=2))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
