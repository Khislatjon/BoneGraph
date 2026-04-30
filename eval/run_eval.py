"""
eval/run_eval.py
================
Retrieval quality benchmark for BoneMind.

Metrics
-------
Recall@k  — fraction of questions where at least one relevant chunk
             appears in the top-k results (k = 1, 3, 5, 10).
MRR       — Mean Reciprocal Rank: mean of 1/rank of the first relevant
             chunk across all questions. 0 if no relevant chunk found.

Relevance definition
--------------------
A chunk is considered relevant if ANY of the question's expected_keywords
appears in the chunk text (case-insensitive substring match).

Usage
-----
    python eval/run_eval.py                  # default top_k=10
    python eval/run_eval.py --top-k 20      # custom top_k
    python eval/run_eval.py --out eval/results.json
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.retriever import BoneMindRetriever


# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--top-k", type=int, default=10)
parser.add_argument("--benchmark", default="eval/benchmark.json")
parser.add_argument("--out", default="eval/results.json")
args = parser.parse_args()

TOP_K = args.top_k
RECALL_K = [1, 3, 5, 10]

# ── Load retriever ─────────────────────────────────────────────────────────────
print("Loading retriever (this takes ~10–20 seconds)...")
retriever = BoneMindRetriever()
retriever.load()
print()

# ── Load benchmark ────────────────────────────────────────────────────────────
with open(args.benchmark) as f:
    questions = json.load(f)

print(f"Benchmark: {len(questions)} questions  |  top_k={TOP_K}\n")
print(f"{'ID':>3}  {'Domain':<16}  {'RR':>6}  {'Rank':>5}  {'Question'}")
print("-" * 90)

# ── Run eval ──────────────────────────────────────────────────────────────────
per_question = []

for q in questions:
    results = retriever.query(q["question"], top_k=TOP_K)
    keywords = [kw.lower() for kw in q["expected_keywords"]]

    first_relevant_rank = None
    for r in results:
        chunk_text = r["text"].lower()
        if any(kw in chunk_text for kw in keywords):
            first_relevant_rank = r["rank"]
            break

    rr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0

    per_question.append({
        "id":                  q["id"],
        "domain":              q["domain"],
        "question":            q["question"],
        "expected_keywords":   q["expected_keywords"],
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank":     rr,
        **{f"recall@{k}": (first_relevant_rank is not None and first_relevant_rank <= k)
           for k in RECALL_K},
    })

    rank_str = str(first_relevant_rank) if first_relevant_rank else "—"
    print(f"{q['id']:>3}  {q['domain']:<16}  {rr:>6.3f}  {rank_str:>5}  {q['question'][:55]}")

# ── Aggregate metrics ─────────────────────────────────────────────────────────
n = len(per_question)
mrr = sum(r["reciprocal_rank"] for r in per_question) / n
recall = {k: sum(r[f"recall@{k}"] for r in per_question) / n for k in RECALL_K}

print("\n" + "=" * 60)
print(f"  Questions evaluated : {n}")
print(f"  MRR                 : {mrr:.3f}")
for k in RECALL_K:
    hits = sum(r[f"recall@{k}"] for r in per_question)
    print(f"  Recall@{k:<2}           : {recall[k]:.3f}  ({hits}/{n})")
print("=" * 60)

# Per-domain breakdown
domains = sorted({r["domain"] for r in per_question})
print("\nPer-domain MRR:")
for d in domains:
    subset = [r for r in per_question if r["domain"] == d]
    d_mrr = sum(r["reciprocal_rank"] for r in subset) / len(subset)
    d_r5  = sum(r["recall@5"] for r in subset) / len(subset)
    print(f"  {d:<16}  MRR={d_mrr:.3f}  Recall@5={d_r5:.3f}  (n={len(subset)})")

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    "top_k": TOP_K,
    "n_questions": n,
    "aggregate": {
        "mrr": round(mrr, 4),
        **{f"recall@{k}": round(recall[k], 4) for k in RECALL_K},
    },
    "per_domain": {
        d: {
            "mrr": round(sum(r["reciprocal_rank"] for r in [x for x in per_question if x["domain"] == d])
                         / len([x for x in per_question if x["domain"] == d]), 4),
        }
        for d in domains
    },
    "questions": per_question,
}

with open(args.out, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {args.out}")
print("\nTarget thresholds for Phase 3 readiness:")
print(f"  Recall@5 ≥ 0.80  →  {'PASS ✓' if recall[5] >= 0.80 else 'FAIL ✗'} ({recall[5]:.3f})")
print(f"  MRR      ≥ 0.60  →  {'PASS ✓' if mrr >= 0.60 else 'FAIL ✗'} ({mrr:.3f})")
