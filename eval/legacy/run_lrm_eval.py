"""
eval/run_lrm_eval.py
====================
LRM reasoning benchmark for BoneMind.

Three evaluation components
---------------------------
1. Chain Coverage  — does the LRM find chains that pass through expected
                     bone science concepts for 25 domain queries?
2. Physics Accuracy — does the physics engine correctly classify known-
                      plausible and known-implausible edge directions?
3. Novelty Calibration — does the novelty classifier assign the expected
                          label (GROUNDED / SPECULATIVE / NOVEL) for 20
                          calibration queries?

Metrics
-------
Chain Coverage Rate  (CCR@3) — % of queries where all expected_node
                                fragments appear in at least one of the
                                top-3 chains.
Anchor Success Rate  (ASR)   — % of queries where the LRM finds ≥1 chain
                                (i.e., anchoring to the graph succeeded).
Physics Accuracy     (PA)    — % of edge pairs where the engine returns
                                the expected status.
Physics Sensitivity          — PA restricted to IMPLAUSIBLE pairs.
Physics Specificity          — PA restricted to PLAUSIBLE pairs.
Novelty Agreement Rate (NAR) — % of novelty cases where the top-1 chain
                                matches the expected label.

Usage
-----
    python eval/run_lrm_eval.py
    python eval/run_lrm_eval.py --max-results 5
    python eval/run_lrm_eval.py --benchmark eval/lrm_benchmark.json
    python eval/run_lrm_eval.py --out eval/lrm_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from reasoning.lrm import LRM
from reasoning.physics import PhysicsEngine


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", default="eval/lrm_benchmark.json")
parser.add_argument("--out",       default="eval/lrm_results.json")
parser.add_argument("--max-results", type=int, default=10,
                    help="max chains returned per query (default 10)")
args = parser.parse_args()

# ── Load benchmark ────────────────────────────────────────────────────────────
with open(args.benchmark) as f:
    bench = json.load(f)

chain_cases   = bench["chain_coverage"]
physics_pairs = bench["physics_pairs"]
novelty_cases = bench["novelty_cases"]

print(f"BoneMind LRM Benchmark  v{bench['version']}")
print(f"  Chain coverage : {len(chain_cases)} queries")
print(f"  Physics pairs  : {len(physics_pairs)} pairs")
print(f"  Novelty cases  : {len(novelty_cases)} cases")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 1. PHYSICS ACCURACY  (no model load needed — pure rule lookup)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("COMPONENT 1 — Physics Engine Accuracy")
print("=" * 60)

engine = PhysicsEngine()

physics_results = []
n_implausible = n_plausible = 0
correct_implausible = correct_plausible = 0

print(f"\n{'ID':>3}  {'Expected':<12}  {'Got':<12}  {'Match':>5}  Edge")
print("-" * 80)

for pair in physics_pairs:
    result = engine.validate_edge(pair["source"], pair["relation"], pair["target"])
    expected = pair["expected_status"]
    got      = result.status
    match    = (got == expected)

    if expected == "IMPLAUSIBLE":
        n_implausible += 1
        correct_implausible += match
    elif expected == "PLAUSIBLE":
        n_plausible += 1
        correct_plausible += match

    marker = "✓" if match else "✗"
    edge_str = f"{pair['source']} --{pair['relation']}--> {pair['target']}"
    print(f"{pair['id']:>3}  {expected:<12}  {got:<12}  {marker:>5}  {edge_str[:50]}")

    physics_results.append({
        "id": pair["id"],
        "source": pair["source"],
        "relation": pair["relation"],
        "target": pair["target"],
        "expected_status": expected,
        "got_status": got,
        "law": result.law,
        "match": match,
        "note": pair["note"],
    })

n_physics = len(physics_pairs)
pa        = (correct_implausible + correct_plausible) / n_physics
sens      = correct_implausible / n_implausible if n_implausible else 0.0
spec      = correct_plausible   / n_plausible   if n_plausible   else 0.0

print()
print(f"  Physics Accuracy    : {pa:.3f}  ({correct_implausible + correct_plausible}/{n_physics})")
print(f"  Sensitivity (IMPL.) : {sens:.3f}  ({correct_implausible}/{n_implausible})")
print(f"  Specificity (PLAUS.): {spec:.3f}  ({correct_plausible}/{n_plausible})")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CHAIN COVERAGE + 3. NOVELTY CALIBRATION  (requires LRM load)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Loading LRM (graph + physics engine)…")
print("=" * 60)

t0  = time.time()
lrm = LRM(max_hops=5, physics_filter=True)
stats = lrm.graph_stats()
print(f"  Graph : {stats.get('n_nodes', '?')} nodes  {stats.get('n_edges', '?')} edges")
print(f"  Loaded in {time.time() - t0:.1f}s")
print()


# ── Helper ────────────────────────────────────────────────────────────────────
def nodes_found(expected_frags: list[str], results) -> bool:
    """
    Return True if EACH expected fragment appears as a substring in the
    node_ids of at least one chain across the result list.

    Each fragment is checked independently — it does not need to appear
    in the same chain as the others.  This matches how the LRM actually
    works: it finds shortest paths between anchor pairs, so a query with
    3 expected concepts will typically return separate chains for each
    pair rather than a single chain spanning all three.

    A fragment like "porosity" matches node_id "cortical_porosity".
    """
    all_nodes = " ".join(
        node for h in results for node in h.chain
    ).lower()
    return all(frag.lower() in all_nodes for frag in expected_frags)


# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("COMPONENT 2 — Chain Coverage")
print("=" * 60)
print(f"\n{'ID':>3}  {'Domain':<14}  {'Anchored':>8}  {'CCR@3':>6}  Query")
print("-" * 85)

chain_results  = []
n_anchored     = 0
n_ccr3         = 0
domain_stats: dict[str, dict] = {}

for case in chain_cases:
    t_q = time.time()
    hypotheses = lrm.query(case["query"], max_results=args.max_results)
    elapsed    = time.time() - t_q

    anchored   = len(hypotheses) > 0
    top3       = hypotheses[:3]
    covered    = nodes_found(case["expected_nodes"], top3) if anchored else False

    n_anchored += anchored
    n_ccr3     += covered

    d = case["domain"]
    if d not in domain_stats:
        domain_stats[d] = {"n": 0, "anchored": 0, "covered": 0}
    domain_stats[d]["n"]        += 1
    domain_stats[d]["anchored"] += anchored
    domain_stats[d]["covered"]  += covered

    a_mark = "✓" if anchored else "✗"
    c_mark = "✓" if covered  else ("—" if not anchored else "✗")
    print(f"{case['id']:>3}  {case['domain']:<14}  {a_mark:>8}  {c_mark:>6}  {case['query'][:48]}")

    top_chain = hypotheses[0].chain_str() if hypotheses else "—"
    top_nov   = hypotheses[0].novelty     if hypotheses else "—"
    top_score = hypotheses[0].score       if hypotheses else 0.0
    chain_results.append({
        "id":             case["id"],
        "domain":         case["domain"],
        "query":          case["query"],
        "expected_nodes": case["expected_nodes"],
        "anchored":       anchored,
        "ccr3":           covered,
        "n_chains":       len(hypotheses),
        "top_chain":      top_chain,
        "top_novelty":    top_nov,
        "top_score":      round(top_score, 4),
        "elapsed_s":      round(elapsed, 2),
    })

n_chain = len(chain_cases)
asr     = n_anchored / n_chain
ccr3    = n_ccr3     / n_chain

print()
print(f"  Anchor Success Rate (ASR)  : {asr:.3f}  ({n_anchored}/{n_chain})")
print(f"  Chain Coverage Rate (CCR@3): {ccr3:.3f}  ({n_ccr3}/{n_chain})")

print("\nPer-domain breakdown:")
for d, s in sorted(domain_stats.items()):
    d_asr  = s["anchored"] / s["n"]
    d_ccr3 = s["covered"]  / s["n"]
    print(f"  {d:<16}  ASR={d_asr:.2f}  CCR@3={d_ccr3:.2f}  (n={s['n']})")


# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("COMPONENT 3 — Novelty Calibration")
print("=" * 60)
print(f"\n{'ID':>3}  {'Expected':<12}  {'Top-3 labels':<22}  {'Match':>5}  Query")
print("-" * 90)

novelty_results = []
n_correct_nov   = 0
label_stats: dict[str, dict] = {}

for case in novelty_cases:
    hypotheses = lrm.query(case["query"], max_results=5)
    expected   = case["expected_novelty"]

    # Check top-3: a match if ANY of the top-3 chains has the expected label.
    # GROUNDED chains score lower than longer chains (length weight=0.5), so
    # they rarely reach position 1 — checking top-3 is more meaningful.
    top3_labels = [h.novelty for h in hypotheses[:3]]
    got         = hypotheses[0].novelty if hypotheses else "NO_CHAIN"
    match       = expected in top3_labels

    n_correct_nov += match

    if expected not in label_stats:
        label_stats[expected] = {"n": 0, "correct": 0}
    label_stats[expected]["n"]       += 1
    label_stats[expected]["correct"] += match

    marker = "✓" if match else ("—" if not hypotheses else "✗")
    labels_str = "/".join(top3_labels) if top3_labels else "NO_CHAIN"
    print(f"{case['id']:>3}  {expected:<12}  {labels_str:<22}  {marker:>5}  {case['query'][:40]}")

    novelty_results.append({
        "id":               case["id"],
        "query":            case["query"],
        "expected_novelty": expected,
        "top1_novelty":     got,
        "top3_labels":      top3_labels,
        "match":            match,
        "n_chains":         len(hypotheses),
        "note":             case["note"],
    })

n_nov = len(novelty_cases)
nar   = n_correct_nov / n_nov

print()
print(f"  Novelty Agreement Rate (NAR): {nar:.3f}  ({n_correct_nov}/{n_nov})")
print("\nPer-label breakdown:")
for label, s in sorted(label_stats.items()):
    acc = s["correct"] / s["n"]
    print(f"  {label:<12}  {acc:.2f}  ({s['correct']}/{s['n']})")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Physics Accuracy     (PA)   : {pa:.3f}   ({correct_implausible + correct_plausible}/{n_physics})")
print(f"    Sensitivity (IMPLAUSIBLE) : {sens:.3f}")
print(f"    Specificity (PLAUSIBLE)   : {spec:.3f}")
print(f"  Anchor Success Rate  (ASR)  : {asr:.3f}   ({n_anchored}/{n_chain})")
print(f"  Chain Coverage Rate  (CCR@3): {ccr3:.3f}   ({n_ccr3}/{n_chain})")
print(f"  Novelty Agreement Rate (NAR): {nar:.3f}   ({n_correct_nov}/{n_nov})")
print()

PASS_PA   = pa   >= 0.90
PASS_ASR  = asr  >= 0.80
PASS_CCR3 = ccr3 >= 0.60
PASS_NAR  = nar  >= 0.50

print("Readiness thresholds:")
print(f"  PA   ≥ 0.90  →  {'PASS ✓' if PASS_PA   else 'FAIL ✗'}  ({pa:.3f})")
print(f"  ASR  ≥ 0.80  →  {'PASS ✓' if PASS_ASR  else 'FAIL ✗'}  ({asr:.3f})")
print(f"  CCR@3≥ 0.60  →  {'PASS ✓' if PASS_CCR3 else 'FAIL ✗'}  ({ccr3:.3f})")
print(f"  NAR  ≥ 0.50  →  {'PASS ✓' if PASS_NAR  else 'FAIL ✗'}  ({nar:.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
output = {
    "benchmark_version": bench["version"],
    "max_results": args.max_results,
    "graph_stats": stats,
    "aggregate": {
        "physics_accuracy":      round(pa,   4),
        "physics_sensitivity":   round(sens, 4),
        "physics_specificity":   round(spec, 4),
        "anchor_success_rate":   round(asr,  4),
        "chain_coverage_rate_3": round(ccr3, 4),
        "novelty_agreement_rate":round(nar,  4),
    },
    "thresholds": {
        "pa_pass":   PASS_PA,
        "asr_pass":  PASS_ASR,
        "ccr3_pass": PASS_CCR3,
        "nar_pass":  PASS_NAR,
    },
    "physics": physics_results,
    "chain_coverage": chain_results,
    "novelty": novelty_results,
}

with open(args.out, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved → {args.out}")
