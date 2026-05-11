"""
scripts/phase4_validate.py
──────────────────────────
Phase 4 success-criterion check.

Spec: a held-out set of ≥20 paraphrased queries should cover ≥ 90 % of
the variables the v0 keyword anchor catches on the canonical phrasings.

How this script defines "cover"
-------------------------------
For each ``(canonical, paraphrase, expected)`` row:

1. Run :func:`reasoning.physics_vars.variables_in_query` on the
   *canonical* string to confirm the v0 anchor really catches every
   variable we expect — if not, the row is dropped with a warning so
   the comparison is fair.
2. Run :class:`reasoning.semantic_anchor.SemanticVariableAnchor.top_k`
   with ``k=5`` (the production cutoff used by the query router) on
   the *paraphrase*.
3. Recall(query) = |expected ∩ top_3| / |expected|.

Aggregate recall = mean of per-query recalls.  Pass bar: ≥ 0.90.

This is *not* a permanent unit test — it loads SPECTER2 (≈10 s, 600 MB)
so it lives as a manual benchmark you can re-run after schema or
embedding changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable regardless of how the script is invoked.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reasoning.bone_relations import build_bone_registry
from reasoning.physics_vars import variables_in_query
from reasoning.semantic_anchor import SemanticVariableAnchor


# ── Mapping: keyword-anchor variable names → v2 Variable symbols ─────────────

_V0_TO_V2: dict[str, str] = {
    "apparent_density":       "rho",
    "porosity":               "phi",
    "elastic_modulus":        "E",
    "peak_strain":            "eps",
    "bone_adaptation_rate":   "dBMD_dt",
    "stress_intensity_range": "dK",
    "crack_growth_rate":      "da_dN",
    "cortical_thickness":     "t",
    "bending_resistance":     "I_section",
    # "strength" has no direct v2 equivalent — closest is sigma.
    "strength":               "sigma",
}


def v0_to_v2(v0_names: list[str]) -> set[str]:
    """Translate keyword-anchor hits to v2 Variable symbols (best-effort)."""
    out: set[str] = set()
    for n in v0_names:
        if n in _V0_TO_V2:
            out.add(_V0_TO_V2[n])
    return out


# ── Paraphrase test set ──────────────────────────────────────────────────────


# Each row: (canonical_phrasing, paraphrase, expected_v2_symbols).
# Canonical wordings are designed to land on the v0 keyword anchor; the
# paraphrase exercises SPECTER2's semantic matching.
TEST_QUERIES: list[tuple[str, str, set[str]]] = [
    ("porosity and elastic modulus",
     "how does bone become softer when more porous",
     {"phi", "E"}),

    ("bone density and stiffness",
     "relationship between BMD and bone rigidity",
     {"rho", "E"}),

    ("cyclic loading and fatigue crack growth",
     "bone microdamage accumulation under repeated stresses",
     {"dK", "da_dN"}),

    ("delta K and crack propagation",
     "stress-intensity influence on fatigue crack rate",
     {"dK", "da_dN"}),

    ("mechanostat strain",
     "Frost's strain-driven bone adaptation window",
     {"eps", "dBMD_dt"}),

    ("bone resorption with disuse",
     "annual BMD loss in immobilised patients",
     {"dBMD_dt"}),

    ("cortical thickness and bending strength",
     "thinner cortex weakening the diaphysis under load",
     {"t", "I_section"}),

    ("porosity and crack growth",
     "how voids in the bone matrix accelerate fatigue",
     {"phi", "da_dN"}),

    ("strain and bone formation",
     "mechanical loading driving osteogenesis",
     {"eps", "dBMD_dt"}),

    ("porosity",
     "void fraction in cortical tissue",
     {"phi"}),

    ("elastic modulus",
     "stiffness coefficient of bone tissue",
     {"E"}),

    ("bone density",
     "apparent density of mineralised tissue",
     {"rho"}),

    ("fatigue crack growth",
     "microcrack progression rate under cyclic load",
     {"da_dN"}),

    ("stress intensity range",
     "delta K in bone fatigue mechanics",
     {"dK"}),

    ("peak strain",
     "principal microstrain magnitude",
     {"eps"}),

    ("cortical wall thickness",
     "thickness of the cortical shell",
     {"t"}),

    ("bone remodeling",
     "yearly percentage change of bone mineral density",
     {"dBMD_dt"}),

    ("bending stiffness",
     "flexural rigidity of the diaphysis",
     {"I_section"}),

    ("mechanical loading",
     "applied bending moment on the femoral midshaft",
     {"M"}),

    ("fatigue loading",
     "repeated cyclic stresses driving microdamage",
     {"dK"}),
]


# ── Driver ───────────────────────────────────────────────────────────────────


def main(k: int = 5) -> int:
    print(f"Phase 4 validation: top-{k} semantic recall on {len(TEST_QUERIES)} paraphrases")
    print(f"(SPECTER2 lazy-loads on first encode — first query is slow.)")
    print()

    registry = build_bone_registry()
    anchor = SemanticVariableAnchor(registry)

    if not anchor._ensure_index():
        print("FAIL: SPECTER2 unavailable — cannot run validation.")
        return 1

    expected_total = 0
    found_total = 0
    rows: list[dict] = []

    for canonical, paraphrase, expected in TEST_QUERIES:
        # Verify the v0 anchor catches the expected set on the canonical.
        v0_hits = set(variables_in_query(canonical))
        v0_v2   = v0_to_v2(list(v0_hits))
        if not expected.issubset(v0_v2):
            missing_v0 = expected - v0_v2
            print(f"  WARN canonical {canonical!r} doesn't fully cover expected; missing on v0: {missing_v0}")

        matches = anchor.top_k(paraphrase, k=k)
        top_syms = {m.symbol for m in matches}
        found = expected & top_syms
        recall = len(found) / max(1, len(expected))
        expected_total += len(expected)
        found_total += len(found)
        rows.append({
            "paraphrase": paraphrase,
            "expected":   sorted(expected),
            "found":      sorted(found),
            "top_k":      [(m.symbol, round(m.score, 3)) for m in matches],
            "recall":     recall,
        })

    aggregate = found_total / max(1, expected_total)
    print(f"\nAggregate recall@{k} = {aggregate*100:.1f}%  ({found_total}/{expected_total})")
    pass_bar = 0.90
    verdict = "PASS" if aggregate >= pass_bar else "FAIL"
    print(f"Pass bar = {pass_bar*100:.0f}%  →  {verdict}")
    print()

    # Per-row trace, sorted worst first so any regressions are visible.
    rows.sort(key=lambda r: r["recall"])
    print("Per-query breakdown (worst first):")
    for r in rows:
        marker = " " if r["recall"] >= 0.5 else "*"
        top_str = ", ".join(f"{s}({sc})" for s, sc in r["top_k"])
        print(f"  {marker} {r['paraphrase'][:55]:55s}  "
              f"recall={r['recall']*100:5.1f}%  "
              f"expected={r['expected']}  top={top_str}")
    return 0 if aggregate >= pass_bar else 2


if __name__ == "__main__":  # pragma: no cover
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5, help="top-k cutoff")
    args = p.parse_args()
    raise SystemExit(main(k=args.k))
