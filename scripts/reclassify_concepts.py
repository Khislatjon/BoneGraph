"""
scripts/reclassify_concepts.py
==============================
Rule-based reclassifier for `concept`-typed nodes in ontology.db.

The extractor LLM defaulted ~86% of nodes to the catch-all `concept` type,
which breaks every type-aware downstream query. This script applies a curated
set of regex rules + hand-overrides for high-leverage nodes, then writes the
result to a separate database file so the original is preserved.

Stages:
  1. Apply EXACT_OVERRIDES for nodes the rules would miss or get wrong.
  2. Apply RULES in priority order; first match wins.
  3. Anything still unmatched stays as `concept`.

Run::
    python scripts/reclassify_concepts.py --dry-run   # preview only
    python scripts/reclassify_concepts.py             # writes data/db/ontology_typed.db
    python scripts/reclassify_concepts.py --output data/db/foo.db
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_INPUT  = Path("data/db/ontology.db")
DEFAULT_OUTPUT = Path("data/db/ontology_typed.db")

# ── Bare-word exact matches (no underscore = suffix rules can't fire) ────────
BARE_WORDS: dict[str, str] = {
    # process
    "osseointegration": "process",
    "physical_stimulation": "process",
    "fall": "process",
    "falls": "process",
    "phagocytosis": "process",
    "endocytosis": "process",
    # property
    "biocompatibility": "property",
    "biodegradability": "property",
    "operability": "property",
    "solubility": "property",
    "porosity": "property",
    "permeability": "property",
    "wettability": "property",
    "crystallinity": "property",
    "gender": "property",
    "sex":    "property",
    "age":    "property",
    # pathology
    "fractures": "pathology",
    "fracture":  "pathology",
    "immune_reaction":     "pathology",
    "disease_transmission": "pathology",
    # clinical
    "fall_prevention": "clinical",
    "suvmax":          "clinical",
    # structure
    "connective_tissue": "structure",
    # factor (recombinant proteins prefixed with rh)
    "rhbmp_2": "factor",
    "rhbmp_7": "factor",
    "rhpth":   "factor",
    "elastin": "factor",
    "collagen": "factor",
    # pathology (bare-word outcomes)
    "disability": "pathology",
    "mortality":  "pathology",
    "immobility": "pathology",
    "inflammation": "process",
    # process
    "strain":     "process",
    # property
    "flexibility":  "property",
    "brittleness":  "property",
    "rib_mobility": "property",
    "vertebral_column_curvature": "property",
    # structure
    "dermis":           "structure",
    "axial_skeleton":   "structure",
    "splanchnocranium": "structure",
}

# ── Hand-curated overrides ────────────────────────────────────────────────────
# Used for high-leverage nodes whose names rules can't classify correctly.
EXACT_OVERRIDES: dict[str, str] = {
    # behavioural / dietary risk factors
    "physical_activity":         "factor",
    "smoking":                   "factor",
    "alcohol_consumption":       "factor",
    "calcium_intake":            "factor",
    "vitamin_d_intake":          "factor",
    "diet":                      "factor",
    "exercise":                  "factor",
    "jumping_exercise":          "factor",
    # mechanical stimuli — processes in the mechanotransduction chain
    "mechanical_load":           "process",
    "mechanical_stress":         "process",
    "mechanical_loading":        "process",
    "mechanical_strain":         "process",
    "mechanical_stimulation":    "process",
    "loading":                   "process",
    "unloading":                 "process",
    "disuse":                    "process",
    # measurements / states
    "temperature":               "property",
    "ph":                        "property",
    # pathology
    "peri_implantitis":          "pathology",
    # clinical
    "treatment_gap":             "clinical",
    "biochemical_markers":       "clinical",
    "fracture_liaison_service":  "clinical",
    # materials
    "hydroxyapatite":            "material",
    # cell types (plurals or abbrevs the seed didn't include)
    "osteoclasts":               "cell",
    "osteoblasts":               "cell",
    "osteocytes":                "cell",
    "mscs":                      "cell",
    "bmscs":                     "cell",
    "hscs":                      "cell",
    "chondrocytes":              "cell",
    "adipocytes":                "cell",
    "macrophages":               "cell",
    "monocytes":                 "cell",
    "fibroblasts":               "cell",
}


def _re(p: str) -> re.Pattern[str]:
    return re.compile(p)


# ── Pattern rules — first match wins ──────────────────────────────────────────
# Order matters: more specific rules go first.
RULES: list[tuple[re.Pattern[str], str]] = [
    # ── factor: bone-specific proteins / receptors (exact names) ─────────────
    (_re(r"^(rankl|rank|opg|sost|sclerostin|runx2|osterix|sp7|dkk1|dkk2|"
         r"lrp5|lrp6|ctsk|trap|alp|ocn|bglap|col1a1|col1a2|p1np|ctx|ntx|"
         r"fgf23|fgfr3|klotho|osteopontin|opn|osteocalcin)$"),         "factor"),
    # ── factor: signalling-family genes / proteins (family + suffix) ─────────
    (_re(r"^(mir|let|lin)_[a-z0-9_]+$"),                               "factor"),
    (_re(r"^(il|tnf|tgf|tgfb|bmp|wnt|fgf|fgfr|igf|igfbp|vegf|pdgf|csf|"
         r"mcsf|ifn|mmp|timp|cxcl|ccl|cxcr|ccr|adam|adamts|integrin)_"
         r"[a-z0-9_]+$"),                                              "factor"),
    # ── factor: drugs by suffix ──────────────────────────────────────────────
    (_re(r"^[a-z]+mab$"),                                              "factor"),  # -mab monoclonals
    (_re(r"^[a-z]+nib$"),                                              "factor"),  # -nib kinase inhibitors
    (_re(r"^[a-z]+fene$"),                                             "factor"),  # raloxifene, bazedoxifene
    (_re(r"^[a-z]+parin$"),                                            "factor"),  # heparin family
    (_re(r"^[a-z]+dronate$"),                                          "factor"),  # bisphosphonates
    (_re(r"^[a-z]+dronic_acid$"),                                      "factor"),
    (_re(r"^bisphosphonates?$"),                                       "factor"),
    (_re(r"^[a-z]+paratide$"),                                         "factor"),  # teriparatide, abaloparatide
    (_re(r"^(denosumab|romosozumab|odanacatib|strontium_ranelate|"
         r"calcitonin|raloxifene|bazedoxifene|teriparatide|abaloparatide)$"), "factor"),
    # ── factor: hormones / vitamins / minerals ───────────────────────────────
    (_re(r"^(estrogen|estrogens|androgen|androgens|testosterone|progesterone|"
         r"cortisol|glucocorticoid|glucocorticoids)$"),                "factor"),
    (_re(r"^(parathyroid_hormone|pth|calcitriol|calcifediol)$"),       "factor"),
    (_re(r"^vitamin_[a-z0-9]+$"),                                      "factor"),
    (_re(r"^(calcium|phosphate|magnesium|zinc|fluoride|strontium|"
         r"silicon|boron)$"),                                          "factor"),
    (_re(r"^anti_[a-z0-9_]+$"),                                        "factor"),  # anti-RANKL, anti-sclerostin
    (_re(r"^.*_factors$"),                                             "factor"),  # growth_factors etc.
    (_re(r"^.*_ions$"),                                                "factor"),
    (_re(r"^.*_lifestyle$"),                                           "factor"),  # sedentary_lifestyle
    # ── cell ─────────────────────────────────────────────────────────────────
    (_re(r"^.*(blasts|clasts|cytes)$"),                                "cell"),
    (_re(r"^.*_cells?$"),                                              "cell"),
    # ── pathology ────────────────────────────────────────────────────────────
    (_re(r"^.*itis$"),                                                 "pathology"),
    (_re(r"^.*_fractures?$"),                                          "pathology"),
    (_re(r"^.*_fractions$"),                                           "pathology"),  # LLM typo for "fractures"
    (_re(r"^.*_(deficiency|insufficiency)$"),                          "pathology"),
    (_re(r"^.*_(pain|injury|complications?|disease|syndrome|"
         r"occurrence|disability|mortality)$"),                        "pathology"),
    (_re(r"^(osteoporosis|osteopenia|osteomalacia|osteonecrosis|"
         r"rickets|paget_disease|fibrous_dysplasia|sarcopenia|"
         r"nonunion|malunion)$"),                                      "pathology"),
    # ── process ──────────────────────────────────────────────────────────────
    (_re(r"^.*_(activity|activation|expression|proliferation|differentiation|"
         r"migration|adhesion|polarization|secretion|apoptosis|autophagy|"
         r"senescence|fusion|invasion|recruitment|maturation|aging)$"), "process"),
    (_re(r"^.*_(resorption|formation|remodeling|remodelling|mineralization|"
         r"calcification|ossification|osteogenesis|chondrogenesis|"
         r"angiogenesis|vasculogenesis|osteoclastogenesis|osteoblastogenesis|"
         r"osseointegration|integration|ingrowth)$"),                  "process"),
    (_re(r"^.*_(synthesis|degradation|absorption|excretion|uptake|release|"
         r"production|metabolism|catabolism|anabolism|homeostasis|"
         r"biosynthesis|phosphorylation|binding|interaction|response|"
         r"inflammation|growth|death|survival|aging|ageing)$"),        "process"),
    (_re(r"^.*_(signaling|signalling|crosstalk)$"),                    "process"),
    (_re(r"^.*genesis$"),                                              "process"),
    (_re(r"^.*_(loss|gain|turnover|repair|healing|regeneration)$"),    "process"),
    (_re(r"^.*_(stress|pressure|force|forces|loading)$"),              "process"),
    # ── mechanism ────────────────────────────────────────────────────────────
    (_re(r"^.*_pathways?$"),                                           "mechanism"),
    # ── property ─────────────────────────────────────────────────────────────
    (_re(r"^.*_(risk|probability|score|index|ratio|burden|prevalence)$"), "property"),
    (_re(r"^.*_(properties|property|resistance|strength|stiffness|"
         r"toughness|modulus|hardness|elasticity|viscoelasticity|"
         r"stability|solubility|roughness|accuracy|intensity|"
         r"performance|ability|biocompatibility)$"),                   "property"),
    (_re(r"^.*_(density|porosity|thickness|volume|area|length|width|"
         r"diameter|height|fraction|content|concentration|level|count|rate|"
         r"size|mass|age)$"),                                          "property"),
    (_re(r"^.*_bmd$"),                                                 "property"),
    (_re(r"^.*_bmc$"),                                                 "property"),
    (_re(r"^.*_temperature$"),                                         "property"),
    # ── factor: dietary / behavioural exposures (suffix-based) ───────────────
    (_re(r"^.*_(intake|consumption|use)$"),                            "factor"),
    # ── clinical ─────────────────────────────────────────────────────────────
    (_re(r"^.*_(treatment|therapy|intervention|prophylaxis|"
         r"supplementation|training)$"),                               "clinical"),
    (_re(r"^.*_(service|program|guideline|protocol|assessment|"
         r"questionnaire|marker|markers|history|diagnosis|costs?)$"),  "clinical"),
    (_re(r"^(frax|dxa|dexa|qct|mri|hr_pqct|hrct)$"),                   "clinical"),
    (_re(r"^serum_[a-z0-9_]+$"),                                       "clinical"),  # serum_calcium, serum_phosphate
    (_re(r"^blood_[a-z0-9_]+$"),                                       "clinical"),
    # ── structure ────────────────────────────────────────────────────────────
    (_re(r"^(cortical|trabecular|cancellous|periosteum|endosteum|"
         r"marrow|lacuna|canaliculi|osteon|haversian_canal)$"),        "structure"),
    (_re(r"^.*_(bone|cortex|trabeculae|pores)$"),                      "structure"),
    (_re(r"^(vertebra|vertebrae|vertebral|femur|femoral|tibia|tibial|"
         r"humerus|hip|spine|lumbar|thoracic|cervical|radius|ulna|"
         r"wrist|forearm|calcaneus|skull|cranium|mandible|maxilla|rib|"
         r"ribs|pelvis|pelvic|skeleton|skeletal_system|"
         r"vertebral_column|dentine|dentin|enamel|cartilage)$"),       "structure"),
    (_re(r"^.*_structure$"),                                           "structure"),
    # ── material ─────────────────────────────────────────────────────────────
    (_re(r"^.*(hydrogel|hydrogels|scaffold|scaffolds|cement|nanoparticle|"
         r"nanoparticles|composite|composites|biomaterial|biomaterials|"
         r"cellulose|polymer|polymers|peek|titanium|ceramics?)$"),     "material"),
    (_re(r"^.*calcium_phosphate$"),                                    "material"),
    (_re(r"^(hyaluronic_acid|bcp|tcp|beta_tcp|plla|pcl|pga|plga|cmc|"
         r"alginate|chitosan|gelatin|fibrin|bioglass)$"),              "material"),
]


def classify(node_id: str) -> str | None:
    """Return new node_type for a concept-typed node, or None to leave it."""
    if node_id in EXACT_OVERRIDES:
        return EXACT_OVERRIDES[node_id]
    if node_id in BARE_WORDS:
        return BARE_WORDS[node_id]
    for pat, ntype in RULES:
        if pat.match(node_id):
            return ntype
    return None


def reclassify(input_db: Path, output_db: Path, dry_run: bool = False) -> None:
    if not input_db.exists():
        raise FileNotFoundError(input_db)

    if not dry_run:
        output_db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_db, output_db)
        target = output_db
    else:
        target = input_db

    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row

    concept_ids = [
        r["node_id"]
        for r in conn.execute(
            "SELECT node_id FROM nodes WHERE node_type='concept'"
        )
    ]

    decisions: dict[str, str] = {}
    for nid in concept_ids:
        new = classify(nid)
        if new:
            decisions[nid] = new

    bucket = Counter(decisions.values())
    print(f"\nconcept-typed nodes:    {len(concept_ids)}")
    print(f"reclassified:           {len(decisions)} "
          f"({len(decisions) / len(concept_ids) * 100:.1f}%)")
    print(f"remain as 'concept':    {len(concept_ids) - len(decisions)}")

    print("\nNew types assigned:")
    for t, n in bucket.most_common():
        print(f"  {t:12s} {n}")

    by_type: dict[str, list[str]] = {}
    for nid, t in decisions.items():
        by_type.setdefault(t, []).append(nid)

    print("\nExamples per assigned type (first 8):")
    for t in sorted(by_type):
        ex = by_type[t][:8]
        print(f"  {t:12s} {', '.join(ex)}")

    leftover = [nid for nid in concept_ids if nid not in decisions]
    print(f"\nSample of {min(20, len(leftover))} nodes still typed 'concept':")
    for nid in leftover[:20]:
        print(f"  {nid}")

    if dry_run:
        print("\n[dry-run] no changes written.")
        conn.close()
        return

    with conn:
        for nid, t in decisions.items():
            conn.execute(
                "UPDATE nodes SET node_type=? WHERE node_id=?", (t, nid)
            )

    print(f"\nFinal node_type histogram in {output_db}:")
    rows = conn.execute(
        "SELECT node_type, COUNT(*) c FROM nodes "
        "GROUP BY node_type ORDER BY c DESC"
    ).fetchall()
    for r in rows:
        print(f"  {r[0]:12s} {r[1]}")
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    reclassify(args.input, args.output, args.dry_run)


if __name__ == "__main__":
    main()
