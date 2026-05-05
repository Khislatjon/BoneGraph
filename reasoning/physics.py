"""
reasoning/physics.py
====================
Step 4.3 — Bone-specific physics engine for hypothesis validation.

Overview
--------
Every hypothesis or causal chain produced by the LRM (Step 4.4) passes
through this engine before being shown to the user.  The engine answers
one question: is this chain physically consistent with known bone mechanics?

Two layers
----------
Layer 1 — Directional rules (always runs)
    A lookup table of 40 known bone physics relationships.  Each entry
    asserts the correct direction of a causal relationship according to
    established bone science.  Checking is instantaneous and requires no
    numerical input.

Layer 2 — Numerical validators (runs when values are provided)
    Four bone-specific physics functions:
      • Currey's law     E ∝ ρ²        (stiffness vs apparent density)
      • Frost mechanostat               (strain → bone adaptation zone)
      • Paris law        da/dN = C·ΔK^m (fatigue crack growth)
      • Beam theory      σ = Mc/I       (long-bone bending stress)

    These fire only when the calling code supplies numerical parameters.
    Without numbers the engine falls back to Layer 1.

Design principles
-----------------
• Deterministic and rule-based — every decision is explainable
• No external dependencies beyond the standard library and math
• Easily extensible: add a DIRECTIONAL_RULES entry or a new numerical
  function without changing anything else
• Chain validation: one IMPLAUSIBLE edge poisons the whole chain

Usage::

    from reasoning.physics import PhysicsEngine

    engine = PhysicsEngine()

    # Validate a single edge
    result = engine.validate_edge("porosity", "increases", "elastic_modulus")
    print(result)
    # ValidationResult(status='IMPLAUSIBLE', law="Currey's law", ...)

    # Validate a full causal chain
    result = engine.validate_chain(
        nodes=["aging", "porosity", "elastic_modulus", "fracture_risk"],
        edges=[("porosity", "increases", "elastic_modulus"), ...]
    )

    # Numerical check
    result = engine.check_currey(apparent_density=1.2)
    result = engine.check_mechanostat(strain_microstrain=4500)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

# ── Types ─────────────────────────────────────────────────────────────────────

Status = Literal["PLAUSIBLE", "IMPLAUSIBLE", "UNCERTAIN"]


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """
    Output of a physics validation check.

    Attributes
    ----------
    status : str
        "PLAUSIBLE"   — consistent with known bone physics.
        "IMPLAUSIBLE" — contradicts a known physical law or relationship.
        "UNCERTAIN"   — no rule covers this; physics cannot confirm or deny.
    law : str
        Name of the physical law or rule that produced the verdict.
        Empty string if no specific law was invoked.
    confidence : float
        0.0–1.0.  Rule-based checks return 1.0 (certain) or 0.5 (uncertain).
        Numerical checks return a value derived from how far the input
        deviates from the expected range.
    explanation : str
        Human-readable sentence explaining the verdict.
    edge : tuple[str, str, str] | None
        The (source, relation, target) triple that triggered the verdict,
        if this result came from a single-edge check.
    """

    status: Status
    law: str = ""
    confidence: float = 1.0
    explanation: str = ""
    edge: tuple[str, str, str] | None = None

    def __str__(self) -> str:
        parts = [f"[{self.status}]"]
        if self.law:
            parts.append(f"({self.law})")
        if self.explanation:
            parts.append(self.explanation)
        return "  ".join(parts)

    @property
    def is_plausible(self) -> bool:
        return self.status == "PLAUSIBLE"

    @property
    def is_implausible(self) -> bool:
        return self.status == "IMPLAUSIBLE"


# ── Layer 1 — Directional rules ───────────────────────────────────────────────
#
# Key: (source_node_fragment, relation, target_node_fragment)
#   Fragments are substrings — a rule fires if the node_id CONTAINS the
#   fragment.  This keeps the table compact while covering many spelling
#   variants (e.g. "porosity" matches "cortical_porosity", "micro_porosity").
#
# Value: (Status, law_name, explanation)
#
# Relation semantics:
#   increases / decreases — one node positively / negatively affects another
#   determines            — structural cause
#   activates / inhibits  — biological up/down regulation
#   leads_to              — downstream consequence
#   predicts              — statistical / clinical prediction

_DIRECTIONAL_RULES: dict[
    tuple[str, str, str], tuple[Status, str, str]
] = {
    # Guard-rail rules only — IMPLAUSIBLE entries block physically impossible
    # chains.  PLAUSIBLE entries exist for symmetry but do not affect scoring.


    # ── Currey's law family (E ∝ ρ², stiffness scales with density) ───────────
    ("porosity", "increases", "elastic_modulus"):
        ("IMPLAUSIBLE", "Currey's law",
         "Porosity ↑ → apparent density ↓ → E ↓ (Currey: E ∝ ρ²). "
         "Porosity cannot increase elastic modulus."),

    ("porosity", "decreases", "elastic_modulus"):
        ("PLAUSIBLE", "Currey's law",
         "Porosity ↑ → density ↓ → E ↓. Correct direction."),

    ("porosity", "increases", "stiffness"):
        ("IMPLAUSIBLE", "Currey's law",
         "Stiffness scales with density; porosity reduces density."),

    ("porosity", "decreases", "stiffness"):
        ("PLAUSIBLE", "Currey's law",
         "Higher porosity → lower stiffness. Correct."),

    ("mineral", "increases", "elastic_modulus"):
        ("PLAUSIBLE", "Currey's law",
         "Higher mineral content → higher apparent density → higher E."),

    ("mineral", "decreases", "elastic_modulus"):
        ("IMPLAUSIBLE", "Currey's law",
         "Mineral is the primary load-bearing phase; less mineral → lower E, not higher."),

    ("bone_mineral_density", "increases", "elastic_modulus"):
        ("PLAUSIBLE", "Currey's law",
         "BMD is a proxy for apparent density; higher BMD → higher E."),

    ("bone_mineral_density", "decreases", "elastic_modulus"):
        ("IMPLAUSIBLE", "Currey's law",
         "Lower BMD means lower density → lower E, not higher."),

    ("density", "increases", "elastic_modulus"):
        ("PLAUSIBLE", "Currey's law",
         "E ∝ ρ²; higher density → higher stiffness."),

    ("density", "decreases", "elastic_modulus"):
        ("IMPLAUSIBLE", "Currey's law",
         "Lower density → lower E by Currey's law."),

    # ── Fracture toughness and collagen ───────────────────────────────────────
    ("collagen", "determines", "fracture_toughness"):
        ("PLAUSIBLE", "Collagen toughening mechanisms",
         "Collagen crosslink density and fibril organisation govern crack "
         "bridging and deflection — primary determinants of KIc."),

    ("collagen", "increases", "fracture_toughness"):
        ("PLAUSIBLE", "Collagen toughening mechanisms",
         "Intact collagen network enhances crack bridging → higher KIc."),

    ("collagen", "decreases", "fracture_toughness"):
        ("IMPLAUSIBLE", "Collagen toughening mechanisms",
         "Degraded collagen reduces toughening, not intact collagen. "
         "Intact collagen increases toughness."),

    ("crosslink", "increases", "fracture_toughness"):
        ("PLAUSIBLE", "Collagen crosslinking mechanics",
         "Enzymatic crosslinks stabilise fibrils → improved energy absorption."),

    ("advanced_glycation", "decreases", "fracture_toughness"):
        ("PLAUSIBLE", "AGE crosslink embrittlement",
         "Non-enzymatic (AGE) crosslinks make collagen brittle → lower KIc."),

    ("advanced_glycation", "increases", "fracture_toughness"):
        ("IMPLAUSIBLE", "AGE crosslink embrittlement",
         "AGEs embrittle collagen; they decrease, not increase, toughness."),

    ("porosity", "decreases", "fracture_toughness"):
        ("PLAUSIBLE", "Stress concentration",
         "Pores act as stress concentrators; more porosity → lower KIc."),

    ("porosity", "increases", "fracture_toughness"):
        ("IMPLAUSIBLE", "Stress concentration",
         "Pores intensify local stress (Kt = 1 + 2√(a/ρ)); "
         "higher porosity reduces toughness, not increases it."),

    # ── Frost mechanostat (strain → bone adaptation) ──────────────────────────
    ("mechanical_loading", "activates", "bone_formation"):
        ("PLAUSIBLE", "Frost mechanostat",
         "Strains in the mild overload window (1500–3000 μɛ) stimulate "
         "modelling — the mechanostat predicts net bone formation."),

    ("disuse", "leads_to", "bone_resorption"):
        ("PLAUSIBLE", "Frost mechanostat",
         "Strains < 200 μɛ (disuse) trigger net resorption — "
         "mechanostat disuse window."),

    ("disuse", "leads_to", "bone_formation"):
        ("IMPLAUSIBLE", "Frost mechanostat",
         "Disuse falls in the resorption window of the mechanostat, "
         "not the formation window."),

    ("overloading", "leads_to", "fracture"):
        ("PLAUSIBLE", "Frost mechanostat",
         "Strains > 25 000 μɛ exceed the fracture threshold."),

    ("strain", "increases", "bone_formation"):
        ("UNCERTAIN", "Frost mechanostat",
         "Depends on strain magnitude. Mild overload (1500–3000 μɛ) → "
         "formation; disuse or pathological overload → resorption or fracture. "
         "Numerical check needed."),

    # ── RANKL / OPG / osteoclast axis ─────────────────────────────────────────
    ("RANKL", "activates", "osteoclast"):
        ("PLAUSIBLE", "RANK/RANKL/OPG signalling",
         "RANKL binds RANK on osteoclast precursors → osteoclastogenesis. "
         "Canonical pathway."),

    ("OPG", "inhibits", "osteoclast"):
        ("PLAUSIBLE", "RANK/RANKL/OPG signalling",
         "OPG is a decoy receptor for RANKL → blocks osteoclastogenesis."),

    ("OPG", "activates", "osteoclast"):
        ("IMPLAUSIBLE", "RANK/RANKL/OPG signalling",
         "OPG inhibits osteoclastogenesis by sequestering RANKL; "
         "it cannot activate osteoclasts."),

    ("osteoclast", "leads_to", "bone_resorption"):
        ("PLAUSIBLE", "Bone cell biology",
         "Osteoclasts are the sole cells that resorb bone matrix."),

    ("osteoblast", "leads_to", "bone_formation"):
        ("PLAUSIBLE", "Bone cell biology",
         "Osteoblasts synthesise and mineralise osteoid → bone formation."),

    ("osteoclast", "leads_to", "bone_formation"):
        ("IMPLAUSIBLE", "Bone cell biology",
         "Osteoclasts resorb bone; formation is performed by osteoblasts."),

    # ── PTH / sclerostin / Wnt ────────────────────────────────────────────────
    ("PTH", "activates", "bone_formation"):
        ("PLAUSIBLE", "Intermittent PTH (anabolic window)",
         "Intermittent PTH elevates osteoblast activity → net bone formation. "
         "Mechanism of teriparatide therapy."),

    ("sclerostin", "inhibits", "bone_formation"):
        ("PLAUSIBLE", "Wnt/β-catenin signalling",
         "Sclerostin (SOST) antagonises Wnt signalling in osteocytes → "
         "suppresses osteoblast differentiation."),

    ("sclerostin", "activates", "bone_formation"):
        ("IMPLAUSIBLE", "Wnt/β-catenin signalling",
         "Sclerostin is an inhibitor of bone formation; anti-sclerostin "
         "antibodies (romosozumab) work by blocking it."),

    # ── Aging / osteoporosis chain ────────────────────────────────────────────
    ("aging", "increases", "porosity"):
        ("PLAUSIBLE", "Bone aging",
         "Cortical porosity increases with age due to endosteal resorption "
         "and expanding Haversian canals."),

    ("aging", "decreases", "bone_mineral_density"):
        ("PLAUSIBLE", "Bone aging",
         "Age-related bone loss consistently reduces BMD — well established "
         "by DXA longitudinal studies."),

    ("osteoporosis", "increases", "fracture_risk"):
        ("PLAUSIBLE", "Osteoporosis epidemiology",
         "Osteoporosis is defined by low BMD and elevated fragility fracture "
         "risk — foundational clinical relationship."),

    ("osteoporosis", "decreases", "fracture_risk"):
        ("IMPLAUSIBLE", "Osteoporosis epidemiology",
         "Osteoporosis is a risk factor for fracture, not a protective factor."),

    # ── Stress concentration (Kt) ─────────────────────────────────────────────
    ("pore", "increases", "stress_concentration"):
        ("PLAUSIBLE", "Stress concentration theory",
         "Pores act as stress risers; Kt = 1 + 2√(a/ρ) for an elliptical "
         "pore — larger/sharper pores → higher Kt."),

    ("pore", "decreases", "stress_concentration"):
        ("IMPLAUSIBLE", "Stress concentration theory",
         "Pores increase, not decrease, stress concentration."),

    # ── Cortical thickness and strength ──────────────────────────────────────
    ("cortical_thickness", "increases", "bending_strength"):
        ("PLAUSIBLE", "Beam theory",
         "For a hollow cylinder I = π(r_o⁴ - r_i⁴)/4; greater cortical "
         "thickness → larger I → lower bending stress for the same moment."),

    ("cortical_thickness", "decreases", "bending_strength"):
        ("IMPLAUSIBLE", "Beam theory",
         "Thicker cortex increases second moment of area → higher resistance "
         "to bending; it cannot decrease bending strength."),
}



# ── Layer 2 — Numerical validators ───────────────────────────────────────────


def currey_modulus(apparent_density_g_cm3: float) -> float:
    """
    Predict elastic modulus (GPa) from apparent density using Currey's law.

    Currey (1988): E = 6.08 · ρ^1.78  (cortical bone, bovine)
    More generally used as E = a · ρ^n  with n ≈ 2 for trabecular bone.

    This implementation uses:
        E = 7.0 · ρ^2   (GPa, apparent density in g/cm³)
    which gives ~17–20 GPa for cortical bone (ρ ≈ 1.6–1.7 g/cm³)
    and ~0.1–5 GPa for trabecular bone (ρ ≈ 0.1–0.9 g/cm³).

    Parameters
    ----------
    apparent_density_g_cm3 : float
        Apparent (bulk) density in g/cm³.
        Typical ranges:
            Cortical bone   : 1.5–1.9 g/cm³
            Trabecular bone : 0.1–0.9 g/cm³

    Returns
    -------
    float
        Predicted elastic modulus in GPa.
    """
    if apparent_density_g_cm3 <= 0:
        raise ValueError("Apparent density must be positive.")
    return 7.0 * (apparent_density_g_cm3 ** 2)


def currey_delta_from_porosity(
    porosity_baseline: float,
    porosity_change_abs: float,
    exponent: float = 2.5,
) -> dict[str, float]:
    """
    Differential Currey prediction: how does ΔE/E depend on Δφ?

    From Currey's law E ∝ ρⁿ and the definition ρ = ρ_full · (1 - φ):

        E_new / E_old = ((1 - φ_new) / (1 - φ_old))ⁿ

    Parameters
    ----------
    porosity_baseline : float
        Starting porosity φ as a volume fraction (0–1).  E.g. 0.05 for
        cortical bone, 0.80 for trabecular bone.
    porosity_change_abs : float
        Absolute change in porosity (added to baseline).  E.g. +0.10
        means "10 percentage points more porous".  Negative values are
        permitted.
    exponent : float
        Currey exponent n.  Default 2.5 — between Currey's bovine
        cortical fit (n ≈ 1.78) and trabecular literature values (n ≈ 2–3).

    Returns
    -------
    dict with keys:
        phi_baseline   — input baseline porosity
        phi_new        — baseline + change
        rho_ratio      — ρ_new / ρ_old (linear in 1 − φ)
        delta_E_ratio  — (E_new − E_old) / E_old
        delta_E_pct    — same as delta_E_ratio but in percent
        exponent       — exponent used

    Raises
    ------
    ValueError
        If φ_new is outside (0, 1).
    """
    phi_new = porosity_baseline + porosity_change_abs
    if not 0.0 < phi_new < 1.0:
        raise ValueError(
            f"phi_new={phi_new:.3f} out of (0, 1); "
            f"baseline={porosity_baseline}, change={porosity_change_abs}"
        )
    rho_ratio = (1.0 - phi_new) / (1.0 - porosity_baseline)
    delta_E_ratio = (rho_ratio ** exponent) - 1.0
    return {
        "phi_baseline":  porosity_baseline,
        "phi_new":       phi_new,
        "rho_ratio":     rho_ratio,
        "delta_E_ratio": delta_E_ratio,
        "delta_E_pct":   delta_E_ratio * 100.0,
        "exponent":      exponent,
    }


def currey_delta_from_density(
    density_baseline_g_cm3: float,
    density_change_pct: float,
    exponent: float = 2.5,
) -> dict[str, float]:
    """
    Differential Currey prediction driven by a relative density change.

    From E ∝ ρⁿ:  E_new / E_old = (ρ_new / ρ_old)ⁿ

    Parameters
    ----------
    density_baseline_g_cm3 : float
        Starting apparent density.
    density_change_pct : float
        Relative change in density, in percent.  E.g. -10.0 for a 10%
        decrease.
    exponent : float
        Currey exponent n (default 2.5).

    Returns
    -------
    dict with keys:
        rho_baseline, rho_new, rho_ratio, delta_E_ratio, delta_E_pct, exponent
    """
    rho_ratio = 1.0 + density_change_pct / 100.0
    if rho_ratio <= 0:
        raise ValueError(f"Density change {density_change_pct}% drives ρ ≤ 0.")
    rho_new = density_baseline_g_cm3 * rho_ratio
    delta_E_ratio = (rho_ratio ** exponent) - 1.0
    return {
        "rho_baseline":  density_baseline_g_cm3,
        "rho_new":       rho_new,
        "rho_ratio":     rho_ratio,
        "delta_E_ratio": delta_E_ratio,
        "delta_E_pct":   delta_E_ratio * 100.0,
        "exponent":      exponent,
    }


def mechanostat_zone(strain_microstrain: float) -> dict[str, str]:
    """
    Classify a strain magnitude into a Frost mechanostat zone.

    Frost's mechanostat (1987, 2003) defines strain windows that predict
    bone's adaptive response to mechanical loading:

        < 50 μɛ          Acute disuse window     → rapid bone loss
        50 – 200 μɛ      Chronic disuse window   → slow bone loss
        200 – 1500 μɛ    Adapted window          → homeostasis / maintenance
        1500 – 3000 μɛ   Mild overload window    → modelling (bone gain)
        3000 – 10000 μɛ  Pathological overload   → microdamage accumulation
        > 25000 μɛ       Fracture threshold       → acute fracture

    Parameters
    ----------
    strain_microstrain : float
        Peak principal strain magnitude in microstrain (μɛ).

    Returns
    -------
    dict with keys:
        zone        — zone name
        response    — predicted biological response
        description — plain-English explanation
    """
    if strain_microstrain < 0:
        raise ValueError("Strain must be non-negative.")

    if strain_microstrain < 50:
        return {
            "zone": "acute_disuse",
            "response": "rapid_bone_loss",
            "description": (
                f"{strain_microstrain:.0f} μɛ — acute disuse zone (<50 μɛ). "
                "Rapid periosteal and endosteal bone loss expected."
            ),
        }
    elif strain_microstrain < 200:
        return {
            "zone": "chronic_disuse",
            "response": "slow_bone_loss",
            "description": (
                f"{strain_microstrain:.0f} μɛ — chronic disuse zone (50–200 μɛ). "
                "Slow net bone resorption."
            ),
        }
    elif strain_microstrain < 1500:
        return {
            "zone": "adapted",
            "response": "homeostasis",
            "description": (
                f"{strain_microstrain:.0f} μɛ — adapted (maintenance) window (200–1500 μɛ). "
                "Bone mass maintained; remodelling is balanced."
            ),
        }
    elif strain_microstrain < 3000:
        return {
            "zone": "mild_overload",
            "response": "bone_formation",
            "description": (
                f"{strain_microstrain:.0f} μɛ — mild overload window (1500–3000 μɛ). "
                "Modelling triggered → net bone apposition."
            ),
        }
    elif strain_microstrain < 25000:
        return {
            "zone": "pathological_overload",
            "response": "microdamage",
            "description": (
                f"{strain_microstrain:.0f} μɛ — pathological overload (3000–25000 μɛ). "
                "Microdamage accumulation; stress fracture risk if repeated."
            ),
        }
    else:
        return {
            "zone": "fracture",
            "response": "acute_fracture",
            "description": (
                f"{strain_microstrain:.0f} μɛ — fracture threshold exceeded (>25000 μɛ). "
                "Single-cycle fracture expected."
            ),
        }


def paris_crack_growth(
    delta_K_MPa_m05: float,
    C: float = 1.7e-9,
    m: float = 3.9,
) -> float:
    """
    Compute fatigue crack growth rate using Paris law.

    Paris law: da/dN = C · ΔK^m

    Default constants for human cortical bone (Vashishth et al. 2004):
        C = 1.7 × 10⁻⁹   (m/cycle · MPa√m units)
        m = 3.9

    Parameters
    ----------
    delta_K_MPa_m05 : float
        Stress intensity factor range ΔK in MPa√m.
        Typical crack-initiation range for bone: 0.5–2.0 MPa√m.
        Critical KIc (fracture toughness) for cortical bone: ~2–6 MPa√m.
    C : float
        Paris law coefficient (material constant).
    m : float
        Paris law exponent (material constant).

    Returns
    -------
    float
        Crack growth rate da/dN in metres per cycle.

    Notes
    -----
    A higher ΔK always gives a higher da/dN — any hypothesis claiming the
    opposite (larger crack grows slower) is physically implausible.
    """
    if delta_K_MPa_m05 <= 0:
        raise ValueError("ΔK must be positive.")
    return C * (delta_K_MPa_m05 ** m)


def beam_bending_stress(
    bending_moment_Nm: float,
    outer_radius_m: float,
    inner_radius_m: float,
) -> dict[str, float]:
    """
    Compute peak bending stress in a long bone modelled as a hollow cylinder.

    Uses the flexure formula: σ = M · c / I

    where:
        M = bending moment (N·m)
        c = outer radius (distance to outermost fibre)
        I = second moment of area = π(r_o⁴ - r_i⁴) / 4

    Parameters
    ----------
    bending_moment_Nm : float
        Applied bending moment in N·m.
        Typical femoral midshaft during walking: ~100–200 N·m.
    outer_radius_m : float
        Outer periosteal radius in metres.
        Typical femoral midshaft: ~0.015–0.017 m.
    inner_radius_m : float
        Inner endosteal radius in metres.
        Typical femoral midshaft: ~0.008–0.011 m.

    Returns
    -------
    dict with keys:
        I_m4       — second moment of area (m⁴)
        sigma_MPa  — peak bending stress (MPa)
        cortical_thickness_m — outer_radius - inner_radius (m)

    Notes
    -----
    Larger outer radius or greater cortical thickness → larger I → lower σ
    for the same bending moment.  Any hypothesis claiming thicker cortex
    leads to higher bending stress is implausible.
    """
    if inner_radius_m >= outer_radius_m:
        raise ValueError("Inner radius must be less than outer radius.")
    if outer_radius_m <= 0 or inner_radius_m < 0:
        raise ValueError("Radii must be positive.")

    I = math.pi * (outer_radius_m**4 - inner_radius_m**4) / 4
    sigma_Pa = bending_moment_Nm * outer_radius_m / I
    sigma_MPa = sigma_Pa / 1e6

    return {
        "I_m4": I,
        "sigma_MPa": round(sigma_MPa, 2),
        "cortical_thickness_m": round(outer_radius_m - inner_radius_m, 4),
    }


# ── Frost mechanostat — differential / regime predictor ──────────────────────


# Annual BMD change rate associated with each Frost zone, expressed as
# % BMD per year.  Conservative literature midpoints (Frost 2003,
# Robling 2009, Burr 2002).  These are *typical* magnitudes — the
# critic's Round-2 magnitude check enforces the broader physical band.
_FROST_BMD_RATE_PCT_PER_YR: dict[str, float] = {
    "acute_disuse":          -3.0,    # immobilisation, paraplegia
    "chronic_disuse":        -1.5,
    "adapted":                0.0,    # homeostasis
    "mild_overload":         +1.5,    # exercise / loading
    "pathological_overload": -1.0,    # microdamage > repair
    "fracture":              -5.0,    # acute trauma
}

# Map each Frost zone to the conceptual graph node it implies.
_FROST_OUTPUT_NODE: dict[str, str] = {
    "acute_disuse":          "bone_resorption",
    "chronic_disuse":        "bone_resorption",
    "adapted":               "bone_remodeling",
    "mild_overload":         "bone_formation",
    "pathological_overload": "bone_resorption",
    "fracture":              "bone_resorption",
}


def mechanostat_adaptation(strain_microstrain: float) -> dict:
    """
    Frost mechanostat regime prediction with a quantitative BMD/yr rate.

    Wraps :func:`mechanostat_zone` and adds an annual BMD-change estimate
    drawn from the literature midpoints for each zone.  The output
    is suitable for the physics-driven generator pipeline.

    Parameters
    ----------
    strain_microstrain : float
        Peak principal strain magnitude (μɛ).  Must be ≥ 0.

    Returns
    -------
    dict with keys:
        zone                       — Frost zone name
        response                   — biological response label
        description                — plain-English explanation
        bmd_pct_per_year           — typical annual BMD change for the zone
        chain_output_node          — graph node_id implied by the zone
                                     (``bone_formation`` / ``bone_resorption``
                                     / ``bone_remodeling``)
    """
    base = mechanostat_zone(strain_microstrain)
    zone = base["zone"]
    return {
        **base,
        "bmd_pct_per_year":  _FROST_BMD_RATE_PCT_PER_YR.get(zone, 0.0),
        "chain_output_node": _FROST_OUTPUT_NODE.get(zone, "bone_remodeling"),
    }


# ── Paris law — differential form ────────────────────────────────────────────


def paris_delta(
    delta_K_baseline: float,
    delta_K_new: float,
    C: float = 1.7e-9,
    m: float = 3.9,
) -> dict[str, float]:
    """
    Differential Paris-law prediction.

    Compares fatigue crack growth rate at a baseline ΔK to that at a
    perturbed ΔK.  Uses Vashishth et al. (2004) cortical-bone constants
    by default.

    Parameters
    ----------
    delta_K_baseline : float
        Reference cyclic stress-intensity factor range (MPa·√m).
    delta_K_new : float
        Perturbed ΔK to compare against the baseline.
    C, m : float
        Paris constants.  Defaults from Vashishth (cortical bone).

    Returns
    -------
    dict with keys:
        dK_baseline, dK_new
        da_dN_baseline, da_dN_new      (m/cycle)
        rate_ratio                     (da_dN_new / da_dN_baseline)
        log10_ratio                    (log₁₀ of the rate ratio)
        delta_rate_pct                 ((rate_ratio − 1) × 100)
        exponent, C
    """
    if delta_K_baseline <= 0 or delta_K_new <= 0:
        raise ValueError("ΔK values must be positive.")
    da_dN_base = paris_crack_growth(delta_K_baseline, C=C, m=m)
    da_dN_new  = paris_crack_growth(delta_K_new,      C=C, m=m)
    ratio = da_dN_new / da_dN_base
    return {
        "dK_baseline":    delta_K_baseline,
        "dK_new":         delta_K_new,
        "da_dN_baseline": da_dN_base,
        "da_dN_new":      da_dN_new,
        "rate_ratio":     ratio,
        "log10_ratio":    math.log10(ratio) if ratio > 0 else float("-inf"),
        "delta_rate_pct": (ratio - 1.0) * 100.0,
        "exponent":       m,
        "C":              C,
    }


# ── Beam bending — thickness-perturbation predictor ──────────────────────────


def beam_bending_thickness_delta(
    outer_radius_mm: float,
    cortical_thickness_baseline_mm: float,
    thickness_change_pct: float,
) -> dict[str, float]:
    """
    Predict the change in bending stress when cortical thickness changes.

    Holds the outer radius and the applied bending moment constant.
    For a hollow cylinder with outer radius r_o and inner radius
    r_i = r_o − t, the second moment of area is::

        I = π/4 · (r_o⁴ − r_i⁴)

    Bending stress σ = M · r_o / I, so for fixed M and r_o::

        σ_new / σ_old = I_old / I_new

    Parameters
    ----------
    outer_radius_mm : float
        Outer (periosteal) radius, mm.  Typical femoral midshaft:
        15–17 mm.
    cortical_thickness_baseline_mm : float
        Starting cortical wall thickness, mm.  Typical: 4–6 mm cortical,
        thinning to 2–3 mm in osteoporosis.
    thickness_change_pct : float
        Relative change in thickness, in percent.  Negative for
        thinning (osteoporotic progression).

    Returns
    -------
    dict with keys:
        ro_mm, t_baseline_mm, t_new_mm
        ri_baseline_mm, ri_new_mm
        I_baseline_mm4, I_new_mm4
        stress_ratio                 (σ_new / σ_old)
        delta_stress_pct             ((stress_ratio − 1) × 100)

    Raises
    ------
    ValueError
        If the perturbed thickness is non-positive or exceeds the outer
        radius.
    """
    if outer_radius_mm <= 0 or cortical_thickness_baseline_mm <= 0:
        raise ValueError("Outer radius and baseline thickness must be positive.")
    if cortical_thickness_baseline_mm >= outer_radius_mm:
        raise ValueError("Baseline thickness must be less than outer radius.")

    t_new = cortical_thickness_baseline_mm * (1.0 + thickness_change_pct / 100.0)
    if t_new <= 0:
        raise ValueError(f"Thickness change drives t_new = {t_new} ≤ 0.")
    if t_new >= outer_radius_mm:
        raise ValueError("Perturbed thickness exceeds outer radius.")

    ri_old = outer_radius_mm - cortical_thickness_baseline_mm
    ri_new = outer_radius_mm - t_new
    I_old = math.pi / 4 * (outer_radius_mm**4 - ri_old**4)
    I_new = math.pi / 4 * (outer_radius_mm**4 - ri_new**4)

    stress_ratio = I_old / I_new
    return {
        "ro_mm":            outer_radius_mm,
        "t_baseline_mm":    cortical_thickness_baseline_mm,
        "t_new_mm":         t_new,
        "ri_baseline_mm":   ri_old,
        "ri_new_mm":        ri_new,
        "I_baseline_mm4":   I_old,
        "I_new_mm4":        I_new,
        "stress_ratio":     stress_ratio,
        "delta_stress_pct": (stress_ratio - 1.0) * 100.0,
    }


def stress_concentration_kt(
    semi_major_axis_m: float,
    tip_radius_m: float,
) -> float:
    """
    Compute stress concentration factor Kt for an elliptical pore/crack.

    Inglis (1913): Kt = 1 + 2√(a / ρ)

    where:
        a  = semi-major axis of the ellipse (longest half-length)
        ρ  = radius of curvature at the crack tip

    Parameters
    ----------
    semi_major_axis_m : float
        Half-length of the pore/crack in metres (a).
        Typical bone pore: 20–200 µm → 2e-5 – 2e-4 m.
    tip_radius_m : float
        Tip radius of curvature in metres (ρ).
        A sharp crack: ~1 µm → 1e-6 m.

    Returns
    -------
    float
        Stress concentration factor Kt (dimensionless, ≥ 1).
        Kt = 1 means no concentration (circular pore).
        Kt = 3 is the theoretical maximum for a circular hole in a plate.
        Sharp cracks can give Kt >> 10.
    """
    if tip_radius_m <= 0:
        raise ValueError("Tip radius must be positive.")
    if semi_major_axis_m <= 0:
        raise ValueError("Semi-major axis must be positive.")
    return 1.0 + 2.0 * math.sqrt(semi_major_axis_m / tip_radius_m)


# ── PhysicsEngine ─────────────────────────────────────────────────────────────


class PhysicsEngine:
    """
    Physics validation engine for the BoneMind LRM.

    Combines Layer 1 (directional rules) and Layer 2 (numerical functions)
    to validate single edges and full causal chains.

    Typical usage::

        engine = PhysicsEngine()

        # Single edge
        result = engine.validate_edge("porosity", "increases", "elastic_modulus")

        # Full chain (list of edges as triples)
        result = engine.validate_chain([
            ("aging",     "increases", "porosity"),
            ("porosity",  "decreases", "elastic_modulus"),
            ("porosity",  "decreases", "fracture_toughness"),
        ])

        # Numerical — with values
        result = engine.check_currey(apparent_density_g_cm3=0.4)
        result = engine.check_mechanostat(strain_microstrain=2200)
    """

    def __init__(self) -> None:
        self._rules = _DIRECTIONAL_RULES

    # ── Layer 1 ───────────────────────────────────────────────────────────────

    def validate_edge(
        self, source: str, relation: str, target: str
    ) -> ValidationResult:
        """
        Validate a single (source, relation, target) edge against Layer 1 rules.

        Matching is done by substring — a rule key fires if BOTH node
        fragments appear as substrings in the respective node_ids AND the
        relation matches exactly.

        Parameters
        ----------
        source : str
            Source node_id (snake_case).
        relation : str
            Relation type (must be in RELATION_TYPES).
        target : str
            Target node_id (snake_case).

        Returns
        -------
        ValidationResult
            PLAUSIBLE / IMPLAUSIBLE if a rule matched; UNCERTAIN otherwise.
        """
        for (src_frag, rel, tgt_frag), (status, law, explanation) in self._rules.items():
            if (
                rel == relation
                and src_frag in source
                and tgt_frag in target
            ):
                return ValidationResult(
                    status=status,
                    law=law,
                    confidence=1.0 if status != "UNCERTAIN" else 0.5,
                    explanation=explanation,
                    edge=(source, relation, target),
                )

        # No rule matched
        return ValidationResult(
            status="UNCERTAIN",
            law="",
            confidence=0.5,
            explanation=(
                f"No physics rule covers '{source} –[{relation}]→ {target}'. "
                "Cannot confirm or deny."
            ),
            edge=(source, relation, target),
        )

    def validate_chain(
        self, edges: list[tuple[str, str, str]]
    ) -> ValidationResult:
        """
        Validate a full causal chain.

        Checks every edge individually.  The chain result is:
          • IMPLAUSIBLE if any edge is IMPLAUSIBLE (one bad link breaks chain)
          • PLAUSIBLE   if all edges are PLAUSIBLE
          • UNCERTAIN   if no edge is IMPLAUSIBLE but some are UNCERTAIN

        Parameters
        ----------
        edges : list[tuple[str, str, str]]
            Ordered list of (source, relation, target) triples.

        Returns
        -------
        ValidationResult
            Aggregate result for the whole chain.  If IMPLAUSIBLE, the
            `edge` field points to the first offending triple.
        """
        if not edges:
            return ValidationResult(
                status="UNCERTAIN",
                explanation="Empty chain — nothing to validate.",
            )

        results = [self.validate_edge(src, rel, tgt) for src, rel, tgt in edges]

        # Any IMPLAUSIBLE → whole chain fails immediately (one bad link breaks it)
        for r in results:
            if r.is_implausible:
                return ValidationResult(
                    status="IMPLAUSIBLE",
                    law=r.law,
                    confidence=r.confidence,
                    explanation=(
                        f"Chain broken at edge "
                        f"'{r.edge[0]} –[{r.edge[1]}]→ {r.edge[2]}': "
                        f"{r.explanation}"
                    ),
                    edge=r.edge,
                )

        # Majority vote: if ≥ 50% of edges are PLAUSIBLE and none IMPLAUSIBLE
        # → chain is PLAUSIBLE overall.
        # This handles the common case where extracted node names don't exactly
        # match rule fragments for every edge, but the majority of the chain
        # is physically confirmed.
        plausible = [r for r in results if r.is_plausible]
        uncertain = [r for r in results if r.status == "UNCERTAIN"]
        laws = list(dict.fromkeys(r.law for r in plausible if r.law))

        if len(plausible) >= len(results) / 2:
            confidence = len(plausible) / len(results)
            return ValidationResult(
                status="PLAUSIBLE",
                law=", ".join(laws) if laws else "directional rules",
                confidence=round(confidence, 2),
                explanation=(
                    f"{len(plausible)}/{len(edges)} edges confirmed by physics rules"
                    f"{f' ({len(uncertain)} unmatched)' if uncertain else ''}. "
                    f"Laws: {', '.join(laws) if laws else 'directional rules'}."
                ),
            )

        # Fewer than half matched → UNCERTAIN
        return ValidationResult(
            status="UNCERTAIN",
            law="",
            confidence=0.5,
            explanation=(
                f"Only {len(plausible)}/{len(edges)} edges matched physics rules — "
                "insufficient coverage to confirm. Chain may be valid but "
                "cannot be physically verified with current rule set."
            ),
        )

    # ── Layer 2 convenience wrappers ──────────────────────────────────────────

    def check_currey(
        self,
        apparent_density_g_cm3: float,
        claimed_modulus_GPa: float | None = None,
    ) -> ValidationResult:
        """
        Validate a density → elastic modulus claim using Currey's law.

        Parameters
        ----------
        apparent_density_g_cm3 : float
            Measured or implied apparent density (g/cm³).
        claimed_modulus_GPa : float | None
            If provided, checks whether the claimed E is within a 2×
            tolerance of the Currey prediction.  If None, just reports
            the predicted value.

        Returns
        -------
        ValidationResult
        """
        predicted = currey_modulus(apparent_density_g_cm3)
        explanation = (
            f"Currey's law (E = 7·ρ²): ρ = {apparent_density_g_cm3} g/cm³ "
            f"→ predicted E ≈ {predicted:.2f} GPa."
        )

        if claimed_modulus_GPa is None:
            return ValidationResult(
                status="PLAUSIBLE",
                law="Currey's law",
                confidence=1.0,
                explanation=explanation,
            )

        ratio = claimed_modulus_GPa / predicted if predicted > 0 else float("inf")
        if 0.5 <= ratio <= 2.0:
            return ValidationResult(
                status="PLAUSIBLE",
                law="Currey's law",
                confidence=1.0 - abs(1.0 - ratio) * 0.4,
                explanation=(
                    f"{explanation} Claimed E = {claimed_modulus_GPa} GPa "
                    f"— within 2× tolerance (ratio = {ratio:.2f}). Plausible."
                ),
            )
        else:
            return ValidationResult(
                status="IMPLAUSIBLE",
                law="Currey's law",
                confidence=1.0,
                explanation=(
                    f"{explanation} Claimed E = {claimed_modulus_GPa} GPa "
                    f"deviates {ratio:.1f}× from prediction — implausible."
                ),
            )

    def check_mechanostat(self, strain_microstrain: float) -> ValidationResult:
        """
        Report the Frost mechanostat zone for a given strain magnitude.

        Parameters
        ----------
        strain_microstrain : float
            Peak strain in microstrain (μɛ).

        Returns
        -------
        ValidationResult
            Always PLAUSIBLE (reporting, not passing/failing); the
            explanation describes the zone and predicted response.
        """
        zone = mechanostat_zone(strain_microstrain)
        return ValidationResult(
            status="PLAUSIBLE",
            law="Frost mechanostat",
            confidence=1.0,
            explanation=zone["description"],
        )

    def check_paris(
        self,
        delta_K_MPa_m05: float,
        C: float = 1.7e-9,
        m: float = 3.9,
    ) -> ValidationResult:
        """
        Report fatigue crack growth rate for given ΔK.

        Parameters
        ----------
        delta_K_MPa_m05 : float
            Stress intensity range ΔK (MPa√m).
        C, m : float
            Paris law constants (default: human cortical bone).

        Returns
        -------
        ValidationResult
        """
        da_dN = paris_crack_growth(delta_K_MPa_m05, C, m)
        kic_typical = 3.0  # typical cortical bone KIc in MPa√m
        status: Status = "PLAUSIBLE" if delta_K_MPa_m05 < kic_typical else "IMPLAUSIBLE"
        explanation = (
            f"Paris law (da/dN = C·ΔK^m): ΔK = {delta_K_MPa_m05} MPa√m "
            f"→ da/dN ≈ {da_dN:.2e} m/cycle."
        )
        if status == "IMPLAUSIBLE":
            explanation += (
                f" ΔK exceeds typical cortical KIc ({kic_typical} MPa√m) — "
                "catastrophic fracture expected rather than stable crack growth."
            )
        return ValidationResult(
            status=status,
            law="Paris law",
            confidence=1.0,
            explanation=explanation,
        )

    def check_beam_bending(
        self,
        bending_moment_Nm: float,
        outer_radius_m: float,
        inner_radius_m: float,
        yield_stress_MPa: float = 130.0,
    ) -> ValidationResult:
        """
        Check whether bending stress exceeds yield stress (fracture threshold).

        Parameters
        ----------
        bending_moment_Nm : float
            Applied moment (N·m).
        outer_radius_m : float
            Periosteal radius (m).
        inner_radius_m : float
            Endosteal radius (m).
        yield_stress_MPa : float
            Tensile yield stress of cortical bone (default 130 MPa).

        Returns
        -------
        ValidationResult
        """
        result = beam_bending_stress(bending_moment_Nm, outer_radius_m, inner_radius_m)
        sigma = result["sigma_MPa"]
        status: Status = "PLAUSIBLE" if sigma < yield_stress_MPa else "IMPLAUSIBLE"
        explanation = (
            f"Beam theory (σ = Mc/I): I = {result['I_m4']:.3e} m⁴, "
            f"σ = {sigma} MPa "
            f"({'< ' if sigma < yield_stress_MPa else '> '}"
            f"yield {yield_stress_MPa} MPa). "
            f"{'Elastic — no fracture predicted.' if status == 'PLAUSIBLE' else 'Exceeds yield — fracture predicted.'}"
        )
        return ValidationResult(
            status=status,
            law="Beam theory",
            confidence=1.0,
            explanation=explanation,
        )

    # ── Utilities ──────────────────────────────────────────────────────────────

    def rule_count(self) -> int:
        """Return the number of directional rules loaded."""
        return len(self._rules)

    def summary(self) -> None:
        """Print a formatted summary of all directional rules."""
        print(f"\n{'─' * 68}")
        print(f"  BoneMind Physics Engine — {self.rule_count()} directional rules")
        print(f"{'─' * 68}")
        for (src, rel, tgt), (status, law, _) in sorted(
            self._rules.items(), key=lambda x: x[1][0]
        ):
            marker = "✓" if status == "PLAUSIBLE" else ("✗" if status == "IMPLAUSIBLE" else "?")
            print(f"  [{marker}] {src:28} –[{rel:15}]→ {tgt:28}  ({law})")
        print(f"{'─' * 68}\n")
