"""
reasoning/bone_relations.py
───────────────────────────
Bone-physics Relations for the reasoning pipeline.

:func:`build_bone_registry` wires seven Relations into a single
:class:`~reasoning.relation.RelationRegistry`.  The Relations span the
four canonical bone laws (Currey, Paris, beam bending, Frost), plus
three geometric / constitutive bridges that let the laws compose:

1. **density_from_porosity** — ``ρ = ρ_full · (1 − φ)``.
2. **currey_modulus**        — ``E = a · ρⁿ`` (Currey 1988).
3. **vashishth_paris**       — ``da/dN = C₀ · (ρ_ref / ρ)^k_ρ · ΔKᵐ``
   (Vashishth 2003 / Paris 1963).
4. **cortical_inertia**      — ``I = π/4 · (R⁴ − (R − t)⁴)``.
5. **beam_bending**          — ``σ = M · R / I`` (Euler–Bernoulli).
6. **hookes_law**            — ``ε[µε] = 1000 · σ[MPa] / E[GPa]``.
7. **frost_mechanostat**     — ``dBMD/dt = k · tanh((ε − ε_set)/ε_w)``,
   the smooth analog of Frost's zone-based response (Frost 2003,
   Robling 2009).

Chains emerge from variable sharing alone:

* φ → ρ → E                                — Currey path
* φ → ρ, ΔK → da/dN                        — Paris path
* R, t → I, M, R, I → σ, σ, E → ε → dBMD/dt — full geometric→biological
  chain (six relations, one MC pass)

Nothing in this file or the registry encodes those chain identities;
the BFS in :class:`RelationRegistry` discovers them from the shared
symbols.
"""

from __future__ import annotations

import sympy as sp

from reasoning.relation import (
    CovariateShift, Prior, Relation, RelationRegistry, Variable,
)


# ── Covariate-shift library ───────────────────────────────────────────────────
#
# Each shift is a small, citation-anchored function that adjusts a
# parameter's prior mean/std based on the active covariates supplied at
# inference time.  Covariates are a flat dict; recognised keys are:
#
#   age:     float years (0 means "not supplied" → no age effect)
#   sex:     "M" / "F" / None
#   site:    "femur_cortical" / "tibia_cortical" / "vertebra" / "radius" / None
#   disease: list of strings, recognised tokens:
#               "osteoporosis"
#               "glucocorticoid"
#               "osteogenesis_imperfecta"
#
# Each shift is conservative — it only fires when the relevant covariate
# is actually present.  Composition is left-to-right: a parameter that
# carries (age, sex, site, disease) shifts will see them applied in
# declaration order, so all four can compound on the same Prior.


def _cov_age(cov: dict) -> float:
    age = cov.get("age")
    return float(age) if age is not None else 0.0


def _cov_site(cov: dict) -> str | None:
    site = cov.get("site")
    return str(site) if site else None


def _cov_disease(cov: dict, token: str) -> bool:
    diseases = cov.get("disease") or []
    if isinstance(diseases, str):
        diseases = [diseases]
    return token in diseases


def _shift_currey_a_age() -> CovariateShift:
    """Currey pre-factor declines ≈ 10 %/decade past age 30 (cortical bone)."""
    def apply(mean: float, std: float, cov: dict) -> tuple[float, float]:
        age = _cov_age(cov)
        if age <= 30.0:
            return mean, std
        # Linear, capped at age 95.
        decades_past_30 = min(age, 95.0) - 30.0
        factor = max(0.30, 1.0 - 0.10 * (decades_past_30 / 10.0))
        return mean * factor, std * factor
    return CovariateShift(
        name="currey_a_age",
        description="Modulus pre-factor drops ≈10 %/decade past age 30",
        citation="Burstein 1976; McCalden 1993 (cortical bone modulus vs age)",
        apply=apply,
    )


def _shift_currey_a_sex() -> CovariateShift:
    """Modest female-vs-male offset; literature reports ~5 % lower in females."""
    def apply(mean: float, std: float, cov: dict) -> tuple[float, float]:
        if cov.get("sex") == "F":
            return mean * 0.95, std * 0.95
        return mean, std
    return CovariateShift(
        name="currey_a_sex",
        description="Females ≈5 % lower modulus pre-factor",
        citation="Smith 1976; Riggs 1981 — sex differences in cortical modulus",
        apply=apply,
    )


def _shift_currey_a_site() -> CovariateShift:
    """Trabecular sites (vertebra) have a much lower pre-factor."""
    def apply(mean: float, std: float, cov: dict) -> tuple[float, float]:
        site = _cov_site(cov)
        if site == "vertebra":
            # Trabecular bone modulus prefactor is roughly half that
            # of cortical at the same apparent density.
            return mean * 0.55, std * 0.55
        if site == "radius":
            return mean * 0.9, std * 0.9
        return mean, std
    return CovariateShift(
        name="currey_a_site",
        description="Trabecular sites have lower modulus pre-factor (~0.55× cortical)",
        citation="Carter–Hayes 1977; Keller 1994 — site-specific Currey constants",
        apply=apply,
    )


def _shift_currey_n_site() -> CovariateShift:
    """Trabecular sites have steeper density-modulus exponent (n ≈ 3 vs 2.5)."""
    def apply(mean: float, std: float, cov: dict) -> tuple[float, float]:
        site = _cov_site(cov)
        if site == "vertebra":
            return mean + 0.5, std       # ≈ 3.0 instead of 2.5
        return mean, std
    return CovariateShift(
        name="currey_n_site",
        description="Trabecular bone uses n ≈ 3.0 vs cortical n ≈ 2.5",
        citation="Keller 1994; Rho 1995 — power-law exponent across sites",
        apply=apply,
    )


def _shift_currey_a_oi() -> CovariateShift:
    """Osteogenesis imperfecta reduces matrix modulus dramatically."""
    def apply(mean: float, std: float, cov: dict) -> tuple[float, float]:
        if _cov_disease(cov, "osteogenesis_imperfecta"):
            return mean * 0.6, std * 0.6
        return mean, std
    return CovariateShift(
        name="currey_a_oi",
        description="OI lowers modulus pre-factor by ≈40 %",
        citation="Imbert 2014 — modulus reduction in OI cortical bone",
        apply=apply,
    )


def _shift_paris_c0_age() -> CovariateShift:
    """
    Paris pre-factor rises with age — fatigue crack growth ≈ doubles per
    decade past 50.  C_0 is stored in *log space* (lognormal), so we add
    log(2) per decade to the log-mean.

    log(2) ≈ 0.693, so 0.069 per year past 50.
    """
    def apply(mean: float, std: float, cov: dict) -> tuple[float, float]:
        age = _cov_age(cov)
        if age <= 50.0:
            return mean, std
        decades_past_50 = (min(age, 95.0) - 50.0) / 10.0
        return mean + 0.693 * decades_past_50, std
    return CovariateShift(
        name="paris_c0_age",
        description="da/dN pre-factor roughly doubles per decade past 50",
        citation="Diab & Vashishth 2005 — fatigue resistance vs donor age",
        apply=apply,
    )


def _shift_paris_c0_osteoporosis() -> CovariateShift:
    """Osteoporotic bone fatigues faster at fixed ΔK (≈1.5× rate)."""
    def apply(mean: float, std: float, cov: dict) -> tuple[float, float]:
        if _cov_disease(cov, "osteoporosis"):
            # log(1.5) ≈ 0.405 added in log space (lognormal mean).
            return mean + 0.405, std
        return mean, std
    return CovariateShift(
        name="paris_c0_osteoporosis",
        description="Osteoporotic bone ≈1.5× faster crack growth at same ΔK",
        citation="Vashishth 2003 — fatigue in osteoporotic cortical bone",
        apply=apply,
    )


def _shift_frost_eps_set_age() -> CovariateShift:
    """Frost setpoint rises with age (mechanostat desensitisation)."""
    def apply(mean: float, std: float, cov: dict) -> tuple[float, float]:
        age = _cov_age(cov)
        if age <= 40.0:
            return mean, std
        decades_past_40 = (min(age, 95.0) - 40.0) / 10.0
        return mean + 150.0 * decades_past_40, std
    return CovariateShift(
        name="frost_eps_set_age",
        description="Adaptation setpoint rises ≈150 µε/decade past 40",
        citation="Frost 2003 — mechanostat desensitisation with age",
        apply=apply,
    )


def _shift_frost_k_form_age() -> CovariateShift:
    """Anabolic capacity halves by age 75."""
    def apply(mean: float, std: float, cov: dict) -> tuple[float, float]:
        age = _cov_age(cov)
        if age <= 30.0:
            return mean, std
        # Linear ramp from 1.0 at age 30 to 0.5 at age 75, then capped.
        t = min(max((age - 30.0) / 45.0, 0.0), 1.0)
        factor = 1.0 - 0.5 * t
        return mean * factor, std * factor
    return CovariateShift(
        name="frost_k_form_age",
        description="Max BMD adaptation rate halves between ages 30 and 75",
        citation="Lanyon 1994; Frost 2003 — age-related anabolic capacity",
        apply=apply,
    )


def _shift_frost_eps_set_osteoporosis() -> CovariateShift:
    """Osteoporosis raises the Frost setpoint substantially."""
    def apply(mean: float, std: float, cov: dict) -> tuple[float, float]:
        if _cov_disease(cov, "osteoporosis"):
            return mean + 500.0, std
        return mean, std
    return CovariateShift(
        name="frost_eps_set_osteoporosis",
        description="Osteoporotic setpoint shifts +500 µε (resorption-biased)",
        citation="Robling 2009 — disuse-osteopenia setpoint shift",
        apply=apply,
    )


def _shift_frost_k_form_glucocorticoid() -> CovariateShift:
    """Chronic glucocorticoid use suppresses formation to ≈20 % of baseline."""
    def apply(mean: float, std: float, cov: dict) -> tuple[float, float]:
        if _cov_disease(cov, "glucocorticoid"):
            return mean * 0.2, std * 0.2
        return mean, std
    return CovariateShift(
        name="frost_k_form_glucocorticoid",
        description="Glucocorticoid therapy cuts anabolic rate to ≈20 %",
        citation="Weinstein 2001 — glucocorticoid-induced osteoblast apoptosis",
        apply=apply,
    )


def _shift_density_from_porosity_site() -> CovariateShift:
    """
    Trabecular tissue is slightly less mineralised than cortical (≈ 1.80
    vs 1.90 g/cm³ matrix density).  Affects the apparent-density bridge.
    """
    def apply(mean: float, std: float, cov: dict) -> tuple[float, float]:
        if _cov_site(cov) == "vertebra":
            return 1.80, std
        return mean, std
    return CovariateShift(
        name="density_rhofull_site",
        description="Trabecular tissue density ≈ 1.80 g/cm³ vs cortical 1.90",
        citation="Cowin 2001 — site-specific matrix densities",
        apply=apply,
    )


# ── Variable definitions ──────────────────────────────────────────────────────


_VARIABLES: list[Variable] = [
    Variable(
        symbol="phi",
        display_symbol="φ",
        name="porosity",
        unit="",
        lo=0.0, hi=0.95,
        typical_ranges={
            "cortical":     (0.02, 0.15),   # healthy → mildly aged
            "transitional": (0.15, 0.50),   # severe cortical osteoporosis
            "trabecular":   (0.50, 0.95),
        },
        description=(
            "Volume fraction of pores (porosity, void fraction) in the bone "
            "matrix. Cortical bone is roughly 5–15 % porous; trabecular "
            "regions span 50–95 %."
        ),
    ),
    Variable(
        symbol="rho",
        display_symbol="ρ",
        name="apparent density",
        unit="g/cm³",
        lo=0.05, hi=2.10,
        typical_ranges={
            "cortical":     (1.70, 2.00),
            "transitional": (1.20, 1.70),
            "trabecular":   (0.10, 0.60),
        },
        description=(
            "Apparent density — mass per unit total volume including pores. "
            "Clinically reported as bone mineral density (BMD); structural "
            "stiffness scales as a power of this quantity."
        ),
    ),
    Variable(
        symbol="E",
        display_symbol="E",
        name="elastic modulus",
        unit="GPa",
        lo=0.001, hi=30.0,
        typical_ranges={
            "cortical":     (15.0, 25.0),
            "transitional": (5.0, 15.0),
            "trabecular":   (0.05, 2.0),
        },
        description=(
            "Elastic (Young's) modulus — the stiffness coefficient relating "
            "stress to strain under uniaxial loading. A measure of how rigid "
            "or soft the bone tissue is."
        ),
    ),
    Variable(
        symbol="dK",
        display_symbol="ΔK",
        name="stress-intensity range",
        unit="MPa·√m",
        lo=0.0, hi=6.0,
        description=(
            "Cyclic load parameter ΔK (delta K) — the stress-intensity range "
            "experienced by a bone microcrack under each loading cycle. The "
            "input variable of Paris-law fatigue analysis: higher cyclic "
            "load → larger ΔK → faster fatigue damage. Used for predicting "
            "stiffness loss or fracture risk under repeated mechanical "
            "loading."
        ),
    ),
    Variable(
        symbol="da_dN",
        display_symbol="da/dN",
        name="crack growth rate",
        unit="m/cycle",
        lo=1.0e-14, hi=1.0e-3,
        description=(
            "Fatigue crack growth rate da/dN — how far a microcrack extends "
            "per cycle of cyclic mechanical loading in cortical bone. The "
            "outcome variable of Paris-law fatigue. Drives long-term "
            "stiffness loss and bone-failure risk under repeated loading."
        ),
    ),
    # Cross-section geometry — femoral midshaft regime.
    Variable(
        symbol="R",
        display_symbol="R",
        name="cortical outer radius",
        unit="mm",
        lo=5.0, hi=25.0,
        description=(
            "Periosteal (outer) radius of the cortical shell. Sets the "
            "diaphyseal cross-section's outer geometry."
        ),
    ),
    Variable(
        symbol="t",
        display_symbol="t",
        name="cortical thickness",
        unit="mm",
        lo=0.5, hi=8.0,
        description=(
            "Cortical wall thickness — the thickness of the cortical shell "
            "(outer radius minus inner radius). Thins progressively in "
            "osteoporosis."
        ),
    ),
    Variable(
        symbol="I_section",
        display_symbol="I",
        name="second moment of area",
        unit="mm⁴",
        lo=1.0, hi=1.0e5,
        description=(
            "Second moment of area — geometric stiffness of a hollow "
            "cortical cross-section. Drives bending stiffness and flexural "
            "rigidity of the diaphysis."
        ),
    ),
    Variable(
        symbol="M",
        display_symbol="M",
        name="bending moment",
        unit="N·mm",
        lo=0.0, hi=1.0e6,
        description=(
            "Applied bending moment on the cross-section — the mechanical "
            "loading magnitude during gait, fall, or exercise."
        ),
    ),
    Variable(
        symbol="sigma",
        display_symbol="σ",
        name="bending stress",
        unit="MPa",
        lo=0.0, hi=300.0,
        description=(
            "Maximum fibre bending stress — the peak compressive or tensile "
            "stress at the outer surface under bending. Related to bending "
            "strength."
        ),
    ),
    Variable(
        symbol="eps",
        display_symbol="ε",
        name="peak strain",
        unit="µε",
        lo=0.0, hi=10000.0,
        description=(
            "Peak principal strain magnitude in microstrain (µε). The input "
            "to Frost's mechanostat — sets whether the bone responds with "
            "formation, homeostasis, or resorption."
        ),
    ),
    Variable(
        symbol="dBMD_dt",
        display_symbol="ΔBMD/Δt",
        name="bone adaptation rate",
        unit="%/yr",
        lo=-5.0, hi=5.0,
        description=(
            "Annual percentage rate of bone-density change. Captures bone "
            "remodeling outcomes: net formation, net resorption, or "
            "homeostasis. The Frost-mechanostat outcome variable."
        ),
    ),
]


# ── Relations ─────────────────────────────────────────────────────────────────


def _density_from_porosity() -> Relation:
    phi, rho, rho_full = sp.symbols("phi rho rho_full", positive=True)
    return Relation(
        name="density_from_porosity",
        equation=rho - rho_full * (1 - phi),
        output="rho",
        inputs=("phi",),
        parameters={
            "rho_full": Prior(
                mean=1.90, std=0.0, distribution="fixed",
                citation="Cowin 2001 — bone matrix density ≈ 1.90 g/cm³",
                shifts=(_shift_density_from_porosity_site(),),
            ),
        },
        citation="Geometric definition (ρ = ρ_full · (1 − φ))",
        description="Apparent density falls off linearly with porosity.",
    )


def _currey_modulus() -> Relation:
    rho, E, a, n = sp.symbols("rho E a n", positive=True)
    return Relation(
        name="currey_modulus",
        equation=E - a * rho**n,
        output="E",
        inputs=("rho",),
        parameters={
            "a": Prior(
                mean=7.0, std=0.8, distribution="normal",
                citation="Currey 1988 — pre-factor for the E–ρ relation",
                shifts=(
                    _shift_currey_a_site(),
                    _shift_currey_a_age(),
                    _shift_currey_a_sex(),
                    _shift_currey_a_oi(),
                ),
            ),
            "n": Prior(
                mean=2.5, std=0.20, distribution="normal",
                citation="Currey 1988 / Hernandez 2001 — exponent ≈ 2.5",
                shifts=(_shift_currey_n_site(),),
            ),
        },
        citation="Currey 1988 (E ∝ ρⁿ)",
        description="Elastic modulus scales as a power of apparent density.",
    )


def _vashishth_paris() -> Relation:
    rho, dK, da_dN = sp.symbols("rho dK da_dN", positive=True)
    C_0, m, k_rho, rho_ref = sp.symbols("C_0 m k_rho rho_ref", positive=True)
    return Relation(
        name="vashishth_paris",
        equation=da_dN - C_0 * (rho_ref / rho)**k_rho * dK**m,
        output="da_dN",
        inputs=("rho", "dK"),
        parameters={
            # C_0 has a wide log-normal: literature values for cortical
            # bone span roughly one order of magnitude.
            "C_0": Prior(
                mean=-20.20, std=0.40, distribution="lognormal",
                citation="Vashishth 2004 — C ≈ 1.7×10⁻⁹ m/cycle baseline",
                shifts=(
                    _shift_paris_c0_age(),
                    _shift_paris_c0_osteoporosis(),
                ),
            ),
            "m": Prior(
                mean=3.9, std=0.25, distribution="normal",
                citation="Vashishth 2004 — Paris exponent for cortical bone",
            ),
            "k_rho": Prior(
                mean=2.0, std=0.40, distribution="normal",
                citation="Vashishth 2003 — density modulation of C",
            ),
            "rho_ref": Prior(
                mean=1.90, std=0.0, distribution="fixed",
                citation="Reference cortical density ≈ 1.90 g/cm³",
            ),
        },
        citation="Vashishth 2003 / Paris 1963 — density-modulated Paris law",
        description=(
            "Fatigue crack growth rate scales as ΔKᵐ with a "
            "density-dependent prefactor."
        ),
    )


def _cortical_inertia() -> Relation:
    R, t, I_section = sp.symbols("R t I_section", positive=True)
    return Relation(
        name="cortical_inertia",
        # Hollow circular cross section about a diameter:
        #   I = π/4 · (R⁴ − (R − t)⁴)
        equation=I_section - sp.pi / 4 * (R**4 - (R - t)**4),
        output="I_section",
        inputs=("R", "t"),
        parameters={},
        citation="Standard mechanics of materials — hollow circular cylinder I",
        description=(
            "Second moment of area for a hollow circular shell of outer "
            "radius R and wall thickness t."
        ),
    )


def _beam_bending() -> Relation:
    M, R, I_section, sigma = sp.symbols("M R I_section sigma", positive=True)
    return Relation(
        name="beam_bending",
        equation=sigma - M * R / I_section,
        output="sigma",
        inputs=("M", "R", "I_section"),
        parameters={},
        citation="Euler–Bernoulli beam bending — σ = M·c/I",
        description=(
            "Maximum fibre bending stress for a beam under moment M, with "
            "outer fibre at radius R and section inertia I_section."
        ),
    )


def _hookes_law() -> Relation:
    """
    Strain from stress and modulus, with units in microstrain.

    σ in MPa, E in GPa → σ/E = 10⁻³ strain = 10³ µε.  The
    factor of 1000 lifts the ratio into the microstrain regime the
    mechanostat is calibrated in.
    """
    sigma, E, eps = sp.symbols("sigma E eps", positive=True)
    return Relation(
        name="hookes_law",
        equation=eps - 1000 * sigma / E,
        output="eps",
        inputs=("sigma", "E"),
        parameters={},
        citation="Hooke's law (1-D) — ε = σ / E",
        description=(
            "Linear-elastic strain from bending stress and elastic modulus, "
            "expressed in microstrain so it matches Frost's setpoint scale."
        ),
    )


def _frost_mechanostat() -> Relation:
    """
    Smooth analog of Frost's piecewise mechanostat.

    The canonical mechanostat partitions strain into disuse, adapted,
    and overload zones with discontinuous BMD-change rates.  For a
    SymPy-friendly composable form we use a tanh squashing centred on
    the adaptation setpoint::

        dBMD/dt = k_form · tanh((eps − eps_set) / eps_width)

    Behaviour
    ---------
    * eps ≪ eps_set                   → dBMD/dt → −k_form (disuse)
    * eps ≈ eps_set                   → dBMD/dt ≈ 0       (homeostasis)
    * eps ≫ eps_set (but sub-woven)   → dBMD/dt → +k_form (mild overload)

    Calibration draws on Frost 2003 / Robling 2009 / Burr 2002 — typical
    BMD swings span ±1.5–3 %/yr across the zones.
    """
    eps, dBMD_dt = sp.symbols("eps dBMD_dt", real=True)
    k_form, eps_set, eps_width = sp.symbols(
        "k_form eps_set eps_width", positive=True,
    )
    return Relation(
        name="frost_mechanostat",
        equation=dBMD_dt - k_form * sp.tanh((eps - eps_set) / eps_width),
        output="dBMD_dt",
        inputs=("eps",),
        parameters={
            "k_form": Prior(
                mean=2.0, std=0.5, distribution="normal",
                citation="Frost 2003 / Burr 2002 — saturation BMD rate ≈ ±2 %/yr",
                shifts=(
                    _shift_frost_k_form_age(),
                    _shift_frost_k_form_glucocorticoid(),
                ),
            ),
            "eps_set": Prior(
                mean=1000.0, std=200.0, distribution="normal",
                citation="Frost 2003 — adapted-window centre ≈ 1000 µε",
                shifts=(
                    _shift_frost_eps_set_age(),
                    _shift_frost_eps_set_osteoporosis(),
                ),
            ),
            "eps_width": Prior(
                mean=700.0, std=150.0, distribution="normal",
                citation="Robling 2009 — width of the adaptation transition",
            ),
        },
        citation="Frost 2003 / Robling 2009 — mechanostat (smooth analog)",
        description=(
            "Annual BMD change rate as a smooth, monotonic function of "
            "peak strain, centred on the Frost setpoint."
        ),
    )


# ── Covariate UI metadata ─────────────────────────────────────────────────────


COVARIATE_SCHEMA: dict = {
    "age": {
        "type":  "number",
        "label": "Patient age",
        "unit":  "years",
        "min":   18,
        "max":   95,
        "step":  1,
        "default": 30,
        "description": (
            "Drives age-dependent shifts in Currey's pre-factor, Paris C₀ "
            "(≈ ×2/decade past 50, Diab & Vashishth 2005), the Frost setpoint "
            "(rises with desensitisation), and the anabolic rate (halves by 75)."
        ),
    },
    "sex": {
        "type":    "enum",
        "label":   "Sex",
        "default": "M",
        "options": [
            {"value": "M", "label": "Male"},
            {"value": "F", "label": "Female"},
        ],
        "description": "Modest baseline modulus offset for females (≈ −5 %).",
    },
    "site": {
        "type":    "enum",
        "label":   "Anatomical site",
        "default": "femur_cortical",
        "options": [
            {"value": "femur_cortical", "label": "Femur — cortical midshaft"},
            {"value": "tibia_cortical", "label": "Tibia — cortical midshaft"},
            {"value": "radius",         "label": "Distal radius"},
            {"value": "vertebra",       "label": "Lumbar vertebra (trabecular)"},
        ],
        "description": (
            "Site sets trabecular vs cortical parameterisation: vertebra uses "
            "Currey's trabecular constants (a ≈ 0.55× cortical, n ≈ 3.0)."
        ),
    },
    "disease": {
        "type":    "multi",
        "label":   "Disease state",
        "default": [],
        "options": [
            {"value": "osteoporosis",
             "label": "Osteoporosis",
             "description": "Setpoint +500 µε; Paris C₀ ×1.5 (Vashishth 2003)."},
            {"value": "glucocorticoid",
             "label": "Chronic glucocorticoid therapy",
             "description": "Anabolic rate cut to ≈20 % (Weinstein 2001)."},
            {"value": "osteogenesis_imperfecta",
             "label": "Osteogenesis imperfecta",
             "description": "Modulus pre-factor ×0.6 (Imbert 2014)."},
        ],
    },
}


# ── Factory ───────────────────────────────────────────────────────────────────


def build_bone_registry() -> RelationRegistry:
    """Return a fresh RelationRegistry pre-populated with the bone laws."""
    reg = RelationRegistry()
    for var in _VARIABLES:
        reg.register_variable(var)
    for rel_factory in (
        _density_from_porosity,
        _currey_modulus,
        _vashishth_paris,
        _cortical_inertia,
        _beam_bending,
        _hookes_law,
        _frost_mechanostat,
    ):
        reg.register(rel_factory())
    return reg
