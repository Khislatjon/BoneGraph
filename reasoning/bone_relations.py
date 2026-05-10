"""
reasoning/bone_relations.py
───────────────────────────
Bone-physics Relations for the v2 reasoning pipeline.

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

from reasoning.relation import Prior, Relation, RelationRegistry, Variable


# ── Variable definitions ──────────────────────────────────────────────────────


_VARIABLES: list[Variable] = [
    Variable(
        symbol="phi",
        name="porosity",
        unit="",
        lo=0.0, hi=0.95,
        description="Volume fraction of pores in the tissue.",
    ),
    Variable(
        symbol="rho",
        name="apparent density",
        unit="g/cm³",
        lo=0.05, hi=2.10,
        description="Mass per unit total volume (including pores).",
    ),
    Variable(
        symbol="E",
        name="elastic modulus",
        unit="GPa",
        lo=0.001, hi=30.0,
        description="Stiffness under uniaxial loading.",
    ),
    Variable(
        symbol="dK",
        name="stress-intensity range",
        unit="MPa·√m",
        lo=0.0, hi=6.0,
        description="Per-cycle stress-intensity factor driving fatigue.",
    ),
    Variable(
        symbol="da_dN",
        name="crack growth rate",
        unit="m/cycle",
        lo=1.0e-14, hi=1.0e-3,
        description="Fatigue crack extension per loading cycle.",
    ),
    # Cross-section geometry — femoral midshaft regime.
    Variable(
        symbol="R",
        name="cortical outer radius",
        unit="mm",
        lo=5.0, hi=25.0,
        description="Periosteal radius of the cortical shell.",
    ),
    Variable(
        symbol="t",
        name="cortical thickness",
        unit="mm",
        lo=0.5, hi=8.0,
        description="Wall thickness of the cortical shell (R − R_inner).",
    ),
    Variable(
        symbol="I_section",
        name="second moment of area",
        unit="mm⁴",
        lo=1.0, hi=1.0e5,
        description="Geometric stiffness of the hollow-cylinder cross section.",
    ),
    Variable(
        symbol="M",
        name="bending moment",
        unit="N·mm",
        lo=0.0, hi=1.0e6,
        description="Applied bending moment on the cross section.",
    ),
    Variable(
        symbol="sigma",
        name="bending stress",
        unit="MPa",
        lo=0.0, hi=300.0,
        description="Maximum fibre stress under bending.",
    ),
    Variable(
        symbol="eps",
        name="peak strain",
        unit="µε",
        lo=0.0, hi=10000.0,
        description="Peak principal strain magnitude in microstrain.",
    ),
    Variable(
        symbol="dBMD_dt",
        name="BMD adaptation rate",
        unit="%/yr",
        lo=-5.0, hi=5.0,
        description="Annual change in bone mineral density "
                    "(positive = formation, negative = resorption).",
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
            ),
            "n": Prior(
                mean=2.5, std=0.20, distribution="normal",
                citation="Currey 1988 / Hernandez 2001 — exponent ≈ 2.5",
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
            ),
            "eps_set": Prior(
                mean=1000.0, std=200.0, distribution="normal",
                citation="Frost 2003 — adapted-window centre ≈ 1000 µε",
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
