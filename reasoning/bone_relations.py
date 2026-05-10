"""
reasoning/bone_relations.py
───────────────────────────
Bone-physics Relations for the v2 reasoning pipeline.

The :func:`build_bone_registry` factory wires three Relations into a
single :class:`~reasoning.relation.RelationRegistry`:

1. **density_from_porosity** — geometric bridge that lets every
   porosity-driven query reach the variables of the laws below.
   Equation: ``ρ = ρ_full · (1 − φ)``.

2. **currey_modulus** — Currey's law for elastic modulus.
   Equation: ``E = a · ρⁿ``.  Citation: Currey 1988.

3. **vashishth_paris** — Paris fatigue-crack growth with a
   density-dependent prefactor (Vashishth 2003 extension of
   Paris 1963).  Equation: ``da/dN = C₀ · (ρ_ref / ρ)^k_ρ · ΔK^m``.

The three Relations share ``ρ`` (apparent density), which is the only
shared symbol they need to compose.  ``phi → ρ → E`` (Currey path)
and ``phi → ρ → da/dN`` (Paris path) both fall out of variable sharing
without anything in this file or the registry knowing those chains
exist.
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
    ):
        reg.register(rel_factory())
    return reg
