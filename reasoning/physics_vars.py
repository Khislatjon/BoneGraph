"""
reasoning/physics_vars.py
─────────────────────────
Registry mapping bone-physics variables to their graph-node proxies
and to typical numerical ranges for cortical and trabecular bone.

Why this exists
---------------
The Reasoning tab generates hypotheses by invoking physical laws
(Currey, Frost mechanostat, Paris).  Each law operates on physics
variables (apparent density ρ, porosity φ, elastic modulus E, …) but
the knowledge graph speaks in graph node IDs (`porosity`, `bmd`,
`elastic_modulus`).  This module is the bridge: it tells the generator
which graph nodes proxy which physical variable, and what numerical
ranges are physically reasonable for bone.

Used by
-------
- reasoning/physics_gen.py — anchors a query to physics variables and
  picks node IDs to materialise the hypothesis chain.
- reasoning/critic.py      — sanity-checks predicted magnitudes against
  the bone-physical ranges below.

The registry is hand-curated.  It is small on purpose: a few variables
covered well beats many variables covered approximately.  Add new
entries when you add a new physical law that needs them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PhysicsVariable:
    """
    A bone-physics variable plus the graph-node IDs that proxy it.

    Attributes
    ----------
    name : str
        Internal identifier, e.g. ``"apparent_density"``.
    symbol : str
        Symbol used in equations, e.g. ``"ρ"``.
    units : str
        SI / engineering units for the value, e.g. ``"g/cm³"``.
    description : str
        One-line explanation for UI display.
    cortical_range : tuple[float, float]
        Typical numerical range for human cortical bone.
    trabecular_range : tuple[float, float]
        Typical numerical range for human trabecular bone.
    node_ids : list[str]
        Graph node_ids in the cleaned ontology that proxy this variable.
        The first entry is the canonical proxy used when materialising
        hypothesis chains.
    inverse_of : str | None
        If set, this variable is the inverse of another (e.g. porosity is
        the inverse of apparent density via ρ = ρ_full · (1 - φ)).  The
        generator uses this to convert between the two.
    keywords : list[str]
        Whole-word/phrase triggers a free-text query must contain to
        anchor this variable.  Matched case-insensitively with word
        boundaries — so ``"bone"`` does not match ``"bonemind"`` and
        ``"strength"`` does not bleed across every variable that mentions
        the word.  Curated; do not auto-derive from ``node_ids``.
    """

    name: str
    symbol: str
    units: str
    description: str
    cortical_range: tuple[float, float]
    trabecular_range: tuple[float, float]
    node_ids: list[str] = field(default_factory=list)
    inverse_of: str | None = None
    keywords: list[str] = field(default_factory=list)


# ── The registry ──────────────────────────────────────────────────────────────
#
# All node_ids here must exist (or be expected to exist) in the cleaned
# ontology.db.  The generator silently skips variables whose node_ids are
# missing from a particular graph, so the registry can be a superset.

VARS: dict[str, PhysicsVariable] = {
    "apparent_density": PhysicsVariable(
        name="apparent_density",
        symbol="ρ",
        units="g/cm³",
        description="Apparent (bulk) bone density — mass per unit total volume.",
        cortical_range=(1.5, 2.0),
        trabecular_range=(0.1, 0.9),
        node_ids=[
            "bone_mineral_density",
            "bone_density",
            "mineral_density",
            "tissue_mineral_density",
            "cortical_bone_mineral_density",
            "cortical_bone_density",
            "cortical_bmd",
            "trabecular_bmd",
            "bmd",
            "lumbar_spine_bmd",
            "femoral_neck_bmd",
            "areal_bmd",
        ],
        keywords=[
            "apparent density", "bone density", "bone mineral density",
            "mineral density", "bmd",
        ],
    ),
    "porosity": PhysicsVariable(
        name="porosity",
        symbol="φ",
        units="–",        # dimensionless volume fraction
        description="Volume fraction of pores within the bone matrix.",
        cortical_range=(0.03, 0.15),       # cortical: ~3–15%
        trabecular_range=(0.50, 0.95),     # trabecular: ~50–95%
        node_ids=[
            "porosity",
            "cortical_porosity",
            "trabecular_porosity",
            "microporosity",
            "vascular_porosity",
            "subchondral_bone_porosity",
        ],
        inverse_of="apparent_density",
        keywords=["porosity", "porous", "pores", "void fraction"],
    ),
    "elastic_modulus": PhysicsVariable(
        name="elastic_modulus",
        symbol="E",
        units="GPa",
        description="Stiffness — Young's modulus of bone tissue.",
        cortical_range=(15.0, 25.0),
        trabecular_range=(0.05, 5.0),
        node_ids=[
            "elastic_modulus",
            "stiffness",
            "mechanical_stiffness",
        ],
        keywords=[
            "elastic modulus", "young's modulus", "youngs modulus",
            "stiffness", "modulus",
        ],
    ),
    "strength": PhysicsVariable(
        name="strength",
        symbol="σ",
        units="MPa",
        description="Ultimate strength — failure stress of bone tissue.",
        cortical_range=(100.0, 200.0),     # cortical compressive
        trabecular_range=(1.0, 20.0),
        node_ids=[
            "mechanical_strength",
            "compressive_strength",
            "bone_strength",
            "tensile_strength",
        ],
        keywords=[
            "ultimate strength", "compressive strength", "tensile strength",
            "yield strength", "failure stress", "bone strength",
        ],
    ),

    # ── Frost mechanostat variables ──────────────────────────────────────────
    "peak_strain": PhysicsVariable(
        name="peak_strain",
        symbol="ε",
        units="µε",                # microstrain
        description="Peak principal strain magnitude during loading.",
        # The "range" here is the full Frost band the generator may sweep.
        cortical_range=(50.0, 5000.0),
        trabecular_range=(50.0, 5000.0),
        node_ids=[
            "strain",
            "mechanical_loading",
            "mechanical_load",
            "cyclic_loading",
            "loading",
            "fatigue_loading",
            "minimum_effective_strain",
        ],
        keywords=[
            "strain", "microstrain", "peak strain", "mechanostat",
            "mechanical loading", "exercise loading", "disuse",
        ],
    ),
    "bone_adaptation_rate": PhysicsVariable(
        name="bone_adaptation_rate",
        symbol="ΔBMD/yr",
        units="%/year",
        description=(
            "Annual relative change in bone mineral density driven by "
            "the Frost mechanostat regime."
        ),
        # Negative = net resorption, positive = net formation.
        cortical_range=(-3.0, 3.0),
        trabecular_range=(-5.0, 5.0),
        node_ids=[
            "bone_formation",
            "bone_resorption",
            "bone_remodeling",
            "bone_loss",
        ],
        keywords=[
            "bone formation", "bone resorption", "bone remodeling",
            "bone remodelling", "bone loss", "bone gain",
            "bone adaptation",
        ],
    ),

    # ── Paris-law variables (fatigue crack growth) ───────────────────────────
    "stress_intensity_range": PhysicsVariable(
        name="stress_intensity_range",
        symbol="ΔK",
        units="MPa·√m",
        description=(
            "Cyclic stress-intensity factor range at a crack tip — the "
            "input that drives Paris-law fatigue crack growth."
        ),
        cortical_range=(0.3, 2.0),         # physiological cyclic loading
        trabecular_range=(0.1, 1.0),
        node_ids=[
            "cyclic_loading",
            "fatigue_loading",
            "mechanical_loading",
            "loading",
        ],
        keywords=[
            "cyclic loading", "cyclic stress", "fatigue loading",
            "stress intensity", "delta k", "ΔK",
        ],
    ),
    "crack_growth_rate": PhysicsVariable(
        name="crack_growth_rate",
        symbol="da/dN",
        units="m/cycle",
        description="Fatigue crack growth rate predicted by Paris law.",
        # Wide range — physiological growth is ~1e-10 m/cycle, near-failure
        # rates approach 1e-6 m/cycle.  We span the full band.
        cortical_range=(1e-12, 1e-6),
        trabecular_range=(1e-12, 1e-6),
        node_ids=[
            "fatigue_crack_growth_rate",
            "crack_propagation",
            "microcrack",
            "bone_fatigue",
        ],
        keywords=[
            "crack growth", "crack propagation", "microcrack",
            "fatigue crack", "bone fatigue", "da/dn",
        ],
    ),

    # ── Beam-bending variables ───────────────────────────────────────────────
    "cortical_thickness": PhysicsVariable(
        name="cortical_thickness",
        symbol="t",
        units="mm",
        description="Cortical wall thickness of a long-bone diaphysis.",
        cortical_range=(1.5, 6.0),         # human femoral midshaft typical
        trabecular_range=(0.1, 0.5),       # trabecular elements
        node_ids=[
            "cortical_thickness",
            "cortical_bone_thickness",
            "thickness",
        ],
        keywords=[
            "cortical thickness", "cortex thickness", "wall thickness",
            "cortical wall",
        ],
    ),
    "bending_resistance": PhysicsVariable(
        name="bending_resistance",
        symbol="EI",
        units="% of baseline",
        description=(
            "Section bending resistance (∝ second moment of area I, "
            "scaled to a baseline of 100%).  Thicker cortex → larger I → "
            "more resistance to bending."
        ),
        # Wide bounds so Round 2 is essentially a sanity check rather
        # than a regime gate — the law's domain is enforced by Round 3.
        cortical_range=(5.0, 1000.0),
        trabecular_range=(5.0, 1000.0),
        node_ids=[
            "bending_stiffness",
            "bending_strength",
            "mechanical_strength",
            "bone_strength",
        ],
        keywords=[
            "bending strength", "bending stiffness", "bending resistance",
            "flexural", "second moment of area", "section modulus",
        ],
    ),
}


# ── Lookup helpers ────────────────────────────────────────────────────────────


def variable_for_node(node_id: str) -> str | None:
    """Return the physics-variable name that a graph node_id proxies, if any."""
    for var_name, var in VARS.items():
        if node_id in var.node_ids:
            return var_name
    return None


def canonical_node(var_name: str, available_nodes: set[str]) -> str | None:
    """
    Pick the best graph node_id for a physics variable.

    Walks the variable's node_ids in priority order (the registry order is
    chosen so the most-canonical proxy comes first) and returns the first
    one that exists in the available set.
    """
    var = VARS.get(var_name)
    if var is None:
        return None
    for nid in var.node_ids:
        if nid in available_nodes:
            return nid
    return None


def variables_in_query(query: str) -> list[str]:
    """
    Identify which physics variables a free-text query touches.

    Matches each variable's curated ``keywords`` list with word-boundary
    regex (case-insensitive).  Multi-word phrases must appear as a
    contiguous span — "bending strength" matches but "bending and
    compressive strength" does not anchor the bending variable on the
    word "strength" alone.  This keeps the gating tight: a query about
    porosity and modulus no longer pulls in beam bending just because
    the word "strength" appears somewhere.
    """
    q = query.lower()
    hits: list[str] = []
    for var_name, var in VARS.items():
        for kw in var.keywords:
            kw_lower = kw.lower()
            # \b doesn't behave well around non-word characters such as
            # "/" or "Δ"; pad with optional whitespace boundaries instead
            # of relying solely on \b.
            pattern = r"(?<!\w)" + re.escape(kw_lower) + r"(?!\w)"
            if re.search(pattern, q):
                hits.append(var_name)
                break
    return hits


def in_physical_range(
    var_name: str,
    value: float,
    scenario: str = "cortical",
    tolerance: float = 0.20,
) -> bool:
    """
    Is `value` within the physical range for this variable?

    A `tolerance` (default ±20%) widens the canonical range so reasonable
    edge cases (paediatric, severely osteoporotic) still pass.
    """
    var = VARS.get(var_name)
    if var is None:
        return False
    lo, hi = var.cortical_range if scenario == "cortical" else var.trabecular_range
    span = hi - lo
    return (lo - tolerance * span) <= value <= (hi + tolerance * span)
