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
    """

    name: str
    symbol: str
    units: str
    description: str
    cortical_range: tuple[float, float]
    trabecular_range: tuple[float, float]
    node_ids: list[str] = field(default_factory=list)
    inverse_of: str | None = None


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

    Matches by substring against each variable's node_ids and against
    the variable name itself, so queries can use either graph-style
    ("bmd", "porosity") or domain-style ("density", "modulus") wording.
    """
    q = query.lower()
    hits: list[str] = []
    for var_name, var in VARS.items():
        # Match variable name's tokens
        name_tokens = var_name.split("_")
        if any(t in q for t in name_tokens if len(t) > 3):
            hits.append(var_name)
            continue
        # Match any of the variable's node_ids by token
        if any(
            any(t in q for t in nid.split("_") if len(t) > 3)
            for nid in var.node_ids
        ):
            hits.append(var_name)
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
