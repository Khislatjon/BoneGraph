"""
reasoning/physics_gen.py
────────────────────────
Physics-driven hypothesis generator.

This module is the heart of the new Reasoning tab.  Instead of walking
the knowledge graph to retrieve chains the corpus already states, the
generator runs each applicable physical law over a small set of
plausible perturbations and emits **predicted edges with quantitative
magnitudes**.  The graph is consulted only afterwards (in
``reasoning/critic.py`` and ``reasoning/novelty.py``) to corroborate or
falsify the prediction.

Why this is novel
-----------------
Buehler-style graph reasoning (Buehler 2024, arXiv:2403.11996) feeds
multi-path subgraphs to an LLM and asks it to be creative.  That works
for materials in general but treats physics as decorative vocabulary.
For bone we have *quantitative* physical laws — Currey, Frost
mechanostat, Paris — so we can let physics *generate* hypotheses with
predicted numbers, not just qualitative directions.

v1 covers Currey's law only.  Frost mechanostat and Paris extend the
same generator pattern.

Pipeline per query
------------------
1. Anchor — identify which physics variables (ρ, φ, E, σ) the query
   touches via :func:`reasoning.physics_vars.variables_in_query`.
2. Choose a bone-tissue scenario (cortical, trabecular, or both)
   based on which proxy nodes the query mentions.
3. For each applicable law, sweep a small set of physically meaningful
   perturbations (e.g. Δφ = +5pp, +10pp, +20pp) and compute the
   predicted output magnitude.
4. Materialise each perturbation as a :class:`PhysicsHypothesis` whose
   ``chain`` references concrete graph node_ids so the front-end can
   render it like the old chain-card UI.

Output
------
Each hypothesis is a structured dataclass with the predicted ΔY/Y, the
assumed baseline inputs, the law that fired, and the chain of graph
nodes the hypothesis is *about*.  Downstream stages add critic results
and novelty status.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

from reasoning.physics import (
    currey_delta_from_porosity,
    currey_delta_from_density,
)
from reasoning.physics_vars import (
    VARS,
    canonical_node,
    variable_for_node,
    variables_in_query,
)

logger = logging.getLogger(__name__)


# ── Hypothesis dataclass ──────────────────────────────────────────────────────


@dataclass
class PhysicsHypothesis:
    """
    A single physics-derived hypothesis with a quantitative prediction.

    Attributes
    ----------
    chain : list[str]
        Graph node_ids spanning input → output.  Used to render the
        chain card in the UI and to look up corpus evidence.
    chain_labels : list[str]
        Human-readable labels for the same nodes (parallel to ``chain``).
    relations : list[str]
        Edge relation between consecutive chain nodes (``"increases"``
        or ``"decreases"``).  ``len(relations) == len(chain) - 1``.
    law : str
        Short name of the firing law (``"Currey's law"``).
    law_form : str
        The mathematical form, e.g. ``"E ∝ ρ^2.5"``.
    scenario : str
        ``"cortical"`` or ``"trabecular"`` — which tissue regime the
        baseline values came from.
    input_var : str
        Physics-variable name being perturbed (e.g. ``"porosity"``).
    output_var : str
        Physics-variable name being predicted (e.g. ``"elastic_modulus"``).
    perturbation : str
        Human-readable perturbation, e.g. ``"+10 pp porosity"``.
    prediction : str
        Human-readable predicted output, e.g. ``"-23.4% E"``.
    delta_input : dict
        Structured input perturbation.
    delta_output : dict
        Structured output prediction including magnitude and direction.
    assumed_inputs : dict
        Every assumed numerical input the prediction depends on.
    confidence : float
        0–1.  Reflects how well the proxy nodes cover the variables and
        whether the perturbation lies in a calibrated range.
    """

    chain: list[str]
    chain_labels: list[str]
    relations: list[str]

    law: str
    law_form: str
    scenario: str
    input_var: str
    output_var: str

    perturbation: str
    prediction: str
    delta_input: dict
    delta_output: dict
    assumed_inputs: dict

    confidence: float = 1.0

    # Fields filled in by later stages
    critique: "CritiqueResult | None" = None  # noqa: F821 (forward ref)
    novelty: str = ""
    novelty_reason: str = ""
    corpus_disclaimer: str | None = None
    score: float = 0.0

    # ── Display helpers ──────────────────────────────────────────────────────

    def chain_str(self) -> str:
        """Render chain as a readable ``A → B`` string (used by novelty.py)."""
        return " → ".join(self.chain_labels)

    @property
    def summary(self) -> str:
        """One-sentence textual summary used by the novelty classifier."""
        return (
            f"Currey's law predicts that {self.perturbation} "
            f"in {self.scenario} bone yields {self.prediction} "
            f"({self.chain_labels[0]} → {self.chain_labels[-1]})."
        )

    def __str__(self) -> str:
        return (
            f"[{self.law}]  {self.chain_str()}\n"
            f"  perturbation : {self.perturbation}\n"
            f"  prediction   : {self.prediction}\n"
            f"  scenario     : {self.scenario}"
        )


# ── Currey perturbation grid ──────────────────────────────────────────────────
#
# Each row is a (Δφ_abs, Δρ_pct, Δstrength_pct) sweep level.  We
# deliberately stay modest — physics-driven hypotheses are most useful
# in the small-perturbation regime where the power-law fit is reliable.
# Over-large perturbations push the proxy ρ outside its fitted range
# and the prediction loses meaning.

_CURREY_PERTURBATIONS: list[tuple[str, float, float]] = [
    # (label,           delta_phi_abs,  delta_rho_pct)
    ("small",            0.05,           5.0),
    ("moderate",         0.10,          10.0),
    ("large",            0.20,          20.0),
]


# Currey's law applies to the σ ∝ ρⁿ scaling for compressive strength
# as well, with an exponent close to 2 (slightly below E).  We re-use
# the same machinery and tag the output variable as ``strength``.
_CURREY_OUTPUTS: list[tuple[str, float]] = [
    # (output_variable_name, exponent_used)
    ("elastic_modulus", 2.5),
    ("strength",        2.0),
]


# ── Generator ─────────────────────────────────────────────────────────────────


class PhysicsGenerator:
    """
    Generate quantitative hypotheses by applying physical laws.

    Construction is cheap; reuse one instance per process.

    Parameters
    ----------
    available_nodes : set[str]
        node_ids present in the active graph.  The generator picks
        canonical proxies from this set so chains always reference real
        graph nodes.
    """

    def __init__(self, available_nodes: set[str]) -> None:
        self._available = available_nodes

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        max_per_law: int = 6,
    ) -> list[PhysicsHypothesis]:
        """
        Run all applicable physical laws against ``query``.

        Parameters
        ----------
        query : str
            Free-text user query.
        max_per_law : int
            Cap per law to keep the response compact.

        Returns
        -------
        list[PhysicsHypothesis]
        """
        query_vars = variables_in_query(query)
        scenarios = self._scenarios_for_query(query)

        if not query_vars:
            logger.info("Query %r matched no physics variables.", query)
            return []

        results: list[PhysicsHypothesis] = []

        # — Currey's law —
        currey = self._apply_currey(query_vars, scenarios)
        results.extend(currey[:max_per_law])

        return results

    # ── Scenario selection ────────────────────────────────────────────────────

    def _scenarios_for_query(self, query: str) -> list[str]:
        """
        Decide which bone-tissue regime(s) the query implies.

        If the query explicitly says "cortical" / "trabecular", honour
        that.  Otherwise emit hypotheses for both and let the UI rank.
        """
        q = query.lower()
        if "cortical" in q and "trabecular" not in q:
            return ["cortical"]
        if "trabecular" in q and "cortical" not in q:
            return ["trabecular"]
        return ["cortical", "trabecular"]

    # ── Currey's law (Items 1 + 2) ────────────────────────────────────────────

    def _apply_currey(
        self,
        query_vars: list[str],
        scenarios: list[str],
    ) -> list[PhysicsHypothesis]:
        """
        Currey's law:  E ∝ ρⁿ, with ρ = ρ_full · (1 - φ).

        Generates predictions for:
          • porosity-driven ΔE/E and Δσ/σ
          • density-driven  ΔE/E and Δσ/σ

        Skips the porosity branch unless ``porosity`` (or its inverse,
        ``apparent_density``) is present in ``query_vars`` — same for
        the density branch.
        """
        out: list[PhysicsHypothesis] = []

        # The two branches — input variable to perturb.
        wants_porosity = "porosity" in query_vars or "apparent_density" in query_vars
        wants_density  = "apparent_density" in query_vars or "porosity" in query_vars

        # We want at least one of E, σ on the output side.
        wants_E        = "elastic_modulus" in query_vars
        wants_strength = "strength" in query_vars
        # If the query gave neither, default to E (most fundamental).
        if not (wants_E or wants_strength):
            wants_E = True

        for scenario in scenarios:
            phi_baseline    = self._baseline("porosity",         scenario)
            rho_baseline    = self._baseline("apparent_density", scenario)

            for output_var, exponent in _CURREY_OUTPUTS:
                if output_var == "elastic_modulus" and not wants_E:
                    continue
                if output_var == "strength" and not wants_strength:
                    continue

                if wants_porosity:
                    out.extend(
                        self._currey_porosity_sweep(
                            scenario=scenario,
                            phi_baseline=phi_baseline,
                            rho_baseline=rho_baseline,
                            output_var=output_var,
                            exponent=exponent,
                        )
                    )
                if wants_density:
                    out.extend(
                        self._currey_density_sweep(
                            scenario=scenario,
                            rho_baseline=rho_baseline,
                            output_var=output_var,
                            exponent=exponent,
                        )
                    )

        return out

    # ── Currey sub-routines ───────────────────────────────────────────────────

    def _currey_porosity_sweep(
        self,
        scenario: str,
        phi_baseline: float,
        rho_baseline: float,
        output_var: str,
        exponent: float,
    ) -> list[PhysicsHypothesis]:
        out: list[PhysicsHypothesis] = []
        input_node  = canonical_node("porosity", self._available)
        output_node = canonical_node(output_var, self._available)
        if input_node is None or output_node is None:
            return out

        for label, dphi, _ in _CURREY_PERTURBATIONS:
            try:
                pred = currey_delta_from_porosity(
                    porosity_baseline=phi_baseline,
                    porosity_change_abs=dphi,
                    exponent=exponent,
                )
            except ValueError:
                continue

            # Relation reflects the *causal sign between the two variables*,
            # not the perturbation direction.  Currey: porosity has a
            # negative effect on E (ρ ↓ when φ ↑, so E ↓ via E ∝ ρⁿ).
            # The perturbation's sign lives in the prediction string.
            out.append(self._build(
                input_node=input_node,
                output_node=output_node,
                relation="decreases",
                law="Currey's law",
                law_form=f"{VARS[output_var].symbol} ∝ ρ^{exponent}  (with ρ = ρ_full·(1−φ))",
                scenario=scenario,
                input_var="porosity",
                output_var=output_var,
                perturbation=(
                    f"+{int(dphi*100)} pp porosity "
                    f"(φ: {phi_baseline:.2f} → {pred['phi_new']:.2f})"
                ),
                prediction=self._fmt_pct(pred["delta_E_pct"], output_var),
                delta_input={
                    "variable":      "porosity",
                    "change_abs":    dphi,
                    "from":          phi_baseline,
                    "to":            pred["phi_new"],
                    "magnitude":     label,
                },
                delta_output={
                    "variable":      output_var,
                    "change_pct":    pred["delta_E_pct"],
                    "rho_ratio":     pred["rho_ratio"],
                    "exponent":      exponent,
                },
                assumed_inputs={
                    "phi_baseline":  phi_baseline,
                    "rho_baseline":  rho_baseline,
                    "exponent":      exponent,
                    "scenario":      scenario,
                },
            ))
        return out

    def _currey_density_sweep(
        self,
        scenario: str,
        rho_baseline: float,
        output_var: str,
        exponent: float,
    ) -> list[PhysicsHypothesis]:
        out: list[PhysicsHypothesis] = []
        input_node  = canonical_node("apparent_density", self._available)
        output_node = canonical_node(output_var, self._available)
        if input_node is None or output_node is None:
            return out

        for label, _, drho_pct in _CURREY_PERTURBATIONS:
            for sign_pct in (drho_pct, -drho_pct):
                try:
                    pred = currey_delta_from_density(
                        density_baseline_g_cm3=rho_baseline,
                        density_change_pct=sign_pct,
                        exponent=exponent,
                    )
                except ValueError:
                    continue

                # Currey: density has a *positive* effect on E.  The
                # perturbation can be ±, but the variable-level relation
                # is always "increases".
                out.append(self._build(
                    input_node=input_node,
                    output_node=output_node,
                    relation="increases",
                    law="Currey's law",
                    law_form=f"{VARS[output_var].symbol} ∝ ρ^{exponent}",
                    scenario=scenario,
                    input_var="apparent_density",
                    output_var=output_var,
                    perturbation=(
                        f"{sign_pct:+.0f}% density "
                        f"(ρ: {rho_baseline:.2f} → {pred['rho_new']:.2f} g/cm³)"
                    ),
                    prediction=self._fmt_pct(pred["delta_E_pct"], output_var),
                    delta_input={
                        "variable":   "apparent_density",
                        "change_pct": sign_pct,
                        "from":       rho_baseline,
                        "to":         pred["rho_new"],
                        "magnitude":  label,
                    },
                    delta_output={
                        "variable":   output_var,
                        "change_pct": pred["delta_E_pct"],
                        "rho_ratio":  pred["rho_ratio"],
                        "exponent":   exponent,
                    },
                    assumed_inputs={
                        "rho_baseline": rho_baseline,
                        "exponent":     exponent,
                        "scenario":     scenario,
                    },
                ))
        return out

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _baseline(self, var_name: str, scenario: str) -> float:
        """Pick a representative baseline for a variable in a given scenario."""
        var = VARS[var_name]
        lo, hi = var.cortical_range if scenario == "cortical" else var.trabecular_range
        # Use the geometric mean — a good representative point on a power-law axis.
        return (lo * hi) ** 0.5

    def _fmt_pct(self, pct: float, output_var: str) -> str:
        """Format a percentage change for display."""
        sym = VARS[output_var].symbol
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}% Δ{sym}/{sym}"

    def _build(
        self,
        input_node: str,
        output_node: str,
        relation: str,
        **fields,
    ) -> PhysicsHypothesis:
        """Materialise a one-step chain hypothesis."""
        chain = [input_node, output_node]
        labels = [n.replace("_", " ") for n in chain]
        return PhysicsHypothesis(
            chain=chain,
            chain_labels=labels,
            relations=[relation],
            **fields,
        )
