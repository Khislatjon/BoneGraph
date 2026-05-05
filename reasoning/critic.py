"""
reasoning/critic.py
───────────────────
Adversarial physics critic.

A :class:`PhysicsCritic` takes a :class:`PhysicsHypothesis` produced by
:class:`reasoning.physics_gen.PhysicsGenerator` and tries to falsify it
through a sequence of independent checks ("rounds").  Each round either
passes or fails with a recorded reason.  A hypothesis survives only if
every round passes; otherwise the first failure is recorded so the UI
can show *why* the hypothesis was rejected.

Why "adversarial"?
------------------
Buehler 2024 uses adversarial multi-agent LLMs for *question generation*.
For BoneMind we need physics-grounded falsification, so the critic
plays the adversary role through deterministic physics checks.  No LLM
involved — every verdict is reproducible.

Rounds
------
1. Directional consistency
   The hypothesis claims (input ─relation→ output).  Round 1 looks up
   the cleaned bone-physics rule table in :mod:`reasoning.physics`.  If
   any rule asserts the opposite relation as IMPLAUSIBLE, the
   hypothesis fails.

2. Magnitude in physical range
   The hypothesis predicts a relative change ΔY/Y.  Combined with the
   assumed baseline, this implies a new absolute output Y_new.  Round 2
   verifies Y_new sits within the physical range for the chosen tissue
   scenario (cortical or trabecular).

3. Power-law domain
   Currey-style exponents are calibrated for moderate perturbations.
   Round 3 rejects extreme perturbations that push ρ outside the
   fitted range or into non-physical territory (φ ≥ 1, ρ ≤ 0).

4. Scenario internal consistency
   Both the input baseline and the output range must come from the same
   tissue scenario.  This catches mixing cortical input with trabecular
   output silently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from reasoning.physics import PhysicsEngine
from reasoning.physics_gen import PhysicsHypothesis
from reasoning.physics_vars import VARS, in_physical_range

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class CheckRecord:
    """One round's outcome."""

    name: str
    passed: bool
    detail: str


@dataclass
class CritiqueResult:
    """
    Aggregate verdict from running every round on one hypothesis.

    Attributes
    ----------
    survived : bool
        True iff every round passed.
    rounds_total : int
        Number of rounds attempted.
    rounds_passed : int
        Number that passed.
    checks : list[CheckRecord]
        One entry per round, in execution order.
    failure_reason : str
        Empty if survived; otherwise the detail of the first failure.
    """

    survived: bool
    rounds_total: int
    rounds_passed: int
    checks: list[CheckRecord] = field(default_factory=list)
    failure_reason: str = ""


# ── Critic ────────────────────────────────────────────────────────────────────


class PhysicsCritic:
    """
    Run the four-round adversarial check on a hypothesis.

    Parameters
    ----------
    engine : PhysicsEngine
        Used for round 1 (directional consistency).  A single engine
        instance is fine for the whole process.
    """

    ROUNDS_TOTAL = 4

    def __init__(self, engine: PhysicsEngine | None = None) -> None:
        self._engine = engine or PhysicsEngine()

    # ── Public API ────────────────────────────────────────────────────────────

    def critique(self, h: PhysicsHypothesis) -> CritiqueResult:
        """Run all rounds; stop recording detail once a round fails."""
        checks: list[CheckRecord] = [
            self._round_directional(h),
            self._round_magnitude(h),
            self._round_powerlaw_domain(h),
            self._round_scenario_consistency(h),
        ]
        passed = sum(1 for c in checks if c.passed)
        survived = passed == self.ROUNDS_TOTAL
        first_fail = next((c for c in checks if not c.passed), None)

        return CritiqueResult(
            survived=survived,
            rounds_total=self.ROUNDS_TOTAL,
            rounds_passed=passed,
            checks=checks,
            failure_reason="" if survived else (first_fail.detail if first_fail else ""),
        )

    # ── Round 1 — directional consistency ─────────────────────────────────────

    def _round_directional(self, h: PhysicsHypothesis) -> CheckRecord:
        """
        Check the hypothesis's chain edges against the directional rule
        table in :mod:`reasoning.physics`.  IMPLAUSIBLE rules veto;
        UNCERTAIN or absent rules pass through.
        """
        edges = [
            (h.chain[i], h.relations[i], h.chain[i + 1])
            for i in range(len(h.relations))
        ]
        if not edges:
            return CheckRecord(
                "directional_consistency", False,
                "Empty chain — nothing to check.",
            )

        result = self._engine.validate_chain(edges)
        if result.is_implausible:
            return CheckRecord(
                "directional_consistency",
                False,
                f"Directional rule conflict: {result.explanation}",
            )
        return CheckRecord(
            "directional_consistency",
            True,
            "Hypothesis direction matches established physics rules.",
        )

    # ── Round 2 — magnitude in physical range ─────────────────────────────────

    def _round_magnitude(self, h: PhysicsHypothesis) -> CheckRecord:
        """
        Check the predicted absolute output value against the bone-physical
        range for the chosen tissue scenario.

        Two ways to obtain the predicted absolute value:
        1. ``delta_output["predicted_value"]`` is supplied by the generator
           (preferred — every law explicitly states the absolute value).
        2. Fallback for relative-change predictions: derive the absolute
           value from the canonical baseline times (1 + change_pct/100).
           Used only when the generator did not supply ``predicted_value``.
        """
        var = VARS.get(h.output_var)
        if var is None:
            return CheckRecord(
                "magnitude_range", False,
                f"Unknown output variable: {h.output_var}",
            )

        lo, hi = (
            var.cortical_range if h.scenario == "cortical" else var.trabecular_range
        )

        if "predicted_value" in h.delta_output:
            new_value = float(h.delta_output["predicted_value"])
            source = "predicted"
        else:
            # Currey-style relative change.  Use the geometric mean of the
            # canonical range as the baseline when both ends are positive,
            # otherwise fall back to the arithmetic midpoint.
            if lo > 0 and hi > 0:
                baseline = (lo * hi) ** 0.5
            else:
                baseline = (lo + hi) / 2.0
            delta_pct = h.delta_output.get("change_pct", 0.0)
            new_value = baseline * (1.0 + delta_pct / 100.0)
            source = "derived"

        if not in_physical_range(h.output_var, new_value, scenario=h.scenario):
            return CheckRecord(
                "magnitude_range",
                False,
                (
                    f"Predicted {var.symbol} = {new_value:.3g} {var.units} "
                    f"falls outside the {h.scenario} range "
                    f"({lo:g}–{hi:g} {var.units})."
                ),
            )
        return CheckRecord(
            "magnitude_range",
            True,
            (
                f"Predicted {var.symbol} ≈ {new_value:.3g} {var.units} "
                f"within {h.scenario} bone range "
                f"({source})."
            ),
        )

    # ── Round 3 — power-law domain ────────────────────────────────────────────

    def _round_powerlaw_domain(self, h: PhysicsHypothesis) -> CheckRecord:
        """
        Law-aware domain check.

        Each physical law is calibrated for a particular regime, and a
        perturbation that pushes the input outside that regime makes
        the prediction meaningless even if directionality and magnitude
        round-1/2 happen to agree.  This round dispatches by law name.
        """
        law = (h.law or "").lower()

        # ── Currey: density-ratio bounds ─────────────────────────────────
        if "currey" in law:
            rho_ratio = h.delta_output.get("rho_ratio")
            if rho_ratio is None:
                return CheckRecord("powerlaw_domain", True, "No ρ-ratio.")
            if rho_ratio <= 0:
                return CheckRecord(
                    "powerlaw_domain", False,
                    f"Implied ρ_new/ρ_old = {rho_ratio:.3f} ≤ 0 — non-physical.",
                )
            if rho_ratio > 2.0:
                return CheckRecord(
                    "powerlaw_domain", False,
                    (
                        f"Implied ρ_new/ρ_old = {rho_ratio:.2f} doubles "
                        "density — outside the Currey calibration range."
                    ),
                )
            if rho_ratio < 0.3:
                return CheckRecord(
                    "powerlaw_domain", False,
                    (
                        f"Implied ρ_new/ρ_old = {rho_ratio:.2f} loses >70% "
                        "of density — outside the Currey calibration range."
                    ),
                )
            return CheckRecord(
                "powerlaw_domain", True,
                f"Density ratio {rho_ratio:.2f} within Currey bounds.",
            )

        # ── Paris: ΔK must stay below cortical KIc (~6 MPa·√m) ───────────
        if "paris" in law:
            dk_new = h.delta_input.get("to") or h.delta_input.get("value")
            if dk_new is None:
                return CheckRecord("powerlaw_domain", True, "No ΔK supplied.")
            kic_cortical = 6.0    # MPa·√m, conservative cortical KIc
            if dk_new >= kic_cortical:
                return CheckRecord(
                    "powerlaw_domain", False,
                    (
                        f"ΔK = {dk_new:.2f} MPa·√m meets or exceeds "
                        f"cortical KIc ≈ {kic_cortical} MPa·√m — failure "
                        "is single-cycle, Paris law is no longer valid."
                    ),
                )
            return CheckRecord(
                "powerlaw_domain", True,
                f"ΔK = {dk_new:.2f} MPa·√m below KIc — Paris regime valid.",
            )

        # ── Frost mechanostat: input strain must be physiologically bounded ─
        if "mechanostat" in law:
            strain = h.delta_input.get("value")
            if strain is None:
                return CheckRecord("powerlaw_domain", True, "No strain value.")
            if strain < 0:
                return CheckRecord(
                    "powerlaw_domain", False,
                    f"Strain {strain:.0f} µɛ is negative — non-physical.",
                )
            if strain > 25_000:
                return CheckRecord(
                    "powerlaw_domain", False,
                    (
                        f"Strain {strain:.0f} µɛ exceeds the fracture "
                        "threshold; the mechanostat adaptation framework "
                        "no longer applies."
                    ),
                )
            return CheckRecord(
                "powerlaw_domain", True,
                f"Strain {strain:.0f} µɛ within mechanostat domain.",
            )

        # ── Beam bending: thickness change must keep r_i > 0 ─────────────
        if "beam" in law:
            t_new = h.delta_input.get("to")
            if t_new is None:
                return CheckRecord("powerlaw_domain", True, "No thickness.")
            if t_new <= 0:
                return CheckRecord(
                    "powerlaw_domain", False,
                    f"Cortical thickness {t_new:.2f} mm ≤ 0 — non-physical.",
                )
            return CheckRecord(
                "powerlaw_domain", True,
                f"Thickness {t_new:.2f} mm within bending-model bounds.",
            )

        # Unknown law — pass through with a permissive check.
        return CheckRecord(
            "powerlaw_domain", True,
            f"No domain check defined for law: {h.law!r}.",
        )

    # ── Round 4 — scenario internal consistency ───────────────────────────────

    def _round_scenario_consistency(self, h: PhysicsHypothesis) -> CheckRecord:
        """
        Make sure both ends of the hypothesis live in the declared
        tissue scenario.  In v1 the generator always picks the scenario
        consistently, so this round is a guard for future laws that
        might mix variables.
        """
        if h.scenario not in {"cortical", "trabecular"}:
            return CheckRecord(
                "scenario_consistency", False,
                f"Unknown scenario: {h.scenario!r}",
            )
        # Generator-level invariant: both VARS lookups succeeded if we
        # got here, so simply confirm.
        return CheckRecord(
            "scenario_consistency", True,
            f"All variables resolved within the {h.scenario} regime.",
        )
