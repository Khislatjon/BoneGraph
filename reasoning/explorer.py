"""
reasoning/explorer.py
─────────────────────
Phase 5 — active exploration over the v2 equation graph.

The Explorer answers the "where does surprise come from?" critique from
the original planning session.  It runs without a user query.  For each
of a small set of physically meaningful sweeps over the variable graph,
it propagates predictions through forward inference, asks the corpus
how attested that direction is, and ranks candidates by

    surprise = magnitude × physics_confidence × corpus_weight

This is **not** numerical-claim extraction.  We do not parse "da/dN at
porosity 0.20 = 5e-9 from Vashishth 2003" out of papers.  What this
delivers is:

* a physics-derived directional prediction (e.g. "porosity ↑ ⇒ da/dN ↑")
  with a quantitative magnitude over a sweep,
* a corpus-presence label (GROUNDED / SPECULATIVE / NOVEL) for the
  free-text rendering of that prediction,
* a single ranked list so the user can scan for the most promising
  candidate hypotheses without writing a query.

Surprises with high magnitude AND high corpus attestation are the
priority — those are claims where physics says something concrete and
the literature is actively discussing the same topic.  Disagreement, if
present, will be detectable; agreement is the dull case.

The sweep design is hand-picked but not hard-coded chains.  Each sweep
declares ``target`` and ``sweep_var``; the chain itself is discovered
by :meth:`RelationRegistry.forward` from variable sharing, exactly as
the rest of the v2 pipeline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from reasoning.relation import ForwardResult, RelationRegistry

logger = logging.getLogger(__name__)


# ── Sweep specifications ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SweepSpec:
    """One hand-picked, principled exploration sweep."""

    target: str                        # output state variable
    sweep_var: str                     # input being varied
    sweep_values: tuple[float, ...]    # 3–5 physically meaningful points
    held: dict[str, float]             # other chain roots, pinned
    corpus_query: str                  # academic-flavoured text for SPECTER2
    title: str                         # short label for the UI


# Each spec touches a different part of the graph, so the seven specs
# together exercise every relation at least once.  Held values come
# from the same literature midpoints the v2 ask-endpoint defaults to.

_SWEEPS: tuple[_SweepSpec, ...] = (
    _SweepSpec(
        target="E",
        sweep_var="phi",
        sweep_values=(0.05, 0.15, 0.30),
        held={},
        title="porosity → elastic modulus",
        corpus_query=(
            "Cortical porosity and elastic modulus of bone tissue; "
            "density-modulated stiffness via Currey's power law."
        ),
    ),
    _SweepSpec(
        target="da_dN",
        sweep_var="phi",
        sweep_values=(0.05, 0.15, 0.30),
        held={"dK": 1.0},
        title="porosity → fatigue crack growth",
        corpus_query=(
            "Cortical porosity and fatigue crack propagation rate in "
            "cortical bone under cyclic loading; Paris law with "
            "density-modulated prefactor."
        ),
    ),
    _SweepSpec(
        target="da_dN",
        sweep_var="dK",
        sweep_values=(0.4, 1.0, 2.0),
        held={"phi": 0.10},
        title="stress-intensity range → crack growth",
        corpus_query=(
            "Stress intensity range and Paris law fatigue crack growth "
            "rate in cortical bone."
        ),
    ),
    _SweepSpec(
        target="sigma",
        sweep_var="t",
        sweep_values=(3.0, 5.0, 7.0),
        held={"R": 13.0, "M": 40_000.0},
        title="cortical thickness → bending stress",
        corpus_query=(
            "Cortical thickness and bending stress in long bone "
            "diaphysis; Euler–Bernoulli beam mechanics."
        ),
    ),
    _SweepSpec(
        target="eps",
        sweep_var="M",
        sweep_values=(20_000.0, 40_000.0, 80_000.0),
        held={"phi": 0.10, "R": 13.0, "t": 5.0},
        title="applied moment → mechanical strain",
        corpus_query=(
            "Applied bending moment, peak strain magnitude, and "
            "elastic response in cortical bone."
        ),
    ),
    _SweepSpec(
        target="dBMD_dt",
        sweep_var="M",
        sweep_values=(10_000.0, 40_000.0, 80_000.0),
        held={"phi": 0.10, "R": 13.0, "t": 5.0},
        title="loading → BMD adaptation rate",
        corpus_query=(
            "Mechanical loading magnitude and bone mineral density "
            "adaptation rate; Frost mechanostat regulation of remodelling."
        ),
    ),
    _SweepSpec(
        target="dBMD_dt",
        sweep_var="phi",
        sweep_values=(0.05, 0.20, 0.40),
        held={"R": 13.0, "t": 5.0, "M": 40_000.0},
        title="porosity → BMD adaptation rate",
        corpus_query=(
            "Cortical porosity, mechanostat strain setpoint, and "
            "bone remodelling adaptation rate."
        ),
    ),
)


# ── Output dataclass ──────────────────────────────────────────────────────────


@dataclass
class ExplorationCandidate:
    """One ranked exploration hypothesis."""

    title: str
    target: str
    sweep_var: str
    held: dict[str, float]
    chain_vars: list[str]
    citations: list[str]

    sweep_values: list[float]
    sweep_means: list[float]
    sweep_p5: list[float]
    sweep_p95: list[float]

    direction: str           # "increases" | "decreases" | "non-monotonic"
    relative_change: float   # (end - start) / |start|
    physics_confidence: float
    magnitude: float

    corpus_label: str        # GROUNDED | SPECULATIVE | NOVEL | UNCERTAIN
    corpus_score: float
    corpus_similarity: float
    corpus_keyword_hits: int

    surprise_score: float
    summary: str
    corpus_query: str
    error: Optional[str] = None


# ── Explorer ──────────────────────────────────────────────────────────────────


_CORPUS_WEIGHTS: dict[str, float] = {
    # Higher weight when the corpus actively discusses this region of
    # the variable space — that is where physics-vs-literature
    # disagreement, if any, becomes detectable.  NOVEL (no corpus
    # contact) is the least actionable for surprise-hunting and gets
    # the lowest weight.
    "GROUNDED":    1.00,
    "SPECULATIVE": 0.65,
    "NOVEL":       0.30,
    "UNCERTAIN":   0.50,
}


class Explorer:
    """
    Walk the variable graph, score candidate hypotheses, and rank them.

    Parameters
    ----------
    registry : RelationRegistry
        The active equation-graph registry.
    novelty_classifier : optional
        :class:`NoveltyClassifier` (or anything that exposes
        ``classify_text(str) → NoveltyResult``).  When omitted, all
        candidates score with corpus_label="UNCERTAIN".
    n_samples : int
        Monte-Carlo samples per forward call.  2000 is a good balance
        between uncertainty resolution and exploration latency.
    seed : int
        Shared RNG seed so the ranking is deterministic across runs.
    """

    def __init__(
        self,
        registry: RelationRegistry,
        novelty_classifier=None,
        *,
        n_samples: int = 2000,
        seed: int = 271_828,
    ) -> None:
        self._registry = registry
        self._novelty = novelty_classifier
        self._n_samples = n_samples
        self._seed = seed

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> list[ExplorationCandidate]:
        """Evaluate every sweep, rank candidates by surprise score."""
        candidates: list[ExplorationCandidate] = []
        for spec in _SWEEPS:
            cand = self._evaluate(spec)
            if cand is not None:
                candidates.append(cand)
        candidates.sort(key=lambda c: c.surprise_score, reverse=True)
        return candidates

    # ── Per-sweep evaluation ──────────────────────────────────────────────────

    def _evaluate(self, spec: _SweepSpec) -> ExplorationCandidate | None:
        """Run all sweep points; build a single candidate."""
        try:
            results = self._run_sweep(spec)
        except Exception as exc:
            logger.warning(
                "Exploration sweep %s/%s failed: %s",
                spec.target, spec.sweep_var, exc,
            )
            return self._failure_candidate(spec, str(exc))

        means  = [r.mean for r in results]
        p5s    = [r.p5   for r in results]
        p95s   = [r.p95  for r in results]
        rel_uncs = [
            r.relative_uncertainty for r in results
            if math.isfinite(r.relative_uncertainty)
        ]

        # Direction over the sweep.  Tolerate small monotone noise.
        direction = self._classify_direction(means)

        # Relative change end-to-end (clamped for log).
        if abs(means[0]) > 1e-30:
            rel_change = (means[-1] - means[0]) / abs(means[0])
        else:
            rel_change = float("inf")

        # Physics confidence — narrower bands → higher confidence.  Cap
        # the relative uncertainty at 5× so a 1000% band doesn't push
        # everything to zero.
        avg_unc = (sum(rel_uncs) / len(rel_uncs)) if rel_uncs else 1.0
        avg_unc = min(avg_unc, 5.0)
        physics_confidence = 1.0 / (1.0 + avg_unc)

        # Magnitude on a log scale (so a 10% change does not get drowned
        # by a 1000% one).  We bound the relative change to keep the
        # ranking robust to occasional pathological sweeps.
        bounded_rel = min(abs(rel_change), 50.0) if math.isfinite(rel_change) else 50.0
        magnitude = math.log10(1.0 + bounded_rel * 100.0) / math.log10(5001.0)
        # magnitude now lies in [0, 1].

        # Corpus contact.
        summary = self._summary_text(spec, direction, rel_change)
        corpus_label, corpus_score, corpus_sim, corpus_hits = \
            self._corpus_lookup(spec.corpus_query)

        corpus_weight = _CORPUS_WEIGHTS.get(corpus_label, 0.5)

        surprise = magnitude * physics_confidence * corpus_weight

        # Use the first result's chain metadata — every sweep point
        # follows the same path through the graph.
        chain_vars = list(results[0].chain_vars)
        citations  = list(results[0].citations)

        return ExplorationCandidate(
            title=spec.title,
            target=spec.target,
            sweep_var=spec.sweep_var,
            held=dict(spec.held),
            chain_vars=chain_vars,
            citations=citations,
            sweep_values=list(spec.sweep_values),
            sweep_means=means,
            sweep_p5=p5s,
            sweep_p95=p95s,
            direction=direction,
            relative_change=rel_change,
            physics_confidence=physics_confidence,
            magnitude=magnitude,
            corpus_label=corpus_label,
            corpus_score=corpus_score,
            corpus_similarity=corpus_sim,
            corpus_keyword_hits=corpus_hits,
            surprise_score=surprise,
            summary=summary,
            corpus_query=spec.corpus_query,
        )

    def _run_sweep(self, spec: _SweepSpec) -> list[ForwardResult]:
        out: list[ForwardResult] = []
        for value in spec.sweep_values:
            given = {**spec.held, spec.sweep_var: float(value)}
            out.append(self._registry.forward(
                spec.target,
                given=given,
                n_samples=self._n_samples,
                seed=self._seed,
            ))
        return out

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_direction(means: list[float]) -> str:
        """Return a coarse monotonicity label."""
        if len(means) < 2:
            return "flat"
        diffs = [b - a for a, b in zip(means, means[1:])]
        if all(d > 0 for d in diffs):
            return "increases"
        if all(d < 0 for d in diffs):
            return "decreases"
        return "non-monotonic"

    def _summary_text(
        self, spec: _SweepSpec, direction: str, rel_change: float,
    ) -> str:
        """Human-readable summary for the UI."""
        sv = self._registry.variable(spec.sweep_var)
        tv = self._registry.variable(spec.target)
        sv_name = sv.name if sv else spec.sweep_var
        tv_name = tv.name if tv else spec.target
        lo, hi = spec.sweep_values[0], spec.sweep_values[-1]
        unit_sv = (sv.unit if sv else "") or ""
        if math.isfinite(rel_change):
            mag = f"{rel_change:+.0%}"
        else:
            mag = "divergent"
        return (
            f"Sweeping {sv_name} from {lo:g} to {hi:g} {unit_sv} "
            f"{direction} {tv_name} by {mag} (mean trajectory)."
        )

    def _corpus_lookup(
        self, query: str,
    ) -> tuple[str, float, float, int]:
        """Return (label, score, max_similarity, keyword_hits)."""
        if self._novelty is None:
            return "UNCERTAIN", 0.5, 0.0, 0
        try:
            nr = self._novelty.classify_text(query)
        except Exception as exc:                       # pragma: no cover
            logger.warning("Novelty classify_text failed: %s", exc)
            return "UNCERTAIN", 0.5, 0.0, 0
        return (
            nr.label,
            float(nr.score),
            float(getattr(nr, "max_similarity", 0.0)),
            int(getattr(nr, "keyword_hits", 0)),
        )

    @staticmethod
    def _failure_candidate(spec: _SweepSpec, msg: str) -> ExplorationCandidate:
        """Surface failed sweeps so the UI can show them rather than silently dropping."""
        return ExplorationCandidate(
            title=spec.title,
            target=spec.target,
            sweep_var=spec.sweep_var,
            held=dict(spec.held),
            chain_vars=[],
            citations=[],
            sweep_values=list(spec.sweep_values),
            sweep_means=[],
            sweep_p5=[],
            sweep_p95=[],
            direction="error",
            relative_change=float("nan"),
            physics_confidence=0.0,
            magnitude=0.0,
            corpus_label="UNCERTAIN",
            corpus_score=0.5,
            corpus_similarity=0.0,
            corpus_keyword_hits=0,
            surprise_score=0.0,
            summary=f"Sweep failed: {msg}",
            corpus_query=spec.corpus_query,
            error=msg,
        )
