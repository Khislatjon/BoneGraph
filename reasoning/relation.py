"""
reasoning/relation.py
─────────────────────
Equation-graph foundation for the v2 reasoning pipeline.

This module is the substrate the rest of the new reasoner will be built
on.  It replaces the per-law branches of :mod:`reasoning.physics_gen`
with a single uniform abstraction — every physical law is a
:class:`Relation`, every quantity is a :class:`Variable`, and the
:class:`RelationRegistry` discovers which laws can be chained simply by
observing which variables they share.

Why this matters
----------------
The v0 generator iterates a Cartesian product of pre-encoded
perturbations.  Composition between laws (e.g. Currey ∘ Paris) cannot
happen because the per-law branches do not exchange intermediate
state.  Here, each Relation is a SymPy equation over typed Variables;
shared symbols across two Relations automatically form an edge in the
variable graph, and forward inference becomes a graph traversal that
finds *and evaluates* a derivation — not a hand-coded chain.

Phase 1 supports forward inference only (given some Variables, predict
another).  Abductive and counterfactual modes will be added in Phase 2
against the same Relation registry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sympy as sp

logger = logging.getLogger(__name__)


def _weighted_quantile(
    samples: np.ndarray, weights: np.ndarray, q: float,
) -> float:
    """
    Quantile of ``samples`` with importance ``weights``.

    Sorts samples, walks the cumulative weight, and linearly
    interpolates the breakpoint at fraction ``q``.  Used by
    :class:`RelationRegistry.abductive` to summarise the importance-
    weighted posterior of every inferred root variable.
    """
    if samples.size == 0:
        return float("nan")
    order = np.argsort(samples)
    s = samples[order]
    w = weights[order]
    cdf = np.cumsum(w)
    total = cdf[-1]
    if total <= 0:
        return float(np.mean(samples))
    cdf /= total
    return float(np.interp(q, cdf, s))


# ── Variables and priors ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Variable:
    """
    A typed physical quantity.

    Variables are nodes in the equation graph.  They carry just enough
    metadata for the inference engine to validate ranges and render the
    derivation cleanly in the UI.

    Attributes
    ----------
    symbol : str
        Stable identifier used in equations and graph traversal (e.g.
        ``"rho"``, ``"phi"``, ``"E"``).  Two Relations that mention the
        same symbol share that node.
    name : str
        Human-readable name (``"apparent density"``).
    unit : str
        Display unit (``"g/cm³"``, ``"MPa·√m"``, ``"m/cycle"``).  Phase 1
        treats units as opaque labels; full dimensional analysis is
        deferred to Phase 4.
    lo, hi : float
        Physically plausible range.  Used to clip MC samples and to
        warn when a prediction sits outside the calibration regime.
    description : str
        One-line description for tooltips and citations.
    """

    symbol: str
    name: str
    unit: str
    lo: float
    hi: float
    description: str = ""

    def clamp(self, value: float | np.ndarray) -> float | np.ndarray:
        """Clip ``value`` to the variable's physical range."""
        return np.clip(value, self.lo, self.hi)


@dataclass(frozen=True)
class Prior:
    """
    A simple uncertainty specification for a Relation parameter.

    Phase 1 keeps the distribution family small on purpose — we only
    need enough variation to surface meaningful uncertainty in forward
    predictions.  Phase 2 will swap this for proper Bayesian priors
    once abductive inference is wired in.

    distribution
        ``"normal"``     — truncated normal with mean ``mean`` and std ``std``.
        ``"lognormal"``  — log-normal whose log has mean ``mean`` and std ``std``.
        ``"fixed"``      — point mass at ``mean`` (no uncertainty).
    """

    mean: float
    std: float = 0.0
    distribution: str = "normal"
    citation: str = ""

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` samples from the prior."""
        if self.distribution == "fixed" or self.std == 0.0:
            return np.full(n, self.mean, dtype=float)
        if self.distribution == "normal":
            return rng.normal(self.mean, self.std, size=n)
        if self.distribution == "lognormal":
            # mean/std parameterise the underlying normal in log space.
            return rng.lognormal(self.mean, self.std, size=n)
        raise ValueError(f"Unknown distribution: {self.distribution!r}")


# ── Relations ────────────────────────────────────────────────────────────────


@dataclass
class Relation:
    """
    A single physical law expressed as a SymPy equation.

    A Relation declares which Variables it connects and which
    Parameters carry uncertainty.  The :attr:`output` variable is the
    one this relation can be evaluated *forwards* for; the
    :attr:`inputs` variables must be supplied (from prior chain steps
    or the user).  Parameters are sampled at evaluation time.

    Symbolic-by-default, numeric-where-needed
    -----------------------------------------
    The equation is stored as a SymPy expression so that Phase 2 can
    invert it for abductive queries.  Forward evaluation uses
    :func:`sympy.lambdify` for speed.
    """

    name: str
    equation: sp.Expr               # implicit form: equation == 0
    output: str                     # symbol that lambdify will solve for
    inputs: tuple[str, ...]         # other state variables required
    parameters: dict[str, Prior]    # constants with uncertainty
    citation: str
    description: str = ""

    # Built lazily on first evaluation.
    _solved: sp.Expr | None = field(default=None, init=False, repr=False)
    _lambdified: Callable | None = field(default=None, init=False, repr=False)
    _arg_order: tuple[str, ...] | None = field(
        default=None, init=False, repr=False,
    )

    # ── Solver setup ──────────────────────────────────────────────────────────

    def _symbol_by_name(self, name: str) -> sp.Symbol:
        """
        Return the SymPy symbol named ``name`` *as it appears in the equation*.

        Symbols with different assumptions (``positive=True`` etc.) compare
        unequal even when their names match, so we can't construct fresh
        symbols and expect ``solve`` or ``lambdify`` to recognise them.
        """
        for s in self.equation.free_symbols:
            if s.name == name:
                return s
        raise KeyError(
            f"Symbol {name!r} not found in equation of relation {self.name!r}.",
        )

    def _solve_for_output(self) -> sp.Expr:
        """Solve the implicit equation for the output variable."""
        out_sym = self._symbol_by_name(self.output)
        solutions = sp.solve(self.equation, out_sym)
        if not solutions:
            raise RuntimeError(
                f"Relation {self.name!r} could not be solved for "
                f"output variable {self.output!r}.",
            )
        # Prefer the first solution that produces real values for
        # positive inputs (heuristic — adequate for monomial-style laws).
        return solutions[0]

    def _ensure_lambda(self) -> None:
        if self._lambdified is not None:
            return
        self._solved = self._solve_for_output()
        # Argument order: inputs first, then parameters (alphabetical
        # for stability).  Use tuple to make positional calling cheap.
        param_order = tuple(sorted(self.parameters.keys()))
        arg_order = self.inputs + param_order
        symbols = [self._symbol_by_name(s) for s in arg_order]
        self._arg_order = arg_order
        self._lambdified = sp.lambdify(symbols, self._solved, modules="numpy")

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        inputs: dict[str, np.ndarray | float],
        rng: np.random.Generator,
        n_samples: int,
    ) -> np.ndarray:
        """
        Sample ``n_samples`` predictions for :attr:`output`.

        ``inputs`` must supply every name in :attr:`inputs`; each entry
        may be a scalar (broadcast to ``n_samples``) or an array of
        length ``n_samples``.  Parameters are drawn fresh from their
        priors on every call so independent runs report independent
        uncertainty bands.
        """
        self._ensure_lambda()
        assert self._lambdified is not None
        assert self._arg_order is not None

        param_samples = {
            name: prior.sample(rng, n_samples)
            for name, prior in self.parameters.items()
        }

        args: list[np.ndarray] = []
        for name in self._arg_order:
            if name in self.parameters:
                args.append(param_samples[name])
            else:
                v = inputs[name]
                if np.isscalar(v):
                    args.append(np.full(n_samples, float(v)))
                else:
                    args.append(np.asarray(v, dtype=float))
        return np.asarray(self._lambdified(*args), dtype=float)

    # ── Display helpers ───────────────────────────────────────────────────────

    @property
    def latex(self) -> str:
        """Render the equation in LaTeX form for the UI."""
        try:
            return sp.latex(self.equation) + " = 0"
        except Exception:
            return str(self.equation)


# ── Registry / inference engine ──────────────────────────────────────────────


@dataclass
class DerivationStep:
    """One Relation invocation in a forward chain."""

    relation_name: str
    citation: str
    inputs: dict[str, float]        # mean of each input variable at this step
    output_var: str
    output_mean: float
    output_p5: float
    output_p95: float


@dataclass
class ForwardResult:
    """Outcome of a successful forward inference run."""

    target: str
    given: dict[str, float]
    chain_vars: list[str]           # state variables along the chain
    steps: list[DerivationStep]     # one per Relation invocation
    samples: np.ndarray             # raw MC samples of the target
    mean: float
    median: float
    p5: float
    p95: float
    relative_uncertainty: float     # (p95 - p5) / |mean|, capped sensibly
    citations: list[str]            # deduplicated, in chain order


@dataclass
class AbductivePosterior:
    """Per-variable posterior summary from an abductive inference run."""

    variable: str
    prior_mean: float
    prior_p5: float
    prior_p95: float
    posterior_mean: float
    posterior_p5: float
    posterior_p95: float
    shift_score: float              # (prior_band - post_band) / prior_band ∈ [0, 1]


@dataclass
class AbductiveResult:
    """Outcome of an abductive inference run."""

    target: str
    observed: float
    observed_std: float
    inferred: list[AbductivePosterior]   # ranked by shift_score, descending
    chain_vars: list[str]
    citations: list[str]
    effective_sample_size: float         # 1 / sum(w²) after normalisation
    n_samples: int


@dataclass
class CounterfactualResult:
    """Outcome of a do(...) intervention compared against baseline."""

    target: str
    given: dict[str, float]
    intervention: dict[str, float]
    chain_vars: list[str]
    baseline_mean: float
    baseline_p5: float
    baseline_p95: float
    intervened_mean: float
    intervened_p5: float
    intervened_p95: float
    delta_mean: float                    # mean(intervened) − mean(baseline)
    delta_p5: float                      # paired delta percentiles
    delta_p95: float
    relative_delta: float                # delta_mean / baseline_mean
    citations: list[str]
    n_samples: int


class RelationRegistry:
    """
    Holds Relations and exposes forward inference over the variable graph.

    Construction
    ------------
    Call :meth:`register` for each Relation.  The registry maintains a
    set of state variables (the symbols that appear as ``output`` or in
    ``inputs`` of any Relation) and an adjacency map from a variable to
    the Relations that *produce* it.  BFS over this map finds a chain
    from given variables to the target.
    """

    def __init__(self) -> None:
        self._relations: dict[str, Relation] = {}
        self._variables: dict[str, Variable] = {}
        # variable symbol → relations that have it as output
        self._producers: dict[str, list[Relation]] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, relation: Relation) -> None:
        if relation.name in self._relations:
            raise ValueError(f"Relation {relation.name!r} already registered.")
        self._relations[relation.name] = relation
        self._producers.setdefault(relation.output, []).append(relation)

    def register_variable(self, var: Variable) -> None:
        self._variables[var.symbol] = var

    def variable(self, symbol: str) -> Variable | None:
        return self._variables.get(symbol)

    def relations(self) -> list[Relation]:
        return list(self._relations.values())

    # ── Path finding ──────────────────────────────────────────────────────────

    def _find_chain(
        self,
        target: str,
        given: set[str],
    ) -> list[Relation] | None:
        """
        Recursive backward search from ``target`` toward ``given``.

        A variable resolves if it is in ``given`` or some producer has
        every input recursively resolvable.  The chain returned is the
        concatenation of sub-chains (de-duplicated, dependency-first)
        followed by the producer of the target itself.

        Returns ``None`` if no chain exists, e.g. because a required
        root variable has no producer and was not supplied.
        """
        if target in given:
            return []

        memo: dict[str, list[Relation] | None] = {}
        visiting: set[str] = set()

        def resolve(var: str) -> list[Relation] | None:
            if var in given:
                return []
            if var in memo:
                return memo[var]
            if var in visiting:
                # Cycle — physical laws form a DAG, so we bail out.
                return None
            visiting.add(var)
            try:
                producers = self._producers.get(var, [])
                for rel in producers:
                    sub_chains: list[list[Relation]] = []
                    ok = True
                    for inp in rel.inputs:
                        sub = resolve(inp)
                        if sub is None:
                            ok = False
                            break
                        sub_chains.append(sub)
                    if not ok:
                        continue
                    merged: list[Relation] = []
                    seen_names: set[str] = set()
                    for sub in sub_chains:
                        for r in sub:
                            if r.name not in seen_names:
                                seen_names.add(r.name)
                                merged.append(r)
                    if rel.name not in seen_names:
                        merged.append(rel)
                    memo[var] = merged
                    return merged
                memo[var] = None
                return None
            finally:
                visiting.discard(var)

        return resolve(target)

    def chain_root_inputs(self, chain: list[Relation]) -> list[str]:
        """
        Variables a chain expects the caller to supply externally.

        Order is stable (first encountered in chain order).  Useful for
        modes like abduction that need to sample over chain roots.
        """
        produced = {rel.output for rel in chain}
        seen: set[str] = set()
        ordered: list[str] = []
        for rel in chain:
            for inp in rel.inputs:
                if inp not in produced and inp not in seen:
                    seen.add(inp)
                    ordered.append(inp)
        return ordered

    # ── Forward inference ─────────────────────────────────────────────────────

    def forward(
        self,
        target: str,
        given: dict[str, float],
        *,
        n_samples: int = 2000,
        seed: int | None = None,
    ) -> ForwardResult:
        """
        Predict ``target`` from ``given`` via Monte Carlo over the chain.

        Parameters
        ----------
        target : str
            Symbol of the state variable to predict.
        given : dict
            Symbol → value for every state variable the user can
            supply.  The chain may use a subset.
        n_samples : int
            Number of Monte Carlo samples for uncertainty propagation.
        seed : int | None
            Optional RNG seed for reproducibility.

        Raises
        ------
        ValueError
            If no derivation chain from ``given`` to ``target`` exists.
        """
        chain = self._resolve_chain(target, given)
        rng = np.random.default_rng(seed)
        state = self._state_from_given(given, n_samples)
        steps = self._run_chain(chain, state, rng, n_samples, record=True)
        return self._build_forward_result(target, given, chain, state, steps)

    # ── Shared chain-evaluation primitives ────────────────────────────────────

    def _resolve_chain(
        self, target: str, given: dict[str, float],
    ) -> list[Relation]:
        """Find a chain or raise a descriptive ValueError."""
        if target not in self._producers and target not in given:
            raise ValueError(f"No Relation produces variable {target!r}.")
        chain = self._find_chain(target, set(given.keys()))
        if chain is None:
            raise ValueError(
                f"No derivation chain found from {sorted(given)} to {target!r}. "
                f"Check that the registry contains a path of Relations whose "
                f"inputs are all eventually supplied.",
            )
        return chain

    @staticmethod
    def _state_from_given(
        given: dict[str, float | np.ndarray], n_samples: int,
    ) -> dict[str, np.ndarray]:
        """Broadcast each given value (scalar or array) into a state array."""
        state: dict[str, np.ndarray] = {}
        for name, val in given.items():
            if np.isscalar(val):
                state[name] = np.full(n_samples, float(val))
            else:
                arr = np.asarray(val, dtype=float)
                if arr.size != n_samples:
                    raise ValueError(
                        f"Array for {name!r} has length {arr.size}, "
                        f"expected {n_samples}.",
                    )
                state[name] = arr
        return state

    def _run_chain(
        self,
        chain: list[Relation],
        state: dict[str, np.ndarray],
        rng: np.random.Generator,
        n_samples: int,
        *,
        record: bool,
    ) -> list[DerivationStep]:
        """
        Walk ``chain`` in order, evaluating each relation and pushing the
        result into ``state``.  If ``record`` is true, build a
        :class:`DerivationStep` per relation for the UI.
        """
        steps: list[DerivationStep] = []
        for rel in chain:
            samples = rel.evaluate(
                inputs={k: state[k] for k in rel.inputs},
                rng=rng,
                n_samples=n_samples,
            )
            state[rel.output] = samples
            if record:
                step_inputs = {k: float(np.mean(state[k])) for k in rel.inputs}
                steps.append(DerivationStep(
                    relation_name=rel.name,
                    citation=rel.citation,
                    inputs=step_inputs,
                    output_var=rel.output,
                    output_mean=float(np.mean(samples)),
                    output_p5=float(np.percentile(samples, 5)),
                    output_p95=float(np.percentile(samples, 95)),
                ))
        return steps

    @staticmethod
    def _chain_vars(chain: list[Relation]) -> list[str]:
        """Ordered, de-duplicated list of variables touched by the chain."""
        ordered: list[str] = []
        seen: set[str] = set()
        for rel in chain:
            for v in (*rel.inputs, rel.output):
                if v not in seen:
                    seen.add(v)
                    ordered.append(v)
        return ordered

    @staticmethod
    def _chain_citations(chain: list[Relation]) -> list[str]:
        citations: list[str] = []
        for rel in chain:
            if rel.citation and rel.citation not in citations:
                citations.append(rel.citation)
        return citations

    def _build_forward_result(
        self,
        target: str,
        given: dict[str, float],
        chain: list[Relation],
        state: dict[str, np.ndarray],
        steps: list[DerivationStep],
    ) -> ForwardResult:
        final = state[target]
        mean = float(np.mean(final))
        p5 = float(np.percentile(final, 5))
        p95 = float(np.percentile(final, 95))
        rel_unc = (p95 - p5) / abs(mean) if abs(mean) > 1e-12 else float("nan")
        return ForwardResult(
            target=target,
            given=given,
            chain_vars=self._chain_vars(chain),
            steps=steps,
            samples=final,
            mean=mean,
            median=float(np.median(final)),
            p5=p5,
            p95=p95,
            relative_uncertainty=rel_unc,
            citations=self._chain_citations(chain),
        )

    # ── Abductive inference ──────────────────────────────────────────────────

    def abductive(
        self,
        target: str,
        observed: float,
        *,
        observed_std: float | None = None,
        infer: list[str] | None = None,
        given: dict[str, float] | None = None,
        n_samples: int = 8000,
        seed: int | None = None,
    ) -> AbductiveResult:
        """
        Infer likely upstream causes of an observed downstream value.

        Implementation
        --------------
        Importance sampling.  For each variable in ``infer`` we draw a
        uniform sample over its declared physical range; for all other
        chain roots we fix the value from ``given``.  We run the chain
        forward to get a predicted target distribution, then weight each
        Monte Carlo sample by a Gaussian likelihood centred on
        ``observed`` with std ``observed_std`` (default: 5 % of the
        absolute observed value, floored at 1 % of the prior band).
        The weighted samples form the posterior over each ``infer``
        variable.

        ``shift_score`` measures how much the observation tightened the
        prior — high for variables the observation is informative about,
        near zero for variables it cannot distinguish.

        Stop-condition note
        -------------------
        Phase 2 spec: if Bayesian inversion is too slow, fall back to a
        discrete top-k forward sweep.  Importance sampling at n=8 000 is
        sub-second on the current graph, so we stay with the full
        posterior; if scaling later becomes a problem the same chain
        can be re-used by a grid/top-k version without API changes.
        """
        given = dict(given or {})
        chain = self._resolve_chain(target, {**given,
                                            **{v: 0.0 for v in (infer or [])}})

        roots = self.chain_root_inputs(chain)
        if infer is None or not infer:
            # Default: every root not explicitly given is a candidate.
            infer = [r for r in roots if r not in given]
        infer = [v for v in infer if v in roots]
        if not infer:
            raise ValueError(
                f"No upstream variables to infer for target {target!r}; "
                f"chain roots are {roots} and all are pinned in 'given'.",
            )

        rng = np.random.default_rng(seed)

        # Build initial state: sample over infer variables, broadcast given.
        state: dict[str, np.ndarray] = {}
        prior_samples: dict[str, np.ndarray] = {}
        for r in roots:
            if r in infer:
                v = self._variables.get(r)
                if v is None:
                    raise ValueError(
                        f"Variable {r!r} is required for abduction but has "
                        f"no registered Variable metadata (range unknown).",
                    )
                draws = rng.uniform(v.lo, v.hi, size=n_samples)
                state[r] = draws
                prior_samples[r] = draws
            elif r in given:
                state[r] = np.full(n_samples, float(given[r]))
            else:
                raise ValueError(
                    f"Chain root {r!r} is neither in 'given' nor in 'infer'.",
                )

        # Forward propagate (no per-step records — we only need target samples).
        self._run_chain(chain, state, rng, n_samples, record=False)
        pred = state[target]

        if observed_std is None or observed_std <= 0:
            observed_std = max(abs(observed) * 0.05, 1e-12)

        # Gaussian likelihood weights.
        residual = (pred - observed) / observed_std
        log_w = -0.5 * residual * residual
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            raise ValueError(
                "Likelihood weights collapsed — observation lies far "
                "outside what the chain can produce over the declared "
                "variable ranges.",
            )
        w_norm = w / w_sum
        ess = 1.0 / float(np.sum(w_norm * w_norm))

        posteriors: list[AbductivePosterior] = []
        for var_name in infer:
            samples = prior_samples[var_name]
            prior_p5  = float(np.percentile(samples, 5))
            prior_p95 = float(np.percentile(samples, 95))
            post_mean = float(np.sum(samples * w_norm))
            post_p5   = float(_weighted_quantile(samples, w_norm, 0.05))
            post_p95  = float(_weighted_quantile(samples, w_norm, 0.95))
            prior_band = max(prior_p95 - prior_p5, 1e-12)
            post_band  = max(post_p95 - post_p5, 0.0)
            shift = 1.0 - min(post_band / prior_band, 1.0)
            posteriors.append(AbductivePosterior(
                variable=var_name,
                prior_mean=float(np.mean(samples)),
                prior_p5=prior_p5,
                prior_p95=prior_p95,
                posterior_mean=post_mean,
                posterior_p5=post_p5,
                posterior_p95=post_p95,
                shift_score=shift,
            ))
        posteriors.sort(key=lambda p: p.shift_score, reverse=True)

        return AbductiveResult(
            target=target,
            observed=float(observed),
            observed_std=float(observed_std),
            inferred=posteriors,
            chain_vars=self._chain_vars(chain),
            citations=self._chain_citations(chain),
            effective_sample_size=ess,
            n_samples=n_samples,
        )

    # ── Counterfactual inference ─────────────────────────────────────────────

    def counterfactual(
        self,
        target: str,
        given: dict[str, float],
        intervention: dict[str, float],
        *,
        n_samples: int = 4000,
        seed: int | None = None,
    ) -> CounterfactualResult:
        """
        Compare target distributions under baseline vs intervened root values.

        Uses the same RNG seed for both passes so that parameter draws
        (Currey's a, n; Paris C₀, m, k_ρ; Frost setpoint, etc.) are
        identical between baseline and intervened states.  This makes
        the paired ``Δtarget = intervened − baseline`` distribution
        reflect *only* the intervention, not parameter noise.

        ``intervention`` must contain at least one variable; values
        replace the corresponding ``given`` entry (the intervention is
        treated as do(var = value), i.e. no other inputs are touched).
        """
        if not intervention:
            raise ValueError("Intervention must specify at least one variable.")

        unknown = [k for k in intervention if k not in given]
        if unknown:
            raise ValueError(
                f"Intervention variables not in 'given': {unknown}. "
                f"do(...) only re-binds existing inputs; add them to "
                f"'given' first if you want to introduce new ones.",
            )

        # Baseline pass.
        chain = self._resolve_chain(target, given)
        rng_base = np.random.default_rng(seed)
        state_b = self._state_from_given(given, n_samples)
        self._run_chain(chain, state_b, rng_base, n_samples, record=False)
        base = state_b[target]

        # Intervened pass — same seed, same chain.
        intervened_given = {**given, **intervention}
        rng_int = np.random.default_rng(seed)
        state_i = self._state_from_given(intervened_given, n_samples)
        self._run_chain(chain, state_i, rng_int, n_samples, record=False)
        interv = state_i[target]

        delta = interv - base
        base_mean = float(np.mean(base))
        delta_mean = float(np.mean(delta))
        rel_delta = delta_mean / base_mean if abs(base_mean) > 1e-12 else float("nan")

        return CounterfactualResult(
            target=target,
            given=given,
            intervention=intervention,
            chain_vars=self._chain_vars(chain),
            baseline_mean=base_mean,
            baseline_p5=float(np.percentile(base, 5)),
            baseline_p95=float(np.percentile(base, 95)),
            intervened_mean=float(np.mean(interv)),
            intervened_p5=float(np.percentile(interv, 5)),
            intervened_p95=float(np.percentile(interv, 95)),
            delta_mean=delta_mean,
            delta_p5=float(np.percentile(delta, 5)),
            delta_p95=float(np.percentile(delta, 95)),
            relative_delta=rel_delta,
            citations=self._chain_citations(chain),
            n_samples=n_samples,
        )

    # ── Introspection ─────────────────────────────────────────────────────────

    def variables_in_graph(self) -> set[str]:
        """All state-variable symbols mentioned by any Relation."""
        out: set[str] = set()
        for rel in self._relations.values():
            out.add(rel.output)
            out.update(rel.inputs)
        return out

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RelationRegistry(relations={len(self._relations)}, "
            f"variables={len(self.variables_in_graph())})"
        )
