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
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sympy as sp

logger = logging.getLogger(__name__)


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
        Backward BFS from ``target`` toward variables in ``given``.

        Returns the chain in *forward* execution order (each Relation's
        inputs are produced by earlier steps or supplied by the user).
        Returns ``None`` if no derivation exists.
        """
        if target in given:
            return []

        # parent[var] = (relation_used, [prev_var, ...])
        parent: dict[str, Relation] = {}
        queue: deque[str] = deque([target])
        visited: set[str] = {target}

        while queue:
            current = queue.popleft()
            for rel in self._producers.get(current, []):
                missing = [v for v in rel.inputs if v not in given]
                # Always record the relation we're considering so we
                # can reconstruct the chain even when more upstream
                # steps are required.
                if current not in parent:
                    parent[current] = rel
                # If every input is given, we can stop expanding this
                # branch — the chain terminates here.
                if not missing:
                    return self._reconstruct(target, parent)
                # Otherwise, queue each missing input for resolution.
                for inp in missing:
                    if inp not in visited:
                        visited.add(inp)
                        queue.append(inp)

        # Loop ended without resolving every input; check if we can
        # still produce a chain whose remaining inputs are all given.
        if target in parent:
            chain = self._reconstruct(target, parent)
            if chain is not None and self._chain_inputs(chain).issubset(given):
                return chain
        return None

    def _reconstruct(
        self, target: str, parent: dict[str, Relation],
    ) -> list[Relation] | None:
        """Walk back from target through ``parent`` and return forward order."""
        if target not in parent:
            return None
        ordered: list[Relation] = []
        seen: set[str] = set()
        stack: list[str] = [target]
        # DFS post-order so every input is emitted before its consumer.
        visiting: set[str] = set()

        def visit(var: str) -> bool:
            if var in seen or var not in parent:
                return var in seen or var not in parent
            if var in visiting:
                # Cycle — should never happen for a DAG of physical
                # laws, but bail out gracefully.
                return False
            visiting.add(var)
            rel = parent[var]
            for inp in rel.inputs:
                if not visit(inp):
                    return True   # input is a leaf (user-supplied)
            ordered.append(rel)
            seen.add(var)
            visiting.discard(var)
            return True

        if not visit(target):
            return None
        return ordered

    @staticmethod
    def _chain_inputs(chain: list[Relation]) -> set[str]:
        """Variables a chain expects to be supplied externally."""
        produced = {rel.output for rel in chain}
        needed: set[str] = set()
        for rel in chain:
            for inp in rel.inputs:
                if inp not in produced:
                    needed.add(inp)
        return needed

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
        if target not in self._producers and target not in given:
            raise ValueError(f"No Relation produces variable {target!r}.")

        chain = self._find_chain(target, set(given.keys()))
        if chain is None:
            raise ValueError(
                f"No derivation chain found from {sorted(given)} to {target!r}. "
                f"Check that the registry contains a path of Relations whose "
                f"inputs are all eventually supplied.",
            )

        rng = np.random.default_rng(seed)
        # Running state: name → np.ndarray of length n_samples
        state: dict[str, np.ndarray] = {
            name: np.full(n_samples, float(val))
            for name, val in given.items()
        }

        steps: list[DerivationStep] = []
        for rel in chain:
            samples = rel.evaluate(
                inputs={k: state[k] for k in rel.inputs},
                rng=rng,
                n_samples=n_samples,
            )
            state[rel.output] = samples
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

        final = state[target]
        mean = float(np.mean(final))
        p5 = float(np.percentile(final, 5))
        p95 = float(np.percentile(final, 95))
        rel_unc = (p95 - p5) / abs(mean) if abs(mean) > 1e-12 else float("nan")

        chain_vars: list[str] = []
        seen_var: set[str] = set()
        for rel in chain:
            for v in (*rel.inputs, rel.output):
                if v not in seen_var:
                    chain_vars.append(v)
                    seen_var.add(v)

        citations: list[str] = []
        for rel in chain:
            if rel.citation and rel.citation not in citations:
                citations.append(rel.citation)

        return ForwardResult(
            target=target,
            given=given,
            chain_vars=chain_vars,
            steps=steps,
            samples=final,
            mean=mean,
            median=float(np.median(final)),
            p5=p5,
            p95=p95,
            relative_uncertainty=rel_unc,
            citations=citations,
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
