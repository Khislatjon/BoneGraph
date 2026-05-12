"""
reasoning/agent_tools.py
────────────────────────
Phase 6 — tool registry and dispatcher for LLM agents.

Each tool wraps an existing deterministic function.  Agents emit JSON
{"action": "tool_name", "action_input": {...}} and the dispatcher
validates, calls, and returns a compact JSON string observation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    name: str
    description: str


LIST_VARIABLES = ToolSpec(
    name="list_variables",
    description=(
        "List every variable in the bone-physics equation graph with "
        "symbol, name, unit, and valid range."
    ),
)

LIST_RELATIONS = ToolSpec(
    name="list_relations",
    description=(
        "List all physical-law relations in the graph with input/output "
        "symbols and citation."
    ),
)

FORWARD = ToolSpec(
    name="forward",
    description=(
        "Run forward inference: predict 'target' symbol given upstream values. "
        "Returns mean, p5, p95 with derivation chain. "
        "Example: {\"target\": \"E\", \"given\": {\"phi\": 0.20}}"
    ),
)

CORPUS_SEARCH = ToolSpec(
    name="corpus_search",
    description=(
        "Semantic search over 248,629 bone-science literature passages. "
        "Returns titles, years, scores, and text snippets."
    ),
)

ALL_TOOLS = [LIST_VARIABLES, LIST_RELATIONS, FORWARD, CORPUS_SEARCH]
TOOL_BY_NAME = {t.name: t for t in ALL_TOOLS}


class ToolDispatcher:
    """Validates action_input and dispatches to the right deterministic function."""

    def __init__(self, registry, retriever) -> None:
        self._registry = registry
        self._retriever = retriever

    def call(self, action: str, action_input: dict) -> str:
        """Dispatch action; return compact JSON string."""
        if not isinstance(action_input, dict):
            action_input = {}
        try:
            if action == "list_variables":
                return self._list_variables()
            if action == "list_relations":
                return self._list_relations()
            if action == "forward":
                return self._forward(action_input)
            if action == "corpus_search":
                return self._corpus_search(action_input)
            return json.dumps({"error": f"Unknown tool '{action}'"})
        except Exception as exc:
            logger.warning("Tool %r failed: %s", action, exc)
            return json.dumps({"error": str(exc)})

    # ── Tool implementations ──────────────────────────────────────────────────

    def _list_variables(self) -> str:
        out = []
        for var in self._registry.iter_variables():
            out.append({
                "symbol":  var.symbol,
                "display": var.display_symbol or var.symbol,
                "name":    var.name,
                "unit":    var.unit or "",
                "range":   [var.lo, var.hi],
            })
        return json.dumps(out)

    def _list_relations(self) -> str:
        out = []
        for rel in self._registry.relations():
            out.append({
                "name":    rel.name,
                "inputs":  list(rel.inputs),
                "output":  rel.output,
                "cite":    rel.citation or "",
            })
        return json.dumps(out)

    def _forward(self, inp: dict) -> str:
        target = str(inp.get("target") or "").strip()
        given_raw = inp.get("given") or {}
        if not target:
            return json.dumps({"error": "'target' is required"})
        if not isinstance(given_raw, dict):
            return json.dumps({"error": "'given' must be a dict"})
        given: dict[str, float] = {}
        for k, v in given_raw.items():
            try:
                given[str(k)] = float(v)
            except (TypeError, ValueError):
                pass
        if not given:
            return json.dumps({"error": "No valid numeric values in 'given'"})
        try:
            r = self._registry.forward(target, given=given, n_samples=500)
            var = self._registry.variable(target)
            return json.dumps({
                "target":    target,
                "given":     given,
                "chain":     r.chain_vars,
                "mean":      round(r.mean, 6),
                "p5":        round(r.p5, 6),
                "p95":       round(r.p95, 6),
                "unit":      (var.unit if var else ""),
                "citations": r.citations,
            })
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def _corpus_search(self, inp: dict) -> str:
        query = str(inp.get("query") or "").strip()
        k = min(int(inp.get("k") or 4), 8)
        if not query:
            return json.dumps({"error": "'query' is required"})
        try:
            hits = self._retriever.query(query, top_k=k)
            passages = []
            for h in hits:
                passages.append({
                    "title":   h.get("title", ""),
                    "year":    h.get("year", ""),
                    "score":   round(float(h.get("score", 0.0)), 3),
                    "snippet": (h.get("text", "") or "")[:350],
                })
            return json.dumps(passages)
        except Exception as exc:
            return json.dumps({"error": str(exc)})
