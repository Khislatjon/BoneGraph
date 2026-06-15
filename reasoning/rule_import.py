"""
reasoning/rule_import.py
========================

A5 — bulk import of physical-grounding rules from a CSV or XLSX file.

The meeting (28 May): Gianluca wants to "pour an Excel file" of physical
groundings into the system. This module parses such a file into the same rule
shapes the feedback loop produces, validates each row, and reports per-row
problems instead of silently dropping them.

Both formats share one table layout (first row = header):

    name, kind, unit, lo, hi, context_terms, value_terms,
    forbidden_terms, exception_terms, explanation,
    first_terms, second_terms, comparator_terms

  - kind ∈ {range, forbid_pattern, comparative}
  - multi-value cells separated by ';'
  - empty cells ignored

A `range` row → {unit, lo, hi, context_terms, value_terms}
A `forbid_pattern` row → {context_terms, forbidden_terms, exception_terms, explanation}
A `comparative` row → {first_terms, second_terms, comparator_terms, explanation}

Public API:
    parse_file(filename, data: bytes) -> list[dict rows]
    validate_row(row: dict) -> (rule_dict | None, error_str | None)
    TEMPLATE_CSV  -> str  (downloadable header + examples)
"""

from __future__ import annotations

import csv
import io

COLUMNS = ["name", "kind", "unit", "lo", "hi", "context_terms",
           "value_terms", "forbidden_terms", "exception_terms", "explanation",
           "first_terms", "second_terms", "comparator_terms"]

MAX_IMPORT_RULES = 50   # practical cap — see docs/reasoning; beyond this the
                        # violation badge becomes noise from overlapping rules.

TEMPLATE_CSV = (
    "name,kind,unit,lo,hi,context_terms,value_terms,forbidden_terms,exception_terms,explanation,first_terms,second_terms,comparator_terms\n"
    "Cortical modulus 10-25 GPa,range,GPa,10,25,cortical,modulus;stiffness;young,,,,,,\n"
    "Trabecular modulus 0.01-3 GPa,range,GPa,0.01,3,trabecular;cancellous,modulus;stiffness,,,,,,\n"
    "Cortical density 1.8-2.0,range,g/cm^3,1.8,2.0,cortical,density,,,,,,\n"
    "Loading strengthens bone,forbid_pattern,,,,bone;loading,,weaken;loses mass;reduces density,disuse;unloading;microgravity,Mechanical loading strengthens bone per Wolff's law,,,\n"
    "Lytic lesion not negligible,forbid_pattern,,,,lytic;lesion;vertebra,,negligible;no effect;minimal,,Even small lytic lesions reduce vertebral failure load,,,\n"
    "Trabecular fails before cortical,comparative,,,,,,,,Trabecular fails before cortical due to higher surface area,trabecular;cancellous,cortical;compact,fails before;fails first;weaker;lower strength\n"
)


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_file(filename: str, data: bytes) -> list[dict]:
    """Dispatch on extension. Returns a list of row dicts keyed by column name.
    Raises ValueError on an unreadable/unsupported file."""
    name = (filename or "").lower()
    if name.endswith(".csv"):
        return _parse_csv(data)
    if name.endswith(".xlsx"):
        return _parse_xlsx(data)
    raise ValueError("Unsupported file type — use .csv or .xlsx")


def _parse_csv(data: bytes) -> list[dict]:
    text = data.decode("utf-8-sig", errors="replace")  # strip BOM if Excel wrote one
    reader = csv.DictReader(io.StringIO(text))
    return [_normalise_keys(r) for r in reader]


def _parse_xlsx(data: bytes) -> list[dict]:
    try:
        import openpyxl
    except ImportError:
        raise ValueError("XLSX import needs openpyxl (pip install openpyxl)")
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    header = [str(c).strip().lower() if c is not None else "" for c in rows[0]]
    out = []
    for raw in rows[1:]:
        if raw is None or all(c is None or str(c).strip() == "" for c in raw):
            continue  # skip blank rows
        d = {header[i]: ("" if v is None else str(v)) for i, v in enumerate(raw) if i < len(header)}
        out.append(_normalise_keys(d))
    return out


def _normalise_keys(row: dict) -> dict:
    return {(k or "").strip().lower(): (v if v is not None else "") for k, v in row.items()}


# ── Validation ────────────────────────────────────────────────────────────────

def _split(cell: str) -> list[str]:
    return [t.strip().lower() for t in str(cell or "").split(";") if t.strip()]


def validate_row(row: dict) -> tuple[dict | None, str | None]:
    """Turn one parsed row into a rule dict {name, kind, params}, or return an
    error string explaining why it was skipped."""
    name = str(row.get("name", "")).strip()
    kind = str(row.get("kind", "")).strip().lower()
    if not name:
        return None, "missing name"
    if kind not in ("range", "forbid_pattern", "comparative"):
        return None, f"kind must be 'range', 'forbid_pattern' or 'comparative' (got '{kind}')"

    if kind == "range":
        unit = str(row.get("unit", "")).strip()
        if not unit:
            return None, "range rule needs a unit"
        try:
            lo = float(str(row.get("lo", "")).strip())
            hi = float(str(row.get("hi", "")).strip())
        except ValueError:
            return None, "range rule needs numeric lo and hi"
        if lo > hi:
            lo, hi = hi, lo
        ctx = _split(row.get("context_terms"))
        if not ctx:
            return None, "range rule needs at least one context_term"
        params = {"unit": unit, "lo": lo, "hi": hi,
                  "context_terms": ctx, "value_terms": _split(row.get("value_terms"))}
        return {"name": name, "kind": "range", "params": params}, None

    if kind == "comparative":
        first = _split(row.get("first_terms"))
        second = _split(row.get("second_terms"))
        comp = _split(row.get("comparator_terms"))
        if not first:
            return None, "comparative needs at least one first_term"
        if not second:
            return None, "comparative needs at least one second_term"
        if not comp:
            return None, "comparative needs at least one comparator_term"
        params = {"first_terms": first, "second_terms": second,
                  "comparator_terms": comp,
                  "explanation": str(row.get("explanation", "")).strip()}
        return {"name": name, "kind": "comparative", "params": params}, None

    # forbid_pattern
    ctx = _split(row.get("context_terms"))
    forb = _split(row.get("forbidden_terms"))
    if not ctx:
        return None, "forbid_pattern needs at least one context_term"
    if not forb:
        return None, "forbid_pattern needs at least one forbidden_term"
    params = {"context_terms": ctx, "forbidden_terms": forb,
              "exception_terms": _split(row.get("exception_terms")),
              "explanation": str(row.get("explanation", "")).strip()}
    return {"name": name, "kind": "forbid_pattern", "params": params}, None


def rule_signature(rule: dict) -> tuple:
    """Stable key for dedup — same kind + same core params = duplicate."""
    p = rule["params"]
    if rule["kind"] == "range":
        return ("range", p["unit"].lower(), p["lo"], p["hi"],
                tuple(sorted(p["context_terms"])), tuple(sorted(p.get("value_terms", []))))
    if rule["kind"] == "comparative":
        return ("comparative", tuple(sorted(p["first_terms"])),
                tuple(sorted(p["second_terms"])), tuple(sorted(p["comparator_terms"])))
    return ("forbid_pattern", tuple(sorted(p["context_terms"])),
            tuple(sorted(p["forbidden_terms"])), tuple(sorted(p.get("exception_terms", []))))
