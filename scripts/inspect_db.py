"""
scripts/inspect_db.py
─────────────────────
Quick summary of everything collected in the papers database.
Run from the project root:

    python scripts/inspect_db.py
    python scripts/inspect_db.py --top 20          # show more rows
    python scripts/inspect_db.py --keyword femur   # filter by keyword

Output sections:
  1. Overall counts
  2. Papers by year
  3. Top journals
  4. Most cited papers
  5. Keyword coverage
  6. Download status
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# ── locate the database relative to this script ──────────────────────────────
# __file__ is this script's path; .parents[1] goes up one level to project root
PROJECT_ROOT = Path(__file__).parents[1]
DB_PATH = PROJECT_ROOT / "data" / "db" / "papers.db"


def separator(title: str) -> None:
    """Print a section header line to visually separate output blocks."""
    width = 70
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def section_overall(conn: sqlite3.Connection) -> None:
    """Print the top-level counts: total papers, open-access PDFs, year span."""
    row = conn.execute(
        "SELECT COUNT(*) as total, "
        "       SUM(has_pdf) as with_pdf_url, "
        "       SUM(CASE WHEN pdf_local_path IS NOT NULL THEN 1 ELSE 0 END) as downloaded, "
        "       MIN(year) as earliest, "
        "       MAX(year) as latest "
        "FROM papers"
    ).fetchone()

    separator("OVERALL")
    print(f"  Total papers collected  : {row['total']:>8,}")
    print(f"  With open-access PDF URL: {row['with_pdf_url']:>8,}")
    print(f"  PDFs downloaded to disk : {row['downloaded']:>8,}")
    print(f"  Year range              : {row['earliest']} – {row['latest']}")


def section_years(conn: sqlite3.Connection, top: int) -> None:
    """Print a distribution of paper counts by publication year."""
    separator(f"PAPERS BY YEAR (top {top})")
    rows = conn.execute(
        "SELECT year, COUNT(*) as n "
        "FROM papers "
        "GROUP BY year "
        "ORDER BY n DESC "
        f"LIMIT {top}"
    ).fetchall()
    for row in rows:
        year_label = str(row["year"]) if row["year"] else "unknown"
        # Build a simple bar chart with '█' characters (1 char = 100 papers)
        bar = "█" * (row["n"] // 100)
        print(f"  {year_label:>7}  {row['n']:>5,}  {bar}")


def section_journals(conn: sqlite3.Connection, top: int) -> None:
    """Print the most frequently appearing publication venues (journals / conferences)."""
    separator(f"TOP {top} JOURNALS / VENUES")
    rows = conn.execute(
        "SELECT venue, COUNT(*) as n "
        "FROM papers "
        "WHERE venue IS NOT NULL AND venue != '' "
        "GROUP BY venue "
        "ORDER BY n DESC "
        f"LIMIT {top}"
    ).fetchall()
    for row in rows:
        print(f"  {row['n']:>5,}  {row['venue'][:65]}")


def section_most_cited(conn: sqlite3.Connection, top: int) -> None:
    """Print the highest-citation-count papers in the database."""
    separator(f"TOP {top} MOST CITED PAPERS")
    rows = conn.execute(
        "SELECT title, year, citation_count "
        "FROM papers "
        "ORDER BY citation_count DESC "
        f"LIMIT {top}"
    ).fetchall()
    for row in rows:
        title = (row["title"] or "")[:65]
        print(f"  [{row['year']}]  {row['citation_count']:>6,} cites  │  {title}")


def section_keywords(conn: sqlite3.Connection, top: int, filter_kw: str | None) -> None:
    """
    Count how many papers were found per search keyword.

    Because one paper can match multiple keywords (stored as a JSON list in
    keywords_matched), we decode each row and tally individually.

    Args:
        filter_kw: if provided, only show keywords that contain this substring.
    """
    separator(f"KEYWORD COVERAGE (top {top})")

    # Load all keywords_matched values from the database
    rows = conn.execute(
        "SELECT keywords_matched FROM papers WHERE keywords_matched IS NOT NULL"
    ).fetchall()

    # Tally occurrences for each keyword string
    counts: dict[str, int] = {}
    for row in rows:
        for kw in json.loads(row["keywords_matched"]):
            counts[kw] = counts.get(kw, 0) + 1

    # Apply optional substring filter
    if filter_kw:
        counts = {k: v for k, v in counts.items() if filter_kw.lower() in k.lower()}

    sorted_kws = sorted(counts.items(), key=lambda x: -x[1])[:top]
    print(f"  {'COUNT':>6}  KEYWORD")
    print(f"  {'─' * 6}  {'─' * 50}")
    for kw, n in sorted_kws:
        print(f"  {n:>6,}  {kw}")


def section_download_status(conn: sqlite3.Connection) -> None:
    """
    Show how many papers have been downloaded vs. still pending.
    Gives an estimate of how much disk space the PDFs occupy.
    """
    separator("PDF DOWNLOAD STATUS")

    total_with_url = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE has_pdf = 1"
    ).fetchone()[0]

    downloaded = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE pdf_local_path IS NOT NULL"
    ).fetchone()[0]

    pending = total_with_url - downloaded

    print(f"  Papers with open-access URL : {total_with_url:>7,}")
    print(f"  Already downloaded          : {downloaded:>7,}")
    print(f"  Pending download            : {pending:>7,}")
    if total_with_url > 0:
        pct = 100 * downloaded / total_with_url
        # Visual progress bar (50 chars wide)
        filled = int(50 * downloaded / total_with_url)
        bar = "█" * filled + "░" * (50 - filled)
        print(f"\n  Progress: [{bar}] {pct:.1f}%")
        if pending > 0:
            # Very rough estimate: average bone paper PDF ≈ 2 MB
            est_gb = pending * 2 / 1024
            print(f"\n  Estimated remaining download: ~{est_gb:.1f} GB  (at ~2 MB/paper)")
            est_hours = (pending * 1.1) / 3600  # 1.1s per file
            print(f"  Estimated time remaining    : ~{est_hours:.1f} hours")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a summary of the BoneMind papers database."
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="How many rows to show in each ranked section (default: 10)"
    )
    parser.add_argument(
        "--keyword", type=str, default=None,
        help="Filter keyword coverage section to keywords containing this string"
    )
    args = parser.parse_args()

    # Verify the database file exists before trying to open it
    if not DB_PATH.exists():
        print(f"ERROR: database not found at {DB_PATH}", file=sys.stderr)
        print("Run the ingestion pipeline first:", file=sys.stderr)
        print("  python -m ingestion.papers.pipeline", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name: row["title"]

    try:
        section_overall(conn)
        section_years(conn, args.top)
        section_journals(conn, args.top)
        section_most_cited(conn, args.top)
        section_keywords(conn, args.top, args.keyword)
        section_download_status(conn)
    finally:
        conn.close()

    print()


if __name__ == "__main__":
    main()
