"""
scripts/download_wiley.py
==========================
Downloads Wiley PDFs using a real Chromium browser via Playwright.

Why Playwright?
---------------
Wiley uses TLS fingerprinting and session cookie checks that block all
programmatic HTTP clients (requests, curl, Postman) regardless of headers.
Playwright drives a real Chrome browser — same TLS handshake, same cookie
handling, indistinguishable from a human — so Wiley allows the download.

Publishers covered
------------------
- Wiley Online Library     (onlinelibrary.wiley.com/doi/pdfdirect/...)
- Wiley Anatomy sub-domain (anatomypubs.onlinelibrary.wiley.com/...)

Safe to run alongside scripts/download_pdfs.py — writes to different rows.

Usage
-----
    python -m scripts.download_wiley

    # Dry run — show counts without downloading
    python -m scripts.download_wiley --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from config.settings import PAPERS_DB_PATH, RAW_PAPERS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Wiley URL prefixes to target
_WILEY_PREFIXES = [
    "https://onlinelibrary.wiley.com/doi/pdfdirect",
    "https://anatomypubs.onlinelibrary.wiley.com",
]

# How long to wait for a PDF to start downloading (ms)
_DOWNLOAD_TIMEOUT = 30_000

# Polite delay between downloads (seconds)
_DELAY = 1.5


def _pdf_path(paper_id: str, year: int | None) -> Path:
    subdir = RAW_PAPERS_DIR / str(year or "unknown")
    subdir.mkdir(parents=True, exist_ok=True)
    safe_id = paper_id.replace("/", "_")
    return subdir / f"{safe_id}.pdf"


def _fetch_papers(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    placeholders = " OR ".join(f"pdf_url LIKE '{p}%'" for p in _WILEY_PREFIXES)
    return conn.execute(
        f"""SELECT paper_id, pdf_url, year
            FROM papers
            WHERE pdf_local_path IS NULL
            AND ({placeholders})"""
    ).fetchall()


def run(dry_run: bool = False) -> None:
    conn = sqlite3.connect(PAPERS_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    rows = _fetch_papers(conn)

    wiley_main = sum(1 for r in rows if "onlinelibrary.wiley.com/doi/pdfdirect" in r["pdf_url"])
    wiley_anat = sum(1 for r in rows if "anatomypubs" in r["pdf_url"])

    logger.info("Wiley papers to download: %d  (main: %d, anatomy: %d)",
                len(rows), wiley_main, wiley_anat)

    if dry_run:
        print(f"\n  Wiley main    : {wiley_main:,}")
        print(f"  Wiley anatomy : {wiley_anat:,}")
        print(f"  Total         : {len(rows):,}")
        conn.close()
        return

    counts = {"downloaded": 0, "failed": 0, "skipped": 0}
    total = len(rows)

    with sync_playwright() as p:
        # headless=False — visible browser window required for Cloudflare.
        # Extra args disable Chrome's automation indicators.
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        context = browser.new_context(
            accept_downloads=True,
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        # Patch navigator.webdriver to undefined before every page load.
        # This is what Cloudflare checks — if it's true, you're flagged as a bot.
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        page = context.new_page()

        for idx, row in enumerate(rows, start=1):
            paper_id = row["paper_id"]
            pdf_url  = row["pdf_url"]
            year     = row["year"]

            dest = _pdf_path(paper_id, year)

            # Skip if already on disk
            if dest.exists():
                conn.execute(
                    "UPDATE papers SET pdf_local_path = ? WHERE paper_id = ?",
                    (str(dest), paper_id),
                )
                conn.commit()
                counts["skipped"] += 1
                continue

            logger.info("[%d/%d] %s", idx, total, pdf_url[:80])

            try:
                pdf_bytes: list[bytes] = []

                def handle_response(resp):
                    try:
                        ct = resp.headers.get("content-type", "")
                        if "application/pdf" in ct or "application/octet-stream" in ct:
                            data = resp.body()
                            if data.startswith(b"%PDF"):
                                pdf_bytes.append(data)
                    except Exception:
                        pass

                # Retry loop — allows re-attempting after CAPTCHA is solved.
                for attempt in range(3):
                    pdf_bytes.clear()
                    page.on("response", handle_response)

                    try:
                        page.goto(pdf_url, wait_until="domcontentloaded", timeout=_DOWNLOAD_TIMEOUT)
                    except PlaywrightTimeout:
                        pass

                    page.wait_for_timeout(3000)
                    page.remove_listener("response", handle_response)

                    if pdf_bytes:
                        break  # Got the PDF — exit retry loop

                    # Check if Cloudflare CAPTCHA is on screen
                    content = page.content().lower()
                    if "challenge" in content or "verify you are human" in content or "captcha" in content:
                        print(f"\n  ⚠️  CAPTCHA detected on paper [{idx}/{total}]")
                        print(f"  Please solve the CAPTCHA in the browser window.")
                        print(f"  Press ENTER here once the PDF has loaded in the browser...")
                        input()
                        # After user solves CAPTCHA, try to grab the already-loaded PDF
                        # by navigating again — Cloudflare sets a cookie so it won't re-challenge
                        continue
                    else:
                        break  # No CAPTCHA but no PDF either — give up

                if pdf_bytes:
                    dest.write_bytes(pdf_bytes[0])
                    conn.execute(
                        "UPDATE papers SET pdf_local_path = ? WHERE paper_id = ?",
                        (str(dest), paper_id),
                    )
                    conn.commit()
                    counts["downloaded"] += 1
                else:
                    logger.warning("No PDF response captured for %s", pdf_url[:70])
                    counts["failed"] += 1

            except Exception as exc:
                logger.error("Failed %s: %s", pdf_url[:70], exc)
                if dest.exists():
                    dest.unlink()
                counts["failed"] += 1

            time.sleep(_DELAY)

        context.close()
        browser.close()

    conn.close()

    print("\n" + "=" * 50)
    print("  Wiley Download Complete")
    print("=" * 50)
    print(f"  Downloaded : {counts['downloaded']:,}")
    print(f"  Failed     : {counts['failed']:,}")
    print(f"  Skipped    : {counts['skipped']:,}  (already on disk)")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Wiley PDFs via Playwright")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show counts without downloading")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
