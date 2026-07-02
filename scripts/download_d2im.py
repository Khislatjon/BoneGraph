"""
scripts/download_d2im.py
────────────────────────
Fetch the trained D2IM model weights used by the Mechanics tab.

The weights are too large for git, so they live in the University of Greenwich
GALA repository and download on demand into data/models/. Run once after a fresh
clone (or whenever you want the data-augmentation variant):

    python -m scripts.download_d2im                       # default D2IM_trained.h5
    python -m scripts.download_d2im --augmented           # data-augmentation variant
    python -m scripts.download_d2im --url <URL> --out <PATH>

Idempotent: skips the download if the target file already exists (use --force to
re-download). Streams to a .part file and renames on success so an interrupted
download never leaves a half-written model in place.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

from config.settings import D2IM_WEIGHTS_PATH, D2IM_WEIGHTS_URL, MODELS_DIR

# The data-augmentation variant lives alongside the default weights on GALA.
AUGMENTED_URL = "https://gala.gre.ac.uk/id/eprint/50955/4/D2IM_trained_data_augmentation.h5"


def download(url: str, out: Path, force: bool = False) -> None:
    if out.exists() and not force:
        print(f"✓ Already present: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
        print("  Use --force to re-download.")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    print(f"Downloading {url}\n        → {out}")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB ({pct:4.1f}%)",
                          end="", flush=True)
    print()
    tmp.replace(out)
    print(f"✓ Saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download D2IM model weights.")
    ap.add_argument("--augmented", action="store_true",
                    help="fetch the data-augmentation variant")
    ap.add_argument("--url", default=None, help="override the download URL")
    ap.add_argument("--out", default=None, help="override the output path")
    ap.add_argument("--force", action="store_true", help="re-download if present")
    args = ap.parse_args()

    if args.augmented:
        url = args.url or AUGMENTED_URL
        out = Path(args.out or (MODELS_DIR / "D2IM_trained_data_augmentation.h5"))
    else:
        url = args.url or D2IM_WEIGHTS_URL
        out = Path(args.out or D2IM_WEIGHTS_PATH)

    try:
        download(url, out, force=args.force)
    except requests.HTTPError as e:
        print(f"✗ Download failed: {e}", file=sys.stderr)
        print("  Check the URL in models/MODELS.md upstream or pass --url.", file=sys.stderr)
        return 1
    except requests.RequestException as e:
        print(f"✗ Network error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
