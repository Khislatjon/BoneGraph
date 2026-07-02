"""
vision/training/cache_features.py
=================================
Run every MURA image through BiomedCLIP once and cache the 512-d vectors, so the
classifier head trains on features instead of re-encoding images every epoch.

This is the only GPU-heavy step — and it is *inference*, not training, which is
why it runs fine on the Jetson (the deployment box) even though that box is not
a training rig. Encode once, cache, and head-training afterwards is trivial and
can run anywhere.

Resumable by design. Feature extraction over 40k images can be interrupted (a
dropped SSH session, contention with the live app). Progress is checkpointed
every ``--flush-every`` images: on restart the script reloads what it has and
continues from where it stopped. A ``--limit`` flag caps images per split for a
quick smoke run (``--limit 2000`` finishes in a few minutes and proves the whole
pipeline before you commit to the full set).

Outputs, per split, under ``--out-dir`` (default ``data/db/vision_features/``):
    features_<split>.npz    X (N,512) float32 · region_idx (N,) · abnormal (N,)
                            · patient (N,) str · path (N,) str
    features_<split>.progress.json   {"n_processed": <records consumed>}

Usage::

    # smoke test — a couple of minutes, proves the pipeline end to end
    python -m vision.training.cache_features --split all --limit 2000

    # full run (the real thing)
    python -m vision.training.cache_features --split all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from config.settings import DATA_DIR
from vision.training.mura_dataset import ImageRecord, load_records, region_counts

DEFAULT_OUT_DIR = DATA_DIR / "db" / "vision_features"


def _paths(out_dir: Path, split: str) -> tuple[Path, Path]:
    return (out_dir / f"features_{split}.npz",
            out_dir / f"features_{split}.progress.json")


def _load_checkpoint(npz_path: Path, prog_path: Path):
    """Return (feats, region_idx, abnormal, patient, path, n_processed) from a
    prior run, or empty accumulators if none exists."""
    if npz_path.exists() and prog_path.exists():
        d = np.load(npz_path, allow_pickle=True)
        n_processed = json.loads(prog_path.read_text()).get("n_processed", 0)
        return (list(d["X"]), list(d["region_idx"]), list(d["abnormal"]),
                list(d["patient"]), list(d["path"]), int(n_processed))
    return [], [], [], [], [], 0


def _flush(npz_path: Path, prog_path: Path, feats, region_idx, abnormal,
           patient, path, n_processed: int) -> None:
    """Atomically persist the cache + progress marker. Write to a temp file and
    rename so an interrupted flush never corrupts the checkpoint."""
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    X = (np.vstack(feats).astype(np.float32) if feats
         else np.empty((0, 512), np.float32))
    # Write via a file handle: np.savez appends ".npz" to a *path* that doesn't
    # already end in .npz, which would break the rename below. A handle is saved
    # to verbatim.
    tmp = npz_path.with_name(npz_path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, X=X,
                 region_idx=np.asarray(region_idx, dtype=np.int64),
                 abnormal=np.asarray(abnormal, dtype=np.int64),
                 patient=np.asarray(patient, dtype=object),
                 path=np.asarray(path, dtype=object))
    tmp.replace(npz_path)
    prog_path.write_text(json.dumps({"n_processed": n_processed}))


def cache_split(split: str, *, limit: int | None, batch_size: int,
                flush_every: int, out_dir: Path) -> None:
    from PIL import Image as PILImage
    from vision import encoder  # imports open_clip/torch lazily — Jetson only

    records: list[ImageRecord] = load_records(split)
    if limit is not None:
        records = records[:limit]
    npz_path, prog_path = _paths(out_dir, split)

    feats, region_idx, abnormal, patient, path, start = _load_checkpoint(npz_path, prog_path)
    total = len(records)
    if start >= total:
        print(f"[{split}] already complete ({start}/{total}) — nothing to do")
        return
    if start:
        print(f"[{split}] resuming from {start}/{total} ({len(feats)} vectors cached)")
    else:
        print(f"[{split}] {total} images; regions: {region_counts(records)}")

    t0 = time.time()
    n_processed = start
    skipped = 0
    pending_imgs: list = []
    pending_meta: list[ImageRecord] = []

    def _encode_pending():
        nonlocal pending_imgs, pending_meta
        if not pending_imgs:
            return
        vecs = encoder.embed_batch(pending_imgs)
        for rec, vec in zip(pending_meta, vecs):
            feats.append(vec.astype(np.float32))
            region_idx.append(rec.region_idx)
            abnormal.append(rec.abnormal)
            patient.append(rec.patient)
            path.append(str(rec.path))
        pending_imgs, pending_meta = [], []

    for rec in records[start:]:
        try:
            img = PILImage.open(rec.path).convert("RGB")
            pending_imgs.append(img)
            pending_meta.append(rec)
        except Exception as e:  # corrupt/missing image — skip, keep index aligned
            skipped += 1
            print(f"[{split}] skip {rec.path}: {e}")
        n_processed += 1

        if len(pending_imgs) >= batch_size:
            _encode_pending()
        if n_processed % flush_every == 0:
            _encode_pending()
            _flush(npz_path, prog_path, feats, region_idx, abnormal, patient, path, n_processed)
            rate = (n_processed - start) / max(1e-6, time.time() - t0)
            eta = (total - n_processed) / max(1e-6, rate)
            print(f"[{split}] {n_processed}/{total}  {rate:.1f} img/s  "
                  f"eta {eta/60:.1f} min  (skipped {skipped})")

    _encode_pending()
    _flush(npz_path, prog_path, feats, region_idx, abnormal, patient, path, n_processed)
    dt = time.time() - t0
    print(f"[{split}] done: {len(feats)} vectors ({skipped} skipped) in {dt/60:.1f} min "
          f"→ {npz_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache BiomedCLIP features for MURA.")
    ap.add_argument("--split", choices=["train", "valid", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap images PER split (quick smoke run, e.g. 2000)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--flush-every", type=int, default=2000,
                    help="checkpoint the cache every N images (resumability)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    splits = ["train", "valid"] if args.split == "all" else [args.split]
    for split in splits:
        cache_split(split, limit=args.limit, batch_size=args.batch_size,
                    flush_every=args.flush_every, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
