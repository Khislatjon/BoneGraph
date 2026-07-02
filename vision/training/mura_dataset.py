"""
vision/training/mura_dataset.py
===============================
Parse the MURA (Stanford musculoskeletal radiograph) dataset into training
records, with a patient-disjoint validation carve-out.

MURA on disk (under ``data/raw/images/MURA/``)::

    MURA-v1.1/train_image_paths.csv     one image path per line
    MURA-v1.1/valid_image_paths.csv
    MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/image1.png
                     └─region─┘  └─patient─┘ └──study+label──┘

Every label we need is encoded in the path:

  * **region**   — the ``XR_<REGION>`` folder (7 upper-limb regions).
  * **patient**  — the ``patientNNNNN`` folder. MURA guarantees no patient
                   appears in both the official train and valid splits, so the
                   official split is already leak-free. We only have to be
                   careful when carving our own val slice out of train.
  * **abnormal** — ``study<N>_positive`` (abnormal, 1) vs ``_negative``
                   (normal, 0).

We treat MURA's official ``train`` as the training pool and MURA's official
``valid`` as the held-out test set (its public test split is not released).
``carve_val`` splits a monitoring slice out of the training pool *by patient*.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from config.settings import DATA_DIR

MURA_ROOT = DATA_DIR / "raw" / "images" / "MURA"

# Paths inside the CSVs are relative to MURA_ROOT (they start with "MURA-v1.1/").
_TRAIN_CSV = MURA_ROOT / "MURA-v1.1" / "train_image_paths.csv"
_VALID_CSV = MURA_ROOT / "MURA-v1.1" / "valid_image_paths.csv"

# The 7 MURA body regions, in a fixed order so the label index is stable across
# runs (index 0 = elbow, ...). Sorted alphabetically — deterministic.
REGIONS = ("elbow", "finger", "forearm", "hand", "humerus", "shoulder", "wrist")
REGION_TO_IDX = {r: i for i, r in enumerate(REGIONS)}


@dataclass(frozen=True)
class ImageRecord:
    path: Path        # absolute path to the .png
    region: str       # e.g. "forearm"
    region_idx: int   # 0..6, index into REGIONS
    patient: str      # e.g. "patient00001"
    abnormal: int     # 1 = positive/abnormal, 0 = negative/normal
    split: str        # "train" | "valid" (MURA's official split)


def _parse_rel_path(rel: str, split: str) -> ImageRecord | None:
    """Turn one CSV line into an ImageRecord, or None if it doesn't match the
    expected ``.../XR_REGION/patientNNNNN/studyN_label/imageN.png`` shape."""
    rel = rel.strip()
    if not rel:
        return None
    parts = Path(rel).parts
    # ("MURA-v1.1", "train", "XR_SHOULDER", "patient00001", "study1_positive", "image1.png")
    if len(parts) < 6:
        return None
    region_dir, patient, study_dir = parts[2], parts[3], parts[4]
    if not region_dir.startswith("XR_"):
        return None
    region = region_dir[3:].lower()
    if region not in REGION_TO_IDX:
        return None
    abnormal = 1 if study_dir.endswith("_positive") else 0
    return ImageRecord(
        path=(MURA_ROOT / rel).resolve(),
        region=region,
        region_idx=REGION_TO_IDX[region],
        patient=patient,
        abnormal=abnormal,
        split=split,
    )


def load_records(split: str) -> list[ImageRecord]:
    """Load all image records for MURA's official ``"train"`` or ``"valid"``
    split, sorted by path so downstream feature caching is deterministic (and
    therefore resumable by index)."""
    csv_path = {"train": _TRAIN_CSV, "valid": _VALID_CSV}[split]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MURA {split} list not found at {csv_path}. Expected the dataset at "
            f"{MURA_ROOT}. Transfer MURA-v1.1/ there (see vision/training/README.md)."
        )
    records: list[ImageRecord] = []
    with open(csv_path, newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            rec = _parse_rel_path(row[0], split)
            if rec is not None:
                records.append(rec)
    records.sort(key=lambda r: str(r.path))
    return records


def carve_val(
    train_records: list[ImageRecord],
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Split the training pool into (train, val) **by patient**, so no patient's
    images land in both. Returns records in their original order within each
    side. ``val_frac`` is a fraction of *patients*, not images."""
    patients = sorted({r.patient for r in train_records})
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_frac))
    val_patients = set(patients[:n_val])
    train_side = [r for r in train_records if r.patient not in val_patients]
    val_side = [r for r in train_records if r.patient in val_patients]
    return train_side, val_side


def region_counts(records: list[ImageRecord]) -> dict[str, int]:
    """Images per region — for logging class imbalance (wrist ≫ humerus)."""
    counts = {r: 0 for r in REGIONS}
    for rec in records:
        counts[rec.region] += 1
    return counts


if __name__ == "__main__":
    # Quick sanity check — runs without any ML deps (no torch / open_clip).
    for split in ("train", "valid"):
        recs = load_records(split)
        print(f"{split}: {len(recs)} images across {len({r.patient for r in recs})} patients")
        for region, n in region_counts(recs).items():
            print(f"    {region:9s} {n}")
