"""
mechanics/d2im.py
=================
D2IM inference adapter for the Mechanics tab.

This is the one model-specific piece of the mechanics pipeline. It turns an
uploaded bone slice into predicted displacement and strain fields, faithfully
reproducing the pre-/post-processing of the original D2IM notebook
(github.com/PeterSoar/D2IM_Prototype) so the numbers match the paper.

Model contract (fixed by the trained weights):
  inputs  — scan: (1, 256, 256, 1) greyscale in [0, 1]
            mask: (1,  20,  20, 1) binary, marks the bone region
  outputs — three (1, 400) vectors = flattened 20×20 displacement fields
            u, v, w in *network units*; multiply by the voxel size (µm) to get
            physical displacement. Axial strain ε_zz (µε) is ∂w/∂z by finite
            difference over the DVC node spacing.

TensorFlow is imported lazily inside ``_load`` — importing this module is cheap
and never fails on a machine without TF, so the rest of BoneGraph keeps working
and the Mechanics tab degrades gracefully (``available`` reports why).

Thread-safety: the Keras model and matplotlib are driven under a single lock —
predictions are a sub-second forward pass, so serialising them is harmless and
avoids TF/pyplot re-entrancy issues across FastAPI's worker threads.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import threading
from pathlib import Path

import numpy as np

from config.settings import (
    D2IM_NODE_SPACING,
    D2IM_VOXEL_SIZE_UM,
    D2IM_WEIGHTS_PATH,
)

# Input geometry the trained weights expect — do not change without retraining.
SCAN_SHAPE = (256, 256)
MASK_SHAPE = (20, 20)
FIELD_SHAPE = (20, 20)  # output displacement / strain grid

_lock = threading.Lock()
_model = None


# ── availability ──────────────────────────────────────────────────────────────

def _weights_path() -> Path:
    return Path(D2IM_WEIGHTS_PATH)


def _tf_installed() -> bool:
    """True if TensorFlow can be imported, without paying the import cost."""
    return importlib.util.find_spec("tensorflow") is not None


def available() -> bool:
    """Whether a prediction can actually run: TF importable *and* weights present.
    Cheap — used by the /status endpoint to enable/disable the tab."""
    return _tf_installed() and _weights_path().exists()


def status() -> dict:
    """Structured availability for the UI: what's missing and how to fix it."""
    has_tf = _tf_installed()
    has_weights = _weights_path().exists()
    if has_tf and has_weights:
        reason = ""
    elif not has_tf and not has_weights:
        reason = "TensorFlow is not installed and the D2IM weights are missing."
    elif not has_tf:
        reason = "TensorFlow is not installed (pip install tensorflow)."
    else:
        reason = "D2IM weights not found — run: python -m scripts.download_d2im"
    return {
        "available": has_tf and has_weights,
        "tensorflow": has_tf,
        "weights": has_weights,
        "weights_path": str(_weights_path()),
        "reason": reason,
        "voxel_size_um": D2IM_VOXEL_SIZE_UM,
        "node_spacing": D2IM_NODE_SPACING,
    }


def _load():
    """Lazily load and cache the Keras model. First caller pays the TF import +
    weight-load cost; others wait on the lock."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            if not _weights_path().exists():
                raise FileNotFoundError(
                    f"D2IM weights not found at {_weights_path()}. "
                    "Run: python -m scripts.download_d2im"
                )
            # The weights are a legacy Keras 2 (.h5) model. On TF ≥ 2.16 the
            # default `tf.keras` is Keras 3, which can't reliably load that
            # format — so prefer the `tf-keras` shim (Keras 2) when present and
            # fall back to `tf.keras` on older TF builds.
            try:
                from tf_keras.models import load_model  # Keras 2 compat shim
            except ImportError:
                from tensorflow.keras.models import load_model

            # compile=False — inference only, so skip rebuilding optimizer/loss.
            _model = load_model(str(_weights_path()), compile=False)
    return _model


# ── pre-processing ────────────────────────────────────────────────────────────

def _to_grayscale_array(contents: bytes, filename: str = "") -> np.ndarray:
    """Decode uploaded bytes into a 2-D float array. Handles 8/16-bit TIFF
    micro-CT via tifffile and ordinary PNG/JPG via PIL."""
    name = (filename or "").lower()
    if name.endswith((".tif", ".tiff")):
        try:
            import tifffile

            arr = tifffile.imread(io.BytesIO(contents))
        except Exception:
            arr = None
        if arr is not None:
            arr = np.asarray(arr)
            if arr.ndim == 3:  # collapse channels / stray Z to a single plane
                arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr[0]
            return arr.astype(np.float32)
    from PIL import Image as PILImage

    img = PILImage.open(io.BytesIO(contents)).convert("L")
    return np.asarray(img, dtype=np.float32)


def _normalise_scan(arr: np.ndarray) -> np.ndarray:
    """Map a greyscale array into [0, 1] the way D2IM expects. 8-bit input is
    divided by 255 (faithful to the notebook); other ranges are min-max scaled
    so 16-bit micro-CT still lands in [0, 1]."""
    arr = np.nan_to_num(arr.astype(np.float32))
    hi = float(arr.max())
    if hi <= 1.0:
        return np.clip(arr, 0.0, 1.0)
    if hi <= 255.0:
        return np.clip(arr / 255.0, 0.0, 1.0)
    lo = float(arr.min())
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def _otsu_threshold(arr: np.ndarray) -> float:
    """Otsu's threshold on a [0, 1] image — used to derive an approximate bone
    mask when the user does not supply one."""
    hist, edges = np.histogram(arr.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    centers = (edges[:-1] + edges[1:]) / 2.0
    w_bg = np.cumsum(hist)
    w_fg = total - w_bg
    cum_mean = np.cumsum(hist * centers)
    total_mean = cum_mean[-1]
    valid = (w_bg > 0) & (w_fg > 0)
    if not valid.any():
        return 0.5
    mean_bg = np.zeros_like(w_bg)
    mean_fg = np.zeros_like(w_fg)
    mean_bg[valid] = cum_mean[valid] / w_bg[valid]
    mean_fg[valid] = (total_mean - cum_mean[valid]) / w_fg[valid]
    between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
    return float(centers[int(np.argmax(between))])


def _prep_scan(arr: np.ndarray) -> np.ndarray:
    """Resize to 256×256 and normalise — returns (1, 256, 256, 1).

    No vertical flip: the scan is fed in its native (uploaded) orientation, which
    matches how D2IM was trained (the training scans are not flipped). Input and
    predicted fields then share the orientation the user sees."""
    from scipy.ndimage import zoom

    arr = _normalise_scan(arr)
    zy, zx = SCAN_SHAPE[0] / arr.shape[0], SCAN_SHAPE[1] / arr.shape[1]
    scan = zoom(arr, (zy, zx), mode="nearest", order=0)
    return scan.reshape(1, SCAN_SHAPE[0], SCAN_SHAPE[1], 1).astype(np.float32)


def _prep_mask(arr: np.ndarray) -> np.ndarray:
    """Resize a supplied mask to 20×20, binarise + dilate — returns (1,20,20,1)."""
    from scipy.ndimage import binary_dilation, zoom

    arr = np.nan_to_num(arr.astype(np.float32))
    zy, zx = MASK_SHAPE[0] / arr.shape[0], MASK_SHAPE[1] / arr.shape[1]
    m = zoom(arr, (zy, zx), order=0, grid_mode=False)
    m = binary_dilation(m > 0)
    return m.reshape(1, MASK_SHAPE[0], MASK_SHAPE[1], 1).astype(np.float32)


def _auto_mask(scan_in: np.ndarray) -> np.ndarray:
    """Derive an approximate 20×20 bone mask from the prepped scan when the user
    supplies none: Otsu-threshold the 256×256 scan, downsample, dilate."""
    from scipy.ndimage import binary_dilation, zoom

    full = scan_in.reshape(SCAN_SHAPE)
    thresh = _otsu_threshold(full)
    binary = (full > thresh).astype(np.float32)
    zy, zx = MASK_SHAPE[0] / SCAN_SHAPE[0], MASK_SHAPE[1] / SCAN_SHAPE[1]
    m = zoom(binary, (zy, zx), order=0)
    m = binary_dilation(m > 0)
    return m.reshape(1, MASK_SHAPE[0], MASK_SHAPE[1], 1).astype(np.float32)


# ── strain ────────────────────────────────────────────────────────────────────

def _axial_strain(disp_w_um: np.ndarray, dx_um: float) -> np.ndarray:
    """ε_zz (µε) from the axial displacement field by central difference along
    the cranio-caudal (z) axis. ε = ∂w/∂z is dimensionless; ×1e6 → microstrain."""
    return np.gradient(disp_w_um, dx_um, axis=0) * 1e6


# ── public inference ──────────────────────────────────────────────────────────

def predict(scan_bytes: bytes, scan_name: str = "",
            mask_bytes: bytes | None = None, mask_name: str = "") -> dict:
    """Run D2IM on one bone slice.

    Returns a dict with displacement/strain arrays (already in physical units),
    summary statistics, and the parameters used. ``mask_source`` records whether
    the bone region came from the user ('uploaded') or Otsu ('auto'). Raises
    FileNotFoundError if weights are missing; lets other exceptions propagate to
    the endpoint, which reports them.
    """
    vs = D2IM_VOXEL_SIZE_UM
    dx = D2IM_NODE_SPACING * vs  # physical spacing between strain grid points (µm)

    scan_in = _prep_scan(_to_grayscale_array(scan_bytes, scan_name))
    if mask_bytes:
        mask_in = _prep_mask(_to_grayscale_array(mask_bytes, mask_name))
        mask_source = "uploaded"
    else:
        mask_in = _auto_mask(scan_in)
        mask_source = "auto"

    model = _load()
    with _lock:
        preds = model.predict([scan_in, mask_in], verbose=0)

    # Displacement fields → physical µm, kept in the scan's native orientation so
    # they overlay the input the user uploaded. Strain magnitudes are identical
    # to the notebook's (∂w/∂z over native displacement); only the notebook's
    # cosmetic display flip is dropped.
    u = np.asarray(preds[0]).reshape(FIELD_SHAPE) * vs
    v = np.asarray(preds[1]).reshape(FIELD_SHAPE) * vs
    w = np.asarray(preds[2]).reshape(FIELD_SHAPE) * vs

    from scipy.ndimage import binary_erosion

    msk = binary_erosion(mask_in.reshape(MASK_SHAPE) > 0, iterations=2)
    ezz = _axial_strain(w, dx)
    ezz_m = np.where(msk, ezz, 0.0)
    w_m = np.where(msk, w, 0.0)

    inside = ezz_m[msk]
    if inside.size == 0:
        inside = ezz_m.ravel()

    stats = {
        "peak_compressive_strain_ue": float(inside.min()),
        "peak_tensile_strain_ue": float(inside.max()),
        "mean_abs_strain_ue": float(np.abs(inside).mean()),
        "w_min_um": float(w_m.min()),
        "w_max_um": float(w_m.max()),
        "w_mean_um": float(w_m[msk].mean()) if msk.any() else 0.0,
    }
    return {
        "scan": scan_in.reshape(SCAN_SHAPE),
        "mask": msk,
        "u_um": u, "v_um": v, "w_um": w_m,
        "ezz_ue": ezz_m,
        "stats": stats,
        "mask_source": mask_source,
        "params": {"voxel_size_um": vs, "node_spacing": D2IM_NODE_SPACING},
    }


def render_figure(result: dict) -> str:
    """Render input scan, axial displacement (µm) and axial strain ε_zz (µε) as a
    single labelled PNG with colorbars; return base64. Uses the matplotlib OO API
    (no pyplot) so it is safe to call from a worker thread."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(13.5, 4.4), dpi=130)
    canvas = FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)

    ax = axes[0]
    ax.imshow(result["scan"], cmap="gray", vmin=0, vmax=1)
    ax.set_title("Input slice", fontsize=11)
    ax.axis("off")

    ax = axes[1]
    im = ax.imshow(result["w_um"], cmap="coolwarm")
    ax.set_title("Predicted axial displacement $w$ (µm)", fontsize=11)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    im = ax.imshow(result["ezz_ue"], cmap="plasma_r")
    ax.set_title("Predicted axial strain $\\varepsilon_{zz}$ (µε)", fontsize=11)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    buf = io.BytesIO()
    canvas.print_png(buf)
    return base64.b64encode(buf.getvalue()).decode()
