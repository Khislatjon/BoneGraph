"""
vision/classifier.py
====================
Bone-region classifier head over frozen BiomedCLIP features — the trained
component that *grounds* the Vision tab's VLM (the "hybrid" from
docs/vision/training.md).

Loads the head trained by ``vision/training/train_head.py``
(``data/models/vision_region_head.pt``) and predicts a body region for an
uploaded image. The prediction is handed to ``llava:13b`` as a hint in
``/api/vision/chat`` so the conversational answer is anchored to a model trained
on bone data rather than the VLM guessing unaided.

Scope / caveats:
  * Trained on **MURA upper-limb X-rays** (elbow, finger, forearm, hand,
    humerus, shoulder, wrist). Inputs outside that distribution — spine, MRI,
    CT, micro-CT, non-bone — are out of scope, and the head will still emit one
    of the seven labels. Callers must frame it as a hint and lean on confidence.
  * Encoder-shared: embeds via ``vision.encoder`` (BiomedCLIP), exactly like
    correction memory, so only the small head is model-specific here. The head
    was trained on L2-normalised embeddings, which is what ``embed_image``
    returns — keep that contract.

Loaded lazily and cached as a process singleton. If the model file is absent the
module degrades gracefully (``is_available()`` is False, ``predict`` returns
None) so the Vision tab still works without the trained head.
"""

from __future__ import annotations

import threading

import numpy as np

from config.settings import MODELS_DIR

MODEL_PATH = MODELS_DIR / "vision_region_head.pt"
# Out-of-distribution reference: a stratified subsample of the training features
# + a nearest-neighbour-cosine threshold, calibrated so ~99% of legitimate MURA
# images pass. Softmax confidence is NOT an OOD detector (the head is forced to
# pick one of seven classes, so it is confident even on a spine micro-CT); this
# reference set answers the real question — "does this image look like anything
# the head was trained on?" — by distance in BiomedCLIP space.
REFSET_PATH = MODELS_DIR / "vision_region_refset.npz"

_lock = threading.Lock()
_model = None
_classes: list[str] | None = None
_target: str | None = None
_refset = None            # (N, 512) float32, L2-normalised
_ood_threshold: float | None = None


def is_available() -> bool:
    """True if the trained head file is present (cheap check, no model load)."""
    return MODEL_PATH.exists()


def _build_head(dim: int, hidden: int, n_classes: int):
    """Reconstruct the head architecture used at training time. Must mirror
    ``vision/training/train_head.py``."""
    import torch.nn as nn

    if hidden and hidden > 0:
        return nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, n_classes)
        )
    return nn.Linear(dim, n_classes)


def _load():
    """Lazily load and cache the head + label list. First call pays the cost."""
    global _model, _classes, _target
    if _model is not None:
        return _model, _classes
    with _lock:
        if _model is None:
            import torch

            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"trained head not found at {MODEL_PATH}")
            # weights_only=False: our own trained checkpoint carries metadata
            # (classes list, target, hidden), not just tensors. It is a trusted,
            # locally-produced file.
            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            classes = list(ckpt["classes"])
            model = _build_head(int(ckpt["dim"]), int(ckpt.get("hidden", 0)), len(classes))
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            _model, _classes, _target = model, classes, ckpt.get("target", "region")
    return _model, _classes


def _load_refset():
    """Lazily load the OOD reference set + threshold. Returns (ref, threshold) or
    (None, None) if the file is absent (guard then treated as always in-scope)."""
    global _refset, _ood_threshold
    if _refset is not None:
        return _refset, _ood_threshold
    with _lock:
        if _refset is None and REFSET_PATH.exists():
            d = np.load(REFSET_PATH)
            _refset = d["ref"].astype(np.float32)  # (N, 512), already L2-normalised
            _ood_threshold = float(d["threshold"])
    return _refset, _ood_threshold


def _ood_check(vec: np.ndarray) -> tuple[float | None, bool]:
    """Nearest-neighbour cosine of ``vec`` to the reference set. Returns
    ``(score, in_scope)``. If no reference set is installed, treats everything as
    in-scope (score None) so the guard degrades gracefully."""
    ref, thr = _load_refset()
    if ref is None:
        return None, True
    score = float((ref @ vec).max())  # both L2-normalised → cosine
    return score, (score >= thr)


def predict(pil_img, top_k: int = 3) -> dict | None:
    """Predict the body region for a PIL image, with an out-of-scope guard.

    Returns ``{"label", "confidence", "topk": [(label, prob), ...], "target",
    "ood_score", "in_scope"}`` or ``None`` if the model is unavailable. When
    ``in_scope`` is False the image does not resemble the head's training data
    (upper-limb X-rays); callers should NOT ground the VLM on ``label`` in that
    case. Never raises on inference errors — a classifier hiccup must not break
    the Vision chat stream.
    """
    try:
        import torch
        from vision import encoder

        model, classes = _load()
        vec = np.asarray(encoder.embed_image(pil_img), dtype=np.float32)  # (512,) L2-normalised
        with torch.no_grad():
            logits = model(torch.tensor(vec).unsqueeze(0))
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        order = np.argsort(probs)[::-1][:top_k]
        topk = [(classes[int(i)], float(probs[int(i)])) for i in order]
        ood_score, in_scope = _ood_check(vec)
        return {"label": topk[0][0], "confidence": topk[0][1], "topk": topk,
                "target": _target, "ood_score": ood_score, "in_scope": in_scope}
    except Exception:
        return None
