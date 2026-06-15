"""
vision/encoder.py
=================

BiomedCLIP image embeddings for the Vision tab's correction memory.

This is the one model-specific piece of the memory layer. `correction_store`
stays encoder-agnostic (it takes vectors); this module is what turns a PIL image
into those vectors. Swapping BiomedCLIP for another encoder later means changing
only this file — as long as `EMBED_DIM` and the vector contract hold.

Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (a ViT-B/16
image tower trained on PubMed figure–caption pairs). 512-d image embeddings.
Loaded lazily and cached as a process singleton — the first call downloads
weights (~400 MB) from the HF hub and is slow; subsequent calls are fast.

Augmentation (the rotation-invariance fix): vanilla CLIP/BiomedCLIP embeddings
shift under rotation, so storing only the original vector would miss a rotated
re-upload of the same scan. `embed_with_augments` returns the original plus a
handful of rotated/flipped copies; the store keeps each correction's
best-matching variant at recall time.
"""

from __future__ import annotations

import threading

import numpy as np

# BiomedCLIP ViT-B/16 image-tower dimensionality. Kept as a module constant so
# callers (and the store) can assert against it without loading the model.
EMBED_DIM = 512

_HF_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# Augmentations applied per image. Rotation is the main invariance we need for
# scans; a horizontal flip covers laterally-mirrored uploads. Names are stored
# in correction_embeddings.variant for debugging which copy matched.
_AUGMENTS = ("original", "rot90", "rot180", "rot270", "hflip")

_lock = threading.Lock()
_model = None
_preprocess = None


def _load():
    """Lazily load and cache the BiomedCLIP model + preprocessing transform.
    Thread-safe: the first caller pays the download/load cost, others wait."""
    global _model, _preprocess
    if _model is not None:
        return _model, _preprocess
    with _lock:
        if _model is None:
            import open_clip
            import torch

            model, preprocess = open_clip.create_model_from_pretrained(_HF_MODEL)
            model.eval()
            _model, _preprocess = model, preprocess
    return _model, _preprocess


def _augment(pil_img):
    """Yield (variant_name, transformed PIL image) for each configured augment."""
    from PIL import Image as PILImage

    for name in _AUGMENTS:
        if name == "original":
            yield name, pil_img
        elif name == "rot90":
            yield name, pil_img.transpose(PILImage.ROTATE_90)
        elif name == "rot180":
            yield name, pil_img.transpose(PILImage.ROTATE_180)
        elif name == "rot270":
            yield name, pil_img.transpose(PILImage.ROTATE_270)
        elif name == "hflip":
            yield name, pil_img.transpose(PILImage.FLIP_LEFT_RIGHT)


def _encode_batch(pil_images: list) -> np.ndarray:
    """Encode a list of PIL images into an (N, EMBED_DIM) float32 array of
    L2-normalised embeddings."""
    import torch

    model, preprocess = _load()
    batch = torch.stack([preprocess(im.convert("RGB")) for im in pil_images])
    # no_grad locally — grad mode is thread-local, so we can't rely on a global
    # toggle holding across FastAPI's worker threads.
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


def embed_image(pil_img) -> np.ndarray:
    """Embed a single image into a 1-D, L2-normalised float32 vector.
    Use this for the recall query (the incoming image)."""
    return _encode_batch([pil_img.convert("RGB")])[0]


def embed_with_augments(pil_img) -> tuple[np.ndarray, list[str]]:
    """Embed the original image plus its augmented copies.

    Returns ``(vectors, variants)`` where ``vectors`` is an
    ``(n_augments, EMBED_DIM)`` array and ``variants`` names each row. Use this
    when *storing* a correction so recall survives rotation/flip of the scan.
    """
    rgb = pil_img.convert("RGB")
    names, images = zip(*_augment(rgb))
    return _encode_batch(list(images)), list(names)
