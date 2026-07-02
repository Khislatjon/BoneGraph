# Mechanics Tab — Architecture

**Status: 🟢 Built (June 2026).** Single-slice D2IM inference ships: upload one
undeformed micro-CT slice → predicted displacement and axial-strain fields.

The Mechanics tab is the **quantitative** counterpart to the
[Vision tab](../vision/architecture.md). Where Vision answers *"what am I
looking at?"* with a VLM, Mechanics answers *"how does it deform?"* by running
**D2IM** — the group's own deep-learning image-mechanics model (Soar, Palanca,
Dall'Ara & Tozzi, *J. Orthop. Translat.* 2024;
[paper](https://www.sciencedirect.com/science/article/pii/S2352431624000828),
[code](https://github.com/PeterSoar/D2IM_Prototype)) — on a single bone image.

> Research and educational use only — **not a clinical tool.**

---

## The one-line idea

A trained CNN maps the **greyscale content of an undeformed XCT slice** to the
**displacement field** it would experience under load; differentiating the axial
component gives the **axial strain field** ε_zz. No finite-element model, no
digital volume correlation at inference — just the image.

---

## Pipeline

```
upload slice ─▶ preprocess ─▶ D2IM CNN ─▶ post-process ─▶ render + stats
 (TIFF/PNG/JPG)   256×256,        (Keras)    µm / µε,         PNG + JSON
   (+ mask?)      /255, mask                  ε_zz = ∂w/∂z
```

1. **Preprocess** (`mechanics/d2im.py`, faithful to the upstream notebook):
   - **Scan** → resize to **256×256** (nearest), normalise to `[0,1]`, orient →
     `(1, 256, 256, 1)`. 8-bit input is divided by 255; other bit-depths (16-bit
     micro-CT) are min-max scaled.
   - **Mask** → resize to **20×20**, binarise, dilate → `(1, 20, 20, 1)`. If the
     user supplies no mask, an **approximate** one is derived from the scan by
     Otsu threshold (reported as `mask_source: "auto"`).
2. **Inference** — one forward pass returns three flattened `20×20` displacement
   fields `(u, v, w)` in network units, gated by the mask.
3. **Post-process** — displacement → µm (× voxel size); axial strain
   ε_zz (µε) = ∂w/∂z by finite difference over the DVC node spacing.
   Calibration constants (`D2IM_VOXEL_SIZE_UM = 39`, `D2IM_NODE_SPACING = 50`)
   come from the paper and live in [`config/settings.py`](../../config/settings.py).
4. **Output** — a labelled PNG (input · displacement · strain) plus summary
   statistics (peak compressive/tensile strain, mean |strain|, displacement
   range) in physical units.

---

## Design notes

- **Isolated, swappable adapter.** Everything depends on the
  `predict` / `render_figure` contract in [`mechanics/d2im.py`](../../mechanics/d2im.py),
  never on TensorFlow. Swapping in **D2IM-Strain** or a retrained model is a
  change to the weights path (`D2IM_WEIGHTS_PATH`) and, if needed, that one file.
- **TensorFlow is lazy and optional.** Importing the module never imports TF;
  it loads only on the first real prediction. If TF or the weights are missing,
  `/api/mechanics/status` reports why and the tab shows setup instructions — the
  other three tabs are unaffected.
- **No streaming.** Unlike the LLM tabs, D2IM is a single sub-second compute, so
  `/api/mechanics/predict` returns plain JSON, not SSE.
- **Scope.** Micro-CT slices, consistent with the Vision tab's micro-CT scope
  lock (micro-CT yes, nano-CT no).
- **Training provenance.** The bundled weights were trained on our own vertebrae
  dataset, so the model generalises best to similar micro-CT slices. The UI and
  the `/predict` disclaimer state this. A CLI to (re)train D2IM on a user's own
  dataset is planned for a future release.
- **Sample slices.** Two bundled vertebra slices with matching bone masks live in
  `frontend/static/samples/` (served at `/static/samples/`) so users can try the
  tab without their own data. Each sample ships as the real `.tif` (scan + mask,
  sent to D2IM) plus a `.png` preview for the browser. Source: the D2IM prototype
  dataset (`S3_INT_UL_AP` — intact, AP view; `S8_LES_UL_ML` — lesion, ML view).

---

## File map

| File | Role |
|---|---|
| [`mechanics/d2im.py`](../../mechanics/d2im.py) | D2IM adapter: preprocessing, inference, strain, rendering |
| [`config/settings.py`](../../config/settings.py) | weights path/URL + voxel/node-spacing calibration |
| [`scripts/download_d2im.py`](../../scripts/download_d2im.py) | fetch trained weights from GALA into `data/models/` |
| `api/main.py` | `/api/mechanics/status` + `/api/mechanics/predict` endpoints |
| `frontend/index.html` | `MechanicsTab` component (upload → figure + stats) |

---

## Setup

The trained weights are not in git (large). Fetch once:

```bash
python -m scripts.download_d2im                 # default D2IM_trained.h5
python -m scripts.download_d2im --augmented     # data-augmentation variant
```

then ensure TensorFlow is installed (`pip install tensorflow`, or a
platform-appropriate build on the Jetson). The tab self-reports readiness via
its status endpoint.
```
