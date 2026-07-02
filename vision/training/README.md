# Vision training — the classifier head (MURA → BiomedCLIP → head)

Trains the small classifier that grounds the Vision tab's VLM. Two stages:

1. **`cache_features`** — run every MURA image through BiomedCLIP once and cache
   the 512-d vectors. GPU-bound; this is the step worth running on the Jetson.
2. **`train_head`** — train a linear/MLP head on the cached vectors and report
   per-region accuracy on MURA's held-out valid split. Trivial; runs anywhere.

Feature extraction is *inference*, not training — which is why the Jetson (the
deployment box) handles it fine. The 64 GB Orin has ample headroom to run this
alongside the live app, but see "Coexisting with the live app" below.

---

## Prerequisites

- **MURA on the Jetson** at `data/raw/images/MURA/MURA-v1.1/` (transfer below).
- **`torch` with CUDA** and **`open_clip`** in the app venv. Verify first:

```bash
cd /home/jon/Documents/PhD/Projects/BoneGraph
.venv/bin/python -c "import torch, open_clip; print('cuda', torch.cuda.is_available())"
```

If `cuda` prints `False`, feature extraction falls back to CPU and will be slow
(hours, not minutes) — install NVIDIA's Jetson CUDA torch wheel first (see
`docs/deployment.md`, "PyTorch on Jetson").

---

## Step 1 — Transfer MURA to the Jetson

From the dev machine (only the `MURA-v1.1/` folder is needed, ~6.4 GB):

```bash
rsync -avP "data/raw/images/MURA/MURA-v1.1" \
  jon@<jetson-ip>:/home/jon/Documents/PhD/Projects/BoneGraph/data/raw/images/MURA/
```

`rsync` is resumable — if the transfer drops, re-run the same command.

## Step 2 — Get this code onto the Jetson

The training code lives on the `feature/vision-training` branch. On the Jetson:

```bash
cd /home/jon/Documents/PhD/Projects/BoneGraph
git fetch origin && git checkout feature/vision-training
```

The branch is **purely additive** (new `vision/training/` package + one new
`embed_batch` helper in `vision/encoder.py`), so it does not change the running
API. The live service keeps running on whatever process is already up until you
restart it.

## Step 3 — Environment (aarch64 shared-lib workarounds)

The same workarounds `run_api.sh` uses are needed to import torch/sklearn here.
Run these once per shell before the commands below:

```bash
cd /home/jon/Documents/PhD/Projects/BoneGraph
export LD_LIBRARY_PATH="/home/jon/miniforge3/lib:${LD_LIBRARY_PATH:-}"
GOMP="$(ls .venv/lib/python3.13/site-packages/scikit_learn.libs/libgomp-*.so.* 2>/dev/null | head -1)"
[ -n "$GOMP" ] && export LD_PRELOAD="$GOMP:${LD_PRELOAD:-}"
```

## Step 4 — Smoke run (do this first)

Cache features for 2,000 images per split (~a few minutes) to prove the whole
pipeline before committing to the full set:

```bash
.venv/bin/python -m vision.training.cache_features --split all --limit 2000
.venv/bin/python -m vision.training.train_head --target region
```

You should see a per-region accuracy table. Numbers will be weak on only 2k
images — the point is that it runs end to end.

## Step 5 — Full run

```bash
# ~30–90 min on the Orin GPU; resumable — re-run if interrupted
.venv/bin/python -m vision.training.cache_features --split all

# seconds; trains the head and reports accuracy on MURA's unseen valid split
.venv/bin/python -m vision.training.train_head --target region
# optional stronger head:
.venv/bin/python -m vision.training.train_head --target region --hidden 256 --epochs 40
```

Outputs:
- `data/db/vision_features/features_{train,valid}.npz` — cached vectors.
- `data/models/vision_region_head.pt` + `.json` — the trained head + metrics.

---

## Coexisting with the live app

Feature extraction shares the GPU with Ollama/the live tabs. On the 64 GB Orin
this is usually fine, but if you see the site lag:

- Run it in a low-traffic window.
- Lower `--batch-size` (e.g. `--batch-size 16`) to reduce peak GPU memory.
- It is **resumable** — safe to Ctrl-C and restart; it continues from the last
  checkpoint (every `--flush-every` images, default 2000).

## Resumability

`cache_features` checkpoints to `features_<split>.progress.json` and rewrites the
`.npz` atomically. If SSH drops or you stop it, just re-run the exact same
command — it reloads and continues. Delete the `.npz` + `.progress.json` for a
split to start that split over.

## Notes

- Labels come straight from the MURA path (`XR_<REGION>/patient.../study_<label>`);
  no manual labelling. Splits are **patient-disjoint** (MURA's own train/valid
  never share a patient; the val carve-out inside train is by patient too).
- `--target abnormal` trains normal/abnormal instead of region. Framing only —
  the Vision tab is **not** a diagnostic tool; keep region as the headline task.
- This head is the "linear probe". If it separates regions well, the features
  are good and a full fine-tune on the uni server is worth trying; if it does
  not, fine-tuning won't rescue it — a cheap early signal.
