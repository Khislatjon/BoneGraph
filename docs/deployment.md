# Deployment — BoneGraph on Jetson AGX Orin

This guide deploys BoneGraph as a 24/7 public beta on an **NVIDIA Jetson AGX Orin
64 GB**, exposed at **bonegraph.org** through a Cloudflare Tunnel. No open ports,
HTTPS automatic, near-zero running cost.

```
Internet ──▶ Cloudflare (DNS + HTTPS + analytics) ──▶ cloudflared tunnel
                                                            │
                                              [Jetson AGX Orin 64 GB]
                                              ├─ run_api.sh → uvicorn api.main:app (127.0.0.1:8000)
                                              ├─ Ollama (huatuogpt-bone, llava:13b, llama3.2:3b)
                                              └─ TensorFlow + D2IM weights (Mechanics tab; optional)
```

The API launches through **`run_api.sh`**, not a bare `uvicorn` line — the script
sets two aarch64 shared-library workarounds (see Step 6) that the app needs in
order to import scikit-learn on the Jetson.

## Target hardware

| | |
|---|---|
| Device | Jetson AGX Orin 64 GB (2048 CUDA cores) |
| Memory | 64 GB unified (CPU+GPU shared) |
| CPU | 12-core ARM64 |
| Storage | 64 GB eMMC (no M.2 SSD — see note) |

**Storage note.** Running on the 64 GB eMMC without an SSD is fine for a beta but
leaves ~10–15 GB headroom after OS + models + Python env + DBs. TensorFlow + the
D2IM weights (~437 MB) eat into that — check `df -h /` before installing them.
Keep only the three required Ollama models (`ollama list` / `ollama rm`), set up
log rotation, and watch `df -h`. Add an M.2 (or USB 3) SSD and move
`OLLAMA_MODELS` + `data/` onto it before scaling up.

## Three things that don't "just clone"

1. **The databases are gitignored.** `data/db/` is not in the repo. You must
   transfer ~1.7 GB (mainly `chunks.db` 1.5 GB + `papers.db` 112 MB) from the
   dev machine to the Jetson (Step 5).
2. **`huatuogpt-bone` is a custom Ollama model** with no Modelfile in the repo.
   `ollama pull huatuogpt-bone` will fail — recreate it from its base + system
   prompt (Step 3).
3. **The Mechanics tab needs TensorFlow + the D2IM weights** (Step 5b), neither of
   which is in `pip install -r requirements.txt` output by default on this box
   (TF is a heavy, platform-specific wheel) nor in git (weights are large). The
   tab **degrades gracefully** — if either is missing, `/api/mechanics/status`
   reports it and the tab shows a "setup needed" panel while the other four tabs
   run normally. So you can ship without it and add it later.

---

## Step 0 — Prep the Jetson

```bash
sudo apt update && sudo apt upgrade -y
sudo nvpmodel -m 0          # MAXN — full power
sudo jetson_clocks          # max clocks (run after MAXN)
df -h /                     # record real free space before installing
sudo systemctl enable --now ssh   # if you'll manage it remotely
```

Ensure active cooling (fan) for sustained 24/7 load.

## Step 1 — Install Ollama (ARM64 / Jetson, CUDA-aware)

```bash
curl -fsSL https://ollama.com/install.sh | sh
systemctl status ollama            # installs as a service; should be active
ollama run llama3.2:3b "hi"        # then watch `sudo tegrastats` for GPU spike
```

## Step 2 — Pull the off-the-shelf models

```bash
ollama pull llava:13b
ollama pull llama3.2:3b
```

## Step 3 — Recreate the custom `huatuogpt-bone` model

On the **dev machine** where the model already exists, export its definition:

```bash
ollama show huatuogpt-bone --modelfile > huatuogpt-bone.Modelfile
```

Copy that file to the Jetson, then build it:

```bash
# on Jetson, in the folder containing the Modelfile
ollama create huatuogpt-bone -f huatuogpt-bone.Modelfile
ollama list                        # confirm all three models present
```

If the Modelfile references a base model you don't have yet, `ollama pull <base>`
first.

## Step 4 — Get the app + Python deps

```bash
git clone https://github.com/<you>/BoneGraph.git
cd BoneGraph
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The live install lives at `~/Documents/PhD/Projects/BoneGraph` on a **Python 3.13**
venv (`.venv/`). The path is arbitrary — just keep it consistent with the systemd
unit's `WorkingDirectory` (Step 6).

**PyTorch on Jetson:** plain `pip install torch` yields a CPU-only build, which is
acceptable for SPECTER2 query embedding (one short query at a time). Only if that
proves too slow, install NVIDIA's Jetson CUDA torch wheel.

## Step 5 — Transfer the databases (from the dev machine)

```bash
# run on the dev machine, from the project root
rsync -avP data/db/ <jetson-user>@<jetson-ip>:~/BoneGraph/data/db/
```

## Step 5b — Mechanics tab: TensorFlow + D2IM weights (optional)

The Mechanics tab runs **D2IM** (a TensorFlow/Keras CNN) on an uploaded micro-CT
slice. Both the framework and the trained weights are needed; skip this step and
the tab shows "setup needed" while everything else works.

**TensorFlow — mind the numpy version.** On **Python 3.13** the only TF wheels are
`tensorflow==2.20.x / 2.21.x`, and they pull `keras>=3.15`, which **requires
numpy 2**. Installing TF therefore upgrades the whole venv's numpy (e.g.
`1.26 → 2.5`). That is shared with the other tabs' embedding stack, so verify the
stack is numpy-2 ready first — a current torch (≥2.3), transformers,
sentence-transformers, scikit-learn (≥1.4) and scipy all are. `tf-keras` is
required to load the legacy Keras-2 `.h5`.

```bash
source .venv/bin/activate
pip install scipy matplotlib tifffile          # D2IM pre/post-processing + rendering
pip install "tensorflow==2.20.0" "tf-keras==2.20.1"   # ← upgrades numpy to 2.x
```

If pip can't find a TF wheel for your interpreter, check `pip index versions
tensorflow`; on aarch64 the wheel is CPU-only, which is fine — D2IM inference is a
sub-second forward pass and does not need the GPU.

**Weights** (`~437 MB`, gitignored) — either download onto the Jetson:

```bash
python -m scripts.download_d2im               # → data/models/D2IM_trained.h5 (GALA)
```

or rsync a copy you already have on the dev machine (faster, no external dep):

```bash
# on the dev machine, from the project root
rsync -avP data/models/D2IM_trained.h5 <jetson-user>@<jetson-ip>:~/BoneGraph/data/models/
```

Verify — expect `available: true`, `tensorflow: true`, `weights: true`:

```bash
python -c "from mechanics import d2im; print(d2im.status())"
```

## Step 6 — Run BoneGraph as a service

The API is launched by **`run_api.sh`** (already in the repo), which exports two
aarch64 workarounds before starting uvicorn:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# 1) miniforge libstdc++ provides CXXABI_1.3.15 that conda's libicu/nltk need
export LD_LIBRARY_PATH="/home/<jetson-user>/miniforge3/lib:${LD_LIBRARY_PATH:-}"
# 2) preload scikit-learn's vendored libgomp to avoid the aarch64
#    "cannot allocate memory in static TLS block" crash at import time
GOMP="$(ls .venv/lib/python3.13/site-packages/scikit_learn.libs/libgomp-*.so.* 2>/dev/null | head -1)"
[ -n "$GOMP" ] && export LD_PRELOAD="$GOMP:${LD_PRELOAD:-}"
exec .venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 "$@"
```

> **Note:** any time you run the app's Python outside the service (e.g. a manual
> `python -c "from retrieval..."` smoke test), replicate that `LD_LIBRARY_PATH` +
> `LD_PRELOAD` or scikit-learn's import will crash with the static-TLS error. It
> is not a numpy or code problem — just the missing preload.

Create `/etc/systemd/system/bonegraph.service`:

```ini
[Unit]
Description=BoneGraph API
After=network-online.target ollama.service
Wants=ollama.service

[Service]
User=<jetson-user>
WorkingDirectory=/home/<jetson-user>/BoneGraph
Environment=FEEDBACK_ADMIN_TOKEN=<pick-a-long-secret>
ExecStart=/home/<jetson-user>/BoneGraph/run_api.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
chmod +x run_api.sh
sudo systemctl daemon-reload
sudo systemctl enable --now bonegraph
curl http://127.0.0.1:8000/            # smoke test — expect 200 once startup completes
```

`--host 127.0.0.1` keeps the app local; the Cloudflare Tunnel is the only path in
from the internet. Startup loads the retriever + graph before the port answers, so
give it a few seconds; watch `journalctl -u bonegraph -f` for
`Application startup complete`.

## Step 7 — Cloudflare Tunnel → bonegraph.org

```bash
# install cloudflared (ARM64)
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 -o cloudflared
sudo install cloudflared /usr/local/bin/

cloudflared tunnel login                       # open the printed URL on any browser; pick bonegraph.org
cloudflared tunnel create bonegraph
cloudflared tunnel route dns bonegraph bonegraph.org
```

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: bonegraph
credentials-file: /home/<jetson-user>/.cloudflared/<tunnel-id>.json
ingress:
  - hostname: bonegraph.org
    service: http://127.0.0.1:8000
  - service: http_status:404
```

Run it as a service:

```bash
sudo cloudflared service install
sudo systemctl enable --now cloudflared
```

## Step 8 — Verify end-to-end

- Visit `https://bonegraph.org` from a phone → all five tabs load over HTTPS.
- Run one query per tab (Chat, Search, Reasoning, Vision); watch `sudo tegrastats`
  for GPU activity.
- **Mechanics:** click a bundled sample vertebra slice → **Predict fields** → a
  displacement/strain figure + stats appears. (Or check `/api/mechanics/status`
  returns `available: true` when logged in.) If TF/weights are absent you'll see a
  "setup needed" panel instead — that's expected, not a failure.
- Feedback endpoint is locked down:
  - `https://bonegraph.org/api/feedback/list` → **403**
  - `…/api/feedback/list?token=<your-secret>` → returns the list

---

## Operations

- **Logs:** `journalctl -u bonegraph -f` and `journalctl -u cloudflared -f`.
  Configure `logrotate` / journald size limits so logs don't fill the eMMC.
- **Disk watch:** `df -h /` regularly; the eMMC has limited headroom.
- **Updates:** `git pull` in the repo, then `sudo systemctl restart bonegraph`.
  If the pull added dependencies, `pip install -r requirements.txt` first (and see
  Step 5b if it touched the Mechanics/TensorFlow stack). Startup re-loads the
  models, so expect a few seconds of downtime on restart.
- **Models:** `ollama list`; remove anything unused with `ollama rm <model>`.
- **Reading feedback:** `curl "http://127.0.0.1:8000/api/feedback/list?token=<secret>"`
  on the Jetson, or via the public URL with the token.

## Pre-launch checklist

- [ ] `FEEDBACK_ADMIN_TOKEN` set in the service environment
- [ ] All three Ollama models present in `ollama list`
- [ ] `data/db/` transferred (chunks.db + papers.db + ontology.db)
- [ ] `run_api.sh` executable; systemd `ExecStart` points at it
- [ ] Both services `enabled` (survive reboot) and `active`
- [ ] `https://bonegraph.org` reachable; all five tabs functional
- [ ] (Optional) Mechanics: TensorFlow + `data/models/D2IM_trained.h5` present;
      `/api/mechanics/status` → `available: true`
- [ ] Feedback list returns 403 without the token
- [ ] Active cooling confirmed; `df -h` headroom acceptable
