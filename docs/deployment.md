# Deployment — BoneGraph (Cloudflare edge + Jetson API)

BoneGraph runs as a **split deployment**: the static frontend is served from
**Cloudflare's global edge**, and the API runs on an **NVIDIA Jetson AGX Orin
64 GB** at home, reached through a Cloudflare Tunnel. No open ports, HTTPS
automatic, near-zero running cost.

```
                       ┌─ bonegraph.org, www.bonegraph.org
Internet ──▶ Cloudflare ┤     └─▶ Worker "bonegraph" (static frontend/ from the edge — always up)
                       │
                       └─ api.bonegraph.org, ssh.bonegraph.org
                             └─▶ cloudflared tunnel
                                       │
                                 [Jetson AGX Orin 64 GB]  (API only)
                                 ├─ run_api.sh → uvicorn api.main:app (127.0.0.1:8000)
                                 ├─ Ollama (huatuogpt-bone, llava:13b, llama3.2:3b)
                                 └─ TensorFlow + D2IM weights (Mechanics tab; optional)
```

**Why the split.** The Jetson is on home Wi-Fi (RTL8822CE) that drops several
times an hour. When one process served both the page and the API, every drop
took the whole domain down (Cloudflare 530). Now the page always loads from the
edge and a Jetson blip degrades a single in-flight query instead of killing the
site. The frontend calls the API cross-origin at `api.bonegraph.org` (see
`API_BASE` in `frontend/index.html`); CORS is already `allow_origins=["*"]`.

Two independent pieces to deploy:

- **The API on the Jetson** — Steps 0–8 below (Ollama, models, DBs, systemd,
  tunnel). This is the bulk of the work.
- **The frontend on Cloudflare** — the "Frontend deployment" section near the
  end. It's a one-time dashboard setup, then auto-deploys on `git push`.

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
2. **`huatuogpt-bone` is a custom Ollama model.** `ollama pull huatuogpt-bone`
   will fail. `huatuogpt-bone.Modelfile` is in the repo, but its `FROM` line
   points at a local `huatuogpt-bone.base.gguf` that is not — recreate the model
   from its base + system prompt (Step 3).
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
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/api/stats   # expect 401
```

The API is **API-only** — it no longer serves any HTML, so `/` returns **404**,
not 200. Smoke-test an API route instead: `/api/stats` returns **401
Not authenticated** (it's auth-gated) once startup is complete — a 401 means the
app is up.

`--host 127.0.0.1` keeps the app local; the Cloudflare Tunnel is the only path in
from the internet. Startup loads the retriever + graph before the port answers, so
give it a few seconds; watch `journalctl -u bonegraph -f` for
`Application startup complete`.

## Step 7 — Cloudflare Tunnel → api.bonegraph.org

The tunnel exposes the **API** (and SSH), not the site. `bonegraph.org` and
`www.bonegraph.org` are served by the Cloudflare Worker (see "Frontend
deployment"); the tunnel owns `api.bonegraph.org` and `ssh.bonegraph.org`.

```bash
# install cloudflared (ARM64)
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 -o cloudflared
sudo install cloudflared /usr/local/bin/

cloudflared tunnel login                       # open the printed URL on any browser; pick bonegraph.org
cloudflared tunnel create bonegraph
cloudflared tunnel route dns bonegraph api.bonegraph.org
cloudflared tunnel route dns bonegraph ssh.bonegraph.org   # optional: SSH over the tunnel
```

The service runs with an explicit config path, so create the config at
**`/etc/cloudflared/config.yml`** (this is the one the service reads — see the
trap note below):

```yaml
tunnel: <tunnel-id>            # e.g. b80de107-6748-4a56-933c-198971050120
credentials-file: /etc/cloudflared/<tunnel-id>.json
ingress:
  - hostname: api.bonegraph.org
    service: http://127.0.0.1:8000
  - hostname: ssh.bonegraph.org
    service: ssh://localhost:22
  # bonegraph.org + www are served by the Worker. These two rules are kept as a
  # rollback path only: delete the Worker routes and traffic falls back here.
  # (Since the API no longer serves HTML, that fallback serves the API, not the
  # page — a true rollback also needs the frontend handlers restored in main.py.)
  - hostname: bonegraph.org
    service: http://127.0.0.1:8000
  - hostname: www.bonegraph.org
    service: http://127.0.0.1:8000
  - service: http_status:404
```

> **Trap — two config files.** `cloudflared service install` may drop a config at
> `~/.cloudflared/config.yml`, but this deployment runs the service with
> `--config /etc/cloudflared/config.yml`. Edit **only** the `/etc/cloudflared/`
> one; the `~/.cloudflared/` copy is stale and ignored. Check which is live with
> `systemctl cat cloudflared | grep ExecStart`.

Run it as a service:

```bash
sudo cloudflared service install
sudo systemctl enable --now cloudflared
curl -s -o /dev/null -w '%{http_code}\n' https://api.bonegraph.org/api/stats   # expect 401
```

## Step 8 — Verify end-to-end

- **Edge vs tunnel** — confirm each hostname is served by the right thing.
  `GET /index.html` is the discriminator: **307** (clean-URL redirect) = the
  Worker; **404** = the Jetson (the API has no such route).
  ```bash
  curl -s -o /dev/null -w '%{http_code}\n' https://bonegraph.org/index.html      # 307 → Worker (edge)
  curl -s -o /dev/null -w '%{http_code}\n' https://api.bonegraph.org/index.html  # 404 → Jetson (API)
  ```
- Visit `https://bonegraph.org` from a phone → the page loads from the edge; all
  five tabs render. API calls go to `api.bonegraph.org`.
- Run one query per tab (Chat, Search, Reasoning, Vision); watch `sudo tegrastats`
  on the Jetson for GPU activity.
- **Mechanics:** click a bundled sample vertebra slice → **Predict fields** → a
  displacement/strain figure + stats appears. (Or check `/api/mechanics/status`
  returns `available: true` when logged in.) If TF/weights are absent you'll see a
  "setup needed" panel instead — that's expected, not a failure.
- **Graceful degradation** — stop the API (`sudo systemctl stop bonegraph`) and
  reload `bonegraph.org`: the page still loads, and each tab shows
  "⚠️ Error: The server seems to be down, please try after a while." Restart with
  `sudo systemctl start bonegraph`.
- Feedback endpoint is locked down (on the API host now):
  - `https://api.bonegraph.org/api/feedback/list` → **403**
  - `…/api/feedback/list?token=<your-secret>` → returns the list

---

## Frontend deployment (Cloudflare Worker — static assets)

The `frontend/` directory (HTML + `static/`, no build step — JSX is transpiled in
the browser by `babel.min.js`) is served as an **assets-only Worker** from
Cloudflare's edge. This is a one-time setup; afterwards it **auto-deploys on every
`git push` to `main`**.

**Config in the repo** (`frontend/wrangler.jsonc`):

```jsonc
{
  "name": "bonegraph",
  "compatibility_date": "2026-07-17",
  "assets": { "directory": "./" }   // assets-only: no "main", nothing runs server-side
}
```

`frontend/.assetsignore` keeps `.wrangler`, `wrangler.jsonc` and `.DS_Store` off
the CDN.

**One-time dashboard setup** (Cloudflare → Workers & Pages → Create → import
`Khislatjon/BoneGraph`):

| Setting | Value | Why |
|---|---|---|
| Root directory | **`/frontend`** | **Critical.** At the repo root the build auto-detects `requirements.txt` and tries to `pip install` torch/tensorflow/CUDA to publish static files — 12 min, then fails. `frontend/` has no Python, so detection finds nothing. |
| Build command | *(empty)* | No build step. |
| Deploy command | `npx wrangler deploy` | Finds `frontend/wrangler.jsonc`. |
| Production branch | `main` | Auto-deploys on push. |

> `SKIP_DEPENDENCY_INSTALL` does **not** fix the pip problem — set via the
> dashboard's Variables it becomes a Worker *runtime* variable the build never
> reads. Setting Root directory is the fix.

A green build (~30 s) publishes to `https://bonegraph.<subdomain>.workers.dev`.

**Point the domain at the Worker.** A Custom Domain refuses while the tunnel CNAME
exists ("delete the DNS record first" → brief outage). Use **Worker Routes**
instead — they intercept the existing proxied DNS record before the tunnel, with
zero downtime:

- `bonegraph.org/*` → Worker `bonegraph` — add from the Worker's **Domains** tab.
- `www.bonegraph.org/*` → Worker `bonegraph` — the Worker-side dialog errors
  ("No zones match www.bonegraph.org"), so add it from the **zone** side:
  Dashboard → bonegraph.org → **Workers Routes** → Add route.

**How the frontend finds the API.** `API_BASE` in `frontend/index.html` and
`frontend/admin.html` resolves to `https://api.bonegraph.org` on the public hosts
(`bonegraph.org`, `www.`, `*.workers.dev`, `*.pages.dev`) and to
`http://localhost:8000` everywhere else. Same-origin is no longer a valid
fallback: since the split the API serves no HTML, so the page and the API are
never on the same origin. To drive a different box — a LAN IP, or this Jetson
from another machine — set `localStorage.setItem('bg_api_base', 'http://<host>:8000')`
in the browser console; that key wins over both branches. CORS is
`allow_origins=["*"]`, so no proxy is needed.

**Shipping frontend changes:** just `git push`. Cloudflare rebuilds and deploys;
no Jetson involvement, and it works even while the Jetson is offline. **Backend
changes still need** `ssh` to the Jetson + `git pull` + restart (Operations below).

## Operations

- **Logs:** `journalctl -u bonegraph -f` and `journalctl -u cloudflared -f`.
  Configure `logrotate` / journald size limits so logs don't fill the eMMC.
- **Disk watch:** `df -h /` regularly; the eMMC has limited headroom.
- **Backend updates:** `git pull` in the repo on the Jetson, then
  `sudo systemctl restart bonegraph`. If the pull added dependencies,
  `pip install -r requirements.txt` first (and see Step 5b if it touched the
  Mechanics/TensorFlow stack). Startup re-loads the models, so expect a few
  seconds of API downtime on restart — the page stays up (it's on the edge).
- **Frontend updates:** `git push` only. Cloudflare rebuilds `frontend/` and
  deploys to the edge automatically; nothing to do on the Jetson.
- **Wi-Fi self-heal (`net-watchdog`).** The Jetson's Wi-Fi (RTL8822CE) drops
  periodically. A systemd timer (`net-watchdog.timer` → `net-watchdog.sh`) pings
  the gateway every few minutes and cycles the radio when it's unreachable; a hard
  hang is caught by the Tegra hardware watchdog. The vendor driver ignores the
  standard powersave setting, so its power management is disabled at the driver
  level via `/etc/modprobe.d/rtl8822ce.conf`
  (`rtw_power_mgnt=0 rtw_ips_mode=0 rtw_lps_level=0 rtw_lps_chk_by_tp=0`). While a
  drop is active the API is unreachable and every tab shows the "server seems to be
  down" message; the page itself stays up.
- **Rollback.** *Frontend:* revert the commit and `git push` (or roll back the
  deployment in the Cloudflare dashboard). *Backend:* `git reset --hard <good-sha>`
  on the Jetson + `sudo systemctl restart bonegraph`.
- **Models:** `ollama list`; remove anything unused with `ollama rm <model>`.
- **Reading feedback:** `curl "http://127.0.0.1:8000/api/feedback/list?token=<secret>"`
  on the Jetson, or `https://api.bonegraph.org/api/feedback/list?token=<secret>`.

## Pre-launch checklist

- [ ] `FEEDBACK_ADMIN_TOKEN` set (loaded from `.env` by `config/settings.py`, or
      as a systemd `Environment=`)
- [ ] All three Ollama models present in `ollama list`
- [ ] `data/db/` transferred (chunks.db + papers.db + ontology.db)
- [ ] `run_api.sh` executable; systemd `ExecStart` points at it
- [ ] Both services `enabled` (survive reboot) and `active`
- [ ] `api.bonegraph.org/api/stats` → 401 (API up via tunnel)
- [ ] Worker deployed; `bonegraph.org` + `www.` routes point at it
      (`GET bonegraph.org/index.html` → 307)
- [ ] `https://bonegraph.org` reachable; all five tabs render and reach the API
- [ ] (Optional) Mechanics: TensorFlow + `data/models/D2IM_trained.h5` present;
      `/api/mechanics/status` → `available: true`
- [ ] Feedback list returns 403 without the token
- [ ] `net-watchdog.timer` active; `/etc/modprobe.d/rtl8822ce.conf` present
- [ ] Active cooling confirmed; `df -h` headroom acceptable
