# Deployment — BoneGraph on Jetson AGX Orin

This guide deploys BoneGraph as a 24/7 public beta on an **NVIDIA Jetson AGX Orin
64 GB**, exposed at **bonegraph.org** through a Cloudflare Tunnel. No open ports,
HTTPS automatic, near-zero running cost.

```
Internet ──▶ Cloudflare (DNS + HTTPS + analytics) ──▶ cloudflared tunnel
                                                            │
                                              [Jetson AGX Orin 64 GB]
                                              ├─ uvicorn api.main:app (127.0.0.1:8000)
                                              └─ Ollama (huatuogpt-bone, llava:13b, llama3.2:3b)
```

## Target hardware

| | |
|---|---|
| Device | Jetson AGX Orin 64 GB (2048 CUDA cores) |
| Memory | 64 GB unified (CPU+GPU shared) |
| CPU | 12-core ARM64 |
| Storage | 64 GB eMMC (no M.2 SSD — see note) |

**Storage note.** Running on the 64 GB eMMC without an SSD is fine for a beta but
leaves ~12–15 GB headroom after OS + models + Python env + DBs. Keep only the
three required models (`ollama list` / `ollama rm`), set up log rotation, and
watch `df -h`. Add an M.2 (or USB 3) SSD and move `OLLAMA_MODELS` + `data/` onto
it before scaling up.

## Two things that don't "just clone"

1. **The databases are gitignored.** `data/db/` is not in the repo. You must
   transfer ~1.7 GB (mainly `chunks.db` 1.5 GB + `papers.db` 112 MB) from the
   dev machine to the Jetson (Step 5).
2. **`huatuogpt-bone` is a custom Ollama model** with no Modelfile in the repo.
   `ollama pull huatuogpt-bone` will fail — recreate it from its base + system
   prompt (Step 3).

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

**PyTorch on Jetson:** plain `pip install torch` yields a CPU-only build, which
is acceptable for SPECTER2 query embedding (one short query at a time). Only if
that proves too slow, install NVIDIA's Jetson CUDA torch wheel.

## Step 5 — Transfer the databases (from the dev machine)

```bash
# run on the dev machine, from the project root
rsync -avP data/db/ <jetson-user>@<jetson-ip>:~/BoneGraph/data/db/
```

## Step 6 — Run BoneGraph as a service

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
ExecStart=/home/<jetson-user>/BoneGraph/.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now bonegraph
curl http://127.0.0.1:8000/api/stats    # smoke test — expect JSON
```

`--host 127.0.0.1` keeps the app local; the Cloudflare Tunnel is the only path
in from the internet.

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

- Visit `https://bonegraph.org` from a phone → all four tabs load over HTTPS.
- Run one query per tab (Chat, Search, Reasoning, Vision); watch `sudo tegrastats`
  for GPU activity.
- Feedback endpoint is locked down:
  - `https://bonegraph.org/api/feedback/list` → **403**
  - `…/api/feedback/list?token=<your-secret>` → returns the list

---

## Operations

- **Logs:** `journalctl -u bonegraph -f` and `journalctl -u cloudflared -f`.
  Configure `logrotate` / journald size limits so logs don't fill the eMMC.
- **Disk watch:** `df -h /` regularly; the eMMC has limited headroom.
- **Updates:** `git pull` in `~/BoneGraph`, then
  `sudo systemctl restart bonegraph`.
- **Models:** `ollama list`; remove anything unused with `ollama rm <model>`.
- **Reading feedback:** `curl "http://127.0.0.1:8000/api/feedback/list?token=<secret>"`
  on the Jetson, or via the public URL with the token.

## Pre-launch checklist

- [ ] `FEEDBACK_ADMIN_TOKEN` set in the service environment
- [ ] All three models present in `ollama list`
- [ ] `data/db/` transferred (chunks.db + papers.db + ontology.db)
- [ ] Both services `enabled` (survive reboot) and `active`
- [ ] `https://bonegraph.org` reachable; all four tabs functional
- [ ] Feedback list returns 403 without the token
- [ ] Active cooling confirmed; `df -h` headroom acceptable
