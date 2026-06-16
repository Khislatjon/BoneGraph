#!/usr/bin/env bash
# BoneGraph API launcher — sets up the conda/aarch64 shared-lib workarounds
# then starts uvicorn. Used by the systemd service.
set -euo pipefail
cd "$(dirname "$0")"

# 1) miniforge libstdc++ provides CXXABI_1.3.15 that conda's libicu/nltk need
export LD_LIBRARY_PATH="/home/jon/miniforge3/lib:${LD_LIBRARY_PATH:-}"

# 2) preload scikit-learn's vendored libgomp to avoid the aarch64
#    "cannot allocate memory in static TLS block" error (resolved at startup)
GOMP="$(ls .venv/lib/python3.13/site-packages/scikit_learn.libs/libgomp-*.so.* 2>/dev/null | head -1)"
[ -n "$GOMP" ] && export LD_PRELOAD="$GOMP:${LD_PRELOAD:-}"

exec .venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 "$@"
