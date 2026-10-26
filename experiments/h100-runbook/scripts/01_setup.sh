#!/usr/bin/env bash
# One-time setup inside the pod (about 10 min plus model downloads).
# Before running: copy the repo tarball into the pod (see README, step 3).
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd /work
[[ -d $REPO ]] || { mkdir -p $REPO && tar xzf /work/tabletune.tgz -C $REPO --strip-components=1; }

# 1. Python 3.12, vLLM at the tested commit (353e491ef) and the package, via the repo's installer
cd $REPO && VENV=/work/venv VLLM_SRC=/work/vllm bash install.sh
source /work/venv/bin/activate
dsd-tuner --version
python -c "import importlib.metadata as m; assert 'dsd_controller' in [e.name for e in m.entry_points(group='vllm.general_plugins')]; print('plugin registered')"
python -m pytest -q -p no:cacheprovider tests | tail -1

# 2. The benchmark client used for every sweep
uv tool install aiperf==0.13.0
~/.local/bin/aiperf --version || true

# 3. Prompt files for the acceptance profiles and the shift experiment
mkdir -p $X
python /work/runbook/scripts/make_prompts.py --repo $REPO --out $X

# 4. Download all models now so later steps don't wait (about 60 GB)
python - <<'EOF'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen3-4B", "AngelSlim/Qwen3-4B_eagle3", "Qwen/Qwen3-4B-AWQ",
          "Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3", "Qwen/Qwen3-8B-AWQ"]:
    print("downloading", m); snapshot_download(m)
EOF

# 5. Smoke test: serve the 4B model with a 3-length table and the controller for one minute
bash /work/runbook/scripts/serve.sh /work/out/smoke.server.log table "[[1,64,3],[65,256,1]]" "$REPO/tables/a10g-eagle3/acceptance.json" "$REPO/tables/a10g-eagle3/costs_k*.json"
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B","prompt":"def fib(n):","max_tokens":32}' | head -c 300; echo
grep -m1 "Dynamic SD controller enabled" /work/out/smoke.server.log
pkill -f "vllm serve" || true
echo "SETUP DONE"
