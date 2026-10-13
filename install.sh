#!/usr/bin/env bash
# Install everything needed to serve Qwen3-4B with the DSD tuner's table and the
# in-engine controller: uv, Python 3.12 in ~/venv, vLLM at the tested commit, and
# this package (which plugs the controller into vLLM). Run it from the repository:
#
#   bash install.sh
#
# Safe to re-run: every step checks first and is skipped when already in place at
# the right version. Override locations with VENV=... and VLLM_SRC=....
set -euo pipefail
cd "$(dirname "$0")"

VLLM_COMMIT=353e491efd3a23057004a3ab68fcfbda4abbee13   # the vLLM commit all results were measured on
PYTHON_VERSION=3.12
VENV=${VENV:-$HOME/venv}
VLLM_SRC=${VLLM_SRC:-$HOME/vllm}

say() { echo "==> $*"; }
ok() { echo "    ok: $*"; }

# 1. System tools: git and curl (installed with apt-get if missing), and an NVIDIA GPU.
for tool in git curl; do
  if command -v "$tool" > /dev/null; then
    ok "$tool"
  elif command -v apt-get > /dev/null; then
    say "installing $tool"
    sudo apt-get install -y -q "$tool"
  else
    echo "!!! $tool is missing; install it and re-run" >&2
    exit 1
  fi
done
if gpu=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2> /dev/null | head -1); then
  ok "GPU: $gpu"
else
  echo "!!! no NVIDIA GPU or driver found (nvidia-smi failed); vLLM needs one" >&2
  exit 1
fi

# 2. uv (manages Python and packages).
if ! command -v uv > /dev/null && [[ -x "$HOME/.local/bin/uv" ]]; then
  export PATH="$HOME/.local/bin:$PATH"
fi
if command -v uv > /dev/null; then
  ok "uv $(uv --version | cut -d' ' -f2)"
else
  say "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 3. Python 3.12 environment.
if [[ -x "$VENV/bin/python" ]] && "$VENV/bin/python" -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))"; then
  ok "Python $PYTHON_VERSION environment at $VENV"
else
  say "creating a Python $PYTHON_VERSION environment at $VENV"
  uv venv --python "$PYTHON_VERSION" "$VENV"
fi
export VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH"
# uv inside the environment too, so `source ~/venv/bin/activate` provides it even in
# the shell that just installed uv (its own PATH change only applies to new shells).
if [[ -x "$VENV/bin/uv" ]]; then
  ok "uv in $VENV"
else
  uv pip install -q uv
  ok "uv in $VENV"
fi

# 4. vLLM at the tested commit (prebuilt kernels, installed from source for the pin).
vllm_ok() {
  [[ "$(git -C "$VLLM_SRC" rev-parse HEAD 2> /dev/null)" == "$VLLM_COMMIT" ]] \
    && python -c "import vllm, sys, pathlib; sys.exit(not str(pathlib.Path(vllm.__file__).resolve()).startswith('$(cd "$VLLM_SRC" 2> /dev/null && pwd)'))" 2> /dev/null
}
if vllm_ok; then
  ok "vLLM ${VLLM_COMMIT:0:9} (from $VLLM_SRC)"
else
  say "installing vLLM ${VLLM_COMMIT:0:9} (~2 min, mostly downloads)"
  [[ -d "$VLLM_SRC/.git" ]] || git clone -q https://github.com/vllm-project/vllm.git "$VLLM_SRC"
  git -C "$VLLM_SRC" fetch -q origin "$VLLM_COMMIT" 2> /dev/null || true
  git -C "$VLLM_SRC" checkout -q "$VLLM_COMMIT"
  VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT="$VLLM_COMMIT" \
    uv pip install -q -e "$VLLM_SRC" --torch-backend=auto
  vllm_ok || { echo "!!! vLLM install check failed" >&2; exit 1; }
fi

# 5. This package: the tuner CLI and the vLLM plugin that attaches the controller.
# Earlier names of this package (vllm-dsd-tuning, and vllm-dsd-tuner before that)
# register the same vLLM plugin under the old `system` import name; left installed,
# vLLM would try to load it next to this one and fail to import it. Remove them.
for old in vllm-dsd-tuning vllm-dsd-tuner; do
  if uv pip show -q "$old" > /dev/null 2>&1; then
    say "removing $old (this package's earlier name)"
    uv pip uninstall -q "$old"
  fi
  # Its build metadata stays in the clone after `git pull` (ignored by git); Python
  # started here would still find the old plugin entry point in it.
  egg_info="${old//-/_}.egg-info"
  if [[ -d "$egg_info" ]]; then
    say "removing $egg_info (build files of the earlier name)"
    rm -rf "$egg_info"
  fi
done
say "installing dsd-tuning"
uv pip install -q -e ".[dev]"
python - << 'EOF'
from importlib.metadata import entry_points
plugins = [e.value for e in entry_points(group="vllm.general_plugins")]
assert "dsd_tuning.controller.vllm_plugin:register" in plugins, \
    "the Dynamic SD controller plugin (dsd_tuning.controller) is not registered"
stale = [v for v in plugins if v.startswith("system.")]
assert not stale, f"an old install of this package is still registered: {stale}; run: uv pip uninstall vllm-dsd-tuning"
EOF
ok "dsd-tuner and its vLLM plugin"

echo
echo "Done. Next:"
echo "  source $VENV/bin/activate"
echo "  dsd-tuner serve"
