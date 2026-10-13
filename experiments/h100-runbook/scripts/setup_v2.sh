#!/usr/bin/env bash
# Environment for the V2-runner studies (experiments/h100/studies, matrix.sh with MATRIX_V2=1): a separate venv at
# $WORK/v2n/venv with the vLLM nightly 43b4aaea (V2 model runner, full CUDA graphs per draft length), this package
# and both baseline controllers (editable), and AIPerf linked from the V1 setup. Run 01_setup.sh first (it provides
# uv in $WORK/venv and AIPerf in $WORK/.local/bin). Idempotent.
set -euo pipefail
WORK=${WORK:-/work}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
VC=43b4aaea3e40e20ef53ce2bfecfa1b72060bf2b5
export HOME=$WORK/v2n
mkdir -p "$HOME/.local/bin" "$WORK/out"
UV=${UV:-$WORK/venv/bin/uv}
LOG=$WORK/out/v2n-install.log
if [[ ! -x $HOME/venv/bin/vllm ]]; then
  "$UV" venv --python 3.12 "$HOME/venv" > "$LOG" 2>&1
  VIRTUAL_ENV=$HOME/venv "$UV" pip install --prerelease=allow --torch-backend=auto vllm \
    --extra-index-url "https://wheels.vllm.ai/$VC" >> "$LOG" 2>&1
fi
VIRTUAL_ENV=$HOME/venv "$UV" pip install -e "$REPO" -e "$REPO/baselines/turbospec_style" \
  -e "$REPO/baselines/nightjar_style" matplotlib pytest pyarrow >> "$LOG" 2>&1
ln -sf "$WORK/.local/bin/aiperf" "$HOME/.local/bin/aiperf"
"$HOME/venv/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
