#!/usr/bin/env bash
# Download the larger targets and their EAGLE-3 drafters into the shared model cache ($WORK/hf).
WORK=${WORK:-/work}
export HF_HOME=$WORK/hf
mkdir -p "$WORK/out"; L=$WORK/out/dl.log; : > "$L"
for m in AngelSlim/Qwen3-a3B_eagle3 AngelSlim/Qwen3-32B_eagle3 Qwen/Qwen3-30B-A3B Qwen/Qwen3-32B; do
  if "$WORK/v2n/venv/bin/hf" download "$m" --exclude "*.pth" --exclude "original/*" > "$WORK/out/dl-${m//\//_}.log" 2>&1; then
    echo "OK $m" >> "$L"; else echo "FAIL $m" >> "$L"; fi
done
echo ALLDONE >> "$L"
