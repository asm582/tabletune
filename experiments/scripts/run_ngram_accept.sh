#!/usr/bin/env bash
# Issue #3, step 2: does n-gram get high acceptance when the output copies the
# prompt (code editing) and low acceptance when it doesn't (code writing)?
# Compares with EAGLE-3 on the same prompts. Acceptance only; no step costs.
# Run on the GPU instance:  bash experiments/scripts/run_ngram_accept.sh
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root
OUT=experiments/results/ngram
mkdir -p "$OUT"
AIPERF_PY=~/.local/share/uv/tools/aiperf/bin/python
N=160
MAXTOK=512

# Code editing: instruction + the code to edit. Code writing: instruction only
# (the shape of aiperf's instruct-coder workload used in the sweeps).
[ -f "$OUT/prompts-edit.jsonl" ] || $AIPERF_PY dsd_tuning/tuner/sample_prompts.py --dataset likaixin/InstructCoder \
    --column instruction --append-column input --n $N --out "$OUT/prompts-edit.jsonl"
[ -f "$OUT/prompts-write.jsonl" ] || $AIPERF_PY dsd_tuning/tuner/sample_prompts.py --dataset likaixin/InstructCoder \
    --column instruction --n $N --out "$OUT/prompts-write.jsonl"

run() {  # name method k prompts temperature
  local name=$1; shift
  echo "=== $name"
  DSD_METHOD=$1 DSD_K_MAX=$2 python dsd_tuning/tuner/profile_acceptance.py --prompts-file "$OUT/$3" \
      --temperature "$4" --max-tokens $MAXTOK --out "$OUT/accept-$name.json" 2>&1 \
    | grep -a "^\[accept\]" | tee -a "$OUT/accept.log"
}
: > "$OUT/accept.log"
run ngram-edit-t0   ngram  5 prompts-edit.jsonl  0.0
run ngram-edit-t1   ngram  5 prompts-edit.jsonl  1.0
run ngram-write-t1  ngram  5 prompts-write.jsonl 1.0
run eagle3-edit-t0  eagle3 3 prompts-edit.jsonl  0.0
run eagle3-edit-t1  eagle3 3 prompts-edit.jsonl  1.0
