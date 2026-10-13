#!/usr/bin/env bash
# Quantization as a deployment change:
# Qwen3-4B-AWQ (4-bit weights) on the T4 with the unchanged EAGLE-3 drafter.
#
#   smoke     AWQ target: no spec, EAGLE-3 fixed K, Dynamic SD table; acceptance
#             (gate: position-1 acceptance > 0.3, else stop). Greedy outputs are
#             compared but not gated: AWQ kernels are not bit-exact across the
#             number of tokens per matmul (no spec itself differs batched vs. single)
#   quality   optional, not in the default run (issue #11 focuses on the pipeline):
#             GSM8K (500 problems, greedy): FP16, AWQ, and AWQ + EAGLE-3
#   pipeline  experiments/scripts/run_gpu_eagle.sh with GPU=t4-awq: acceptance, costs K=0..3,
#             table, sweep (no spec, K=1/2/3, AWQ table, FP16 v3 table), controller
#
# Usage (on the instance, repo root): bash experiments/scripts/run_quant.sh [stage ...]
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
while pgrep -f "run_repeats_specbench.sh|run_repeats.sh|run_draft.sh|run_dflash.sh" > /dev/null; do sleep 60; done
source ~/venv/bin/activate
AWQ=Qwen/Qwen3-4B-AWQ
STAGES=${*:-smoke pipeline}
has() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo "=== $* ($(date +%H:%M:%S)) ==="; }

if has smoke; then
  export SMOKE_OUT=experiments/results/smoke/t4-awq TARGET_MODEL=$AWQ
  for m in nospec eagle3 dsd; do
    log "smoke $m (AWQ)"
    python experiments/bench/smoke_test.py --mode $m > "$SMOKE_OUT.$m.log" 2>&1 || echo "!!! smoke $m failed"
    grep -ahE "\[smoke\] ($m|spec)" "$SMOKE_OUT.$m.log" | cut -c1-230
    grep -ahoE "GPU KV cache size: [0-9,]+ tokens|Model loading took [0-9.]+ GiB|Traceback" "$SMOKE_OUT.$m.log" | sort -u
  done
  python experiments/bench/smoke_test.py --compare | tee "$SMOKE_OUT/compare.txt"
  python - <<'PY' || { echo "!!! AWQ smoke gate failed: stopping"; exit 1; }
import json, sys
r = json.load(open("experiments/results/smoke/t4-awq/eagle3.json"))["spec_counters"]
a1 = r["vllm:spec_decode_num_accepted_tokens_per_pos"][0] / r["vllm:spec_decode_num_drafts"]
ok = open("experiments/results/smoke/t4-awq/compare.txt").read().count("batched 8/8, single 3/3") == 2
print(f"[gate] EAGLE-3 position-1 acceptance on AWQ: {a1:.2f}; greedy match: {ok} (not gated)")
sys.exit(0 if a1 > 0.3 else 1)
PY
  unset SMOKE_OUT TARGET_MODEL
fi

if has quality; then
  [ -f experiments/results/quality/gsm8k_test.jsonl ] || ~/.local/share/uv/tools/aiperf/bin/python experiments/bench/quality_gsm8k.py \
      --export experiments/results/quality/gsm8k_test.jsonl
  for spec in "Qwen/Qwen3-4B fp16" "$AWQ awq" "$AWQ awq_eagle3 --spec eagle3"; do
    set -- $spec
    log "quality GSM8K: $1 $3 $4"
    TARGET_MODEL=$1 python experiments/bench/quality_gsm8k.py --data experiments/results/quality/gsm8k_test.jsonl --n 500 \
      --out experiments/results/quality/gsm8k_$2.json ${3:-} ${4:-} 2>&1 | grep -a "^\[gsm8k\]\|Traceback"
  done
fi

if has pipeline; then
  log "EAGLE-3 pipeline on the AWQ model"
  GPU=t4-awq DTYPE=float16 TARGET_MODEL=$AWQ bash experiments/scripts/run_gpu_eagle.sh
fi
log "quantization run done"
