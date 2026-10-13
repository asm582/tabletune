#!/usr/bin/env bash
# Concurrency sweep against a running server with aiperf.
# Usage: bash experiments/bench/sweep.sh <out_dir> [concurrencies] [dataset]
#   dataset: an aiperf public dataset name, or file:<path> for a JSONL file of
#   {"text": ...} prompts (aiperf single_turn; e.g. from dsd_tuning/tuner/sample_prompts.py --aiperf)
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root
source experiments/bench/configs.sh

OUT=$1
CONCURRENCIES=${2:-"1 2 4 8 16 32 64"}
DATASET=${3:-spec-bench}
PORT=${PORT:-8000}
OSL=${OSL:-256}

# Greedy by default (TEMPERATURE=0), thinking off, responses capped at OSL
# tokens but allowed to stop naturally (forcing a fixed length would distort
# acceptance after EOS).
TEMPERATURE=${TEMPERATURE:-0}
EXTRA="{\"temperature\": $TEMPERATURE, \"chat_template_kwargs\": {\"enable_thinking\": false}}"

if [[ $DATASET == file:* ]]; then
  DATA_ARGS=(--custom-dataset-type single-turn --input-file "${DATASET#file:}")
else
  DATA_ARGS=(--public-dataset "$DATASET")
fi

for c in $CONCURRENCIES; do
  # Enough requests for several full "waves" at each concurrency.
  n=$(( c * 4 )); (( n < 16 )) && n=16
  echo "[sweep] c=$c requests=$n -> $OUT/c$c"
  ~/.local/bin/aiperf profile \
    --model "$TARGET_MODEL" \
    --url "localhost:$PORT" \
    --endpoint-type chat --streaming \
    "${DATA_ARGS[@]}" \
    --dataset-sampling-strategy shuffle \
    --random-seed 42 \
    --osl "$OSL" \
    --extra-inputs "$EXTRA" \
    --use-server-token-count --per-chunk-usage \
    --concurrency "$c" \
    --request-count "$n" \
    --warmup-request-count "$c" \
    --artifact-dir "$OUT/c$c" \
    --no-gpu-telemetry \
    --ui-type none \
    > "$OUT/c$c.log" 2>&1 \
    || { echo "[sweep] c=$c FAILED, see $OUT/c$c.log" >&2; exit 1; }
done
echo "[sweep] done: $OUT"
