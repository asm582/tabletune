#!/usr/bin/env bash
# AIPerf concurrency sweep through the llm-d router (the guide's standalone Envoy+EPP service), plus per-pod
# vLLM metric snapshots around each level (speculative-decoding counters, for draft length and acceptance).
#   llmd_sweep.sh <out_dir> "<concurrencies>" "<pod ips>"
set -uo pipefail
OUT=$1; CONCS=$2; IPS=$3
URL=${URL:-optimized-baseline-epp.vllm-dsd.svc.cluster.local:80}
mkdir -p $OUT
snap() { for ip in $IPS; do curl -s -m 10 http://$ip:8000/metrics | grep -E '^vllm:(spec_decode_[a-z_]*_total|num_requests_running|generation_tokens_total|num_preemptions_total)' > $OUT/c$1.$2.$ip.prom; done; }
for c in $CONCS; do
  n=$(( c * 4 )); (( n < 32 )) && n=32
  snap $c pre
  "${AIPERF:-${WORK:-/work}/.local/bin/aiperf}" profile --model Qwen/Qwen3-32B --url "$URL" --endpoint-type chat --streaming \
    --public-dataset instruct-coder --dataset-sampling-strategy shuffle --random-seed ${SEED:-42} --osl 256 \
    --extra-inputs '{"temperature": 1.0, "chat_template_kwargs": {"enable_thinking": false}}' \
    --use-server-token-count --per-chunk-usage --concurrency $c --request-count $n --warmup-request-count $c \
    --artifact-dir $OUT/c$c --no-gpu-telemetry --no-server-metrics --ui-type none > $OUT/c$c.log 2>&1 \
    || echo "!!! c=$c failed" | tee -a $OUT/FAILED
  snap $c post
  echo "[llmd] $(date -u +%T) c=$c done $(grep -m1 -oE 'Output Token Throughput[^│]*│[^│]*' $OUT/c$c.log | tr -s ' ')"
done
