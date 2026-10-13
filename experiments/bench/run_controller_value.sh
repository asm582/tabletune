#!/usr/bin/env bash
# Does the in-engine controller (VLLM_DSD_CONTROLLER) add value?
# Code workload, temperature 1.0, max_num_seqs 128, vLLM with the controller plugin.
#
#   A  docs example table, controller off        (a copied table)
#   B  docs example table, re-fit + hysteresis   (fixes a copied table live?)
#   C  hand table with K=0, controller off       (boundaries inside the range)
#   D  hand table, hysteresis only               (hysteresis alone)
#   E  hand table, re-fit + hysteresis           (re-fit rescues a bad table?)
#
# Re-fit uses the tuner's short-prompt step costs (tuner-v3-code) and, as its
# prior, the spec-bench acceptance (tuner-v2): a different workload, so the
# controller has to learn code's acceptance from live traffic.
#
# Usage (on the instance, from ~/tabletune): bash experiments/bench/run_controller_value.sh
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
C="4 16 32 40 64 96"
export MAX_SEQS=128 TEMPERATURE=1.0
REFIT=(VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=8 VLLM_DSD_REFIT_STEPS=200
       VLLM_DSD_LOG_STEPS=1000 "VLLM_DSD_COSTS=experiments/results/tuner-v3-code/costs_k*.json"
       VLLM_DSD_PRIOR=experiments/results/tuner-v2/acceptance.json)
HYST=(VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=8 VLLM_DSD_LOG_STEPS=1000)

bash experiments/bench/run_configs.sh ctrlval-off "dsd-docs dsd-rescaled" "$C" instruct-coder   # A, C
env "${REFIT[@]}" bash experiments/bench/run_configs.sh ctrlval-refit "dsd-docs dsd-rescaled" "$C" instruct-coder  # B, E
env "${HYST[@]}" bash experiments/bench/run_configs.sh ctrlval-hyst "dsd-rescaled" "$C" instruct-coder   # D
echo "=== controller value runs done ($(date +%H:%M:%S)) ==="
