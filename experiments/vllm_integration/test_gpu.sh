#!/usr/bin/env bash
# GPU integration test for the vLLM plugin (run on the instance, from ~/tabletune,
# with vLLM at the pinned commit and this package installed in ~/venv).
#
#   1. The plugin is registered, and vLLM's own Dynamic SD and scheduler
#      speculative-decoding tests pass with it installed.
#   2. Correctness: greedy outputs with the controller on (hysteresis + frequent
#      re-fits) must match no speculation token for token.
#   3. Serving: the v3 table with the controller on, on the code workload at
#      temperature 1.0 (c = 4, 32, 64, 96); compare with the run without it and
#      check the controller's log lines.
#   4. N-gram (issue #3): greedy outputs with the controller on must match no
#      speculation, and the controller must track the proposal rate (it must
#      not for EAGLE, checked in step 2).
#
# Step 0 runs the repo's tests, including the EAGLE regression tests.
#
# By default it overwrites committed results: experiments/results/integration/*.log,
# experiments/results/smoke/{dsd,ngram-dsd}_ctrl.json and the sweep ctrl-t4. With
# TEST_OUT set, the logs and smoke results go to $TEST_OUT and the sweep to the new
# folder experiments/results/sweeps/ctrl-t4-test (CTRL_SWEEP=<name> to pick another).
#
# Usage: bash experiments/vllm_integration/test_gpu.sh
#        TEST_OUT=~/dsd-test bash experiments/vllm_integration/test_gpu.sh   (keep the committed results)
set -uo pipefail
[[ -n "${TEST_OUT:-}" ]] && TEST_OUT=$(realpath -m "$TEST_OUT")
cd "$(dirname "$0")/../.."  # repo root
source ~/venv/bin/activate
OUT=${TEST_OUT:-experiments/results/integration}
export SMOKE_OUT=${TEST_OUT:+$TEST_OUT/smoke}
SMOKE_OUT=${SMOKE_OUT:-experiments/results/smoke}   # smoke_test.py writes here
mkdir -p "$OUT"
fail=0

# Pre-flight: the GPU must be free, or results would come from a leftover server.
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "!!! GPU is busy: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"
  exit 1
fi

CTRL_SWEEP=${CTRL_SWEEP:-ctrl-t4${TEST_OUT:+-test}}

echo "=== 0. repo tests ($(date +%H:%M:%S)) ==="
python -m pytest -q -p no:cacheprovider tests 2>&1 | tail -1 | tee "$OUT/repo_tests.log"
grep -qE "passed" "$OUT/repo_tests.log" && ! grep -qE "failed|error" "$OUT/repo_tests.log" \
  || { echo "!!! repo tests failed"; fail=1; }

echo "=== 1. plugin registered; vLLM's own tests ($(date +%H:%M:%S)) ==="
python - <<'EOF' || fail=1
from importlib.metadata import entry_points
names = [e.value for e in entry_points(group="vllm.general_plugins")]
print("vllm.general_plugins:", names)
assert "dsd_tuning.controller.vllm_plugin:register" in names, "the controller plugin (dsd_tuning.controller) is not installed in this environment"
EOF
~/.local/bin/uv pip install -q pytest pytest-asyncio tblib > /dev/null 2>&1  # vLLM test deps
~/.local/bin/uv pip install -q -e ".[tracking]" > /dev/null 2>&1               # plot_pareto.py (step 3)
( cd ~/vllm && python -m pytest -q -p no:cacheprovider \
    tests/v1/spec_decode/test_dynamic_sd.py \
    tests/v1/core/test_scheduler.py -k "dynamic or spec" ) \
  > "$OUT/unit.log" 2>&1
tail -3 "$OUT/unit.log"
grep -qE "[0-9]+ passed" "$OUT/unit.log" && ! grep -qE "[0-9]+ (failed|errors?)\b" "$OUT/unit.log" \
  || { echo "!!! unit tests failed, see $OUT/unit.log"; fail=1; }

echo "=== 2. correctness: controller on, greedy vs no-spec ($(date +%H:%M:%S)) ==="
export VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=2 VLLM_DSD_REFIT_STEPS=20 \
       VLLM_DSD_LOG_STEPS=50 VLLM_DSD_COSTS="experiments/results/tuner-v3-code/costs_k*.json" \
       VLLM_DSD_PRIOR=experiments/results/tuner-v3-code/acceptance.json
python experiments/bench/smoke_test.py --mode dsd --tag _ctrl > "$OUT/smoke.log" 2>&1
grep -E "\[smoke\]" "$OUT/smoke.log" | head -2
grep -oE "Dynamic SD controller[^\"]{0,160}" "$OUT/smoke.log" | sort | uniq -c | head -5
python - <<'EOF' || fail=1
import json, os, sys
ref = json.load(open("experiments/results/smoke/nospec.json"))
got = json.load(open(os.path.join(os.environ["SMOKE_OUT"], "dsd_ctrl.json")))
same = sum(a == b for a, b in zip(ref["token_ids"], got["token_ids"]))
same1 = sum(a == b for a, b in zip(ref["single_token_ids"], got["single_token_ids"]))
n, n1 = len(ref["token_ids"]), len(ref["single_token_ids"])
print(f"greedy match vs no-spec: batched {same}/{n}, single {same1}/{n1}")
sys.exit(0 if (same, same1) == (n, n1) else 1)
EOF
grep -q "Dynamic SD controller enabled" "$OUT/smoke.log" \
  || { echo "!!! controller was not enabled in the engine"; fail=1; }
if grep -q "tracking proposal rate" "$OUT/smoke.log"; then
  echo "!!! EAGLE must not track the proposal rate"; fail=1
else
  echo "EAGLE: proposal rate not tracked (as intended)"
fi

echo "=== 3. serving: v3 table + controller, code T=1.0 ($(date +%H:%M:%S)) ==="
export VLLM_DSD_HYSTERESIS=4 VLLM_DSD_REFIT_STEPS=500 VLLM_DSD_LOG_STEPS=2000
MAX_SEQS=128 TEMPERATURE=1.0 bash experiments/bench/run_configs.sh "$CTRL_SWEEP" "dsd-tuned-v3" \
  "4 32 64 96" instruct-coder
unset VLLM_DSD_CONTROLLER
python experiments/bench/plot_pareto.py "$CTRL_SWEEP" > /dev/null 2>&1
CTRL_SWEEP=$CTRL_SWEEP python - <<'EOF'
import csv, os
new = {r["concurrency"]: float(r["tok_s_gpu"]) for r in csv.DictReader(open(f"experiments/results/sweeps/{os.environ['CTRL_SWEEP']}/summary.csv"))}
old = {r["concurrency"]: float(r["tok_s_gpu"]) for r in csv.DictReader(open("experiments/results/sweeps/code128t-t4/summary.csv")) if r["config"] == "dsd-tuned-v3"}
for c in sorted(new, key=int):
    print(f"c={c:>3}: v3 table {old[c]:7.1f}  | v3 + controller {new[c]:7.1f}  ({new[c]/old[c]-1:+.1%})")
EOF
grep -hoE "Dynamic SD controller[^\"]{0,200}" "experiments/results/sweeps/$CTRL_SWEEP/dsd-tuned-v3/server.log" | tail -4

echo "=== 4. n-gram + controller: greedy vs no-spec, proposal rate ($(date +%H:%M:%S)) ==="
export VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=2 VLLM_DSD_REFIT_STEPS=20 \
       VLLM_DSD_LOG_STEPS=50 VLLM_DSD_COSTS="experiments/results/tuner-v3-code/costs_k*.json" \
       VLLM_DSD_PRIOR=experiments/results/ngram/accept-ngram-edit-t1.json
python experiments/bench/smoke_test.py --mode ngram-dsd --tag _ctrl > "$OUT/smoke_ngram.log" 2>&1
unset VLLM_DSD_CONTROLLER
grep -oE "Dynamic SD controller[^\"]{0,200}" "$OUT/smoke_ngram.log" | grep -v "'steps'" | head -4
python - <<'EOF' || fail=1
import json, os, sys
ref = json.load(open("experiments/results/smoke/nospec.json"))
got = json.load(open(os.path.join(os.environ["SMOKE_OUT"], "ngram-dsd_ctrl.json")))
same = sum(a == b for a, b in zip(ref["token_ids"], got["token_ids"]))
same1 = sum(a == b for a, b in zip(ref["single_token_ids"], got["single_token_ids"]))
n, n1 = len(ref["token_ids"]), len(ref["single_token_ids"])
print(f"n-gram greedy match vs no-spec: batched {same}/{n}, single {same1}/{n1}")
sys.exit(0 if (same, same1) == (n, n1) else 1)
EOF
grep -q "tracking proposal rate" "$OUT/smoke_ngram.log" \
  || { echo "!!! n-gram controller is not tracking the proposal rate"; fail=1; }

echo "=== done ($(date +%H:%M:%S)), failures: $fail ==="
exit $fail
