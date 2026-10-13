# Tuner v2: spec-bench (T4)

The second tuner run, on the Tesla T4 (Qwen3-4B FP16 + EAGLE-3, spec-bench, greedy, up to 64 requests), made on 2026-09-25 after v1's rules were fixed.

| File | What |
|---|---|
| `costs_k0.json` … `costs_k3.json` | Step costs per batch size (1–16, then every 4 up to 64) for draft length 0–3, 320-token prompts |
| `acceptance.json` | Acceptance per position on the 160 spec-bench prompts the baseline sweep did not send |
| `schedule.json`, `schedule_raw.json`, `goodput.csv` | The table `[[1,2,3],[3,5,2],[6,32,3],[33,64,2]]`, before and after the tie and minimum-width rules, and the predicted goodput |

- **Made by:** `bash experiments/scripts/run_tuner.sh experiments/results/sweeps/baseline-t4 experiments/results/tuner-v2` ([script](../../scripts/run_tuner.sh)).
- **Write-up:** [sweeps README, "v1 → v2"](../sweeps/README.md#v1--v2-what-changed-and-why).
- **Used by:** the `dsd-tuned-v2` config in [`experiments/bench/configs.sh`](../../bench/configs.sh) (sweeps `dsd-t4`, `code-t4`); `costs_k*.json` for the code recalibration in [`../tuner-v2-code/`](../tuner-v2-code/README.md); `acceptance.json` as the controller's prior in several controller runs (e.g. [`experiments/bench/run_controller_value.sh`](../../bench/run_controller_value.sh), [`run_repeats.sh`](../../scripts/run_repeats.sh), [`run_gpu_eagle.sh`](../../scripts/run_gpu_eagle.sh)); the tests in `tests/`.
