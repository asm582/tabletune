# Tuner v1: spec-bench (T4)

The first tuner run, on the Tesla T4 (Qwen3-4B FP16 + EAGLE-3, spec-bench, greedy, up to 64 requests), made on 2026-09-25. **Superseded by [`../tuner-v2/`](../tuner-v2/README.md);** kept because the `dsd-t4` sweep measured this table.

| File | What |
|---|---|
| `costs_k0.json` … `costs_k3.json` | Step costs per batch size for draft length 0–3, 256-token prompts, v1's coarse batch grid |
| `acceptance.json` | Acceptance per position on 64 spec-bench prompts the baseline sweep did not send |
| `schedule.json`, `schedule_raw.json`, `goodput.csv` | The table `[[1,2,3],[3,42,2],[43,64,1]]`, before and after v1's rules, and the predicted goodput |

- **Made by:** the first version of `run_tuner.sh` (commit 1526e64), with the calibration prompts taken from the `baseline-t4` sweep. Today's [`experiments/scripts/run_tuner.sh`](../../scripts/run_tuner.sh) uses v2's settings, so running it again gives v2-style results, not these.
- **Write-up:** [sweeps README, "v1 → v2"](../sweeps/README.md#v1--v2-what-changed-and-why) and the corrections log there (v1's "draft length only falls with load" rule was dropped).
- **Used by:** the `dsd-tuned` config in [`experiments/bench/configs.sh`](../../bench/configs.sh) (sweep `dsd-t4`).
