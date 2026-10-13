# Tuner v2, recalibrated on code (T4)

The v2 table rebuilt for the code workload (`likaixin/InstructCoder`) by measuring only acceptance, made on 2026-09-25. It shows that a new workload needs about 2 minutes of GPU time, not a new sweep.

| File | What |
|---|---|
| `prompts.jsonl` | 160 code prompts sampled from the dataset (seed 1234) |
| `acceptance.json`, `acceptance.log` | Acceptance per position on those prompts (greedy) |
| `schedule.json`, `schedule_raw.json`, `goodput.csv` | The table `[[1,2,3],[3,5,2],[6,33,3],[34,64,2]]`, built from [`../tuner-v2/`](../tuner-v2/README.md)'s step costs; one boundary differs from v2 (32 → 33) |

- **Made by:** the three commands under "How to recalibrate for a new workload" in the [sweeps README](../sweeps/README.md#what-recalibration-is-and-why-its-cheaper) (`sample_prompts.py`, `profile_acceptance.py --prompts-file`, `make_schedule.py`).
- **Write-up:** [sweeps README, "Second workload"](../sweeps/README.md#second-workload-code-instruct-coder). The `code-t4` sweep ran v2's table, not this one.
- **Used by:** the tests in `tests/`, which rebuild this table.
