# Tuner v3: code at temperature 1.0, up to 128 requests (T4)

The tuner's output for the short-prompt code workload on the Tesla T4 (Qwen3-4B FP16 + EAGLE-3), made on 2026-09-26. **This is the source of the default T4 table,** [`tables/t4-eagle3`](../../../tables/t4-eagle3/) (same `schedule.json` and `acceptance.json`).

| File | What |
|---|---|
| `costs_k0.json` … `costs_k3.json` | Step costs per batch size (1–128) for draft length 0–3, with 32-token prompts |
| `acceptance.json` | Acceptance per position on code at T=1.0, a copy of [`../screen/code_t1.0.json`](../screen/code_t1.0.json) |
| `schedule.json`, `schedule_raw.json`, `goodput.csv`, `schedule.log` | The table `[[1,18,3],[19,43,2],[44,128,1]]`, before and after the tie and minimum-width rules, and the predicted goodput |

- **Made by:** [`experiments/scripts/run_v3_code.sh`](../../scripts/run_v3_code.sh), which also runs the proof sweep `code128t-t4`.
- **Write-up:** [sweeps README, "Third setup"](../sweeps/README.md#third-setup-code-at-temperature-10-up-to-128-requests-tuner-v3); 3-run means in [`../repeats/code128t-t4.md`](../repeats/code128t-t4.md).
- **Used by:** the `dsd-tuned-v3`, `ngram-v3table`, `draft-v3table` and `dflash-v3table` configs in [`experiments/bench/configs.sh`](../../bench/configs.sh); the controller runs (its step costs, and in some runs its acceptance as the prior); the tests in `tests/`, which rebuild this table and check it matches.
