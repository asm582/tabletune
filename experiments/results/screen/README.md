# Screening: acceptance on code with sampling (T4)

> Paths and commands here use the package's current name, `dsd_tuning/`; when these runs were made it was `system/` (renamed in 0.0.1).

EAGLE-3 acceptance per position on 160 code prompts at temperature 0.8 and 1.0 (Qwen3-4B FP16 on the Tesla T4), measured on 2026-09-25 to check whether sampling lowers acceptance enough to matter.

| File | Temperature | Tokens per step at draft length 3 |
|---|---|---|
| `code_t0.8.json` | 0.8 | 2.13 |
| `code_t1.0.json` | 1.0 | 2.10 |

Greedy, for comparison, was 2.16 ([`../tuner-v2-code/acceptance.json`](../tuner-v2-code/acceptance.json)). Sampling barely lowers acceptance for this model and drafter.

- **Made by:** [`dsd_tuning/tuner/profile_acceptance.py`](../../../dsd_tuning/tuner/profile_acceptance.py) with `--temperature` (added in the same commit, 84508fb). The exact command and which 160 prompts were used are not recorded (unknown).
- **Used by:** `code_t1.0.json` is copied as `acceptance.json` into [`../tuner-v3-code/`](../tuner-v3-code/README.md) by [`experiments/scripts/run_v3_code.sh`](../../scripts/run_v3_code.sh), so it is also the acceptance of the default T4 table.
