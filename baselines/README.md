# Baselines: TurboSpec-style and Nightjar-style draft-length control

Two prior adaptive speculation systems, reimplemented as vLLM plugins so they can be compared with TableTune on
the same engine (vLLM nightly `43b4aaea`, V2 model runner), GPU (H100) and workloads. Neither paper released code
(TurboSpec: arXiv 2406.14066; Nightjar: arXiv 2512.22420), so each package follows its paper's method section.
They are separate systems: neither imports TableTune, and each has its own folder, package, plugin entry point and
environment switch. Both are no-ops unless their switch is set, so all three plugins can be installed together.

| | `turbospec_style/` | `nightjar_style/` |
|---|---|---|
| Decides | the draft length for every step, for the whole batch | the draft length per exact batch size, held for a bin of steps |
| Step cost | linear model fitted offline to profiled step times (`fit.py`): target pass over B(k+1) tokens plus drafting | none: measured online as latency per token |
| Acceptance | one per-token rate, moving average over steps, geometric tokens per step | implicit in the measured latency per token |
| Rule | argmax over allowed k of B·(1−α^(k+1))/(1−α) / T(B,k) | ADA-BINGREEDY: explore with probability 1/b per bin, else the lowest mean loss |
| Switch | `VLLM_TS_CONTROLLER=1`, `VLLM_TS_MODEL`, `VLLM_TS_PRIOR`, `VLLM_TS_EMA` (0.05) | `VLLM_NJ_CONTROLLER=1`, `VLLM_NJ_SWITCH_COST_MS` (0), `VLLM_NJ_SEED` |

Both start from a Dynamic SD table whose lengths are the ones they may choose. Under the V2 runner only those lengths
have full CUDA graphs, the same constraint TableTune's controller has.

Differences from the papers, all forced by running unmodified vLLM:
- TurboSpec: the context-token term of the step-time model is dropped (the scheduler's lookup sees only the batch
  size, and the profiles use one prompt length); the model is fitted to whole profiled steps rather than to separate
  target and draft passes. The per-step rate α is the one whose geometric formula matches the observed tokens per
  draft at the drafted length, and it starts from the tuner's acceptance profile (the prior TableTune gets).
- Nightjar: no drafter offloading to CPU and no re-prefill cost table (engine changes). vLLM keeps running
  EAGLE-style drafters at length 0, so resuming speculation re-prefills nothing and the switching cost is 0.

Install into the serving venv: `uv pip install --no-deps -e turbospec_style -e nightjar_style`.
Tests: `cd turbospec_style && python -m pytest -q tests` (likewise `nightjar_style`).
