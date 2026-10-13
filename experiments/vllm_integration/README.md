# GPU integration test for the controller plugin

Checks the plugin against a real vLLM on a GPU. How the plugin attaches and its settings: [dsd_tuning/controller/README.md](../../dsd_tuning/controller/README.md).

**Setup:** [`install.sh`](../../install.sh) (vLLM at the pinned commit `353e491ef` and this package in `~/venv`), then from the repository root:

```bash
TEST_OUT=~/dsd-test bash experiments/vllm_integration/test_gpu.sh
```

It takes ~16 min on a T4. **Without `TEST_OUT` it overwrites 8 committed result files** (that is how they are refreshed), and adds `dsd-tuned-v3/` and `pareto.png` to the `ctrl-t4` sweep:

- `experiments/results/integration/`: `repo_tests.log`, `unit.log`, `smoke.log`, `smoke_ngram.log`
- `experiments/results/smoke/`: `dsd_ctrl.json`, `ngram-dsd_ctrl.json`
- `experiments/results/sweeps/ctrl-t4/`: `meta.txt`, `summary.csv` (the sweep behind section 3 of the [results](../results/integration/README.md))

With `TEST_OUT`, the logs and smoke results go to that folder, and the serving sweep to a new folder, `experiments/results/sweeps/ctrl-t4-test` (delete it when done; `CTRL_SWEEP=<name>` picks another name). Ran it without `TEST_OUT`? Restore the files and remove the additions:

```bash
git checkout -- experiments/results/integration experiments/results/smoke experiments/results/sweeps/ctrl-t4
rm -r experiments/results/sweeps/ctrl-t4/dsd-tuned-v3 experiments/results/sweeps/ctrl-t4/pareto.png
```

What it checks:

0. **Repo tests**, including the EAGLE regression tests.
1. **Plugin and vLLM tests:** the plugin is registered in the environment, and vLLM's own Dynamic SD and scheduler speculative-decoding tests pass with this package installed.
2. **Correctness:** greedy outputs with the controller on (hysteresis 2, re-fit every 20 steps) must match no speculation token for token (`experiments/bench/smoke_test.py`).
3. **Serving:** the v3 table with the controller on (hysteresis 4, re-fit every 500 steps) on the code workload at temperature 1.0, c = 4, 32, 64, 96, compared with the same table without the controller; the controller's log lines are printed.
4. **N-gram:** greedy outputs with the controller on must match no speculation, and the controller must track the proposal rate.

Results: [`experiments/results/integration/README.md`](../results/integration/README.md).
