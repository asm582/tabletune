# Repeats: means of 3 runs

Mean and spread (min..max) of each sweep over its 3 runs: `<sweep>` (run 1), `<sweep>-r2` and `<sweep>-r3`. One pair of files per sweep: `<sweep>.csv` (every metric) and `<sweep>.md` (output tok/s per GPU, and the spread as a % of the mean).

- **Made by:** `python experiments/bench/aggregate_repeats.py <sweep>` ([script](../../bench/aggregate_repeats.py)) from the sweeps' `summary.csv` files in [`../sweeps/`](../sweeps/). The runs 2 and 3 come from [`run_repeats.sh`](../../scripts/run_repeats.sh), [`run_repeats_specbench.sh`](../../scripts/run_repeats_specbench.sh) and the `SUFFIX=-r2`/`-r3` runs of the study scripts.
- **Used by:** the top-level README's numbers (`code128t-{t4,a10g}.md`, `ctrlval-{off,refit}{,-a10g}.md`) and the results READMEs that quote 3-run means.
