# dsd_tuning: the Dynamic SD tuner and controller

The Python package behind `dsd-tuner` (install name `dsd-tuning`, import name `dsd_tuning`). Two parts, plus what they share:

| | What | Details |
|---|---|---|
| [`tuner/`](tuner/README.md) | **Offline:** measures step time and acceptance on a GPU and builds the table | profilers, `make_schedule.py`, `validate_costmodel.py` |
| [`controller/`](controller/README.md) | **In vLLM:** holds K near table edges and re-fits the table from live acceptance | `controller.py`, the vLLM plugin |
| `core.py` | The formula and table rules both parts use | *goodput = batch size × tokens per step ÷ step time* |
| `tables.py` | The [table catalog](../tables/): list, check, serve, add | `dsd-tuner tables / serve / add-table` |
| `__main__.py` | The `dsd-tuner` command | `dsd-tuner --help` |

Tests (no GPU) mirror this layout: [`tests/tuner/`](../tests/tuner), [`tests/controller/`](../tests/controller), and the shared parts in [`tests/`](../tests). Run `pytest -q` from the repository root.
