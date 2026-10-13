"""Regression tests: the EAGLE-3 results must not change when other drafting
methods (n-gram, issue #3) are added.

Tuner side: recomputes the tables committed under experiments/results/ from their stored
step costs and acceptance. Written before the n-gram changes,
against the code that produced those results.
"""

import json
from pathlib import Path

import pytest

from dsd_tuning.core import goodput_table, load_costs, make_schedule, tokens_per_step

RESULTS = Path(__file__).resolve().parents[2] / "experiments" / "results"


def _acc(name: str) -> list[float]:
    return json.loads((RESULTS / name / "acceptance.json").read_text())["accept_by_pos"]


@pytest.mark.parametrize("acceptance, costs, max_b", [
    ("tuner-v2", "tuner-v2", 64),            # spec-bench, greedy
    ("tuner-v2-code", "tuner-v2", 64),       # code prompts, recalibrated acceptance
    ("tuner-v3-code", "tuner-v3-code", 128), # code, T=1.0, short-prompt costs (the proof sweep)
])
def test_stored_eagle_tables_reproduce(acceptance, costs, max_b):
    stored = json.loads((RESULTS / acceptance / "schedule.json").read_text())
    got = make_schedule(load_costs(str(RESULTS / costs / "costs_k*.json")), _acc(acceptance), max_b)
    assert got == stored


def test_v3_table_is_the_published_t4_table():
    got = make_schedule(load_costs(str(RESULTS / "tuner-v3-code/costs_k*.json")),
                        _acc("tuner-v3-code"), 128)
    assert got == [[1, 18, 3], [19, 43, 2], [44, 128, 1]]


def test_goodput_values_unchanged():
    costs = load_costs(str(RESULTS / "tuner-v3-code/costs_k*.json"))
    g = goodput_table(costs, tokens_per_step(_acc("tuner-v3-code")), 128)
    assert g[1] == pytest.approx({0: 28.029262550102306, 1: 41.9843533321123,
                                  2: 48.18267466653444, 3: 49.744330287391705}, rel=1e-12)
    assert g[64] == pytest.approx({0: 957.8687420489412, 1: 1323.5349460633322,
                                   2: 1195.5246260909685, 3: 1104.902436018568}, rel=1e-12)


def test_make_schedule_script_reproduces_v3_files(tmp_path):
    """dsd_tuning/tuner/make_schedule.py, the script that wrote experiments/results/tuner-v3-code, must
    write byte-identical schedule.json and goodput.csv from the same inputs."""
    from dsd_tuning.tuner.make_schedule import main
    v3 = RESULTS / "tuner-v3-code"
    main(["--costs", str(v3 / "costs_k*.json"), "--acceptance", str(v3 / "acceptance.json"),
          "--out", str(tmp_path), "--max-num-seqs", "128"])
    for f in ("schedule.json", "schedule_raw.json", "goodput.csv"):
        assert (tmp_path / f).read_text() == (v3 / f).read_text(), f
