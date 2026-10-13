"""Regression tests: the EAGLE-3 results must not change when other drafting
methods (n-gram, issue #3) are added.

Controller side: pins the controller's behaviour with EAGLE-style inputs (a
draft on every step at the table's K). Written before the n-gram changes,
against the code that produced those results.
"""

import json
from pathlib import Path

import pytest

from dsd_tuning.controller import DynamicSDController
from dsd_tuning.core import expand, load_costs, make_schedule

RESULTS = Path(__file__).resolve().parents[2] / "experiments" / "results"


def _acc(name: str) -> list[float]:
    return json.loads((RESULTS / name / "acceptance.json").read_text())["accept_by_pos"]


@pytest.mark.parametrize("allowed", [[0, 1, 3], [1, 3]])
def test_controller_rebuild_from_integration_setup(allowed):
    """The rebuild both untuned starting tables converged to (experiments/results/integration §5)."""
    costs = load_costs(str(RESULTS / "tuner-v3-code/costs_k*.json"))
    got = make_schedule(costs, _acc("tuner-v2"), 128, allowed_ks=allowed)
    assert got == [[1, 34, 3], [35, 128, 1]]


def _eagle_controller(lookup: list[int]) -> DynamicSDController:
    return DynamicSDController(
        lookup, 3, hysteresis=4,
        costs=load_costs(str(RESULTS / "tuner-v3-code/costs_k*.json")),
        prior_accept=_acc("tuner-v2"), refit_steps=10**9, prior_weight=2000)


def test_controller_acceptance_estimate_with_eagle_counts():
    """EAGLE drafts K on every step; the estimate is the blended per-draft rate."""
    c = _eagle_controller(expand([[1, 64, 3], [65, 128, 1]], 128))
    for _ in range(3000):
        c.observe(3, 2)          # positions 1, 2 accepted, 3 rejected
    for _ in range(1000):
        c.observe(3, 0)
    prior = _acc("tuner-v2")
    want = [(2000 * prior[0] + 3000) / 6000, (2000 * prior[1] + 3000) / 6000,
            (2000 * prior[2] + 0) / 6000]
    assert c.acceptance_estimate() == pytest.approx(want, rel=1e-12)


def test_controller_ignores_steps_without_drafts_for_eagle():
    """K=0 steps (no drafts by design) must not count as rejected drafts."""
    c = _eagle_controller(expand([[1, 16, 3], [17, 128, 0]], 128))
    before = c.acceptance_estimate()
    c.choose_k(100)              # table says K=0
    for _ in range(5000):
        c.observe(0, 0)
    assert c.acceptance_estimate() == before
    assert c.refit() == [[1, 69, 3], [70, 128, 0]]   # prior only, K in {0, 3}
