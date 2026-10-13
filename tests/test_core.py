"""Unit tests for dsd_tuning/core.py: the formula and the table rules (no GPU)."""

import json
from pathlib import Path

import pytest

from dsd_tuning.core import (build_schedule, compress, expand, fixed_k_losses, goodput_table,
                            interp, load_costs, make_schedule, speculative_config,
                            tokens_per_step)

REPO = Path(__file__).resolve().parent.parent


def test_interp_flat_linear_and_extrapolated():
    xs, ys = [1, 3, 5], [10.0, 20.0, 40.0]
    assert interp(xs, ys, 0) == 10.0          # flat below the first point
    assert interp(xs, ys, 2) == 15.0          # linear between points
    assert interp(xs, ys, 5) == 40.0
    assert interp(xs, ys, 7) == 60.0          # last slope (10 per unit) beyond


def test_tokens_per_step():
    assert tokens_per_step([0.6, 0.3, 0.1]) == pytest.approx({0: 1, 1: 1.6, 2: 1.9, 3: 2.0})


def test_compress_and_expand_round_trip():
    best = {b: (3 if b <= 4 else 2 if b <= 9 else 1) for b in range(1, 13)}
    sched = compress(best)
    assert sched == [[1, 4, 3], [5, 9, 2], [10, 12, 1]]
    assert expand(sched, 12)[1:] == [best[b] for b in range(1, 13)]


def test_expand_carries_k_through_gaps_and_tail():
    lookup = expand([[1, 2, 3], [5, 6, 1]], 8)
    assert lookup[1:] == [3, 3, 3, 3, 1, 1, 1, 1]


def _costs(extra_ms_per_k_per_b: float, base_ms: float = 40.0, slope: float = 1.0):
    """Synthetic step costs: each extra draft token adds a fixed cost plus a
    per-request cost, so deep K pays at small batches and loses at large ones."""
    bs = [1, 16, 32, 64, 128]
    return {k: (bs, [base_ms + slope * b + k * (1.0 + extra_ms_per_k_per_b * b) for b in bs])
            for k in range(4)}


def test_best_k_falls_with_batch_size():
    sched = make_schedule(_costs(0.15), [0.6, 0.35, 0.15], max_b=128)
    ks = [k for _, _, k in sched]
    assert ks[0] == 3                        # deep speculation at small batches
    assert ks[-1] < ks[0]                    # less speculation at large batches
    assert sched[0][0] == 1 and sched[-1][1] == 128


def test_allowed_ks_restricts_choice():
    sched = make_schedule(_costs(0.15), [0.6, 0.35, 0.15], max_b=128, allowed_ks=[1, 3])
    assert {k for _, _, k in sched} <= {1, 3}


def test_rule1_ties_keep_current_k():
    # K=2 is 0.3% better at B=2 only: below eps, so K=3 is kept.
    goodput = {1: {2: 100.0, 3: 110.0}, 2: {2: 100.3, 3: 100.0}, 3: {2: 90.0, 3: 100.0}}
    raw, best = build_schedule(goodput, eps=0.005, min_width_frac=0.0)
    assert raw[2] == 2 and best[2] == 3


def test_rule2_merges_narrow_ranges():
    # A one-batch-size K=2 island at B=20 inside a K=3 range gets merged.
    goodput = {b: {2: 100.0, 3: 101.0} for b in range(1, 41)}
    goodput[20] = {2: 110.0, 3: 100.0}
    _, best = build_schedule(goodput, eps=0.0, min_width_frac=0.25)
    assert compress(best) == [[1, 40, 3]]


def test_fixed_k_losses():
    goodput = {1: {1: 100.0, 2: 90.0}, 2: {1: 80.0, 2: 100.0}}
    losses = fixed_k_losses(goodput)
    assert losses[1]["worst_loss"] == pytest.approx(0.2) and losses[1]["worst_b"] == 2
    assert losses[2]["worst_loss"] == pytest.approx(0.1) and losses[2]["worst_b"] == 1


def test_speculative_config_is_valid_for_vllm():
    cfg = speculative_config([[1, 18, 3], [19, 128, 1]], "drafter/x")
    assert cfg == {"method": "eagle3", "model": "drafter/x", "num_speculative_tokens": 3,
                   "num_speculative_tokens_per_batch_size": [[1, 18, 3], [19, 128, 1]]}
    # vLLM needs num_speculative_tokens >= 1 even if the table is all K=0.
    assert speculative_config([[1, 64, 0]], "d")["num_speculative_tokens"] == 1


@pytest.mark.parametrize("folder,costs_folder,max_b", [
    ("tuner-v2", "tuner-v2", 64),
    ("tuner-v2-code", "tuner-v2", 64),
    ("tuner-v3-code", "tuner-v3-code", 128),
])
def test_reproduces_committed_tables(folder, costs_folder, max_b):
    """The measured profiles in experiments/results/ must still produce the tables we benchmarked."""
    costs = load_costs(str(REPO / "experiments" / "results" / costs_folder / "costs_k*.json"))
    acc = json.loads((REPO / "experiments" / "results" / folder / "acceptance.json").read_text())
    tps = {int(k): v for k, v in acc["tokens_per_step_by_k"].items()}
    _, best = build_schedule(goodput_table(costs, tps, max_b))
    expected = json.loads((REPO / "experiments" / "results" / folder / "schedule.json").read_text())
    assert compress(best) == expected


# ---------------------------------------------------------------- proposal rate
# Drafters that propose only on some steps (n-gram, issue #3).
from dsd_tuning.core import load_acceptance, step_ms  # noqa: E402
from dsd_tuning.tuner.validate_costmodel import step_ms_mean  # noqa: E402

COSTS_P = {0: ([1, 64], [10.0, 20.0]), 1: ([1, 64], [11.0, 30.0]),
           2: ([1, 64], [12.0, 40.0]), 3: ([1, 64], [13.0, 50.0])}


def test_proposal_rate_one_is_the_old_formula():
    acc = [0.6, 0.35, 0.15]
    assert tokens_per_step(acc, proposal_rate=1.0) == tokens_per_step(acc)
    for k in COSTS_P:
        assert step_ms_mean(COSTS_P, k, 30, 1.0) == step_ms(COSTS_P, k, 30)


def test_proposal_rate_scales_tokens_not_cost():
    tps = tokens_per_step([0.5, 0.4], proposal_rate=0.2)
    assert tps == pytest.approx({0: 1.0, 1: 1.1, 2: 1.18})
    # The tuner's model: a step costs the full K, whatever the proposal rate.
    assert step_ms(COSTS_P, 3, 64) == 50.0
    # The rejected average model (validate_costmodel.py only): halfway K=1..K=2.
    assert step_ms_mean(COSTS_P, 3, 64, 0.5) == pytest.approx(35.0)
    assert step_ms_mean(COSTS_P, 0, 64, 0.5) == 20.0


def test_low_proposal_rate_lowers_k_sooner():
    """Only the gain scales with p; the cost of a K step does not. A sparse
    drafter therefore drops to lower K at smaller batches."""
    acc = [0.5, 0.4, 0.3]
    assert make_schedule(COSTS_P, acc, 64) == [[1, 14, 3], [15, 34, 2], [35, 64, 1]]
    sparse = make_schedule(COSTS_P, acc, 64, proposal_rate=0.2)
    assert expand(sparse, 64)[20] <= expand(make_schedule(COSTS_P, acc, 64), 64)[20]


def test_load_acceptance_rates(tmp_path):
    old = tmp_path / "old.json"
    old.write_text('{"accept_by_pos": [0.6, 0.3]}')
    assert load_acceptance(old) == ([0.6, 0.3], 1.0)       # files from before the field
    eagle = tmp_path / "eagle.json"
    eagle.write_text('{"accept_by_pos": [0.6, 0.3], "proposal_rate": 1.0035}')
    assert load_acceptance(eagle) == ([0.6, 0.3], 1.0)     # counting error -> 1
    ngram = tmp_path / "ngram.json"
    ngram.write_text('{"accept_by_pos": [0.55, 0.4], "proposal_rate": 0.2177}')
    assert load_acceptance(ngram) == ([0.55, 0.4], 0.2177)


# ------------------------------------------------------------- robustness


def _write_costs(tmp_path, k, rows):
    (tmp_path / f"costs_k{k}.json").write_text(json.dumps({"rows": rows}))
    return str(tmp_path / "costs_k*.json")


def _row(k, b, ms):
    return {"k": k, "batch_size": b, "step_ms": ms}


@pytest.mark.parametrize("content, match", [
    ("not json", "bad cost file"),
    ('{"rows": []}', "no rows"),
    ('{"table": []}', "missing field 'rows'"),
    ('{"rows": [{"k": 1, "batch_size": 1}]}', "missing field 'step_ms'"),
    ('{"rows": [{"k": 1, "batch_size": 1, "step_ms": 10}, {"k": 2, "batch_size": 2, "step_ms": 12}]}',
     "several K values"),
    ('{"rows": [{"k": 1, "batch_size": 1, "step_ms": 10}, {"k": 1, "batch_size": 1, "step_ms": 11}]}',
     "strictly increasing"),
    ('{"rows": [{"k": 1, "batch_size": 1, "step_ms": -3}]}', "finite and > 0"),
    ('{"rows": [{"k": 1, "batch_size": 1, "step_ms": NaN}]}', "finite and > 0"),
    ('{"rows": [{"k": 1, "batch_size": 0, "step_ms": 3}]}', "integers >= 1"),
])
def test_malformed_cost_files_name_the_file(tmp_path, content, match):
    (tmp_path / "costs_k1.json").write_text(content)
    with pytest.raises(ValueError, match=match) as e:
        load_costs(str(tmp_path / "costs_k*.json"))
    assert "costs_k1.json" in str(e.value)


def test_cost_rows_are_sorted_by_batch_size(tmp_path):
    pattern = _write_costs(tmp_path, 1, [_row(1, 8, 50.0), _row(1, 1, 40.0)])
    assert load_costs(pattern) == {1: ([1, 8], [40.0, 50.0])}


@pytest.mark.parametrize("content, match", [
    ('{"proposal_rate": 1.0}', "missing field 'accept_by_pos'"),
    ('{"accept_by_pos": [0.6, 1.3]}', r"in \[0, 1\]"),
    ('{"accept_by_pos": "0.6"}', "must be a list"),
    ('{"accept_by_pos": [0.6], "proposal_rate": 0}', "proposal rate must be > 0"),
    ('{"accept_by_pos": [0.6], "proposal_rate": "high"}', "proposal rate must be > 0"),
    ('[0.6, 0.3]', "bad acceptance file"),
])
def test_malformed_acceptance_files_are_rejected(tmp_path, content, match):
    f = tmp_path / "acceptance.json"
    f.write_text(content)
    with pytest.raises(ValueError, match=match):
        load_acceptance(f)


def test_extrapolated_step_time_never_falls():
    """A noisy last point (88.2 -> 81.4 ms, T4 AWQ K=0) used to extrapolate to
    negative step times at large batches."""
    xs, ys = [112, 120, 128], [83.1, 88.2, 81.4]
    assert interp(xs, ys, 512) == 81.4
    assert interp(xs, ys, 124) == pytest.approx(84.8)             # interpolation unchanged



def test_merge_tol_keeps_a_narrow_range_that_clearly_wins():
    from dsd_tuning.core import build_schedule
    # K=1 is 10% better on a range narrower than 25% of its start; the original rule merges it away
    goodput = {b: {0: 1.0, 1: (1.1 if 40 <= b <= 44 else 0.9)} for b in range(1, 129)}
    _, merged = build_schedule(goodput)
    _, kept = build_schedule(goodput, merge_tol=0.02)
    assert all(merged[b] == 0 for b in range(40, 45))
    assert all(kept[b] == 1 for b in range(40, 45))


def test_merge_tol_still_merges_a_cheap_narrow_range():
    from dsd_tuning.core import build_schedule
    goodput = {b: {0: 1.0, 1: (1.006 if 40 <= b <= 44 else 0.9)} for b in range(1, 129)}
    _, kept = build_schedule(goodput, merge_tol=0.02)
    assert all(kept[b] == 0 for b in range(40, 45))
