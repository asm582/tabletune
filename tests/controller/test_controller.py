"""Unit tests for dsd_tuning/controller/controller.py: hysteresis and online re-fit (no GPU)."""

import pytest

from dsd_tuning.controller import DynamicSDController
from dsd_tuning.core import expand

TABLE = [[1, 18, 3], [19, 43, 2], [44, 128, 1]]
LOOKUP = expand(TABLE, 128)


def test_without_hysteresis_it_follows_the_table():
    c = DynamicSDController(LOOKUP, 3)
    for b in [1, 18, 19, 43, 44, 128]:
        assert c.choose_k(b) == LOOKUP[b]


def test_batch_hovering_at_a_boundary_does_not_flip():
    c = DynamicSDController(LOOKUP, 3, hysteresis=4)
    c.choose_k(40)                                    # K=2
    ks = [c.choose_k(b) for b in [43, 44, 45, 43, 44, 46, 42, 45]]
    assert ks == [2] * 8                              # 44+ is K=1 in the table, but held
    assert c.stats["switches"] == 0 and c.stats["held"] > 0

    plain = DynamicSDController(LOOKUP, 3)
    plain.choose_k(40)
    ks_plain = [plain.choose_k(b) for b in [43, 44, 45, 43, 44, 46, 42, 45]]
    assert ks_plain == [2, 1, 1, 2, 1, 1, 2, 1]      # without hysteresis it flips


def test_switches_once_well_past_the_boundary():
    c = DynamicSDController(LOOKUP, 3, hysteresis=4)
    c.choose_k(40)
    assert c.choose_k(48) == 1                        # 4+ past the boundary at 44
    assert c.stats["switches"] == 1


def test_jump_far_from_current_range_switches_immediately():
    c = DynamicSDController(LOOKUP, 3, hysteresis=4)
    c.choose_k(10)                                    # K=3
    assert c.choose_k(43) == 2                        # K=3 is nowhere near 43


def test_k_never_exceeds_num_spec_tokens():
    c = DynamicSDController(expand([[1, 64, 3]], 64), 2)
    assert c.choose_k(1) == 2
    assert c.allowed_ks == [2]


def test_out_of_range_batch_sizes_are_clamped():
    c = DynamicSDController(LOOKUP, 3)
    assert c.choose_k(0) == LOOKUP[1]
    assert c.choose_k(10_000) == LOOKUP[128]


# ------------------------------------------------------------ acceptance
def test_estimate_is_the_prior_before_any_observation():
    c = DynamicSDController(LOOKUP, 3, prior_accept=[0.6, 0.3, 0.1])
    assert c.acceptance_estimate() == pytest.approx([0.6, 0.3, 0.1])


def test_estimate_moves_to_live_acceptance_with_enough_data():
    c = DynamicSDController(LOOKUP, 3, prior_accept=[0.6, 0.3, 0.1], prior_weight=100)
    for _ in range(100_000):
        c.observe(3, 3)                               # every draft accepted
    assert all(x > 0.99 for x in c.acceptance_estimate())


def test_unobserved_positions_keep_the_prior():
    c = DynamicSDController(LOOKUP, 3, prior_accept=[0.6, 0.3, 0.1], prior_weight=100)
    for _ in range(10_000):
        c.observe(1, 0)                               # K=1: only position 1 is drafted
    est = c.acceptance_estimate()
    assert est[0] < 0.01                              # position 1 learned: rejected
    assert est[1:] == pytest.approx([0.3, 0.1])       # positions 2-3 never drafted


# ----------------------------------------------------------------- re-fit
def _costs():
    bs = [1, 16, 32, 64, 128]
    return {k: (bs, [40 + b + k * (1 + 0.15 * b) for b in bs]) for k in range(4)}


def test_refit_deepens_k_when_live_acceptance_is_higher():
    lookup = expand([[1, 8, 2], [9, 128, 1]], 128)
    c = DynamicSDController(lookup, 3, costs=_costs(), prior_accept=[0.3, 0.1, 0.02],
                            refit_steps=10, prior_weight=10)
    for _ in range(5000):
        c.observe(2, 2)                               # live traffic accepts everything
    c.refit()
    assert c.stats["table_changes"] == 1
    assert c.lookup[1:] != lookup[1:]
    assert set(c.lookup[1:]) <= {1, 2}                # K=3 has no CUDA graph: not allowed
    assert c.lookup[64] >= lookup[64]                 # more speculation where it now pays


def test_refit_runs_every_refit_steps():
    c = DynamicSDController(LOOKUP, 3, costs=_costs(), prior_accept=[0.6, 0.35, 0.15],
                            refit_steps=5)
    for _ in range(12):
        c.choose_k(10)
    assert c.stats["refits"] == 2


def test_refit_decays_old_evidence():
    c = DynamicSDController(LOOKUP, 3, costs=_costs(), prior_accept=[0.6, 0.35, 0.15],
                            refit_steps=1, decay=0.5)
    c.observe(3, 1)
    c.refit()
    assert c.drafted == pytest.approx([0.5, 0.5, 0.5])
    assert c.accepted == pytest.approx([0.5, 0.0, 0.0])


def test_refit_disabled_without_costs_or_prior():
    c = DynamicSDController(LOOKUP, 3, refit_steps=5)
    assert c.refit_steps == 0 and c.refit() is None
    for _ in range(20):
        c.choose_k(30)
    assert c.stats["refits"] == 0


def test_table_changes_are_logged(caplog):
    lookup = expand([[1, 128, 1]], 128)
    c = DynamicSDController(lookup, 3, costs=_costs(), prior_accept=[0.9, 0.8, 0.7],
                            refit_steps=1)
    c.allowed_ks = [1, 3]
    with caplog.at_level("INFO"):
        c.refit()
    assert "Dynamic SD controller: new table" in caplog.text


# ------------------------------------------------------------ from env
from pathlib import Path  # noqa: E402

from dsd_tuning.controller import controller_from_env  # noqa: E402

V3 = Path(__file__).resolve().parents[2] / "experiments" / "results" / "tuner-v3-code"


def test_disabled_by_default():
    assert controller_from_env(LOOKUP, 3, env={}) is None
    assert controller_from_env(LOOKUP, 3, env={"VLLM_DSD_CONTROLLER": "0"}) is None


def test_hysteresis_only_from_env():
    c = controller_from_env(LOOKUP, 3, env={"VLLM_DSD_CONTROLLER": "1",
                                            "VLLM_DSD_HYSTERESIS": "6"})
    assert c.hysteresis == 6 and c.refit_steps == 0


@pytest.mark.skipif(not V3.exists(), reason="needs the tuner outputs in experiments/results/")
def test_refit_from_env_with_tuner_outputs():
    c = controller_from_env(LOOKUP, 3, env={
        "VLLM_DSD_CONTROLLER": "1", "VLLM_DSD_COSTS": str(V3 / "costs_k*.json"),
        "VLLM_DSD_PRIOR": str(V3 / "acceptance.json"), "VLLM_DSD_REFIT_STEPS": "100"})
    assert c.refit_steps == 100 and c.prior and c.costs
    assert c.describe().startswith("Dynamic SD controller enabled: hysteresis=4, refit=every 100 steps")
    # With the calibrated prior and no live data, a re-fit reproduces the tuner's table.
    c.refit()
    assert c.lookup == LOOKUP and c.stats["table_changes"] == 0


def test_stats_are_logged_periodically(caplog):
    c = DynamicSDController(LOOKUP, 3, log_steps=10)
    with caplog.at_level("INFO"):
        for _ in range(25):
            c.choose_k(30)
    assert len(caplog.records) == 2 and "'steps': 10" in caplog.records[0].getMessage()


# ------------------------------------------------- proposal rate (n-gram, #3)
def test_no_proposal_is_ignored_unless_tracking():
    """EAGLE-style drafters: steps without drafts change nothing (default)."""
    c = DynamicSDController(LOOKUP, 3, costs=_costs(), prior_accept=[0.6, 0.35, 0.15])
    c.choose_k(10)
    for _ in range(1000):
        c.observe_no_proposal()
    assert c.decode_steps == 0 and c.proposal_rate_estimate() == 1.0


def test_proposal_rate_is_learned_when_tracking():
    c = DynamicSDController(LOOKUP, 3, prior_accept=[0.6, 0.35, 0.15], prior_weight=100,
                            track_proposals=True, prior_proposal_rate=1.0)
    c.choose_k(10)                                    # K=3
    for _ in range(2000):
        c.observe(3, 1)
    for _ in range(8000):
        c.observe_no_proposal()
    assert c.proposal_rate_estimate() == pytest.approx((100 + 2000) / (100 + 10_000))
    # Acceptance stays per draft: steps without drafts don't count as rejections.
    assert c.acceptance_estimate()[0] == pytest.approx((100 * 0.6 + 2000) / 2100)


def test_steps_at_k0_are_not_counted():
    c = DynamicSDController(expand([[1, 16, 3], [17, 128, 0]], 128), 3,
                            track_proposals=True)
    c.choose_k(100)                                   # K=0: no drafts expected
    for _ in range(1000):
        c.observe_no_proposal()
    assert c.decode_steps == 0


def test_refit_uses_the_proposal_rate():
    """Same live acceptance, but the n-gram-like drafter proposes on 20% of
    steps: K=3 gains only a fifth as much while a K=3 step costs the same, so
    the rebuilt table drops to K=1 much sooner."""
    bs = [1, 16, 32, 64, 128]
    steep = {k: (bs, [40 + b + k * (1 + 0.8 * b) for b in bs]) for k in range(4)}
    lookup = expand([[1, 16, 3], [17, 128, 1]], 128)
    kw = dict(costs=steep, prior_accept=[0.9, 0.5, 0.5], prior_weight=10)
    every = DynamicSDController(lookup, 3, **kw)
    sparse = DynamicSDController(lookup, 3, track_proposals=True, **kw)
    for c in (every, sparse):
        c.choose_k(10)
        for _ in range(500):                          # live acceptance 0.9 / 0.5 / 0.5
            for n in [3] * 5 + [1] * 4 + [0]:
                c.observe(3, n)
    for _ in range(20_000):
        sparse.observe_no_proposal()
    assert every.refit() == [[1, 31, 3], [32, 128, 1]]
    assert sparse.proposal_rate_estimate() == pytest.approx(5010 / 25_010)
    assert sparse.refit() == [[1, 4, 3], [5, 128, 1]]


def test_refit_decays_proposal_counts():
    c = DynamicSDController(LOOKUP, 3, costs=_costs(), prior_accept=[0.6, 0.35, 0.15],
                            track_proposals=True, decay=0.5)
    c.choose_k(10)
    c.observe(3, 1)
    c.observe_no_proposal()
    c.refit()
    assert (c.decode_steps, c.proposal_steps) == pytest.approx((1.0, 0.5))


def test_tracking_follows_the_method():
    env = {"VLLM_DSD_CONTROLLER": "1"}
    assert controller_from_env(LOOKUP, 3, env=env, method="ngram").track_proposals
    assert not controller_from_env(LOOKUP, 3, env=env, method="eagle3").track_proposals
    assert not controller_from_env(LOOKUP, 3, env=env).track_proposals


# ------------------------------------------------ short priors (n-gram K=5)
from dsd_tuning.controller import extend_prior  # noqa: E402


def test_full_length_prior_is_unchanged():
    assert extend_prior([0.6, 0.35, 0.15], 3) == [0.6, 0.35, 0.15]
    c = DynamicSDController(LOOKUP, 3, prior_accept=[0.6, 0.35, 0.15])
    assert c.prior == [0.6, 0.35, 0.15]


def test_short_prior_is_extended_geometrically():
    ext = extend_prior([0.6, 0.3, 0.15], 5)
    assert ext == pytest.approx([0.6, 0.3, 0.15, 0.075, 0.0375])
    c = DynamicSDController(expand([[1, 128, 5]], 128), 5, prior_accept=[0.6, 0.3, 0.15])
    assert len(c.acceptance_estimate()) == 5 and min(c.acceptance_estimate()) > 0


def test_short_prior_still_allows_deep_k_after_learning():
    """EAGLE's 3-position prior with n-gram at K=5: K=5 must stay reachable."""
    lookup = expand([[1, 16, 1], [17, 64, 3], [65, 128, 5]], 128)
    c = DynamicSDController(lookup, 5, costs={k: ([1, 128], [40.0, 40.0 + 2 * k]) for k in range(6)},
                            prior_accept=[0.6, 0.3, 0.15], prior_weight=10)
    c.choose_k(100)
    for _ in range(5000):
        c.observe(5, 5)                               # every draft accepted
    c.refit()
    assert c.lookup[1:] == [5] * 128


# ------------------------------------------- explore and prior decay (issue #2)
def test_new_options_are_off_by_default():
    """Default path: identical decisions and estimates to a plain controller."""
    kw = dict(hysteresis=4, costs=_costs(), prior_accept=[0.6, 0.35, 0.15], refit_steps=7)
    plain = DynamicSDController(LOOKUP, 3, **kw)
    explicit = DynamicSDController(LOOKUP, 3, explore_every=0, prior_decay=0.0, **kw)
    for i in range(300):
        b = [3, 20, 44, 90][i % 4]
        assert plain.choose_k(b) == explicit.choose_k(b)
        plain.observe(2, i % 3)
        explicit.observe(2, i % 3)
    assert plain.acceptance_estimate() == explicit.acceptance_estimate()
    assert plain.prior_weight == 2000.0 and "explored" not in plain.stats


def test_explore_tries_the_deepest_k_only_when_cheap():
    bs = [1, 16, 128]
    costs = {1: (bs, [40.0, 60.0, 100.0]), 3: (bs, [41.0, 66.0, 200.0])}  # K=3: +2.5%, +10%, +100%
    c = DynamicSDController(expand([[1, 100, 1], [101, 128, 3]], 128), 3,
                            costs=costs, explore_every=2, explore_max_cost=0.25)
    assert c.allowed_ks == [1, 3]
    assert [c.choose_k(1) for _ in range(4)] == [1, 3, 1, 3]      # +2.5%: every 2nd step
    assert [c.choose_k(100) for _ in range(4)] == [1, 1, 1, 1]    # ~+85%: never
    assert c.current_k == 1 and c.stats["explored"] == 2          # hysteresis state untouched


def test_prior_decay_shrinks_the_prior_weight_to_a_floor():
    c = DynamicSDController(LOOKUP, 3, costs=_costs(), prior_accept=[0.6, 0.35, 0.15],
                            prior_decay=0.5)
    for _ in range(3):
        c.refit()
    assert c.prior_weight == 250.0
    for _ in range(20):
        c.refit()
    assert c.prior_weight == 20.0                                  # 1% of the start


NGRAM = Path(__file__).resolve().parents[2] / "experiments" / "results" / "ngram-tuner"


@pytest.mark.skipif(not (NGRAM.exists() and V3.exists()), reason="needs the tuner outputs in experiments/results/")
@pytest.mark.parametrize("opts, want", [
    ({}, [[1, 2, 5], [3, 128, 3]]),                                # run (b): stuck
    (dict(explore_every=20, prior_decay=0.5), [[1, 20, 5], [21, 128, 3]]),  # the tuner's table
])
def test_run_b_ratchet_and_its_fix(opts, want):
    """Simulates n-gram run (b): untuned start, EAGLE's 3-position prior, true
    acceptance 0.79/0.71/0.66/0.62/0.59 with proposal rate 0.49, load levels
    4..128. Without the options positions 4-5 never get data; with both, the
    controller rebuilds the table the tuner builds for K in {1,3,5}."""
    import json
    import random

    from dsd_tuning.core import compress, load_costs
    rng = random.Random(0)
    true, p = [0.79, 0.71, 0.66, 0.62, 0.59], 0.49
    c = DynamicSDController(
        expand([[1, 16, 1], [17, 64, 3], [65, 128, 5]], 128), 5, hysteresis=4,
        costs=load_costs(str(NGRAM / "costs_k*.json")),
        prior_accept=json.loads((V3 / "acceptance.json").read_text())["accept_by_pos"],
        refit_steps=200, track_proposals=True, **opts)
    for s in range(4000):
        b = [4, 8, 16, 32, 64, 96, 128][(s // 500) % 7]
        k = c.choose_k(b)
        for _ in range(b):
            if rng.random() > p:
                c.observe_no_proposal()
                continue
            n = 0
            while n < k and rng.random() < true[n] / (true[n - 1] if n else 1):
                n += 1
            c.observe(k, n)
    assert compress({b: c.lookup[b] for b in range(1, 129)}) == want


# --------------------------------------------------- any K (issue #2)
def test_any_k_allows_every_k_up_to_num_spec_tokens():
    lookup = expand([[1, 64, 3], [65, 128, 1]], 128)
    assert DynamicSDController(lookup, 3).allowed_ks == [1, 3]
    assert DynamicSDController(lookup, 3, any_k=True).allowed_ks == [0, 1, 2, 3]


def test_any_k_only_with_mrv1(caplog):
    env = {"VLLM_DSD_CONTROLLER": "1", "VLLM_DSD_ANY_K": "1"}
    assert controller_from_env(LOOKUP, 3, env=env).allowed_ks == [1, 2, 3]
    assert "VLLM_DSD_ANY_K ignored" in caplog.text            # runner not pinned to MRv1 (a warning)
    mrv1 = controller_from_env(LOOKUP, 3, env={**env, "VLLM_USE_V2_MODEL_RUNNER": "0"})
    assert mrv1.allowed_ks == [0, 1, 2, 3]
    assert controller_from_env(LOOKUP, 3, env={"VLLM_DSD_CONTROLLER": "1",
                                               "VLLM_USE_V2_MODEL_RUNNER": "0"}).allowed_ks == [1, 2, 3]


def test_any_k_refit_can_pick_a_k_missing_from_the_table():
    """The EAGLE case from integration §5: a table with K in {1,3} cannot use K=2."""
    bs = [1, 32, 64, 128]
    costs = {0: (bs, [40, 60, 80, 120]), 1: (bs, [42, 64, 88, 136]),
             2: (bs, [43, 66, 92, 150]), 3: (bs, [44, 80, 120, 200])}
    lookup = expand([[1, 16, 3], [17, 128, 1]], 128)
    kw = dict(costs=costs, prior_accept=[0.62, 0.35, 0.16], refit_steps=10**9)
    fixed = DynamicSDController(lookup, 3, **kw).refit()
    free = DynamicSDController(lookup, 3, any_k=True, **kw).refit()
    assert 2 not in {k for _, _, k in fixed}
    assert 2 in {k for _, _, k in free}


# ------------------------------------------------ table stability (issue #2 A)
# ------------------------------------------------------------- robustness
# Failure policy (controller.py docstring): bad settings stop vLLM at startup;
# nothing raises from choose_k() while serving.
import sys  # noqa: E402


@pytest.mark.parametrize("kwargs, match", [
    (dict(decay=1.5), "decay"),
    (dict(decay=-0.1), "decay"),
    (dict(prior_decay=1.0), "prior_decay"),
    (dict(prior_weight=-1), "prior_weight"),
    (dict(prior_weight=float("nan")), "prior_weight"),
    (dict(explore_max_cost=float("inf")), "explore_max_cost"),
    (dict(refit_steps=-5), "refit_steps"),
    (dict(explore_every=2.5), "explore_every"),
    (dict(prior_proposal_rate=0.0), "prior_proposal_rate"),
    (dict(prior_accept=[0.6, float("nan"), 0.1]), "prior"),
    (dict(prior_accept=[1.2, 0.3, 0.1]), "prior"),
])
def test_invalid_options_are_rejected_at_startup(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DynamicSDController(LOOKUP, 3, **kwargs)


def test_costs_must_cover_every_k_in_the_table():
    costs = {k: v for k, v in _costs().items() if k != 2}
    with pytest.raises(ValueError, match=r"table uses K=\[1, 2, 3\]"):
        DynamicSDController(LOOKUP, 3, costs=costs, prior_accept=[0.6, 0.3, 0.1])


def _broken_make_schedule(monkeypatch, result=None):
    """Replace make_schedule inside the controller module (works for the vLLM copy too)."""
    calls = []

    def fake(*args, **kwargs):
        calls.append(1)
        if result is None:
            raise RuntimeError("boom")
        return result
    monkeypatch.setattr(sys.modules[DynamicSDController.__module__], "make_schedule", fake)
    return calls


def test_failed_refit_keeps_serving_the_current_table(monkeypatch, caplog):
    c = DynamicSDController(LOOKUP, 3, costs=_costs(), prior_accept=[0.6, 0.35, 0.15],
                            refit_steps=10)
    calls = _broken_make_schedule(monkeypatch)
    ks = [c.choose_k(b) for b in [5, 30, 100] * 20]          # 60 steps: 6 re-fit attempts
    assert ks == [LOOKUP[b] for b in [5, 30, 100]] * 20       # table unchanged, no exception
    assert len(calls) == 3                                    # turned off after 3 in a row
    assert c.refit_steps == 0 and c.stats["refit_errors"] == 3
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert "re-fit failed (RuntimeError: boom)" in warnings[0] and "turned off" in warnings[-1]


def test_refit_never_switches_to_a_k_without_a_cuda_graph(monkeypatch):
    c = DynamicSDController(LOOKUP, 3, costs=_costs(), prior_accept=[0.6, 0.35, 0.15],
                            refit_steps=1)
    c.allowed_ks = [1, 2]                                     # e.g. K=3 has no graph
    _broken_make_schedule(monkeypatch, result=[[1, 128, 3]])
    assert c.choose_k(10) == 3                                # hysteresis on the old table
    assert c.lookup == LOOKUP and c.stats["refit_errors"] == 1


def test_zero_prior_weight_does_not_divide_by_zero():
    c = DynamicSDController(LOOKUP, 3, costs=_costs(), prior_accept=[0.6, 0.35, 0.15],
                            prior_weight=0, track_proposals=True, prior_proposal_rate=0.5)
    assert c.acceptance_estimate() == [0.6, 0.35, 0.15]      # no data: the prior
    assert c.proposal_rate_estimate() == 0.5
    c.choose_k(10)
    c.observe(1, 1)
    assert c.acceptance_estimate() == [1.0, 0.35, 0.15]      # data only where drafted
    for _ in range(100):
        c.observe_no_proposal()
    assert c.refit() is not None                              # tiny rate, not a crash


@pytest.mark.parametrize("env, match", [
    ({"VLLM_DSD_HYSTERESIS": "four"}, "VLLM_DSD_HYSTERESIS='four': expected an integer"),
    ({"VLLM_DSD_DECAY": "half"}, "VLLM_DSD_DECAY='half': expected a number"),
    ({"VLLM_DSD_DECAY": "1.5"}, "decay must be in"),
    ({"VLLM_DSD_ANY_K": "maybe"}, "VLLM_DSD_ANY_K='maybe': expected 0 or 1"),
    ({"VLLM_DSD_ANY_K": "2"}, "VLLM_DSD_ANY_K"),
    ({"VLLM_DSD_COSTS": "x*"}, "VLLM_DSD_COSTS is set but VLLM_DSD_PRIOR is not"),
    ({"VLLM_DSD_PRIOR": "x.json"}, "VLLM_DSD_PRIOR is set but VLLM_DSD_COSTS is not"),
    ({"VLLM_DSD_COSTS": "/nonexistent/costs_k*.json", "VLLM_DSD_PRIOR": "x.json"},
     "VLLM_DSD_COSTS: no cost files match"),
])
def test_env_misconfiguration_names_the_variable(env, match):
    with pytest.raises(ValueError, match=match):
        controller_from_env(LOOKUP, 3, env={"VLLM_DSD_CONTROLLER": "1", **env})


@pytest.mark.parametrize("value, enabled", [
    ("1", True), ("true", True), ("ON", True), (" yes ", True),
    ("0", False), ("false", False), ("", False),
])
def test_env_switch_spellings(value, enabled):
    c = controller_from_env(LOOKUP, 3, env={"VLLM_DSD_CONTROLLER": value})
    assert (c is not None) == enabled

