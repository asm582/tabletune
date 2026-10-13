import json

from turbospec_style.controller import (StepTimeModel, TurboSpecStyleController, alpha_for,
                                        prior_alpha_from_acceptance, tokens_per_draft)
from turbospec_style.fit import fit


def test_geometric_roundtrip():
    for a in (0.1, 0.5, 0.8):
        for k in (1, 3, 8):
            assert abs(alpha_for(tokens_per_draft(a, k), k) - a) < 1e-6


def test_fit_recovers_linear_model():
    true = StepTimeModel([4.0, 0.002, 0.5, 0.001], "sequential")
    rows = [(b, k, true.step_ms(b, k)) for b in (1, 2, 4, 8, 16, 32, 64, 128) for k in (0, 1, 2, 3)]
    m, err = fit(rows, "sequential")
    assert err < 1e-9 and all(abs(x - y) < 1e-6 for x, y in zip(m.coef, true.coef))


def table(ranges, n=128):
    lk = [0] * (n + 1)
    for s, e, k in ranges:
        for b in range(s, e + 1):
            lk[b] = k
    return lk


def test_picks_long_drafts_at_low_load_and_short_at_high():
    # step time grows fast with batched tokens: speculation pays only at small batches
    m = StepTimeModel([5.0, 0.05, 0.3, 0.0], "sequential")
    c = TurboSpecStyleController(table([[1, 40, 3], [41, 100, 1], [101, 128, 0]]), 3, m, prior_alpha=0.7)
    assert c.allowed_ks == [0, 1, 3]
    assert c.choose_k(1) == 3 and c.choose_k(128) < 3


def test_moving_average_follows_acceptance():
    m = StepTimeModel([5.0, 0.01, 0.3, 0.0], "sequential")
    c = TurboSpecStyleController(table([[1, 128, 3]]), 3, m, prior_alpha=0.2, ema=0.2)
    for _ in range(50):
        c.choose_k(4)
        for _ in range(4):
            c.observe(3, 3)          # every guess accepted
        c.end_step()
    assert c.alpha > 0.9


def test_prior_from_tuner_profile(tmp_path):
    p = tmp_path / "acceptance.json"
    p.write_text(json.dumps({"accept_by_pos": [0.8, 0.64, 0.512, 0.0]}))
    assert abs(prior_alpha_from_acceptance(p) - 0.8) < 1e-3
