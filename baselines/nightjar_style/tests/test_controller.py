from nightjar_style.controller import NightjarStyleController


def table(ranges, n=128):
    lk = [0] * (n + 1)
    for s, e, k in ranges:
        for b in range(s, e + 1):
            lk[b] = k
    return lk


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def run(c, clock, b, step_ms, accepted_per_req, steps):
    """Drive the controller: step time and acceptance depend on the chosen gamma."""
    ks = []
    for _ in range(steps):
        k = c.choose_k(b)
        ks.append(k)
        for _ in range(b):
            c.observe(k, accepted_per_req(k))
        clock.t += step_ms(k) / 1000
        c.end_step()
    return ks


def test_arms_are_the_tables_lengths():
    c = NightjarStyleController(table([[1, 40, 3], [41, 100, 1], [101, 128, 0]]), 3)
    assert c.arms == [0, 1, 3]


def test_converges_to_the_best_arm_per_batch_size():
    clock = Clock()
    c = NightjarStyleController(table([[1, 40, 3], [41, 100, 1], [101, 128, 0]]), 3, clock=clock)
    # B=4: long drafts pay (cheap steps); B=100: speculation costs more than it gains
    low = run(c, clock, 4, lambda k: 5 + 0.2 * k, lambda k: min(k, 2), 3000)
    high = run(c, clock, 100, lambda k: 10 + 12 * k, lambda k: min(k, 1), 3000)
    assert low[-500:].count(3) > 400
    assert high[-500:].count(0) > 400


def test_batch_sizes_learn_independently():
    clock = Clock()
    c = NightjarStyleController(table([[1, 128, 3], [1, 1, 1]]), 3, clock=clock)
    run(c, clock, 8, lambda k: 5, lambda k: k, 100)
    assert 8 in c.timelines and 9 not in c.timelines
