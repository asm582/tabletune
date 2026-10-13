"""Nightjar-style draft-length control (Li et al., arXiv 2512.22420), reimplemented as a vLLM plugin.

Nightjar has no public code; this follows the paper's planner (ADA-BINGREEDY):

- Context: the exact batch size B. Each B has its own timeline; only steps with that B advance it.
- Arms: the draft lengths gamma, including 0 (speculation off) when the starting table has it. Under
  vLLM's V2 runner only lengths with captured CUDA graphs can run: those in the starting table.
- Loss: latency per token, the step's wall time divided by the tokens it produced (accepted drafts plus
  one per request). The planner minimizes the empirical mean loss of each (B, gamma), plus a switching
  cost C_switch / gamma when speculation resumes after gamma = 0 (Eq. 4).
- Schedule: time is split into blocks, bins and rounds (one round = one decode step at that B). An arm is
  chosen at the start of a bin and held for the bin; a bin is exploratory with probability 1/b
  (b = bin index in the block; uniform arm), else greedy. A bin ends after sqrt(H) rounds, a block after
  sqrt(H) bins, and the next block has H = 2^(j-1) for block index j.

Not reproduced, because they need engine changes: moving the drafter to CPU memory while speculation is
off, and the measured re-prefill cost table. vLLM keeps running EAGLE-style drafters at length 0, so
nothing needs re-prefilling and C_switch defaults to 0.

No vLLM imports; vllm_plugin.py calls choose_k(batch_size) once per scheduled step, observe(drafted,
accepted) per request, and end_step() once per finished step.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class _Timeline:
    """ADA-BINGREEDY state for one batch size."""
    j: int = 1          # block index
    h: int = 1          # block horizon H
    b: int = 0          # bin index within the block
    tau: int = 0        # round within the bin; 0 = a new bin starts at the next round
    arm: int = 0
    loss_sum: dict[int, float] = field(default_factory=dict)
    loss_n: dict[int, int] = field(default_factory=dict)

    def mean(self, arm: int) -> float:
        n = self.loss_n.get(arm, 0)
        return self.loss_sum[arm] / n if n else -math.inf   # untried arms first


class NightjarStyleController:

    def __init__(self, lookup: list[int], num_spec_tokens: int, switch_cost_ms: float = 0.0,
                 seed: int = 0, max_step_ms: float = 2000.0, log_steps: int = 0,
                 clock=time.perf_counter):
        self.max_b = len(lookup) - 1
        self.arms = sorted({max(0, min(k, num_spec_tokens)) for k in lookup[1:]})
        self.switch_cost_ms = switch_cost_ms
        self.rng = random.Random(seed)
        self.max_step_ms = max_step_ms
        self.log_steps = log_steps
        self.clock = clock
        self.timelines: dict[int, _Timeline] = {}
        self.last_k: int | None = None
        self.pending: deque[tuple[int, int]] = deque()   # (batch size, gamma) of scheduled, unfinished steps
        self.last_end: float | None = None
        self.step_accepted = 0
        self.steps = 0
        self.stats: dict[str, int] = {"steps": 0, "explore_bins": 0, "greedy_bins": 0, "losses": 0}

    def describe(self) -> str:
        return (f"Nightjar-style controller enabled: arms={self.arms}, switch cost {self.switch_cost_ms} ms, "
                f"per-batch-size ADA-BINGREEDY")

    def _start_bin(self, t: _Timeline) -> None:
        t.b += 1
        if t.b > math.sqrt(t.h):          # block over: the next one is twice as long
            t.j += 1
            t.h = 2 ** (t.j - 1)
            t.b = 1
        if self.rng.random() < 1.0 / t.b:
            t.arm = self.rng.choice(self.arms)
            self.stats["explore_bins"] += 1
        else:
            def score(g: int) -> float:
                penalty = self.switch_cost_ms / g if (self.last_k == 0 and g > 0) else 0.0
                return t.mean(g) + penalty
            t.arm = min(self.arms, key=score)
            self.stats["greedy_bins"] += 1
        t.tau = 0

    def choose_k(self, batch_size: int) -> int:
        self.steps += 1
        self.stats["steps"] += 1
        b = min(max(batch_size, 1), self.max_b)
        t = self.timelines.setdefault(b, _Timeline(arm=self.arms[-1]))
        if t.tau == 0:
            self._start_bin(t)
        t.tau += 1
        k = t.arm
        if t.tau >= math.sqrt(t.h):       # this round ends the bin (bins of ceil(sqrt(H)) rounds)
            t.tau = 0
        self.last_k = k
        self.pending.append((b, k))
        while len(self.pending) > 8:      # steps whose output never came back: forget them
            self.pending.popleft()
        if self.log_steps and self.steps % self.log_steps == 0:
            logger.info("Nightjar-style controller: %s, %d batch sizes seen", dict(self.stats), len(self.timelines))
        return k

    def observe(self, num_drafted: int, num_accepted: int) -> None:
        if num_drafted > 0:
            self.step_accepted += max(0, min(num_accepted, num_drafted))

    def end_step(self) -> None:
        """The oldest scheduled step finished: record its latency per token for its (B, gamma)."""
        now = self.clock()
        b, k = self.pending.popleft() if self.pending else (0, 0)
        if b and self.last_end is not None:
            ms = (now - self.last_end) * 1000.0
            tokens = b + self.step_accepted
            if 0 < ms <= self.max_step_ms and tokens > 0:   # skip idle gaps
                t = self.timelines[b]
                t.loss_sum[k] = t.loss_sum.get(k, 0.0) + ms / tokens
                t.loss_n[k] = t.loss_n.get(k, 0) + 1
                self.stats["losses"] += 1
        self.last_end = now
        self.step_accepted = 0
