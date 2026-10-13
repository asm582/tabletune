"""TurboSpec-style draft-length control (Liu et al., arXiv 2406.14066), reimplemented as a vLLM plugin.

TurboSpec has no public code; this follows the paper's method section:

- Step time is a linear model fitted offline to profiled runs (fit.py):
  T(B, k) = delta + gamma * B * (k + 1) + k * (delta_d + gamma_d * B)   (sequential drafters, e.g. EAGLE-3)
  T(B, k) = delta + gamma * B * (k + 1) + [k > 0] * delta_d + gamma_d * B * k   (block drafters, DFlash)
  i.e. a target pass over B * (k + 1) batched tokens plus the drafting passes. The paper's context-token
  term (alpha * N_context) is dropped: the scheduler's lookup sees only the batch size, and the profiles
  use one prompt length.
- Acceptance is one global per-token rate alpha, a moving average over past steps, and the expected
  tokens per request and step is the geometric l(alpha, k) = (1 - alpha^(k+1)) / (1 - alpha).
- Every step, for the whole batch, it picks k = argmax_k B * l(alpha, k) / T(B, k) (no table, no
  hysteresis), over the lengths that have CUDA graphs: those in the starting table.

alpha per step is the rate that makes l(alpha, k) equal the observed mean tokens per draft, so the
geometric model reproduces what was measured at the drafted length. It starts from the same prior the
TableTune controller gets (the tuner's acceptance profile).

No vLLM imports; vllm_plugin.py calls choose_k(batch_size) once per scheduled step, observe(drafted,
accepted) per request, and end_step() once per finished step.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


def tokens_per_draft(alpha: float, k: int) -> float:
    """Expected tokens per request and step with k drafts at per-token acceptance alpha (geometric)."""
    if k <= 0:
        return 1.0
    if alpha >= 1.0:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def alpha_for(tokens: float, k: int) -> float:
    """The alpha with tokens_per_draft(alpha, k) == tokens (bisection; tokens in [1, k + 1])."""
    if k <= 0:
        raise ValueError("k must be > 0")
    tokens = min(max(tokens, 1.0), k + 1.0)
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if tokens_per_draft(mid, k) < tokens:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class StepTimeModel:
    """The fitted linear step-time model (fit.py writes it as JSON)."""

    def __init__(self, coef: list[float], form: str):
        if form not in ("sequential", "block"):
            raise ValueError(f"unknown model form {form!r}")
        if len(coef) != 4:
            raise ValueError("expected 4 coefficients")
        self.coef, self.form = [float(c) for c in coef], form

    @staticmethod
    def features(b: float, k: int, form: str) -> list[float]:
        if form == "sequential":
            return [1.0, b * (k + 1), float(k), b * k]
        return [1.0, b * (k + 1), 1.0 if k > 0 else 0.0, b * k]

    def step_ms(self, b: float, k: int) -> float:
        t = sum(c * x for c, x in zip(self.coef, self.features(b, k, self.form)))
        return max(t, 1e-3)

    @classmethod
    def load(cls, path: str | Path) -> "StepTimeModel":
        d = json.loads(Path(path).read_text())
        return cls(d["coef"], d["form"])


class TurboSpecStyleController:

    def __init__(self, lookup: list[int], num_spec_tokens: int, model: StepTimeModel,
                 prior_alpha: float, ema: float = 0.05, log_steps: int = 0):
        if not 0 < ema <= 1:
            raise ValueError(f"ema must be in (0, 1], got {ema}")
        if not 0 <= prior_alpha < 1:
            raise ValueError(f"prior_alpha must be in [0, 1), got {prior_alpha}")
        self.max_b = len(lookup) - 1
        self.num_spec_tokens = num_spec_tokens
        self.allowed_ks = sorted({max(0, min(k, num_spec_tokens)) for k in lookup[1:]})
        self.model = model
        self.alpha = prior_alpha
        self.ema = ema
        self.log_steps = log_steps
        self.pending: deque[tuple[int, int]] = deque()   # (batch size, k) of scheduled, unfinished steps
        self.step_drafts = 0                              # requests with drafts in the finishing step
        self.step_tokens = 0.0                            # their tokens (accepted + 1)
        self.steps = 0
        self.stats: dict[str, int] = {"steps": 0, "updates": 0}
        self.k_hist: dict[int, int] = {k: 0 for k in self.allowed_ks}

    def describe(self) -> str:
        return (f"TurboSpec-style controller enabled: allowed K={self.allowed_ks}, model={self.model.form} "
                f"{[round(c, 5) for c in self.model.coef]}, prior alpha={self.alpha:.3f}, ema={self.ema}")

    def best_k(self, b: int) -> int:
        return max(self.allowed_ks, key=lambda k: b * tokens_per_draft(self.alpha, k) / self.model.step_ms(b, k))

    def choose_k(self, batch_size: int) -> int:
        self.steps += 1
        self.stats["steps"] += 1
        b = min(max(batch_size, 1), self.max_b)
        k = self.best_k(b)
        self.k_hist[k] += 1
        self.pending.append((b, k))
        while len(self.pending) > 8:      # steps whose output never came back (e.g. aborted): forget them
            self.pending.popleft()
        if self.log_steps and self.steps % self.log_steps == 0:
            logger.info("TurboSpec-style controller: alpha=%.3f K histogram %s %s", self.alpha,
                        dict(self.k_hist), dict(self.stats))
        return k

    def observe(self, num_drafted: int, num_accepted: int) -> None:
        if num_drafted > 0:
            self.step_drafts += 1
            self.step_tokens += 1 + max(0, min(num_accepted, num_drafted))

    def end_step(self) -> None:
        """The oldest scheduled step finished: fold its acceptance into the moving average."""
        b, k = self.pending.popleft() if self.pending else (0, 0)
        if self.step_drafts and k > 0:
            obs = alpha_for(self.step_tokens / self.step_drafts, k)
            self.alpha = (1 - self.ema) * self.alpha + self.ema * obs
            self.alpha = min(self.alpha, 0.999)
            self.stats["updates"] += 1
        self.step_drafts, self.step_tokens = 0, 0.0


def prior_alpha_from_acceptance(path: str | Path) -> float:
    """Per-token alpha matching the tuner's acceptance profile at its longest length."""
    d = json.loads(Path(path).read_text())
    acc = d.get("accept_by_pos") or d.get("acceptance_by_position")
    while acc and acc[-1] <= 0:      # positions the profile never drafted
        acc = acc[:-1]
    if not acc:
        raise ValueError(f"{path}: no accept_by_pos")
    k = len(acc)
    return alpha_for(1.0 + sum(acc), k)
