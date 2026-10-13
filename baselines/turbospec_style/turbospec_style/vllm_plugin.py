"""vLLM plugin: answer the Dynamic SD lookup with the TurboSpec-style controller.

A no-op unless VLLM_TS_CONTROLLER=1. Then it wraps three Scheduler methods (no edits to vLLM):
__init__ swaps the Dynamic SD lookup for one that asks the controller, make_spec_decoding_stats passes
each request's drafted/accepted counts, and update_from_output closes a step. The server must start
with a Dynamic SD table: its lengths are the ones with CUDA graphs, so the only ones allowed.

  VLLM_TS_CONTROLLER=1  VLLM_TS_MODEL=model.json (fit.py)  VLLM_TS_PRIOR=acceptance.json (tuner)
  VLLM_TS_EMA=0.05 (moving-average weight per step)  VLLM_TS_LOG_STEPS=0
"""

from __future__ import annotations

import functools
import logging
import os

from .controller import StepTimeModel, TurboSpecStyleController, prior_alpha_from_acceptance

logger = logging.getLogger(__name__)
ENV = "VLLM_TS_CONTROLLER"


class _Lookup(list):
    def __init__(self, lookup, controller):
        super().__init__(lookup)
        self.controller = controller

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.controller.choose_k(index)
        return super().__getitem__(index)


def _controller(lookup: list[int], num_spec_tokens: int) -> TurboSpecStyleController:
    for name in ("VLLM_TS_MODEL", "VLLM_TS_PRIOR"):
        if not os.environ.get(name):
            raise ValueError(f"{ENV} is set but {name} is not")
    return TurboSpecStyleController(
        lookup, num_spec_tokens, StepTimeModel.load(os.environ["VLLM_TS_MODEL"]),
        prior_alpha_from_acceptance(os.environ["VLLM_TS_PRIOR"]),
        ema=float(os.environ.get("VLLM_TS_EMA", "0.05")),
        log_steps=int(os.environ.get("VLLM_TS_LOG_STEPS", "0")))


def install(cls: type) -> None:
    if getattr(cls, "_ts_installed", False):
        return
    orig_init, orig_stats, orig_update = cls.__init__, cls.make_spec_decoding_stats, cls.update_from_output

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.ts_controller = None
        lookup = getattr(self, "dynamic_sd_lookup", None)
        if lookup is None:
            raise RuntimeError(f"{ENV} is set but the server has no Dynamic SD table")
        c = _controller(list(lookup), self.num_spec_tokens)
        logger.info(c.describe())
        self.ts_controller = c
        self.dynamic_sd_lookup = _Lookup(lookup, c)

    @functools.wraps(orig_stats)
    def make_spec_decoding_stats(self, stats, num_draft_tokens, num_accepted_tokens, *args, **kwargs):
        c = getattr(self, "ts_controller", None)
        if c is not None:
            c.observe(num_draft_tokens, num_accepted_tokens)
        return orig_stats(self, stats, num_draft_tokens, num_accepted_tokens, *args, **kwargs)

    @functools.wraps(orig_update)
    def update_from_output(self, scheduler_output, model_runner_output):
        out = orig_update(self, scheduler_output, model_runner_output)
        c = getattr(self, "ts_controller", None)
        if c is not None and scheduler_output.num_scheduled_tokens:
            c.end_step()
        return out

    cls.__init__, cls.make_spec_decoding_stats, cls.update_from_output = __init__, make_spec_decoding_stats, update_from_output
    cls._ts_installed = True


def register() -> None:
    if os.environ.get(ENV, "").lower() not in ("1", "true", "yes", "on"):
        return
    from vllm.v1.core.sched.scheduler import Scheduler
    ours, theirs = logging.getLogger(__package__), logging.getLogger("vllm")
    if not ours.handlers:
        for h in theirs.handlers:
            ours.addHandler(h)
        ours.setLevel(theirs.level)
        ours.propagate = False
    install(Scheduler)
