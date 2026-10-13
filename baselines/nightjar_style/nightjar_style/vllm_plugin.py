"""vLLM plugin: answer the Dynamic SD lookup with the Nightjar-style controller.

A no-op unless VLLM_NJ_CONTROLLER=1. Then it wraps three Scheduler methods (no edits to vLLM):
__init__ swaps the Dynamic SD lookup for one that asks the controller, make_spec_decoding_stats passes
each request's drafted/accepted counts, and update_from_output closes a step (its wall time is the loss).
The server must start with a Dynamic SD table: its lengths are the arms, since only they have CUDA graphs.

  VLLM_NJ_CONTROLLER=1  VLLM_NJ_SWITCH_COST_MS=0  VLLM_NJ_SEED=0  VLLM_NJ_LOG_STEPS=0
"""

from __future__ import annotations

import functools
import logging
import os

from .controller import NightjarStyleController

logger = logging.getLogger(__name__)
ENV = "VLLM_NJ_CONTROLLER"


class _Lookup(list):
    def __init__(self, lookup, controller):
        super().__init__(lookup)
        self.controller = controller

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.controller.choose_k(index)
        return super().__getitem__(index)


def _controller(lookup: list[int], num_spec_tokens: int) -> NightjarStyleController:
    return NightjarStyleController(
        lookup, num_spec_tokens, switch_cost_ms=float(os.environ.get("VLLM_NJ_SWITCH_COST_MS", "0")),
        seed=int(os.environ.get("VLLM_NJ_SEED", "0")),
        log_steps=int(os.environ.get("VLLM_NJ_LOG_STEPS", "0")))


def install(cls: type) -> None:
    if getattr(cls, "_nj_installed", False):
        return
    orig_init, orig_stats, orig_update = cls.__init__, cls.make_spec_decoding_stats, cls.update_from_output

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.nj_controller = None
        lookup = getattr(self, "dynamic_sd_lookup", None)
        if lookup is None:
            raise RuntimeError(f"{ENV} is set but the server has no Dynamic SD table")
        c = _controller(list(lookup), self.num_spec_tokens)
        logger.info(c.describe())
        self.nj_controller = c
        self.dynamic_sd_lookup = _Lookup(lookup, c)

    @functools.wraps(orig_stats)
    def make_spec_decoding_stats(self, stats, num_draft_tokens, num_accepted_tokens, *args, **kwargs):
        c = getattr(self, "nj_controller", None)
        if c is not None:
            c.observe(num_draft_tokens, num_accepted_tokens)
        return orig_stats(self, stats, num_draft_tokens, num_accepted_tokens, *args, **kwargs)

    @functools.wraps(orig_update)
    def update_from_output(self, scheduler_output, model_runner_output):
        out = orig_update(self, scheduler_output, model_runner_output)
        c = getattr(self, "nj_controller", None)
        if c is not None and scheduler_output.num_scheduled_tokens:
            c.end_step()
        return out

    cls.__init__, cls.make_spec_decoding_stats, cls.update_from_output = __init__, make_spec_decoding_stats, update_from_output
    cls._nj_installed = True


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
