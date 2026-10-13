"""vLLM plugin: attach the Dynamic SD controller to vLLM's scheduler.

Registered as a ``vllm.general_plugins`` entry point, so vLLM calls register()
in every process before it builds the scheduler. Nothing changes unless
VLLM_DSD_CONTROLLER is set; then three Scheduler methods are wrapped (no edits
to vLLM's source):

- ``__init__`` builds the controller and swaps in ControllerLookup, so
  ``lookup[batch_size]`` asks the controller for K;
- ``make_spec_decoding_stats`` passes each request's drafted/accepted counts;
- ``update_from_output`` counts decode steps without drafts (n-gram only).

Written against vLLM 353e491ef; a missing hook point fails at start-up.
"""

from __future__ import annotations

import functools
import logging
import os

from .controller import DynamicSDController, controller_from_env

logger = logging.getLogger(__name__)

ENV_SWITCH = "VLLM_DSD_CONTROLLER"
HOOK_METHODS = ("__init__", "make_spec_decoding_stats", "update_from_output")


class ControllerLookup(list):
    """The Dynamic SD lookup vLLM indexes by batch size, answered by the controller.

    Keeps the original table as list contents, so anything else reading the
    lookup sees vLLM's values; only ``lookup[batch_size]`` goes to the controller.
    """

    def __init__(self, lookup: list[int], controller: DynamicSDController):
        super().__init__(lookup)
        self.controller = controller

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.controller.choose_k(index)
        return super().__getitem__(index)


def install(scheduler_cls: type) -> None:
    """Add the controller hooks to a scheduler class (idempotent)."""
    if getattr(scheduler_cls, "_dsd_controller_installed", False):
        return
    missing = [m for m in HOOK_METHODS if not hasattr(scheduler_cls, m)]
    if missing:
        raise RuntimeError(f"{ENV_SWITCH} is set but this vLLM's Scheduler has no {missing}; "
                           "the dsd-tuning plugin supports vLLM 353e491ef")

    orig_init = scheduler_cls.__init__
    orig_stats = scheduler_cls.make_spec_decoding_stats
    orig_update = scheduler_cls.update_from_output

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.dynamic_sd_controller = None
        if not hasattr(self, "dynamic_sd_lookup"):
            raise RuntimeError(f"{ENV_SWITCH} is set but this vLLM's Scheduler has no "
                               "dynamic_sd_lookup; the dsd-tuning plugin supports vLLM 353e491ef")
        lookup = self.dynamic_sd_lookup
        if lookup is None:          # no Dynamic SD table in --speculative-config
            return
        spec = self.vllm_config.speculative_config
        controller = controller_from_env(list(lookup), self.num_spec_tokens,
                                         method=getattr(spec, "method", None))
        if controller is not None:
            logger.info(controller.describe())
            self.dynamic_sd_controller = controller
            self.dynamic_sd_lookup = ControllerLookup(lookup, controller)

    @functools.wraps(orig_stats)
    def make_spec_decoding_stats(self, spec_decoding_stats, num_draft_tokens,
                                 num_accepted_tokens, *args, **kwargs):
        controller = getattr(self, "dynamic_sd_controller", None)
        if controller is not None:
            controller.observe(num_draft_tokens, num_accepted_tokens)
        return orig_stats(self, spec_decoding_stats, num_draft_tokens,
                          num_accepted_tokens, *args, **kwargs)

    @functools.wraps(orig_update)
    def update_from_output(self, scheduler_output, model_runner_output):
        controller = getattr(self, "dynamic_sd_controller", None)
        if controller is not None and controller.track_proposals:
            _observe_no_proposals(self, controller, scheduler_output, model_runner_output)
        return orig_update(self, scheduler_output, model_runner_output)

    scheduler_cls.__init__ = __init__
    scheduler_cls.make_spec_decoding_stats = make_spec_decoding_stats
    scheduler_cls.update_from_output = update_from_output
    scheduler_cls._dsd_controller_installed = True


def _observe_no_proposals(scheduler, controller, scheduler_output, model_runner_output) -> None:
    """Count decode steps that sampled tokens without drafts (proposal rate, n-gram)."""
    sampled = model_runner_output.sampled_token_ids
    if not sampled:
        return
    drafts = scheduler_output.scheduled_spec_decode_tokens
    index = model_runner_output.req_id_to_index
    for req_id in scheduler_output.num_scheduled_tokens:
        request = scheduler.requests.get(req_id)
        if request is None or request.is_finished() or drafts.get(req_id):
            continue
        i = index.get(req_id)
        if i is not None and sampled[i] and request.num_output_tokens > 0:  # not prefill
            controller.observe_no_proposal()


def register() -> None:
    """Entry point vLLM calls in each process; a no-op unless the controller is enabled."""
    if not os.environ.get(ENV_SWITCH):
        return
    from vllm.v1.core.sched.scheduler import Scheduler  # also sets up vLLM's "vllm" logger

    # vLLM only configures its own "vllm" logger; send this package's records
    # through the same handlers and level, so they appear in vLLM's log.
    ours, theirs = logging.getLogger(__package__.split(".")[0]), logging.getLogger("vllm")
    if not ours.handlers:
        for handler in theirs.handlers:
            ours.addHandler(handler)
        ours.setLevel(theirs.level)
        ours.propagate = False
    install(Scheduler)
