"""The vLLM plugin's hooks, on a stand-in for vLLM's Scheduler (no GPU, no vLLM)."""

from types import SimpleNamespace

import pytest

from dsd_tuning.controller.vllm_plugin import ControllerLookup, install

TABLE = [3] * 17 + [2] * 16 + [1] * 32       # lookup[B] for B = 0..64 (index 0 unused)


def make_scheduler_class():
    """A fresh stand-in with the three methods and attributes the plugin uses."""

    class Scheduler:
        def __init__(self, method="eagle3", lookup=TABLE):
            self.vllm_config = SimpleNamespace(speculative_config=SimpleNamespace(method=method))
            self.num_spec_tokens = 3
            self.dynamic_sd_lookup = list(lookup) if lookup is not None else None
            self.requests = {}
            self.stats_calls = 0

        def make_spec_decoding_stats(self, stats, num_draft_tokens, num_accepted_tokens,
                                     num_invalid_spec_tokens, request_id):
            self.stats_calls += 1
            return stats

        def update_from_output(self, scheduler_output, model_runner_output):
            return "vllm-output"

        def schedule_k(self, batch_size):          # what vLLM's schedule() does
            return self.dynamic_sd_lookup[batch_size]

    return Scheduler


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("VLLM_DSD_CONTROLLER", "1")
    monkeypatch.setenv("VLLM_DSD_HYSTERESIS", "0")


def test_k_comes_from_the_controller_and_the_table_is_kept(enabled, caplog):
    cls = make_scheduler_class()
    install(cls)
    with caplog.at_level("INFO"):
        s = cls()
    assert "Dynamic SD controller enabled: hysteresis=0" in caplog.text   # logged once at startup
    assert isinstance(s.dynamic_sd_lookup, ControllerLookup)
    assert list(s.dynamic_sd_lookup) == TABLE                  # contents unchanged
    assert [s.schedule_k(b) for b in (1, 16, 17, 33, 64)] == [3, 3, 2, 1, 1]
    assert s.dynamic_sd_controller.stats["steps"] == 5         # each lookup went to the controller


def test_accepted_counts_reach_the_controller(enabled):
    cls = make_scheduler_class()
    install(cls)
    s = cls()
    s.make_spec_decoding_stats(None, 3, 2, None, "r1")
    assert s.stats_calls == 1                                  # vLLM's own stats still run
    assert s.dynamic_sd_controller.drafted[:3] == [1, 1, 1]


def test_steps_without_drafts_are_counted_only_for_ngram(enabled):
    def step(s):
        s.schedule_k(2)                         # K > 0 this step, as in vLLM
        s.requests = {"decode": SimpleNamespace(is_finished=lambda: False, num_output_tokens=5),
                      "prefill": SimpleNamespace(is_finished=lambda: False, num_output_tokens=0)}
        out = SimpleNamespace(scheduled_spec_decode_tokens={},
                              num_scheduled_tokens={"decode": 1, "prefill": 30})
        runner = SimpleNamespace(sampled_token_ids=[[7], [8]], req_id_to_index={"decode": 0, "prefill": 1})
        assert s.update_from_output(out, runner) == "vllm-output"

    cls = make_scheduler_class()
    install(cls)
    ngram, eagle = cls(method="ngram"), cls(method="eagle3")
    step(ngram)
    step(eagle)
    assert ngram.dynamic_sd_controller.decode_steps == 1       # decode only, not prefill
    assert eagle.dynamic_sd_controller.decode_steps == 0


def test_disabled_controller_leaves_vllm_unchanged(monkeypatch):
    monkeypatch.setenv("VLLM_DSD_CONTROLLER", "0")
    cls = make_scheduler_class()
    install(cls)
    s = cls()
    assert s.dynamic_sd_controller is None
    assert type(s.dynamic_sd_lookup) is list and s.schedule_k(33) == 1
    assert cls(lookup=None).dynamic_sd_controller is None     # no Dynamic SD table at all


def test_missing_hook_point_fails_at_startup(enabled):
    class OtherVllm:
        def __init__(self):
            pass

    with pytest.raises(RuntimeError, match="make_spec_decoding_stats"):
        install(OtherVllm)


def test_installing_twice_wraps_once(enabled):
    cls = make_scheduler_class()
    install(cls)
    install(cls)                                               # e.g. two plugin entry points
    s = cls()
    s.make_spec_decoding_stats(None, 3, 2, None, "r1")
    assert s.stats_calls == 1
    assert s.dynamic_sd_controller.drafted[:3] == [1, 1, 1]    # observed once, not twice


def test_register_twice_is_harmless(enabled, monkeypatch):
    """vLLM may call register() more than once per process; hooks and log handlers are added once."""
    import logging
    import sys
    import types

    from dsd_tuning.controller import vllm_plugin

    cls = make_scheduler_class()
    stub = types.ModuleType("vllm.v1.core.sched.scheduler")
    stub.Scheduler = cls
    for name in ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.sched"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.scheduler", stub)
    vllm_log, ours = logging.getLogger("vllm"), logging.getLogger("dsd_tuning")
    handler = logging.NullHandler()
    monkeypatch.setattr(vllm_log, "handlers", [handler])
    monkeypatch.setattr(ours, "handlers", [])
    monkeypatch.setattr(ours, "propagate", True)
    monkeypatch.setattr(ours, "level", ours.level)             # register() copies vLLM's level

    vllm_plugin.register()
    vllm_plugin.register()
    assert ours.handlers == [handler]
    s = cls()
    s.make_spec_decoding_stats(None, 3, 2, None, "r1")
    assert s.dynamic_sd_controller.drafted[:3] == [1, 1, 1]
