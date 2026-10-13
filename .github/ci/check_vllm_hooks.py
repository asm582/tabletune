"""Check that vLLM at the pinned commit still has what dsd_tuning/controller/vllm_plugin.py hooks.

The plugin wraps three Scheduler methods and replaces the Dynamic SD lookup; it
also relies on the engine loading plugins before it builds the scheduler, and on
AsyncScheduler not overriding the wrapped methods. This reads the source files
(no vLLM install needed) so a vLLM upgrade that moves a hook point fails CI.

Usage: python .github/ci/check_vllm_hooks.py <path/to/vllm/clone>
"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
sched = (root / "vllm/v1/core/sched/scheduler.py").read_text()
async_sched = (root / "vllm/v1/core/sched/async_scheduler.py").read_text()
core = (root / "vllm/v1/engine/core.py").read_text()

checks = {
    "Scheduler.__init__ builds dynamic_sd_lookup": "self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(" in sched,
    "schedule() indexes the lookup by batch size": re.search(r"self\.dynamic_sd_lookup\[\s*len\(num_scheduled_tokens\)", sched) is not None,
    "make_spec_decoding_stats(stats, num_draft_tokens, num_accepted_tokens, ...)":
        re.search(r"def make_spec_decoding_stats\(\s*self,\s*spec_decoding_stats[^,]*,\s*num_draft_tokens[^,]*,\s*num_accepted_tokens", sched) is not None,
    "update_from_output(scheduler_output, model_runner_output)":
        re.search(r"def update_from_output\(\s*self,\s*scheduler_output[^,]*,\s*model_runner_output", sched) is not None,
    "scheduler_output.scheduled_spec_decode_tokens / req_id_to_index / sampled_token_ids":
        all(t in sched for t in ("scheduled_spec_decode_tokens", "req_id_to_index", "sampled_token_ids")),
    "AsyncScheduler does not override the wrapped methods":
        not re.search(r"def (make_spec_decoding_stats|update_from_output)\(", async_sched),
    "EngineCore loads general plugins": "load_general_plugins()" in core,
}
for name, ok in checks.items():
    print(("ok   " if ok else "FAIL ") + name)
sys.exit(0 if all(checks.values()) else 1)
