"""Tests for the dsd-tuner command (no GPU: the profiling steps are faked)."""

import json
from pathlib import Path

import pytest

import dsd_tuning.__main__ as cli

REPO = Path(__file__).resolve().parents[2]
V3 = REPO / "experiments" / "results" / "tuner-v3-code"


def test_schedule_command_writes_table_and_config(tmp_path, capsys):
    cli.main(["schedule", "--costs", str(V3 / "costs_k*.json"),
              "--acceptance", str(V3 / "acceptance.json"), "--max-num-seqs", "128",
              "--out", str(tmp_path)])
    table = json.loads((tmp_path / "schedule.json").read_text())
    cfg = json.loads((tmp_path / "speculative_config.json").read_text())
    assert table == json.loads((V3 / "schedule.json").read_text())
    assert cfg["num_speculative_tokens_per_batch_size"] == table
    assert cfg["model"] == "AngelSlim/Qwen3-4B_eagle3"          # read from acceptance.json
    assert "--speculative-config" in capsys.readouterr().out


def test_screen_command_reports_best_fixed_k(capsys):
    cli.main(["screen", "--costs", str(V3 / "costs_k*.json"),
              "--acceptance", str(V3 / "acceptance.json"), "--max-num-seqs", "128"])
    out = capsys.readouterr().out
    assert "best single fixed K" in out and "K=3:" in out


def _fake_run(v3_costs=V3):
    """Stand-in for the GPU steps: copy measured profiles to where they'd be written."""
    calls = []

    def run(cmd, env, log):
        calls.append((cmd, env))
        out = Path(cmd[cmd.index("--out") + 1])
        if "profile_costs.py" in cmd[1]:
            k = cmd[cmd.index("--k") + 1]
            out.write_text((v3_costs / f"costs_k{k}.json").read_text())
        else:
            out.write_text((V3 / "acceptance.json").read_text())
    return run, calls


def test_run_command_profiles_every_k_then_builds_the_table(tmp_path, monkeypatch):
    run, calls = _fake_run()
    monkeypatch.setattr(cli, "_run", run)
    cli.main(["run", "--prompts-file", "p.jsonl", "--isl", "32", "--max-num-seqs", "128",
              "--temperature", "1.0", "--out", str(tmp_path)])
    cost_calls = [c for c, _ in calls if "profile_costs.py" in c[1]]
    assert [c[c.index("--k") + 1] for c in cost_calls] == ["0", "1", "2", "3"]
    assert all(c[c.index("--isl") + 1] == "32" for c in cost_calls)
    acc_cmd = [c for c, _ in calls if "profile_acceptance.py" in c[1]][0]
    assert acc_cmd[acc_cmd.index("--temperature") + 1] == "1.0"
    assert calls[0][1]["TARGET_MODEL"] == "Qwen/Qwen3-4B"      # model passed via env
    assert json.loads((tmp_path / "schedule.json").read_text()) == [[1, 18, 3], [19, 43, 2],
                                                                    [44, 128, 1]]


def test_recalibrate_reuses_costs_and_only_measures_acceptance(tmp_path, monkeypatch):
    run, calls = _fake_run()
    monkeypatch.setattr(cli, "_run", run)
    cli.main(["recalibrate", "--costs", str(V3 / "costs_k*.json"), "--prompts-file", "p.jsonl",
              "--max-num-seqs", "128", "--out", str(tmp_path)])
    assert len(calls) == 1 and "profile_acceptance.py" in calls[0][0][1]
    assert (tmp_path / "speculative_config.json").exists()


def test_prompts_or_sweep_is_required():
    with pytest.raises(SystemExit):
        cli.main(["recalibrate", "--costs", "x", "--out", "y"])


# ------------------------------------------------------------- robustness
def test_bad_batch_sizes_fail_before_any_gpu_work(monkeypatch):
    monkeypatch.setattr(cli, "_run", lambda *a: pytest.fail("profiling must not start"))
    for bad in ["1,8,4", "0,1", "1,,2"]:
        with pytest.raises(SystemExit) as e:
            cli.main(["run", "--prompts-file", "p", "--out", "o", "--batch-sizes", bad])
        assert e.value.code == 2


@pytest.mark.parametrize("costs, acceptance, match", [
    ("/nonexistent/costs_k*.json", str(V3 / "acceptance.json"), "no cost files match"),
    (str(V3 / "costs_k*.json"), "/nonexistent/acceptance.json", "No such file"),
])
def test_missing_files_give_one_line_errors(costs, acceptance, match):
    with pytest.raises(SystemExit) as e:
        cli.main(["schedule", "--costs", costs, "--acceptance", acceptance])
    assert str(e.value.code).startswith("[dsd-tuner] error:") and match in str(e.value.code)



@pytest.mark.parametrize("table", sorted(p.name for p in (REPO / "tables").iterdir() if p.is_dir()))
def test_cli_and_script_build_the_committed_table(table, tmp_path):
    """Issue #21: `dsd-tuner schedule` and make_schedule.py must agree, on every catalog table."""
    from dsd_tuning.tuner.make_schedule import main as make_schedule
    t = REPO / "tables" / table
    args = ["--costs", str(t / "costs_k*.json"), "--acceptance", str(t / "acceptance.json"),
            "--max-num-seqs", "128"]
    cli.main(["schedule", *args, "--out", str(tmp_path / "cli")])
    make_schedule([*args, "--out", str(tmp_path / "script")])
    committed = json.loads((t / "schedule.json").read_text())
    assert json.loads((tmp_path / "cli" / "schedule.json").read_text()) == committed
    assert json.loads((tmp_path / "script" / "schedule.json").read_text()) == committed


def test_stored_tokens_per_step_is_validated(tmp_path):
    from dsd_tuning.core import load_tokens_per_step
    f = tmp_path / "acceptance.json"
    f.write_text(json.dumps({"accept_by_pos": [0.6], "tokens_per_step_by_k": {"0": 1.0, "1": 0.5}}))
    with pytest.raises(ValueError, match="must be >= 1"):
        load_tokens_per_step(f)
    f.write_text(json.dumps({"accept_by_pos": [0.6], "proposal_rate": 0.5,
                             "tokens_per_step_by_k": {"0": 1.0, "1": 1.6}}))
    assert load_tokens_per_step(f) == ({0: 1.0, 1: 1.3}, 0.5)   # n-gram: recomputed with p
