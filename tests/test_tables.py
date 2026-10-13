"""The table catalog: every table in tables/ is valid and serves its committed table (no GPU)."""

import json

import pytest

import dsd_tuning.__main__ as cli
from dsd_tuning import tables
from dsd_tuning.controller import controller_from_env
from dsd_tuning.core import expand


@pytest.mark.parametrize("name", list(tables.load_all()))
def test_table_serves_its_committed_schedule_and_the_controller_accepts_it(name):
    env, argv = tables.serve_command(name, controller=True)
    spec = json.loads(argv[argv.index("--speculative-config") + 1])
    p = tables.get(name)
    assert spec["num_speculative_tokens_per_batch_size"] == json.loads((p["path"] / "schedule.json").read_text())
    assert argv[:3] == ["vllm", "serve", p["target"]] and env["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    lookup = expand(spec["num_speculative_tokens_per_batch_size"], p["max_num_seqs"])
    assert controller_from_env(lookup, spec["num_speculative_tokens"], env=env,
                               method=p["method"]) is not None


def test_serve_dry_run_prints_the_command(capsys):
    cli.main(["serve", "t4-eagle3", "--dry-run"])
    out = capsys.readouterr().out
    assert "vllm serve Qwen/Qwen3-4B" in out and "--max-model-len 4096" in out


def test_unknown_table_is_a_one_line_error():
    with pytest.raises(SystemExit, match="unknown table"):
        cli.main(["serve", "nope", "--dry-run"])


@pytest.mark.parametrize("gpu, want", [("Tesla T4", "t4-eagle3"), ("NVIDIA A10G", "a10g-eagle3")])
def test_serve_picks_the_preset_for_this_gpu(monkeypatch, gpu, want):
    monkeypatch.setattr(tables.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": gpu + "\n"})())
    assert tables.detect() == want


def test_unknown_gpu_points_to_tuning(monkeypatch):
    monkeypatch.setattr(tables.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": "NVIDIA H100\n"})())
    with pytest.raises(ValueError, match="docs/TUNING.md"):
        tables.detect()


def test_every_table_in_the_catalog_passes_the_checks():
    assert {name: p for name, p in tables.check_all().items() if p} == {}


def test_add_table_packages_a_tuning_run_and_checks_it(tmp_path, monkeypatch):
    monkeypatch.setenv("DSD_TABLES", str(tmp_path))
    src = tables.REPO / "experiments/results/tuner-v3-code"
    dest = tables.add(str(src), "t4-test", "Tesla T4", ["T4"], "float16", "test", "me", default=False)
    table = json.loads((dest / "table.json").read_text())
    assert table["max_model_len"] == 4096 and not tables.check("t4-test", dict(table, path=dest))
    assert list(tables.load_all()) == ["t4-test"]
    with pytest.raises(ValueError, match="already exists"):
        tables.add(str(src), "t4-test", "Tesla T4", ["T4"], "float16", "test", "me", default=False)


def test_check_catches_a_broken_table(tmp_path):
    t = dict(tables.get("t4-eagle3"), num_speculative_tokens=2, max_num_seqs=64)
    problems = tables.check("t4-eagle3", t)
    assert any("K=3 > num_speculative_tokens=2" in p for p in problems)
    assert any("1..64" in p for p in problems)
    assert tables.check("Bad Name", dict(t, path=tmp_path)) == ["name: use lower-case letters, digits and dashes"]


def test_add_table_without_costs_is_a_one_line_error(tmp_path, monkeypatch):
    monkeypatch.setenv("DSD_TABLES", str(tmp_path / "tables"))
    src = tmp_path / "run"
    src.mkdir()
    (src / "acceptance.json").write_text('{"accept_by_pos": [0.5]}')
    with pytest.raises(SystemExit, match=r"^\[dsd-tuner\] error: no costs_k\*\.json"):
        cli.main(["add-table", "--from", str(src), "--name", "x", "--gpu", "X", "--match", "X",
                  "--dtype", "float16", "--measured", "m", "--contributor", "c"])


def _catalog_with(tmp_path, monkeypatch, *names):
    """A catalog in tmp_path with copies of the named committed tables."""
    for n in names:
        (tmp_path / "tables" / n).mkdir(parents=True)
        for f in tables.get(n)["path"].glob("*.json"):
            (tmp_path / "tables" / n / f.name).write_text(f.read_text())
    monkeypatch.setenv("DSD_TABLES", str(tmp_path / "tables"))
    return tmp_path / "tables"


ADD = ["add-table", "--from", str(tables.REPO / "experiments/results/tuner-v3-code"), "--gpu", "Tesla T4",
       "--match", "T4", "--dtype", "float16", "--measured", "m", "--contributor", "c"]


def test_add_table_default_refuses_a_second_default_for_the_gpu(tmp_path, monkeypatch, capsys):
    """Issue #85: one default per GPU, as `tables --check` requires; nothing is written."""
    catalog = _catalog_with(tmp_path, monkeypatch, "t4-eagle3", "a10g-eagle3")
    monkeypatch.setattr(tables.shutil, "copy", lambda *a: pytest.fail("wrote files"))
    with pytest.raises(SystemExit, match=r"t4-eagle3 is already the default for GPU 'T4'.*"
                                         r"Drop --default, or first set \"default\": false in .*t4-eagle3/table.json"):
        cli.main([*ADD, "--name", "t4-mine", "--default"])
    assert "added" not in capsys.readouterr().out
    assert not (catalog / "t4-mine").exists()
    assert {n: p for n, p in tables.check_all().items() if p} == {}


def test_add_table_default_works_once_the_old_default_is_unset(tmp_path, monkeypatch):
    catalog = _catalog_with(tmp_path, monkeypatch, "t4-eagle3")
    meta = catalog / "t4-eagle3/table.json"
    meta.write_text(json.dumps(dict(json.loads(meta.read_text()), default=False)))
    cli.main([*ADD, "--name", "t4-mine", "--default"])
    assert {n: p for n, p in tables.check_all().items() if p} == {}
    monkeypatch.setattr(tables.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "Tesla T4\n"})())
    assert tables.detect() == "t4-mine"


def test_add_table_without_default_next_to_a_default_is_fine(tmp_path, monkeypatch):
    _catalog_with(tmp_path, monkeypatch, "t4-eagle3")
    cli.main([*ADD, "--name", "t4-mine"])
    assert sorted(tables.load_all()) == ["t4-eagle3", "t4-mine"] and not any(tables.check_all().values())


def test_add_table_runs_the_catalog_check_before_writing(tmp_path, monkeypatch):
    """A table that `tables --check` would fail is refused before any file is written."""
    catalog = _catalog_with(tmp_path, monkeypatch, "t4-eagle3")
    monkeypatch.setattr(tables.shutil, "copy", lambda *a: pytest.fail("wrote files"))
    with pytest.raises(SystemExit, match=r"tables --check` would fail: Bad_Name: name: use lower-case"):
        cli.main([*ADD, "--name", "Bad_Name"])
    (catalog / "t4-eagle3/schedule.json").write_text("[]")       # a broken table already in the catalog
    with pytest.raises(SystemExit, match=r"t4-eagle3: schedule.json is empty"):
        cli.main([*ADD, "--name", "t4-mine"])
    assert sorted(p.name for p in catalog.iterdir()) == ["t4-eagle3"]


def test_empty_schedule_is_reported_not_raised(tmp_path):
    t = tables.get("t4-eagle3")
    for f in t["path"].glob("*.json"):
        (tmp_path / f.name).write_text(f.read_text())
    (tmp_path / "schedule.json").write_text("[]")
    assert tables.check("t4-eagle3", dict(t, path=tmp_path)) == ["schedule.json is empty"]


def test_the_catalog_is_found_outside_the_clone(tmp_path, monkeypatch):
    """A non-editable install has no tables/ next to the code: use ./tables or DSD_TABLES."""
    monkeypatch.setattr(tables, "REPO", tmp_path / "site-packages")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DSD_TABLES", raising=False)
    with pytest.raises(ValueError, match="DSD_TABLES"):
        tables.load_all()
    (tmp_path / "tables").mkdir()
    assert tables.tables_dir() == tmp_path / "tables"


def test_serve_uses_the_tables_max_model_len(tmp_path, monkeypatch):
    src = tables.get("t4-eagle3")["path"]
    dest = tmp_path / "t4-long"
    dest.mkdir()
    for f in src.glob("*.json"):
        (dest / f.name).write_text(f.read_text())
    t = json.loads((dest / "table.json").read_text())
    (dest / "table.json").write_text(json.dumps(dict(t, max_model_len=8192)))
    monkeypatch.setenv("DSD_TABLES", str(tmp_path))
    _, argv = tables.serve_command("t4-long")
    assert argv[argv.index("--max-model-len") + 1] == "8192"


@pytest.mark.parametrize("flags, controller", [([], True), (["--controller"], True), (["--no-controller"], False)])
def test_serve_runs_the_controller_by_default(capsys, flags, controller):
    cli.main(["serve", "t4-eagle3", "--dry-run", *flags])
    assert ("VLLM_DSD_CONTROLLER=1" in capsys.readouterr().out) == controller
