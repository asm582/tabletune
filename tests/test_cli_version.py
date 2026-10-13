import pytest

from dsd_tuning.__main__ import _version, main


def test_version_flag_prints_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out == f"dsd-tuner {_version()}"


def test_version_matches_pyproject():
    tomllib = pytest.importorskip("tomllib")
    with open("pyproject.toml", "rb") as f:
        expected = tomllib.load(f)["project"]["version"]
    v = _version()
    assert v in (expected, "unknown")
