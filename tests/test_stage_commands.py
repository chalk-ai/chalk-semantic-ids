"""Direct stage commands must show accurate help without starting remote work."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import runpy
import sys

import pytest

from semantic_ids import cli, project, stages


@pytest.mark.parametrize("script", [
    "train/prepare.py", "train/run.py", "host/run.py", "host/verify.py",
    "materialize/run.py", "train-recommender/run.py", "serve/run.py", "serve/import_model.py",
])
def test_direct_help_never_reads_receipts_or_loads_a_stage(script, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("--help must exit before loading stage code or saved state")

    monkeypatch.setattr(project, "read_project", forbidden)
    monkeypatch.setattr(cli, "read_project", forbidden)
    monkeypatch.setattr(stages, "load_stage", forbidden)
    monkeypatch.setattr(sys, "argv", [script, "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(project.ROOT / script), run_name="__main__")
    assert stopped.value.code == 0
    usage = capsys.readouterr().out.splitlines()[0]
    assert usage.startswith("usage: ")
    # A direct stage command takes flags, not the old CLI's subcommand name.
    assert usage.split()[2] == "[-h]"


def test_importing_shared_cli_does_not_import_hosting_stage(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Importing the CLI must not load a stage")

    monkeypatch.setattr(stages, "load_stage", forbidden)
    spec = spec_from_file_location("semantic_ids._cli_import_check", project.ROOT / "semantic_ids" / "cli.py")
    spec.loader.exec_module(module_from_spec(spec))


def test_chalk_apply_excludes_remote_action_scripts():
    root = Path(__file__).resolve().parents[1]
    ignores = set((root / ".chalkignore").read_text().splitlines())
    assert {"train", "host", "materialize", "train-recommender", "serve", "semantic_ids"} <= ignores
