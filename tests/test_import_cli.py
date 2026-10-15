"""Import lifecycle with fake remote boundaries; never contacts Chalk."""
from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from semantic_ids import cli, importing, import_cli, project
from semantic_ids.project import Target


@pytest.fixture
def setup(tmp_path, monkeypatch):
    target = Target("ftqa-test", "https://target.example.test")
    state = {"format_version": 1, "target": asdict(target), "snapshot": "test", "datasets": {},
             "training": {"tokenizer": {"model_version": 1}}}
    monkeypatch.setattr(project, "RECEIPT", tmp_path / "project.json")
    monkeypatch.setattr(import_cli, "ROOT", tmp_path)
    monkeypatch.setattr(import_cli, "IMPORT_RECEIPT", tmp_path / "imported-recommender.json")
    for key in ("CHALK_ENVIRONMENT", "CHALK_ENVIRONMENT_ID", "CHALK_API_SERVER"):
        monkeypatch.delenv(key, raising=False)
    project.write_project(state)
    calls = []

    def download(**kwargs):
        calls.append("download")
        path = Path(kwargs["download_dir"]) / "weights.pt"
        path.write_bytes(b"immutable test weights")
        return SimpleNamespace(downloaded_model_files=[str(path)], downloaded_additional_files=[])

    summary = {"catalog_count": 2, "catalog_fingerprint": "fingerprint", "sample_history_ids": [4, 9],
               "source_kind": "synthetic_demo"}
    monkeypatch.setattr(importing, "source_client", lambda *args: SimpleNamespace(download_model_artifact=download))
    monkeypatch.setattr(importing, "validate_artifact", lambda artifact: summary)
    monkeypatch.setattr(importing, "registration_kwargs", lambda *args, **kwargs: {"metadata": {}})
    monkeypatch.setattr(cli, "_ensure_namespace", lambda *args: calls.append("namespace"))

    def register(**kwargs):
        calls.append("register")
        return SimpleNamespace(model_name="semantic-id-recommender", model_version=7,
                               artifact=SimpleNamespace(id="imported-artifact"))

    destination = SimpleNamespace(register_model_version=register)
    monkeypatch.setattr(Target, "client", lambda self: destination)
    args = Namespace(source_environment="source-env", source_api_server="https://source.example.test",
                     source_model="source-recommender", source_version=4, dry_run=False,
                     model_version=None, retry_registration=False)
    return target, state, calls, destination, args


def test_import_separate_receipt_and_idempotent_rerun(setup):
    target, state, calls, _, args = setup
    original = project.RECEIPT.read_bytes()
    import_cli.import_recommender(args)
    saved = import_cli.read_import(target)
    assert saved["model_version"] == 7
    assert saved["source"]["model_version"] == 4
    assert saved["source_kind"] == "synthetic_demo"
    assert project.RECEIPT.read_bytes() == original
    import_cli.import_recommender(args)
    assert calls == ["download", "namespace", "register"]
    assert "recommender" not in state["training"]


def test_dry_run_never_registers_or_changes_receipts(setup):
    _, _, calls, _, args = setup
    args.dry_run = True
    original = project.RECEIPT.read_bytes()
    import_cli.import_recommender(args)
    assert calls == ["download"]
    assert not import_cli.IMPORT_RECEIPT.exists()
    assert project.RECEIPT.read_bytes() == original


def test_uncertain_registration_requires_verified_reconnection(setup):
    target, _, calls, destination, args = setup
    def interrupted(**kwargs):
        calls.append("register")
        raise RuntimeError("connection interrupted after registration")
    destination.register_model_version = interrupted
    with pytest.raises(RuntimeError, match="Inspect the destination"):
        import_cli.import_recommender(args)
    saved = import_cli.read_import(target)
    with pytest.raises(ValueError, match="registration was interrupted"):
        import_cli.import_recommender(args)
    assert calls.count("register") == 1
    args.model_version = 7
    destination.get_model_version = lambda **kw: SimpleNamespace(
        model_name="semantic-id-recommender", version=7,
        model_artifact=SimpleNamespace(metadata={"semantic_ids_import_id": "wrong"}))
    with pytest.raises(ValueError, match="exact imported artifact"):
        import_cli.import_recommender(args)
    destination.get_model_version = lambda **kw: SimpleNamespace(
        model_name="semantic-id-recommender", version=7,
        model_artifact=SimpleNamespace(metadata={"semantic_ids_import_id": saved["import_id"]}))
    import_cli.import_recommender(args)
    assert import_cli.read_import(target)["model_version"] == 7
    assert calls.count("register") == 1


def test_catalog_failure_happens_before_destination_mutations(setup, monkeypatch):
    _, _, calls, _, args = setup
    def reject(artifact):
        raise ValueError("catalog mismatch")
    monkeypatch.setattr(importing, "validate_artifact", reject)
    with pytest.raises(ValueError, match="catalog mismatch"):
        import_cli.import_recommender(args)
    assert calls == ["download"]
    assert not import_cli.IMPORT_RECEIPT.exists()


def test_recommend_selects_imported_without_claiming_native_training(setup):
    target, state, _, _, args = setup
    import_cli.import_recommender(args)
    assert import_cli.selected_recommender(state, target)["model_version"] == 7
    with pytest.raises(ValueError, match="Train the recommender"):
        import_cli.selected_recommender(state, target, "trained")
    state["training"]["recommender"] = {"model_version": 8}
    assert import_cli.selected_recommender(state, target)["model_version"] == 8
    assert import_cli.selected_recommender(state, target, "imported")["model_version"] == 7


def test_recommend_sends_imported_version_and_selected_branch(setup, monkeypatch, capsys):
    _, _, _, destination, args = setup
    import_cli.import_recommender(args)
    requests = []
    destination.query = lambda **kwargs: requests.append(kwargs) or SimpleNamespace(
        errors=[], get_feature_value=lambda name: '{"items": []}')
    cli.recommend(cli.parser().parse_args(["recommend", "--model-source", "imported", "--history", "4", "9",
                                          "--branch", "semantic"]))
    assert requests[0]["input"]["recommendation.model_version"] == 7
    assert requests[0]["branch"] == "semantic"


def test_cross_environment_receipt_rejected(setup):
    target, _, _, _, args = setup
    import_cli.import_recommender(args)
    with pytest.raises(ValueError, match="different Chalk target"):
        import_cli.read_import(Target("elsewhere", target.api_server))


def test_concurrent_import_is_rejected(setup):
    *_, args = setup
    with import_cli._import_lock():
        with pytest.raises(ValueError, match="Another recommender import"):
            import_cli.import_recommender(args)
