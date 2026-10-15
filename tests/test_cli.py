"""No remote executions: fake service boundaries exercise stage transitions."""
from argparse import Namespace
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from semantic_ids import cli, project
from semantic_ids.project import Target


@pytest.fixture
def receipt(tmp_path, monkeypatch):
    path = tmp_path / "project.json"
    monkeypatch.setattr(project, "RECEIPT", path)
    monkeypatch.setattr(cli, "RECEIPT", path)
    from semantic_ids import import_cli
    monkeypatch.setattr(import_cli, "IMPORT_RECEIPT", tmp_path / "imported-recommender.json")
    for key in ("CHALK_ENVIRONMENT", "CHALK_ENVIRONMENT_ID", "CHALK_API_SERVER"):
        monkeypatch.delenv(key, raising=False)
    return path


def saved_project():
    return {"format_version": 1, "target": {"environment": "ftqa-test", "api_server": "https://api.example.test"},
            "snapshot": "test", "datasets": {
                "titles": {"name": "titles-test", "revision_id": "titles-revision", "rows": 2000},
                "examples": {"name": "examples-test", "revision_id": "examples-revision"}},
            "training": {"tokenizer": {"model_version": 2}}}


def test_saved_target_rejects_another_environment(receipt, monkeypatch):
    project.write_project(saved_project())
    monkeypatch.setenv("CHALK_ENVIRONMENT", "conference-demo")
    with pytest.raises(ValueError, match="differs"):
        project.read_project()


def test_partial_prepare_reuses_completed_upload(receipt, tmp_path, monkeypatch):
    import chalkcompute

    titles = tmp_path / "titles.csv"
    watches = tmp_path / "watches.tsv"
    titles.write_text("1,one\n2,two\n3,three\n4,four\n")
    watches.write_text("u\t1 2 3 4\n")
    target = Target("ftqa-test", "https://api.example.test")
    monkeypatch.setattr(Target, "selected", lambda *args: target)
    monkeypatch.setattr(Target, "client", lambda self: object())
    calls = []

    def upload(name, rows):
        calls.append(name)
        if len(calls) == 2:
            raise RuntimeError("interrupted examples upload")
        return SimpleNamespace(revision_id=name + "-revision")

    monkeypatch.setattr(chalkcompute, "DatasetClient", lambda **kw: SimpleNamespace(upload=upload))
    args = Namespace(environment="ftqa-test", api_server=None, titles=titles, watches=watches, max_users=1, seed=0)
    with pytest.raises(RuntimeError, match="interrupted"):
        cli.prepare(args)
    first = json.loads(receipt.read_text())["datasets"]["titles"]
    cli.prepare(args)
    result = json.loads(receipt.read_text())
    assert result["datasets"]["titles"] == first
    assert len([name for name in calls if "titles" in name]) == 1
    assert result["datasets"]["examples"]["rows"] == 3


@pytest.mark.parametrize("branch", [None, "semantic"])
def test_materialization_accepts_arrow_lists_and_checks_complete_catalog(receipt, monkeypatch, branch):
    import chalkcompute

    state = saved_project()
    project.write_project(state)
    rows = [{"id": 0, "title": "zero"}, {"id": 1, "title": "one"}]
    frame = pd.DataFrame({"video.id": [0, 1], "video.title": ["zero", "one"],
                          "video.tokenizer_version": [2, 2],
                          "video.codes": [np.array([1, 2, 3]), np.array([1, 2, 3])]})
    queried = []
    dataset = SimpleNamespace(errors=[], revisions=[SimpleNamespace(revision_id="catalog-revision")],
                              to_pandas=lambda **kw: frame)
    monkeypatch.setattr(Target, "client", lambda self: SimpleNamespace(
        offline_query=lambda **kw: queried.append(kw) or dataset))
    monkeypatch.setattr(chalkcompute, "DatasetClient", lambda **kw: SimpleNamespace(
        read=lambda revision: SimpleNamespace(to_pylist=lambda: rows)))
    args = cli.parser().parse_args(["materialize", *(["--branch", branch] if branch else [])])
    cli.materialize(args)
    result = json.loads(receipt.read_text())
    assert result["datasets"]["catalog"]["rows"] == 2
    assert queried[0]["input"]["video.tokenizer_version"] == [2, 2]
    assert queried[0]["recompute_features"] is True
    assert "video.codes" in queried[0]["required_output"]
    assert queried[0]["branch"] == branch
    assert result["target"] == state["target"]


@pytest.mark.parametrize("existing_catalog", [False, True])
@pytest.mark.parametrize("missing_row", [False, True])
def test_materialization_sample_limits_inference_without_changing_training_catalog(
    receipt, monkeypatch, capsys, existing_catalog, missing_row
):
    import chalkcompute
    import pyarrow as pa

    state = saved_project()
    if existing_catalog:
        state["datasets"]["catalog"] = {"revision_id": "full-catalog", "rows": 2000}
    project.write_project(state)
    saved = receipt.read_bytes()
    source = [{"id": i, "title": f"title {i}"} for i in range(8)]
    returned = source[:4 if missing_row else 5]
    frame = pd.DataFrame({"video.id": [row["id"] for row in returned],
                          "video.title": [row["title"] for row in returned],
                          "video.tokenizer_version": [2] * len(returned),
                          "video.codes": [np.array([1, 2, 3])] * len(returned)})
    queried = []
    dataset = SimpleNamespace(errors=[], revisions=[SimpleNamespace(revision_id="sample-revision")],
                              to_pandas=lambda **kw: frame)
    monkeypatch.setattr(Target, "client", lambda self: SimpleNamespace(
        offline_query=lambda **kw: queried.append(kw) or dataset))
    monkeypatch.setattr(chalkcompute, "DatasetClient", lambda **kw: SimpleNamespace(
        read=lambda revision: pa.Table.from_pylist(source)))
    ticks = iter([0.0, 1.0, 1.0, 8.0])
    from semantic_ids.stages import load_stage
    monkeypatch.setattr(load_stage("materialize"), "perf_counter", lambda: next(ticks))
    args = cli.parser().parse_args(["materialize", "--branch", "semantic", "--limit", "5"])
    if missing_row:
        with pytest.raises(ValueError, match="every requested video"):
            cli.materialize(args)
    else:
        cli.materialize(args)
    assert queried[0]["input"] == {
        "video.id": list(range(5)), "video.title": [f"title {i}" for i in range(5)],
        "video.tokenizer_version": [2] * 5}
    assert queried[0]["dataset_name"].startswith("semantic-catalog-sample-test-")
    assert queried[0]["recompute_features"] is True
    assert queried[0]["branch"] == "semantic"
    assert "video.codes" in queried[0]["required_output"]
    assert receipt.read_bytes() == saved
    output = capsys.readouterr().out
    assert "Title Dataset read: 1.00s" in output
    assert "Chalk query elapsed: 7.00s" in output
    assert ("Sample complete: 5 rows" in output) is not missing_row


@pytest.mark.parametrize("branch", [None, "semantic"])
def test_recommend_routes_feature_query_to_selected_branch(receipt, monkeypatch, capsys, branch):
    state = saved_project()
    state["training"]["recommender"] = {"model_version": 3}
    project.write_project(state)
    queried = []
    result = SimpleNamespace(errors=[], get_feature_value=lambda feature: '{"items": []}')
    monkeypatch.setattr(Target, "client", lambda self: SimpleNamespace(
        query=lambda **kw: queried.append(kw) or result))
    args = cli.parser().parse_args(["recommend", "--history", "0", "1",
                                   *(["--branch", branch] if branch else [])])
    cli.recommend(args)
    assert queried[0]["branch"] == branch
    assert queried[0]["input"]["recommendation.model_version"] == 3
    assert queried[0]["input"]["recommendation.history_ids"] == [0, 1]
    assert queried[0]["output"] == ["recommendation.result"]
    assert json.loads(capsys.readouterr().out) == {"items": []}
    assert json.loads(receipt.read_text()) == state


def test_reconnect_rejects_another_catalog_before_registration(receipt, monkeypatch):
    import chalkcompute

    state = saved_project()
    state["datasets"]["catalog"] = {"revision_id": "catalog-current"}
    config = {"dataset_revision": "examples-revision", "catalog_revision": "catalog-current", "tokenizer_version": 2,
              "epochs": 1, "batch_size": 4, "seed": 0}
    state["training"]["recommender"] = {"config": config, "checkpoint_key": "unique", "run_id": "native-run"}
    project.write_project(state)
    normalized = {"threads": 2, "checkpoint_every": 1, **config}
    report = {"run_id": "native-run", "dataset_revision": "examples-revision",
              "config_sha256": hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "catalog_revision": "catalog-old", "tokenizer_version": 1}
    artifact = SimpleNamespace(id="artifact", metadata={"kind": "recommender", "completed": True,
                                                       "report_json": json.dumps(report)})
    monkeypatch.setattr(Target, "client", lambda self: object())
    monkeypatch.setattr(chalkcompute, "ConnectClient", lambda **kw: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(chalkcompute, "TrainingRunHandle", lambda *args: SimpleNamespace(
        run_id="native-run", wait=lambda: None, latest_checkpoint=lambda: artifact))
    with pytest.raises(ValueError, match="different semantic feature catalog"):
        cli.train(Namespace(run_id=None), "recommender")


def test_cli_help_does_not_import_deployment(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["semantic_ids.cli", "--help"])
    before = "semantic_ids.deploy" in sys.modules
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 0
    assert ("semantic_ids.deploy" in sys.modules) == before


def test_status_reads_receipt_without_contacting_chalk(receipt, monkeypatch, capsys):
    def forbidden(self):
        pytest.fail("Local status must not construct a Chalk client")

    monkeypatch.setattr(Target, "client", forbidden)
    cli.status(Namespace())
    assert "No saved project" in capsys.readouterr().out
    project.write_project(saved_project())
    cli.status(Namespace())
    output = capsys.readouterr().out
    assert "does not query live Chalk status" in output
    assert "titles-revision" in output
    assert "registered version 2" in output
    assert "Model recommender: not submitted" in output


def test_failed_build_requires_explicit_retry_and_preserves_config(receipt, monkeypatch):
    import sys
    import chalkcompute

    state = saved_project()
    state["training"] = {}
    project.write_project(state)
    submissions = []

    def build_failure(**kwargs):
        submissions.append(kwargs["config"])
        raise RuntimeError("image build failed before submission")

    function = SimpleNamespace(run_training=build_failure)
    monkeypatch.setitem(sys.modules, "semantic_ids.training", SimpleNamespace(
        train_tokenizer=function, train_recommender=function))
    monkeypatch.setattr(Target, "client", lambda self: object())
    monkeypatch.setattr(chalkcompute, "ConnectClient", lambda **kw: SimpleNamespace(close=lambda: None))
    args = cli.parser().parse_args(["train-tokenizer"])
    with pytest.raises(RuntimeError, match="image build failed"):
        cli.train(args, "tokenizer")
    saved = json.loads(receipt.read_text())["training"]["tokenizer"]
    assert saved["submission_started"] is True
    assert "run_id" not in saved
    with pytest.raises(ValueError, match="--retry-submission"):
        cli.train(args, "tokenizer")
    assert len(submissions) == 1
    retry = cli.parser().parse_args(["train-tokenizer", "--retry-submission", "--epochs", "1"])
    with pytest.raises(RuntimeError, match="image build failed"):
        cli.train(retry, "tokenizer")
    assert submissions == [saved["config"], saved["config"]]
    assert json.loads(receipt.read_text())["training"]["tokenizer"]["checkpoint_key"] == saved["checkpoint_key"]


def test_retry_submission_cannot_replace_a_saved_run(receipt):
    state = saved_project()
    state["training"]["tokenizer"] = {"run_id": "original-run", "submission_started": True}
    project.write_project(state)
    with pytest.raises(ValueError, match="without a saved run ID"):
        cli.train(cli.parser().parse_args(["train-tokenizer", "--retry-submission"]), "tokenizer")


def test_restart_failed_preserves_history_and_guards_interrupted_replacement(receipt, monkeypatch):
    import os
    import sys
    import chalkcompute

    state = saved_project()
    config = {"dataset_revision": "titles-revision", "epochs": 400, "batch_size": 1024, "seed": 0}
    old = {"run_id": "failed-run", "checkpoint_key": "old-key", "submission_started": True, "config": config}
    state["training"]["tokenizer"] = old
    project.write_project(state)
    monkeypatch.setattr(Target, "client", lambda self: object())
    monkeypatch.setattr(chalkcompute, "ConnectClient", lambda **kw: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(chalkcompute, "TrainingRunHandle", lambda *args: SimpleNamespace(
        status=lambda: SimpleNamespace(name="FAILED"), latest_checkpoint=lambda: None))
    submissions = []

    def submit(**kwargs):
        saved = json.loads(receipt.read_text())
        record = saved["training"]["tokenizer"]
        assert saved["training_history"]["tokenizer"] == [old]
        assert saved["datasets"] == state["datasets"]
        assert record["checkpoint_key"] != "old-key"
        assert os.environ["SEMANTIC_TRAINING_KEY"] == record["checkpoint_key"]
        assert record["submission_started"] is True
        submissions.append(kwargs["config"])
        raise RuntimeError("replacement submission interrupted")

    function = SimpleNamespace(run_training=submit)
    monkeypatch.setitem(sys.modules, "semantic_ids.training", SimpleNamespace(
        train_tokenizer=function, train_recommender=function))
    args = cli.parser().parse_args(["train-tokenizer", "--restart-failed", "--epochs", "1"])
    with pytest.raises(RuntimeError, match="replacement submission interrupted"):
        cli.train(args, "tokenizer")
    assert submissions == [config]  # Preserve the original run's parameters.
    with pytest.raises(ValueError, match="--retry-submission"):
        cli.train(cli.parser().parse_args(["train-tokenizer"]), "tokenizer")
    assert submissions == [config]


@pytest.mark.parametrize("status,checkpoint", [
    ("QUEUED", None), ("WORKING", None), ("COMPLETED", None),
    ("CANCELED", None), ("FAILED", object()),
])
def test_restart_does_not_replace_unfailed_or_checkpointed_runs(receipt, monkeypatch, status, checkpoint):
    import chalkcompute

    state = saved_project()
    state["training"]["tokenizer"] = {"run_id": "original-run", "config": {}, "checkpoint_key": "original-key"}
    project.write_project(state)
    before = receipt.read_text()
    monkeypatch.setattr(chalkcompute, "TrainingRunHandle", lambda *args: SimpleNamespace(
        status=lambda: SimpleNamespace(name=status), latest_checkpoint=lambda: checkpoint))
    with pytest.raises(ValueError, match="FAILED|checkpoint"):
        cli._restart_failed(state, "tokenizer", object())
    assert receipt.read_text() == before


def test_restart_does_not_treat_checkpoint_lookup_failure_as_no_checkpoint(receipt, monkeypatch):
    import chalkcompute

    state = saved_project()
    state["training"]["tokenizer"] = {"run_id": "original-run", "config": {}, "checkpoint_key": "original-key"}
    project.write_project(state)
    before = receipt.read_text()

    def unavailable():
        raise RuntimeError("checkpoint service unavailable")

    monkeypatch.setattr(chalkcompute, "TrainingRunHandle", lambda *args: SimpleNamespace(
        status=lambda: SimpleNamespace(name="FAILED"), latest_checkpoint=unavailable))
    with pytest.raises(RuntimeError, match="checkpoint service unavailable"):
        cli._restart_failed(state, "tokenizer", object())
    assert receipt.read_text() == before


@pytest.mark.parametrize("publication", ["model_version", "artifact_id", "registration_started"])
def test_restart_rejects_publication_evidence(receipt, publication):
    state = saved_project()
    state["training"]["tokenizer"] = {"run_id": "original-run", publication: True}
    project.write_project(state)
    with pytest.raises(ValueError, match="without model publication"):
        cli.train(cli.parser().parse_args(["train-tokenizer", "--restart-failed"]), "tokenizer")


def test_registration_recovery_validates_report_without_creating_version(receipt):
    state = saved_project()
    record = {"run_id": "original-run", "registration_started": True}
    state["training"]["tokenizer"] = record
    report = {"run_id": "original-run", "config_sha256": "config", "dataset_revision": "titles-revision"}
    registered = SimpleNamespace(version=7, model_name=cli.TOKENIZER_MODEL,
                                 model_artifact=SimpleNamespace(metadata={
                                     "kind": "tokenizer", "completed": True, "report_json": json.dumps(report)}))
    # The client exposes no create/promote method: attachment must be read-only.
    client = SimpleNamespace(get_model_version=lambda **kw: registered)
    cli._register_model(client, state, record, "tokenizer", "original-run",
                        SimpleNamespace(id="final-artifact"), report, version=7)
    saved = json.loads(receipt.read_text())["training"]["tokenizer"]
    assert saved["model_version"] == 7
    assert saved["model_name"] == cli.TOKENIZER_MODEL


def test_registration_recovery_rejects_another_run(receipt):
    state = saved_project()
    record = {"run_id": "original-run", "registration_started": True}
    state["training"]["tokenizer"] = record
    project.write_project(state)
    registered = SimpleNamespace(version=7, model_name=cli.TOKENIZER_MODEL,
                                 model_artifact=SimpleNamespace(metadata={
                                     "kind": "tokenizer", "completed": True,
                                     "report_json": json.dumps({"run_id": "other-run"})}))
    with pytest.raises(ValueError, match="does not match"):
        cli._register_model(SimpleNamespace(get_model_version=lambda **kw: registered), state, record,
                            "tokenizer", "original-run", SimpleNamespace(id="artifact"),
                            {"run_id": "original-run"}, version=7)
    assert "model_version" not in json.loads(receipt.read_text())["training"]["tokenizer"]


def test_wrong_external_run_id_does_not_trap_receipt(receipt, monkeypatch):
    import sys
    import chalkcompute

    state = saved_project()
    state["training"]["tokenizer"] = {"config": {}, "checkpoint_key": "key", "submission_started": True}
    project.write_project(state)
    attempts = []

    def unavailable():
        raise RuntimeError("run lookup failed")

    def handle(run_id, connection):
        attempts.append(run_id)
        return SimpleNamespace(run_id=run_id, wait=unavailable)

    monkeypatch.setitem(sys.modules, "semantic_ids.training", SimpleNamespace(
        train_tokenizer=object(), train_recommender=object()))
    monkeypatch.setattr(Target, "client", lambda self: object())
    monkeypatch.setattr(chalkcompute, "ConnectClient", lambda **kw: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(chalkcompute, "TrainingRunHandle", handle)
    for run_id in ("mistyped-id", "corrected-id"):
        with pytest.raises(RuntimeError, match="run lookup failed"):
            cli.train(cli.parser().parse_args(["train-tokenizer", "--run-id", run_id]), "tokenizer")
        assert "run_id" not in json.loads(receipt.read_text())["training"]["tokenizer"]
    assert attempts == ["mistyped-id", "corrected-id"]


def test_registration_failure_needs_explicit_retry(receipt, monkeypatch):
    state = saved_project()
    record = {"run_id": "original-run", "registration_started": True}
    state["training"]["tokenizer"] = record
    artifact = SimpleNamespace(id="final-artifact")
    calls = []
    monkeypatch.setattr(cli, "_ensure_namespace", lambda *args: None)

    def promote(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(artifact=artifact, model_version=7)

    client = SimpleNamespace(promote_model_artifact=promote)
    with pytest.raises(ValueError, match="--retry-registration"):
        cli._register_model(client, state, record, "tokenizer", "original-run", artifact, {})
    assert calls == []
    cli._register_model(client, state, record, "tokenizer", "original-run", artifact, {}, retry=True)
    assert calls == [{"name": cli.TOKENIZER_MODEL, "run_id": "original-run"}]
    assert json.loads(receipt.read_text())["training"]["tokenizer"]["model_version"] == 7
