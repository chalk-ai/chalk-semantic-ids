"""Local numerical recovery tests; Chalk checkpoint transport and MiniLM are mocked.

These tests intentionally execute small CPU training loops when explicitly run.
They never launch a native job, download an encoder, or contact Chalk.
"""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from semantic_ids import data, model, training_runtime as runtime
from semantic_ids.stages import load_stage


class Interrupted(RuntimeError):
    pass


class Checkpoints:
    """Simulate durable sidecars plus a worker exit immediately after persistence."""
    def __init__(self, directory, monkeypatch, *, interrupt_epoch=None):
        self.directory = directory
        self.latest = None
        self.reports, self.metrics, self.paths = [], [], []
        self.interrupt_epoch = interrupt_epoch
        monkeypatch.setattr(runtime, "checkpoint", self.save)
        monkeypatch.setattr(runtime, "last_checkpoint_path", lambda: self.latest)
        monkeypatch.setattr(runtime, "log_metrics", self.log)

    def log(self, metrics, *, tags):
        self.metrics.append((metrics, tags))
        torch.rand(3)  # Logging must not perturb numerical recovery.

    def save(self, trained, *, additional_files, metadata, run_name, path):
        assert path not in self.paths
        self.paths.append(path)
        target = self.directory / path
        target.mkdir(parents=True)
        for source in additional_files:
            shutil.copy2(source, target / Path(source).name)
        torch.save(trained.state_dict(), target / "model.pt")
        self.latest = str(target / "model.pt")
        report = json.loads((target / "report.json").read_text())
        assert json.loads(metadata["report_json"]) == report
        assert metadata["completed"] == report["completed"]
        assert run_name == report["run_id"]
        self.reports.append(report)
        torch.rand(5)  # Model serialization may also use the global generator.
        if not report["completed"] and report["epoch"] == self.interrupt_epoch:
            self.interrupt_epoch = None
            raise Interrupted("Worker exited after its durable checkpoint")
        return SimpleNamespace(artifact_id=f"artifact-{len(self.reports)}")

    @property
    def final_directory(self):
        return Path(self.latest).parent


@pytest.fixture(autouse=True)
def native_environment(monkeypatch):
    monkeypatch.setenv("CHALK_TRAINING_RUN_ID", "native-run-1")
    monkeypatch.setenv("CHALK_DATASET_REVISION_ID", "primary-revision-1")


@pytest.fixture
def tokenizer_frame():
    return pd.DataFrame({"video.id": range(1, 21), "video.title": [f"Title {index}" for index in range(20)]})


@pytest.fixture
def tokenizer_config():
    # Integral protobuf doubles must be restored before calling torch.
    return {"dataset_revision": "primary-revision-1", "epochs": 3.0, "holdout": 4.0,
            "batch_size": 4.0, "hidden_dims": [8.0], "latent_dim": 4.0,
            "codebook_size": 4.0, "seed": 7.0, "threads": 1.0}


@pytest.fixture
def encoder(monkeypatch):
    calls = []
    def encode(titles, config):
        calls.append((titles, config["encoder_model"], config["encoder_revision"]))
        vectors = np.random.default_rng(41).normal(size=(len(titles), 384)).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    monkeypatch.setattr(load_stage("train"), "_encode_titles", encode)
    return calls


@pytest.fixture
def catalog_rows():
    return [{"id": index, "title": f"Video {index}", "codes": [index // 3, index % 2, 0]}
            for index in range(1, 9)]


@pytest.fixture
def recommender_frame():
    return pd.DataFrame(data.parse_watch_examples(b"a\t1 2 3 4 5 6 7\nb\t8 7 6 5 4 3 2\n", set(range(1, 9))))


@pytest.fixture
def recommender_config(catalog_rows):
    return {"dataset_revision": "primary-revision-1", "catalog_revision": "catalog-revision-1",
            "catalog_rows": catalog_rows, "epochs": 3.0, "batch_size": 3.0, "hidden_dim": 8.0,
            "seed": 7.0, "threads": 1.0, "top_k": 3.0, "beam_width": 4.0}


def assert_same_weights(first, second):
    assert first.state_dict().keys() == second.state_dict().keys()
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name


def test_tokenizer_rejects_unservable_title_before_encoding(tokenizer_frame, tokenizer_config, encoder):
    tokenizer_frame.loc[0, "video.title"] = "x" * (runtime.MAX_TITLE_LENGTH + 1)
    with pytest.raises(ValueError, match="at most"):
        runtime.train_tokenizer(tokenizer_frame, tokenizer_config)
    assert encoder == []


def test_tokenizer_interrupted_training_matches_uninterrupted_weights_and_metrics(
    tmp_path, monkeypatch, tokenizer_frame, tokenizer_config, encoder,
):
    complete = Checkpoints(tmp_path / "complete", monkeypatch)
    expected = runtime.train_tokenizer(tokenizer_frame, tokenizer_config)
    interrupted = Checkpoints(tmp_path / "interrupted", monkeypatch, interrupt_epoch=2)
    with pytest.raises(Interrupted):
        runtime.train_tokenizer(tokenizer_frame, tokenizer_config)
    saved = torch.load(interrupted.final_directory / "recovery_state.pt", weights_only=True)
    assert saved["state"]["epoch"] == 2
    assert saved["state"]["optimizer_state"]["state"]
    assert saved["state"]["quantizer_initialized"] == [True, True, True]
    resumed = runtime.train_tokenizer(tokenizer_frame, tokenizer_config)
    assert_same_weights(expected, resumed)
    assert complete.reports[-1] == interrupted.reports[-1]
    assert [report["epoch"] for report in interrupted.reports if not report["completed"]] == [1, 2, 3]
    artifact = torch.load(interrupted.final_directory / "rqvae.pt", weights_only=True)
    assert artifact["metadata"]["completed"] is True
    assert artifact["metadata"]["encoder_revision"] == runtime.ENCODER_REVISION
    assert artifact["quantizer_initialized"] == [True, True, True]
    assert all(call[1:] == (runtime.ENCODER_MODEL, runtime.ENCODER_REVISION) for call in encoder)


def test_recommender_interrupted_training_matches_uninterrupted_and_saves_reconstructable_artifacts(
    tmp_path, monkeypatch, recommender_frame, recommender_config,
):
    complete = Checkpoints(tmp_path / "complete", monkeypatch)
    expected = runtime.train_recommender(recommender_frame, recommender_config)
    interrupted = Checkpoints(tmp_path / "interrupted", monkeypatch, interrupt_epoch=1)
    with pytest.raises(Interrupted):
        runtime.train_recommender(recommender_frame, recommender_config)
    resumed = runtime.train_recommender(recommender_frame, recommender_config)
    assert_same_weights(expected, resumed)
    assert complete.reports[-1] == interrupted.reports[-1]
    report = interrupted.reports[-1]
    assert report["test_used_for_selection"] is False
    assert report["metrics"]["training"]["training_examples"] == 8
    assert report["metrics"]["validation"]["examples"] == report["metrics"]["test"]["examples"] == 2
    catalog = {row["video_id"]: tuple(row["semantic_tokens"])
               for row in json.loads((interrupted.final_directory / "catalog.json").read_text())}
    loaded = model.load_model(interrupted.final_directory / "recommender.pt", catalog=catalog)
    assert_same_weights(resumed, loaded)
    assert model.recommend(loaded, catalog, [1, 2], top_k=3) == model.recommend(resumed, catalog, [1, 2], top_k=3)
    assert {tags["split"] for _, tags in interrupted.metrics} == {"train", "validation", "test"}


@pytest.mark.parametrize("change", ["run", "revision", "config", "data", "embeddings"])
def test_tokenizer_rejects_unrelated_latest_checkpoint(
    tmp_path, monkeypatch, tokenizer_frame, tokenizer_config, encoder, change,
):
    store = Checkpoints(tmp_path, monkeypatch, interrupt_epoch=1)
    with pytest.raises(Interrupted):
        runtime.train_tokenizer(tokenizer_frame, tokenizer_config)
    config, frame = copy.deepcopy(tokenizer_config), tokenizer_frame.copy()
    if change == "run":
        monkeypatch.setenv("CHALK_TRAINING_RUN_ID", "unrelated-run")
    elif change == "revision":
        config["dataset_revision"] = "different-revision"
        monkeypatch.setenv("CHALK_DATASET_REVISION_ID", "different-revision")
    elif change == "config":
        config["epochs"] += 1
    elif change == "data":
        frame.loc[0, "video.title"] = "Changed source content"
    else:
        monkeypatch.setattr(load_stage("train"), "_encode_titles", lambda titles, config: np.ones((len(titles), 384), dtype=np.float32))
    with pytest.raises(ValueError, match="different|differ"):
        runtime.train_tokenizer(frame, config)
    assert len(store.reports) == 1


def test_recommender_rejects_changed_catalog_after_interruption(
    tmp_path, monkeypatch, recommender_frame, recommender_config,
):
    store = Checkpoints(tmp_path, monkeypatch, interrupt_epoch=1)
    with pytest.raises(Interrupted):
        runtime.train_recommender(recommender_frame, recommender_config)
    changed = copy.deepcopy(recommender_config)
    changed["catalog_rows"][0]["codes"] = [9, 9, 9]
    with pytest.raises(ValueError, match="different run, configuration, or dataset"):
        runtime.train_recommender(recommender_frame, changed)
    assert len(store.reports) == 1


@pytest.mark.parametrize("change", ["overlap", "missing", "history", "unknown", "event"])
def test_chronological_boundaries_are_checked_before_training(recommender_frame, catalog_rows, change):
    frame = recommender_frame.copy(deep=True)
    if change == "overlap":
        frame.loc[frame["split"] == "validation", "split"] = "train"
    elif change == "missing":
        frame = frame.iloc[1:]
    elif change == "history":
        frame.at[1, "history_ids"] = [8, 1]
    elif change == "unknown":
        frame.at[0, "target_id"] = 999
    else:
        frame.at[1, "id"] = frame.at[0, "id"]
    catalog = {row["video_id"]: tuple(row["semantic_tokens"]) for row in data.build_catalog(catalog_rows)}
    with pytest.raises(ValueError):
        runtime._examples(frame, catalog)


def test_catalog_fetch_uses_exact_revision_and_normalizes_arrow_lists(monkeypatch, catalog_rows):
    import chalk.client
    calls = []
    frame = pd.DataFrame([{f"video.{key}": np.asarray(value) if key == "codes" else value
                           for key, value in row.items()} | {"video.tokenizer_version": 2} for row in catalog_rows])
    def get_dataset(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(to_pandas=lambda **options: frame)
    monkeypatch.setattr(chalk.client, "ChalkClient", lambda: SimpleNamespace(get_dataset=get_dataset))
    actual = runtime._catalog({"catalog_revision": "exact-revision", "tokenizer_version": 2})
    assert calls == [{"revision_id": "exact-revision"}]
    assert actual == data.build_catalog(catalog_rows)
    with pytest.raises(ValueError, match="tokenizer version"):
        runtime._catalog({"catalog_revision": "exact-revision", "tokenizer_version": 3})


def test_primary_revision_mismatch_fails_before_embedding(monkeypatch, tokenizer_frame, tokenizer_config):
    monkeypatch.setenv("CHALK_DATASET_REVISION_ID", "wrong-revision")
    monkeypatch.setattr(load_stage("train"), "_encode_titles", lambda *args: pytest.fail("Do not encode a mismatched dataset"))
    with pytest.raises(ValueError, match="different dataset revision"):
        runtime.train_tokenizer(tokenizer_frame, tokenizer_config)


def test_missing_recovery_sidecar_is_not_silently_ignored(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    path.touch()
    monkeypatch.setattr(runtime, "last_checkpoint_path", lambda: str(path))
    with pytest.raises(ValueError, match="missing its numerical recovery state"):
        runtime._restore({})
