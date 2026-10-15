"""Offline SDK checkpoint compatibility, adapted from the original training tests.

Use the installed PyTorch serializer, checkpoint volume writer's local fallback,
and latest-pointer reader. Authentication and remote checkpoint calls are mocked;
these tests do not establish managed-worker or hosted execution.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from semantic_ids import model, rqvae, training_runtime as runtime


@pytest.fixture
def sdk_checkpoints(monkeypatch, tmp_path):
    from chalk.client.client_grpc import _write_checkpoint_to_volume
    from chalk.client.serialization.model_serialization import MODEL_SERIALIZERS
    from chalk.ml import chalk_train
    from chalk.ml.utils import ModelType

    def no_auth(*args, **kwargs):
        raise AssertionError("Checkpoint tests must not load credentials")

    monkeypatch.setattr(chalk_train, "load_token", no_auth)
    monkeypatch.delenv("CHALK_TRAINING_RUN_NAME", raising=False)
    monkeypatch.setenv("CHALK_TRAINING_RUN_ID", "fixture-native-run")
    root = tmp_path / "checkpoints"
    monkeypatch.setenv("CHALK_CHECKPOINT_DIR", str(root))
    serializer = MODEL_SERIALIZERS[ModelType.PYTORCH]
    records = []

    def persist(*, training_run_id, model, additional_files, metadata, path):
        serialized = tmp_path / "model.safetensors"
        serializer.serialize_fn(model, str(serialized))
        paths = {Path(value).name: value for value in additional_files}
        paths[serialized.name] = str(serialized)
        _write_checkpoint_to_volume(path, paths, serialized.name)
        records.append({"training_run_id": training_run_id, "metadata": dict(metadata),
                        "directory": root / path})
        return SimpleNamespace(artifact_id=f"fixture-artifact-{len(records)}")

    latest = MagicMock(return_value=None)
    client = SimpleNamespace(get_latest_checkpoint=latest,
                             _checkpoint_training_run=MagicMock(side_effect=persist))
    checkpointer = chalk_train.ClientCheckpointer()
    checkpointer._client = client
    monkeypatch.setattr(chalk_train, "CheckpointClass", checkpointer)
    return SimpleNamespace(root=root, latest=latest, records=records, serializer=serializer)


def test_sdk_latest_pointer_avoids_remote_lookup_after_local_write(sdk_checkpoints, tmp_path):
    from chalk.client.client_grpc import _write_checkpoint_to_volume
    from chalk.ml import chalk_train

    assert chalk_train.last_checkpoint_path() is None
    sdk_checkpoints.latest.assert_called_once_with(training_run_id="fixture-native-run")
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"fixture checkpoint bytes")
    _write_checkpoint_to_volume("fixture-native-run/epoch-1", {source.name: str(source)}, source.name)
    expected = sdk_checkpoints.root / "fixture-native-run" / "epoch-1" / source.name
    assert chalk_train.last_checkpoint_path() == str(expected)
    assert expected.read_bytes() == source.read_bytes()
    sdk_checkpoints.latest.assert_called_once()


@pytest.mark.parametrize("kind", ["tokenizer", "recommender"])
def test_runtime_sidecars_restore_with_sdk_serialization_and_pointer(sdk_checkpoints, tmp_path, kind):
    from chalk.ml import chalk_train

    if kind == "tokenizer":
        trained = rqvae.RQVAE(input_dim=4, latent_dim=4, codebook_size=4, num_levels=3,
                              hidden_dims=(8,), commitment_weight=0.25)
    else:
        trained = model.SemanticRecommender({1: (0, 0, 0, 0), 2: (1, 0, 0, 0)}, hidden_dim=4)
    optimizer = torch.optim.Adam(trained.parameters())
    sum(parameter.square().sum() for parameter in trained.parameters()).backward()
    optimizer.step()
    state = {"epoch": 1, "model_state": trained.state_dict(), "optimizer_state": optimizer.state_dict(),
             "torch_rng_state": torch.get_rng_state(),
             "shuffle_rng_state": torch.Generator().manual_seed(7).get_state()}
    identity = {"kind": kind, "run_id": "fixture-native-run", "dataset_revision": "fixture-revision",
                "config_sha256": "config", "data_sha256": "data", "catalog_sha256": "catalog"}
    report = {**identity, "completed": True, "epoch": 1}
    workdir = tmp_path / "work"
    workdir.mkdir()
    serving_artifact = workdir / ("rqvae.pt" if kind == "tokenizer" else "recommender.pt")
    serving_artifact.write_bytes(b"fixture serving artifact")
    rng_before = torch.get_rng_state()

    # Both functions imported by the runtime remain the actual SDK wrappers.
    runtime._save(trained, workdir, identity, state, report, [serving_artifact])
    assert torch.equal(torch.get_rng_state(), rng_before)
    saved = sdk_checkpoints.records[-1]
    directory = saved["directory"]
    assert chalk_train.last_checkpoint_path() == str(directory / "model.safetensors")
    assert saved["training_run_id"] == identity["run_id"]
    assert saved["metadata"]["completed"] is True
    assert json.loads(saved["metadata"]["report_json"]) == report
    assert json.loads((directory / "report.json").read_text()) == report
    assert (directory / serving_artifact.name).read_bytes() == serving_artifact.read_bytes()
    serialized_weights = sdk_checkpoints.serializer.load_fn(str(directory / "model.safetensors"))
    assert serialized_weights.keys() == trained.state_dict().keys()
    assert all(torch.equal(value, serialized_weights[name]) for name, value in trained.state_dict().items())

    restored = runtime._restore(identity)
    assert restored["epoch"] == 1
    assert all(torch.equal(value, restored["model_state"][name]) for name, value in state["model_state"].items())
    assert restored["optimizer_state"]["param_groups"] == state["optimizer_state"]["param_groups"]
    assert state["optimizer_state"]["state"]
    for parameter, values in state["optimizer_state"]["state"].items():
        for name, value in values.items():
            assert torch.equal(value, restored["optimizer_state"]["state"][parameter][name])
    for name in ("torch_rng_state", "shuffle_rng_state"):
        assert torch.equal(state[name], restored[name])
    sdk_checkpoints.latest.assert_not_called()
    with pytest.raises(ValueError, match="different run"):
        runtime._restore({**identity, "run_id": "another-run"})
    (directory / "recovery_state.pt").unlink()
    with pytest.raises(ValueError, match="missing its numerical recovery state"):
        runtime._restore(identity)
