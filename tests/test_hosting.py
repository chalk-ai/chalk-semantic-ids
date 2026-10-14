"""Offline inference/packaging guards; no deployment or remote calls."""

import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from semantic_ids import hosting


@pytest.fixture(autouse=True)
def clean_caches(monkeypatch):
    monkeypatch.setattr(hosting, "_artifacts", {})
    monkeypatch.setattr(hosting, "_models", {})
    monkeypatch.setattr(hosting, "_encoder", None)
    monkeypatch.setattr(hosting, "_client", None)
    previous_stage = sys.modules.pop("semantic_ids_host", None)
    yield
    sys.modules.pop("semantic_ids_host", None)
    if previous_stage is not None:
        sys.modules["semantic_ids_host"] = previous_stage


@pytest.mark.parametrize("entrypoint", ["legacy", "stage"])
def test_deploy_help_exits_before_decorators(monkeypatch, entrypoint):
    import chalkcompute

    def forbidden(*args, **kwargs):
        pytest.fail("--help must not register a Function")

    monkeypatch.setattr(chalkcompute, "function", forbidden)
    monkeypatch.setattr(sys, "argv", ["semantic_ids.deploy", "--help"])
    with pytest.raises(SystemExit) as stopped:
        if entrypoint == "legacy":
            runpy.run_module("semantic_ids.deploy", run_name="__main__")
        else:
            runpy.run_path(str(Path(__file__).resolve().parents[1] / "host" / "run.py"), run_name="__main__")
    assert stopped.value.code == 0


def test_remote_module_import_cannot_deploy(monkeypatch):
    import chalkcompute

    def forbidden(*args, **kwargs):
        pytest.fail("Worker import must not deploy another Function")

    monkeypatch.setenv("CHALK_REMOTE_FUNCTION", "1")
    monkeypatch.setattr(chalkcompute.RemoteFunction, "deploy", forbidden)
    monkeypatch.setattr(chalkcompute.RemoteFunction, "_start_background_deploy", forbidden)
    namespace = runpy.run_module("semantic_ids.deploy", run_name="semantic_ids_worker")
    assert "semantic_tokenize" in namespace
    assert "semantic_recommend" in namespace


def test_direct_host_command_packages_functions_under_stable_module(monkeypatch):
    import chalkcompute
    from semantic_ids import project

    functions = []
    waits = []

    def capture_function(**options):
        def decorate(function):
            functions.append(function)
            function.wait_ready = lambda *, timeout: waits.append((function.__name__, timeout))
            return function
        return decorate

    monkeypatch.setattr(chalkcompute, "function", capture_function)
    monkeypatch.delenv("CHALK_REMOTE_FUNCTION", raising=False)
    monkeypatch.setattr(project, "read_project", lambda: (
        {"training": {"tokenizer": {"model_version": 1}}}, None))
    monkeypatch.setattr(sys, "argv", ["host/run.py", "--ready-timeout", "12"])
    path = Path(__file__).resolve().parents[1] / "host" / "run.py"
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(path), run_name="__main__")
    assert stopped.value.code == 0
    assert [function.__module__ for function in functions] == ["semantic_ids_host", "semantic_ids_host"]
    assert waits == [("semantic_tokenize", 12.0), ("semantic_recommend", 12.0)]


def test_checkpoint_download_is_cached_by_immutable_version(tmp_path, monkeypatch):
    path = tmp_path / "rqvae.pt"
    path.write_bytes(b"placeholder")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(downloaded_model_files=[], downloaded_additional_files=[str(path)])

    monkeypatch.setattr(hosting, "_client", SimpleNamespace(download_model_artifact=download))
    first = hosting._checkpoint_files("tokenizer", 1, ["rqvae.pt"])
    assert hosting._checkpoint_files("tokenizer", 1, ["rqvae.pt"]) == first
    hosting._checkpoint_files("tokenizer", 2, ["rqvae.pt"])
    assert [call["version"] for call in calls] == [1, 2]


def test_tokenizer_rejects_encoder_mismatch_before_loading_encoder(tmp_path, monkeypatch):
    import torch

    path = tmp_path / "rqvae.pt"
    torch.save({"metadata": {"encoder_model": "wrong", "encoder_revision": "wrong"}}, path)
    monkeypatch.setattr(hosting, "_checkpoint_files", lambda *args: {"rqvae.pt": path})
    with pytest.raises(ValueError, match="encoder differ"):
        hosting.tokenize_title("a new title", 1)
    assert hosting._encoder is None


def test_tokenizer_preserves_codebooks_and_loads_encoder_offline(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from semantic_ids.rqvae import RQVAE

    config = {"input_dim": 384, "latent_dim": 4, "codebook_size": 8,
              "num_levels": 3, "hidden_dims": [8], "commitment_weight": 0.25}
    model = RQVAE(**config)
    path = tmp_path / "rqvae.pt"
    torch.save({"state_dict": model.state_dict(), "config": config,
                "quantizer_initialized": [True, True, True],
                "metadata": {"completed": True, "encoder_model": hosting.ENCODER_MODEL,
                             "encoder_revision": hosting.ENCODER_REVISION}}, path)
    monkeypatch.setattr(hosting, "_checkpoint_files", lambda *args: {"rqvae.pt": path})
    calls = []

    def encoder(location, **options):
        calls.append((location, options))
        return SimpleNamespace(encode=lambda *args, **kwargs: np.ones((1, 384), dtype=np.float32))

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=encoder))
    first = hosting.tokenize_title("first call", 1)
    assert hosting.tokenize_title("another call", 1) == first
    assert calls == [("/opt/encoder", {"device": "cpu", "local_files_only": True})]
    loaded = hosting._models[(hosting.TOKENIZER_MODEL, 1)]
    assert all(quantizer._initialized for quantizer in loaded.quantizers)
    for name, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)


def test_recommender_rejects_catalog_drift(tmp_path, monkeypatch):
    from semantic_ids.model import SemanticRecommender, save_model

    catalog = {1: (1, 2, 3, 0), 2: (4, 5, 6, 0)}
    path = tmp_path / "recommender.pt"
    save_model(SemanticRecommender(catalog), path, catalog=catalog, metadata={})
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps([
        {"video_id": 1, "title": "first", "semantic_tokens": [1, 2, 3, 0]},
        {"video_id": 2, "title": "second", "semantic_tokens": [4, 5, 7, 0]},
    ]))
    monkeypatch.setattr(hosting, "_checkpoint_files", lambda *args: {
        "recommender.pt": path, "catalog.json": catalog_path,
    })
    with pytest.raises(ValueError, match="Catalog fingerprint"):
        hosting.recommend_videos([1], 1, 2)


def test_recommendation_serves_real_model_with_titles(monkeypatch):
    from semantic_ids.model import SemanticRecommender

    catalog = {1: (1, 2, 3, 0), 2: (4, 5, 6, 0)}
    model = SemanticRecommender(catalog).eval()
    monkeypatch.setattr(hosting, "_recommender", lambda version: (model, catalog, {1: "first", 2: "second"}))
    result = json.loads(hosting.recommend_videos([1], 7, 2))
    assert result["model_version"] == 7
    assert result["items"][0]["video_id"] == 2
    assert result["items"][0]["title"] == "second"
    with pytest.raises(ValueError, match="outside"):
        hosting.recommend_videos([999], 7, 2)


@pytest.mark.parametrize("version", [0, -1, True, "1"])
def test_invalid_versions_fail_before_io(version):
    with pytest.raises(ValueError, match="positive integer"):
        hosting.tokenize_title("a title", version)
