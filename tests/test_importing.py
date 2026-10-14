"""Import validates the original catalog and keeps source/target auth separate."""
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
import torch

from semantic_ids import importing
from semantic_ids.model import SemanticRecommender, catalog_fingerprint, save_model


def _profile(environment="source-env", api_server="https://api.chalk.ai", secret="source-secret"):
    return SimpleNamespace(activeEnvironment=environment, apiServer=api_server,
                           clientId="source-client", clientSecret=secret)


def _saved_profiles(monkeypatch, profiles):
    import chalk.config.auth_config
    monkeypatch.setattr(chalk.config.auth_config, "_load_global_config_uncached",
                        lambda: SimpleNamespace(tokens=profiles))
    monkeypatch.delenv("CHALK_WEB_IDENTITY_TOKEN_FILE", raising=False)


def test_source_auth_ignores_target_environment_and_deduplicates_profiles(monkeypatch):
    import chalk.client
    token = _profile()
    _saved_profiles(monkeypatch, {"first": token, "duplicate": copy.copy(token),
                                "target": _profile("target-env", "https://api.staging.chalk.dev", "target-secret")})
    monkeypatch.setenv("CHALK_CLIENT_ID", "target-client")
    monkeypatch.setenv("CHALK_CLIENT_SECRET", "target-secret")
    monkeypatch.setenv("CHALK_ENVIRONMENT", "target-env")
    monkeypatch.setenv("CHALK_API_SERVER", "https://api.staging.chalk.dev")
    before = dict(os.environ)
    monkeypatch.setattr(chalk.client, "ChalkClient", lambda **kwargs: kwargs)
    result = importing.source_client("source-env", "https://api.chalk.ai/")
    assert result == {"client_id": "source-client", "client_secret": "source-secret",
                      "environment": "source-env", "api_server": "https://api.chalk.ai", "branch": None}
    assert dict(os.environ) == before


@pytest.mark.parametrize("profiles", [
    {},
    {"wrong-env": _profile("target-env")},
    {"wrong-host": _profile(api_server="https://api.staging.chalk.dev")},
    {"first": _profile(), "second": _profile(secret="other-secret")},
])
def test_source_auth_refuses_missing_or_ambiguous_profiles(monkeypatch, profiles):
    _saved_profiles(monkeypatch, profiles)
    with pytest.raises(ValueError, match="one distinct saved credential") as error:
        importing.source_client("source-env", "https://api.chalk.ai")
    assert "secret" not in str(error.value)


def test_source_auth_refuses_web_identity_override(monkeypatch):
    _saved_profiles(monkeypatch, {"source": _profile()})
    monkeypatch.setenv("CHALK_WEB_IDENTITY_TOKEN_FILE", "/target/identity")
    with pytest.raises(ValueError, match="web-identity override"):
        importing.source_client("source-env", "https://api.chalk.ai")


@pytest.fixture
def artifact(tmp_path):
    from chalk.ml import ModelClass, ModelEncoding, ModelType

    catalog = {101: (0, 0, 0, 0), 202: (0, 1, 0, 0),
               303: (1, 0, 0, 0), 404: (1, 1, 0, 0)}
    rows = [{"video_id": video_id, "semantic_tokens": list(tokens), "title": f"Video {video_id}"}
            for video_id, tokens in catalog.items()]
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(rows))
    checkpoint = tmp_path / "recommender.pt"
    save_model(SemanticRecommender(catalog, hidden_dim=4), checkpoint,
               catalog=catalog, metadata={"completed": True})
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"opaque source model file; importer must not reserialize")
    report = {"catalog_fingerprint": catalog_fingerprint(catalog), "source_kind": "synthetic_demo",
              "tokenizer_version": 2}
    return SimpleNamespace(
        model_name="semantic-demo-recommender", model_version=4,
        downloaded_model_files=[str(model_file)],
        downloaded_additional_files=[str(checkpoint), str(catalog_path)],
        model_artifact=SimpleNamespace(
            model_type=ModelType.PYTORCH, model_class=ModelClass.CLASSIFICATION,
            model_encoding=ModelEncoding.SAFETENSOR,
            input_schema=[([64], pa.float64())], output_schema=[([5], pa.float64())],
            input_features=["viewer.history_ids"], output_features=["viewer.recommendations"],
            metadata={"kind": "recommender", "completed": True,
                      "semantic_demo_report_json": json.dumps(report)},
            dependencies=["torch==2.9.1"],
        ),
    )


def test_artifact_loads_and_checks_local_recommendations(artifact):
    summary = importing.validate_artifact(artifact)
    assert summary["catalog_count"] == 4
    assert summary["sample_history_ids"] == [101, 202]
    assert summary["source_kind"] == "synthetic_demo"
    assert summary["tokenizer_version"] == 2


@pytest.mark.parametrize("provenance,source_kind", [
    ({"source_kind": "microlens"}, "microlens"),
    ({"feedback_source": "historical_watches"}, "historical_watches"),
    ({}, "unknown"),
])
def test_native_report_preserves_tokenizer_lineage_and_provenance(artifact, provenance, source_kind):
    metadata = artifact.model_artifact.metadata
    metadata.pop("semantic_demo_report_json")
    # The native training report does not currently label source_kind. Preserve
    # explicit provenance when present; do not infer real-data provenance from
    # a report key or silently reuse the conference-demo label.
    metadata["report_json"] = json.dumps({
        "kind": "recommender", "completed": True, "catalog_revision": "native-catalog",
        "tokenizer_version": 7, **provenance,
    })
    summary = importing.validate_artifact(artifact)
    assert summary["tokenizer_version"] == 7
    assert summary["source_kind"] == source_kind


def test_native_report_fingerprint_is_checked(artifact):
    metadata = artifact.model_artifact.metadata
    metadata.pop("semantic_demo_report_json")
    metadata["report_json"] = json.dumps({"catalog_fingerprint": "another-catalog"})
    with pytest.raises(ValueError, match="fingerprint differs from the source artifact report"):
        importing.validate_artifact(artifact)


@pytest.mark.parametrize("mutation,message", [
    (lambda rows: rows.append(rows[0]), "unique nonnegative"),
    (lambda rows: rows[0].update(video_id=True), "unique nonnegative"),
    (lambda rows: rows[0].update(title=None), "string titles"),
    (lambda rows: rows[0].update(semantic_tokens=[1, 2, 3]), "four nonnegative"),
    (lambda rows: rows[0].update(semantic_tokens=[True, 0, 0, 0]), "four nonnegative"),
    (lambda rows: rows[0].update(semantic_tokens=[-1, 0, 0, 0]), "four nonnegative"),
    (lambda rows: rows[1].update(semantic_tokens=rows[0]["semantic_tokens"]), "must be unique"),
])
def test_rejects_invalid_catalog(artifact, mutation, message):
    path = Path(artifact.downloaded_additional_files[-1])
    rows = json.loads(path.read_text())
    mutation(rows)
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match=message):
        importing.validate_artifact(artifact)


def test_rejects_catalog_from_another_model(artifact):
    path = Path(artifact.downloaded_additional_files[-1])
    rows = json.loads(path.read_text())
    rows[0]["semantic_tokens"] = [0, 0, 0, 1]
    path.write_text(json.dumps(rows))
    artifact.model_artifact.metadata.pop("semantic_demo_report_json")
    with pytest.raises(ValueError, match="fingerprint differs from the model checkpoint"):
        importing.validate_artifact(artifact)


def test_rejects_nonfinite_checkpoint(artifact):
    path = artifact.downloaded_additional_files[0]
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["state_dict"]["bos"][0] = float("nan")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="nonfinite weights"):
        importing.validate_artifact(artifact)


def test_rejects_duplicate_or_missing_download_files(artifact, tmp_path):
    artifact.downloaded_additional_files.append(artifact.downloaded_additional_files[0])
    with pytest.raises(ValueError, match="filenames must be unique"):
        importing.validate_artifact(artifact)
    artifact.downloaded_additional_files.pop()
    Path(artifact.downloaded_additional_files[0]).unlink()
    with pytest.raises(ValueError, match="regular local files"):
        importing.validate_artifact(artifact)


def test_registration_preserves_files_and_roundtrips_sdk_schemas(artifact):
    from chalk.client.serialization.model_serialization import ModelSerializer

    files = artifact.downloaded_model_files + artifact.downloaded_additional_files
    hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files}
    before_metadata = copy.deepcopy(artifact.model_artifact.metadata)
    summary = importing.validate_artifact(artifact)
    kwargs = importing.registration_kwargs(
        artifact, source_environment="source-env", source_api_server="https://api.chalk.ai/",
        source_model_name="semantic-demo-recommender", source_model_version=4, summary=summary,
    )
    assert kwargs["model_paths"] == artifact.downloaded_model_files
    assert kwargs["additional_files"] == artifact.downloaded_additional_files
    assert kwargs["model_encoding"] == artifact.model_artifact.model_encoding
    assert kwargs["model_class"] == artifact.model_artifact.model_class
    assert kwargs["input_features"] == artifact.model_artifact.input_features
    assert kwargs["output_features"] == artifact.model_artifact.output_features
    for schema in ("input_schema", "output_schema"):
        original_proto = ModelSerializer.convert_schema(getattr(artifact.model_artifact, schema))
        transferred_proto = ModelSerializer.convert_schema(kwargs[schema])
        assert original_proto == transferred_proto
    assert kwargs["skip_volume_upload"] is True
    assert kwargs["metadata"]["semantic_demo_report_json"] == before_metadata["semantic_demo_report_json"]
    assert json.loads(kwargs["metadata"]["semantic_ids_import_json"])["source_kind"] == "synthetic_demo"
    ModelSerializer.convert_metadata_to_protobuf(kwargs["metadata"])
    assert artifact.model_artifact.metadata == before_metadata
    assert {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files} == hashes


def test_registration_rejects_wrong_source_version(artifact):
    with pytest.raises(ValueError, match="does not match"):
        importing.registration_kwargs(
            artifact, source_environment="source-env", source_api_server="https://api.chalk.ai",
            source_model_name="semantic-demo-recommender", source_model_version=5, summary={},
        )
