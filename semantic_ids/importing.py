"""Validate and copy a registered recommender with its original catalog.

This module performs no network requests at import time. The caller owns the
explicit download, destination registration, and receipt update.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from .project import Target


IMPORT_METADATA_KEY = "semantic_ids_import_json"


def source_client(environment: str, api_server: str):
    """Select one exact saved source identity without changing target settings.

    Chalk's normal ``load_token`` selects by working directory, not environment.
    Read the SDK's parsed saved config and require one distinct credential/API
    pair explicitly associated with the requested source environment instead.
    Credentials remain in memory and are never included in errors or receipts.
    """
    from chalk.client import ChalkClient
    from chalk.config.auth_config import _load_global_config_uncached
    from chalk.config.web_identity import web_identity_token_file_from_env

    target = Target(environment, api_server)
    if web_identity_token_file_from_env() is not None:
        raise ValueError("A web-identity override is active; source credentials cannot be selected safely")
    origin = target.api_server.rstrip("/")
    try:
        config = _load_global_config_uncached()
        tokens = (config.tokens or {}) if config else {}
        matches = {
            (token.clientId, token.clientSecret, (token.apiServer or "https://api.chalk.ai").rstrip("/"))
            for token in tokens.values()
            if token.activeEnvironment == environment
            and (token.apiServer or "https://api.chalk.ai").rstrip("/") == origin
        }
    except Exception:
        raise ValueError("Could not read saved Chalk source authentication") from None
    if len(matches) != 1:
        raise ValueError("Need one distinct saved credential/API pair matching the source environment")
    client_id, client_secret, _ = next(iter(matches))
    if not client_id or not client_secret:
        raise ValueError("The saved source profile has no usable credentials")
    try:
        return ChalkClient(client_id=client_id, client_secret=client_secret,
                           environment=environment, api_server=origin, branch=None)
    except Exception:
        raise ValueError("Could not initialize the source Chalk client") from None


def _files(result):
    model_files = [Path(path) for path in result.downloaded_model_files]
    additional_files = [Path(path) for path in result.downloaded_additional_files]
    if not model_files:
        raise ValueError("The downloaded artifact must contain a model file")
    all_files = model_files + additional_files
    names = [path.name for path in all_files]
    if len(names) != len(set(names)):
        raise ValueError("Downloaded artifact filenames must be unique")
    if any(not path.is_file() or path.is_symlink() for path in all_files):
        raise ValueError("Downloaded artifact files must be regular local files")
    selected = {path.name: path for path in all_files}
    if not {"recommender.pt", "catalog.json"}.issubset(selected):
        raise ValueError("The artifact must include recommender.pt and catalog.json")
    return model_files, additional_files, selected


def _report(metadata):
    # Native training in this repository writes the unprefixed report_json key.
    value = metadata.get("semantic_demo_report_json",
                         metadata.get("semantic_ids_report_json", metadata.get("report_json", {})))
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Source training report must be a JSON object")
    return value


def validate_artifact(result, *, expected_fingerprint: str | None = None) -> dict:
    """Check catalog/weights identity and perform two local inference checks.

    Returns primitive, credential-free receipt fields. This checks compatibility
    with the artifact's own catalog, not quality on other users or datasets.
    ``load_model`` uses ``torch.load(weights_only=True)``.
    """
    import torch
    from .model import catalog_fingerprint, load_model, recommend

    _, _, files = _files(result)
    metadata = result.model_artifact.metadata
    # ChalkPy 2.160 serializes bool metadata as a numeric protobuf value.
    # Downloaded completed=True can therefore arrive as integer 1.
    if metadata.get("kind") != "recommender" or metadata.get("completed") not in (True, 1):
        raise ValueError("Source artifact must be a completed recommender")
    report = _report(metadata)
    rows = json.loads(files["catalog.json"].read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Catalog must be a nonempty list")
    catalog = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each catalog row must be an object")
        video_id, codes = row.get("video_id"), row.get("semantic_tokens")
        if type(video_id) is not int or video_id < 0 or video_id in catalog:
            raise ValueError("Catalog requires unique nonnegative integer video IDs")
        if not isinstance(row.get("title"), str):
            raise ValueError("Catalog requires string titles")
        if (not isinstance(codes, list) or len(codes) != 4
                or any(type(token) is not int or token < 0 for token in codes)):
            raise ValueError("Catalog entries require four nonnegative integer semantic tokens")
        catalog[video_id] = tuple(codes)
    fingerprint = catalog_fingerprint(catalog)
    for claimed in (expected_fingerprint, report.get("catalog_fingerprint")):
        if claimed is not None and claimed != fingerprint:
            raise ValueError("Catalog fingerprint differs from the source artifact report")
    model = load_model(files["recommender.pt"], catalog=catalog)
    if model.metadata.get("completed") is False:
        raise ValueError("Recommender checkpoint is incomplete")
    if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise ValueError("Recommender checkpoint contains nonfinite weights")
    for history in ([], [next(iter(catalog))]):
        expected = min(3, len(catalog) - len(history))
        if expected == 0:
            continue
        items = recommend(model, catalog, history, top_k=expected)
        ids = [item["video_id"] for item in items]
        if len(ids) != expected or len(set(ids)) != len(ids):
            raise ValueError("Local inference returned an invalid recommendation count")
        for item in items:
            video_id = item["video_id"]
            if (video_id not in catalog or video_id in history
                    or tuple(item["semantic_tokens"]) != catalog[video_id]
                    or not math.isfinite(item["score"])):
                raise ValueError("Local inference returned an invalid recommendation")
    return {
        "catalog_count": len(catalog),
        "catalog_fingerprint": fingerprint,
        "sample_history_ids": list(catalog)[:2],
        "source_kind": report.get("source_kind", report.get("feedback_source", "unknown")),
        "tokenizer_version": report.get("tokenizer_version"),
    }


def registration_kwargs(result, *, source_environment: str, source_api_server: str,
                        source_model_name: str, source_model_version: int, summary: dict) -> dict:
    """Prepare a file-based registration; this function makes no remote writes.

    Schemas returned by Chalk's download API are already in its public accepted
    format (tensor shape/dtype pairs or tabular columns). Preserve those schemas
    and serialized files instead of reserializing or modifying the model.
    """
    source = Target(source_environment, source_api_server)
    if type(source_model_version) is not int or source_model_version < 1:
        raise ValueError("Source model version must be a positive integer")
    if result.model_name != source_model_name or result.model_version != source_model_version:
        raise ValueError("Downloaded artifact does not match the requested source model version")
    model_files, additional_files, _ = _files(result)
    spec = result.model_artifact
    if spec.model_type is None or spec.model_encoding is None:
        raise ValueError("Source artifact must declare its model type and serialization encoding")
    if spec.input_schema is None or spec.output_schema is None:
        raise ValueError("Source artifact must declare input and output schemas")
    metadata = copy.deepcopy(dict(spec.metadata))
    if any(not isinstance(value, (str, int, float, bool)) for value in metadata.values()):
        raise ValueError("Source metadata includes values unsupported by Chalk file registration")
    metadata[IMPORT_METADATA_KEY] = json.dumps({
        "source_environment": source.environment,
        "source_api_server": source.api_server.rstrip("/"),
        "source_model_name": source_model_name,
        "source_model_version": source_model_version,
        "catalog_count": summary["catalog_count"],
        "catalog_fingerprint": summary["catalog_fingerprint"],
        "source_kind": summary["source_kind"],
    }, sort_keys=True, allow_nan=False)
    return {
        "model_type": spec.model_type,
        "model_class": spec.model_class,
        "model_encoding": spec.model_encoding,
        "model_paths": [str(path) for path in model_files],
        "additional_files": [str(path) for path in additional_files],
        "input_schema": copy.deepcopy(spec.input_schema),
        "output_schema": copy.deepcopy(spec.output_schema),
        "metadata": metadata,
        "input_features": list(spec.input_features),
        "output_features": list(spec.output_features),
        "dependencies": list(spec.dependencies),
        # Existing Functions download these artifacts directly. No serving
        # volume or model deployment is required to register a usable copy.
        "skip_volume_upload": True,
    }
