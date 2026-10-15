"""Shared native checkpoint recovery and training identity validation.

Numerical training bodies live in train/run.py and train-recommender/run.py.
The forwarding functions retain the existing local Python interface.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import uuid

import pandas as pd
import torch
from chalk.ml.chalk_train import checkpoint, last_checkpoint_path, log_metrics

from .settings import ENCODER_MODEL, ENCODER_REVISION, MAX_TITLE_LENGTH


def _normalize(value):
    # Protobuf Struct transports integer hyperparameters as doubles.
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _frame(df):
    if hasattr(df, "collect"):
        df = df.collect()
    if not isinstance(df, pd.DataFrame) and hasattr(df, "to_pandas"):
        df = df.to_pandas()
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Training requires a nonempty pinned dataset")
    frame = df.copy()
    frame.columns = [str(column).rsplit(".", 1)[-1] for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Dataset has ambiguous feature column names")
    return frame


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _config(config):
    if not isinstance(config, dict):
        raise ValueError("Training config must be an object")
    config = _normalize(config)
    for key in ("dataset_revision",):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"An explicit {key} is required")
    for key, default, minimum in (("seed", 0, 0), ("threads", 2, 1), ("checkpoint_every", 1, 1)):
        config[key] = _integer(config.get(key, default), key, minimum)
    return config


def _identity(kind, config, rows, *, catalog=None):
    run_id = os.environ.get("CHALK_TRAINING_RUN_ID")
    if not run_id:
        raise ValueError("CHALK_TRAINING_RUN_ID is required inside native training")
    if os.environ.get("CHALK_DATASET_REVISION_ID") != config["dataset_revision"]:
        raise ValueError("Native training received a different dataset revision")
    return {"kind": kind, "run_id": run_id, "dataset_revision": config["dataset_revision"],
            "config_sha256": _digest(config), "data_sha256": _digest(rows),
            "catalog_sha256": _digest(catalog)}


def _restore(identity):
    latest = last_checkpoint_path()
    if latest is None:
        return None
    path = Path(latest)
    root = path if path.is_dir() else path.parent
    recovery = root / "recovery_state.pt"
    if not path.exists() or not recovery.is_file():
        raise ValueError("Latest checkpoint is missing its numerical recovery state")
    saved = torch.load(recovery, map_location="cpu", weights_only=True)
    if (not isinstance(saved, dict) or saved.get("format_version") != 1 or saved.get("identity") != identity
            or not isinstance(saved.get("state"), dict)):
        # Storage is scoped to a training name; its latest checkpoint may be from
        # another run. Never silently warm-start from that unrelated checkpoint.
        raise ValueError("Checkpoint belongs to a different run, configuration, or dataset")
    return saved["state"]


def _save(model, workdir, identity, state, report, files=()):
    recovery = workdir / "recovery_state.pt"
    report_path = workdir / "report.json"
    torch.save({"format_version": 1, "identity": identity, "state": state}, recovery)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    metadata = {"kind": identity["kind"], "completed": report["completed"],
                "report_json": json.dumps(report, sort_keys=True, allow_nan=False)}
    for key in ("encoder_model", "encoder_revision"):
        if key in report:
            metadata[key] = report[key]
    random_state = torch.get_rng_state()
    try:
        artifact = checkpoint(model, additional_files=[*map(str, files), str(recovery), str(report_path)],
            metadata=metadata, run_name=identity["run_id"], path=f"{identity['kind']}-{uuid.uuid4().hex}")
        if not getattr(artifact, "artifact_id", None):
            raise RuntimeError("Chalk did not return a checkpoint artifact ID")
    finally:
        torch.set_rng_state(random_state)


def train_tokenizer(df, config):
    """Compatibility entry point for the tokenizer stage's numerical body."""
    from .stages import load_stage

    return load_stage("train").run_body(df, config)


def train_recommender(df, config):
    """Compatibility entry point for the recommender stage's numerical body."""
    from .stages import load_stage

    return load_stage("train-recommender").run_body(df, config)


def _examples(df, catalog):
    from .stages import load_stage

    return load_stage("train-recommender")._examples(df, catalog)


def _catalog(config):
    from .stages import load_stage

    return load_stage("train-recommender")._catalog(config)
