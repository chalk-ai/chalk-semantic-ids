"""Train the next-video recommender in Chalk using its native training API.

Run from the repository root: uv run python train-recommender/run.py
"""
from __future__ import annotations

# Let the CLI establish the saved target, Dataset names and checkpoint namespace
# before importing this file under its stable worker module name.
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from semantic_ids.cli import main

    raise SystemExit(main(sys.argv[1:], stage="train-recommender"))

import math
import os
from pathlib import Path
import tempfile
import json

import numpy as np
import pandas as pd
import torch
from chalkcompute import Image
# ChalkCompute 2.13.1's worker imports chalkcompute.training.entrypoint, which
# shadows the top-level export. This is the same native decorator.
from chalkcompute._training import training

from semantic_ids import training_runtime as runtime
from semantic_ids.settings import CPU_TORCH_INSTALL, TRAINING_DEPS

IMAGE = (
    Image.debian_slim(python_version="3.12")
    .run_commands(CPU_TORCH_INSTALL)
    .pip_install(TRAINING_DEPS)
    .add_local_python_source("semantic_ids")
)
KEY = os.environ.get("SEMANTIC_TRAINING_KEY", "configure-before-running")
RUNTIME_ENV = {
    key: os.environ[key] for key in (
        "CHALK_ENVIRONMENT", "CHALK_API_SERVER", "SEMANTIC_TRAINING_KEY",
        "SEMANTIC_TOKENIZER_DATASET", "SEMANTIC_RECOMMENDER_DATASET",
    ) if key in os.environ
}


def _examples(df, catalog):
    required = {"id", "user_id", "source_index", "history_ids", "target_id", "split"}
    frame = runtime._frame(df)
    if not required <= set(frame.columns):
        raise ValueError("Examples need event identity, user, chronological position, history, target, and split")
    rows, users, seen = [], {}, set()
    for row in frame.to_dict("records"):
        event_id, user = row["id"], row["user_id"]
        if not isinstance(event_id, str) or not event_id or event_id in seen or not isinstance(user, str) or not user:
            raise ValueError("Examples need unique event IDs and nonempty users")
        position = runtime._integer(row["source_index"], "source_index", 1)
        history = [runtime._integer(video, "history video") for video in row["history_ids"]]
        target = runtime._integer(row["target_id"], "target video")
        if len(history) != position or target not in catalog or any(video not in catalog for video in history):
            raise ValueError("Example history is incomplete or refers to an unknown catalog item")
        item = {"event_id": event_id, "user_id": user, "source_index": position,
                "history": history, "target": target, "split": row["split"]}
        seen.add(event_id)
        users.setdefault(user, []).append(item)
        rows.append(item)
    for user_rows in users.values():
        user_rows.sort(key=lambda row: row["source_index"])
        if len(user_rows) < 3 or [row["source_index"] for row in user_rows] != list(range(1, len(user_rows) + 1)):
            raise ValueError("Examples must retain every chronological training and held-out event")
        if [row["split"] for row in user_rows] != ["train"] * (len(user_rows) - 2) + ["validation", "test"]:
            raise ValueError("Training must precede the final validation and test events")
        for before, after in zip(user_rows, user_rows[1:]):
            if after["history"] != before["history"] + [before["target"]]:
                raise ValueError("Example histories disagree across split boundaries")
    return sorted(rows, key=lambda row: row["event_id"])


def _catalog(config):
    from semantic_ids.data import build_catalog
    if not isinstance(config.get("catalog_revision"), str) or not config["catalog_revision"].strip():
        raise ValueError("An explicit catalog_revision is required")
    if "catalog_rows" in config:
        frame = runtime._frame(pd.DataFrame(config["catalog_rows"]))
    else:
        from chalk.client import ChalkClient
        client = ChalkClient()
        dataset = client.get_dataset(revision_id=config["catalog_revision"])
        frame = runtime._frame(dataset.to_pandas(output_id=False, output_ts=False))
        if "tokenizer_version" not in frame:
            raise ValueError("Pinned Chalk catalog is missing its tokenizer version")
    if "tokenizer_version" in frame:
        expected = runtime._integer(config.get("tokenizer_version"), "tokenizer_version", 1)
        if {runtime._integer(value, "catalog tokenizer_version", 1) for value in frame["tokenizer_version"]} != {expected}:
            raise ValueError("Catalog tokenizer version differs from the pinned configuration")
    records = frame.to_dict("records")
    for row in records:
        # Arrow list columns arrive in pandas as NumPy arrays.
        if isinstance(row.get("codes"), np.ndarray):
            row["codes"] = row["codes"].tolist()
    return build_catalog(records)


def run_body(df, config):
    """Train on earlier watches and report validation/test metrics without selection."""
    from semantic_ids import model as api
    config = runtime._config(config)
    records = _catalog(config)
    catalog = {row["video_id"]: tuple(row["semantic_tokens"]) for row in records}
    rows = _examples(df, catalog)
    identity = runtime._identity("recommender", config, rows, catalog=records)
    state = runtime._restore(identity)
    torch.set_num_threads(config["threads"])
    epochs = runtime._integer(config.get("epochs", 3), "epochs", 1)
    for key, default in (("hidden_dim", 64), ("batch_size", 256), ("max_history", 20), ("top_k", 10), ("beam_width", 32)):
        runtime._integer(config.get(key, default), key, 1)
    learning_rate = config.get("learning_rate", 0.003)
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not 0 < learning_rate < math.inf:
        raise ValueError("learning_rate must be finite and positive")
    with tempfile.TemporaryDirectory(prefix="chalk-recommender-") as directory:
        workdir = Path(directory)
        def on_epoch(model, progress, metrics):
            nonlocal state
            state = progress
            runtime.log_metrics(metrics, tags={"split": "train", "epoch": str(progress["epoch"])})
            if progress["epoch"] == 1 or progress["epoch"] % config["checkpoint_every"] == 0 or progress["epoch"] == epochs:
                runtime._save(model, workdir, identity, state, {**identity, "completed": False, "epoch": progress["epoch"]})
        trained, training_metrics = api.train_model(catalog, [row for row in rows if row["split"] == "train"],
            epochs=epochs, hidden_dim=config.get("hidden_dim", 64), seed=config["seed"],
            batch_size=config.get("batch_size", 256), learning_rate=config.get("learning_rate", 0.003),
            max_history=config.get("max_history", 20), resume_state=state, on_epoch=on_epoch)
        metrics = {"training": training_metrics}
        for split in ("validation", "test"):
            metrics[split] = api.evaluate_model(trained, catalog, [row for row in rows if row["split"] == split],
                top_k=config.get("top_k", 10), beam_width=config.get("beam_width", 32))
            runtime.log_metrics({"loss": metrics[split]["token_nll"], "recall": metrics[split]["recall_at_k"]}, tags={"split": split})
        report = {**identity, "completed": True, "catalog_revision": config["catalog_revision"],
                  "tokenizer_version": config.get("tokenizer_version"), "metrics": metrics,
                  "test_used_for_selection": False, "interpretation": "Historical next-video prediction; not an online engagement experiment."}
        artifact, catalog_path = workdir / "recommender.pt", workdir / "catalog.json"
        api.save_model(trained, artifact, catalog=catalog, metadata=report)
        catalog_path.write_text(json.dumps(records, indent=2, sort_keys=True, allow_nan=False))
        runtime._save(trained, workdir, identity, state, report, [artifact, catalog_path])
        return trained

@training(
    data=os.environ.get("SEMANTIC_RECOMMENDER_DATASET", "semantic-watches-unconfigured"),
    name=f"semantic-recommender-{KEY}", cpu="2", memory="8Gi", image=IMAGE,
    max_retries=2, env=RUNTIME_ENV,
)
def train_recommender(df, config):
    return run_body(df, config)
