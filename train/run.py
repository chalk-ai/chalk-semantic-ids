"""Train the title tokenizer in Chalk using its native training API.

Run from the repository root: uv run python train/run.py
"""
from __future__ import annotations

# Let the CLI establish the saved target, Dataset names and checkpoint namespace
# before importing this file under its stable worker module name.
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from semantic_ids.cli import main

    raise SystemExit(main(sys.argv[1:], stage="train-tokenizer"))

import math
import os
from pathlib import Path
import tempfile
import hashlib

import numpy as np
import torch
from chalkcompute import Image
# ChalkCompute 2.13.1's worker imports chalkcompute.training.entrypoint, which
# shadows the top-level export. This is the same native decorator.
from chalkcompute._training import training

from semantic_ids import training_runtime as runtime
from semantic_ids.settings import CPU_TORCH_INSTALL, TRAINING_DEPS, ENCODER_MODEL, ENCODER_REVISION, MAX_TITLE_LENGTH

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


def _encode_titles(titles, config):
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(config["encoder_model"], revision=config["encoder_revision"], device="cpu")
    return encoder.encode(titles, batch_size=64, normalize_embeddings=True,
                          show_progress_bar=False, convert_to_numpy=True)


def run_body(df, config):
    """Encode raw titles with pinned MiniLM, then train and checkpoint an RQ-VAE."""
    from semantic_ids.rqvae import RQVAE

    config = runtime._config(config)
    config.setdefault("encoder_model", ENCODER_MODEL)
    config.setdefault("encoder_revision", ENCODER_REVISION)
    if (config["encoder_model"] != ENCODER_MODEL or not isinstance(config["encoder_revision"], str)
            or len(config["encoder_revision"]) != 40
            or any(character not in "0123456789abcdef" for character in config["encoder_revision"])):
        raise ValueError("Tokenizer requires MiniLM with an immutable encoder revision")
    frame = runtime._frame(df)
    if not {"id", "title"} <= set(frame.columns) or frame["id"].duplicated().any():
        raise ValueError("Tokenizer data must contain unique video IDs and titles")
    rows = sorted([{"id": runtime._integer(row["id"], "video ID"), "title": row["title"]}
                   for row in frame.to_dict("records")], key=lambda row: row["id"])
    if any(not isinstance(row["title"], str) or len(row["title"]) > MAX_TITLE_LENGTH for row in rows):
        raise ValueError(f"Video titles must be strings of at most {MAX_TITLE_LENGTH:,} characters")
    identity = runtime._identity("tokenizer", config, rows)
    state = runtime._restore(identity)
    torch.set_num_threads(config["threads"])
    values = np.asarray(_encode_titles([row["title"] for row in rows], config), dtype=np.float32)
    if values.shape != (len(rows), 384) or not np.isfinite(values).all():
        raise ValueError("MiniLM must return finite 384-dimensional title embeddings")
    epochs = runtime._integer(config.get("epochs", 400), "epochs", 1)
    batch_size = runtime._integer(config.get("batch_size", 1024), "batch_size", 1)
    holdout = runtime._integer(config.get("holdout", min(1000, max(1, len(rows) // 10))), "holdout", 1)
    if holdout >= len(rows):
        raise ValueError("Tokenizer holdout must leave training rows")
    model_config = {"input_dim": 384, "latent_dim": runtime._integer(config.get("latent_dim", 32), "latent_dim", 1),
                    "codebook_size": runtime._integer(config.get("codebook_size", 256), "codebook_size", 1), "num_levels": 3,
                    "hidden_dims": [runtime._integer(size, "hidden dimension", 1) for size in config.get("hidden_dims", [512, 256, 128])],
                    "commitment_weight": 0.25}
    learning_rate = config.get("learning_rate", 0.001)
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not 0 < learning_rate < math.inf:
        raise ValueError("learning_rate must be finite and positive")
    signature = {"model_config": model_config, "embeddings_sha256": hashlib.sha256(values.tobytes()).hexdigest()}
    torch.manual_seed(config["seed"])
    indices = np.random.default_rng(config["seed"]).permutation(len(rows))
    train, heldout = torch.tensor(values[indices[holdout:]]), torch.tensor(values[indices[:holdout]])
    model = RQVAE(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    shuffle = torch.Generator().manual_seed(config["seed"])
    start = 0
    if state is not None:
        if state.get("signature") != signature:
            raise ValueError("Checkpoint tokenizer architecture or title embeddings differ")
        start = runtime._integer(state.get("epoch"), "checkpoint epoch")
        flags = state.get("quantizer_initialized")
        if start > epochs or not isinstance(flags, list) or len(flags) != 3 or any(type(flag) is not bool for flag in flags):
            raise ValueError("Checkpoint tokenizer progress is invalid")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        for quantizer, initialized in zip(model.quantizers, flags):
            quantizer._initialized = initialized
        shuffle.set_state(state["shuffle_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
    with tempfile.TemporaryDirectory(prefix="chalk-tokenizer-") as directory:
        workdir = Path(directory)
        for epoch in range(start, epochs):
            model.train()
            total = 0.0
            for batch in torch.randperm(len(train), generator=shuffle).split(batch_size):
                _, _, reconstruction, quantization = model(train[batch])
                loss = reconstruction + quantization
                if not torch.isfinite(loss):
                    raise ValueError("Tokenizer training loss is not finite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch)
            if epoch % 20 == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    residual = model.encoder(train[:4096])
                    for quantizer in model.quantizers:
                        quantizer.revive_dead_codes(residual)
                        quantized, _, _ = quantizer(residual)
                        residual = residual - quantized
            state = {"signature": signature, "epoch": epoch + 1, "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(), "quantizer_initialized": [q._initialized for q in model.quantizers],
                     "shuffle_rng_state": shuffle.get_state(), "torch_rng_state": torch.get_rng_state()}
            try:
                runtime.log_metrics({"loss": total / len(train)}, tags={"split": "train", "epoch": str(epoch + 1)})
                if epoch == 0 or (epoch + 1) % config["checkpoint_every"] == 0 or epoch + 1 == epochs:
                    runtime._save(model, workdir, identity, state, {**identity, "completed": False, "epoch": epoch + 1})
            finally:
                torch.set_rng_state(state["torch_rng_state"])
                shuffle.set_state(state["shuffle_rng_state"])
        model.eval()
        with torch.inference_mode():
            mse = float(torch.nn.functional.mse_loss(model(heldout)[0], heldout))
            codes = model.encode_ids(torch.tensor(values)).tolist()
        if not math.isfinite(mse):
            raise ValueError("Tokenizer holdout reconstruction error is not finite")
        report = {**identity, "completed": True, "encoder_model": config["encoder_model"],
                  "encoder_revision": config["encoder_revision"], "model_config": model_config,
                  "metrics": {"reconstruction_mse_holdout": mse, "train_rows": len(train), "holdout_rows": holdout,
                              "epochs": epochs, "codes_used_per_level": [len({row[level] for row in codes}) for level in range(3)]},
                  "holdout_video_ids": [rows[index]["id"] for index in indices[:holdout]]}
        model.metadata = report
        artifact = workdir / "rqvae.pt"
        torch.save({"state_dict": model.state_dict(), "config": model_config, "metadata": report,
                    "quantizer_initialized": [q._initialized for q in model.quantizers]}, artifact)
        runtime.log_metrics({"mse": mse}, tags={"split": "validation"})
        runtime._save(model, workdir, identity, state, report, [artifact])
        return model

@training(
    data=os.environ.get("SEMANTIC_TOKENIZER_DATASET", "semantic-titles-unconfigured"),
    name=f"semantic-tokenizer-{KEY}", cpu="2", memory="8Gi", image=IMAGE,
    max_retries=2, env=RUNTIME_ENV,
)
def train_tokenizer(df, config):
    return run_body(df, config)
