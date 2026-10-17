#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["torch", "sentence-transformers==5.1.2", "onnxruntime", "tokenizers", "numpy", "pandas", "pyarrow"]
# ///
"""Export the RQ-VAE for the torch-free serving path, and prove it assigns the same IDs.

Writes `artifacts/rqvae.npz` (encoder MLP + codebooks; the decoder is training-only),
downloads the pinned ONNX encoder, then runs every MicroLens title through both the
torch path (`SentenceTransformer` + `RQVAE.encode_ids`) and `SemanticIdEncoder`, one
title per call as the deployed function sees them. Exits non-zero unless every title
gets identical tokens: a serving path that drifts from training mints wrong IDs quietly.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rqvae import RQVAE
from semantic_id_runtime import ENCODER_FILES, SemanticIdEncoder, encoder_file_url

HERE = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = HERE / "artifacts"


def load_torch_rqvae(path: pathlib.Path) -> RQVAE:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = RQVAE(
        input_dim=config["input_dim"],
        latent_dim=config["latent_dim"],
        codebook_size=config["codebook_size"],
        num_levels=config["num_levels"],
        hidden_dims=tuple(config["hidden_dims"]),
        commitment_weight=config["commitment_weight"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for quantizer in model.quantizers:
        quantizer._initialized = True
    return model


def export_npz(model: RQVAE, out: pathlib.Path) -> None:
    linears = [layer for layer in model.encoder if isinstance(layer, torch.nn.Linear)]
    arrays: dict[str, np.ndarray] = {
        "encoder_layers": np.array(len(linears)),
        "num_levels": np.array(len(model.quantizers)),
    }
    for index, layer in enumerate(linears):
        arrays[f"encoder_{index}_weight"] = layer.weight.detach().numpy().astype(np.float32)
        arrays[f"encoder_{index}_bias"] = layer.bias.detach().numpy().astype(np.float32)
    for level, quantizer in enumerate(model.quantizers):
        arrays[f"codebook_{level}"] = quantizer.embedding.weight.detach().numpy().astype(np.float32)
    np.savez(out, **arrays)


def fetch_encoder(directory: pathlib.Path) -> None:
    for filename in ENCODER_FILES:
        target = directory / filename
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(encoder_file_url(filename), target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=ARTIFACTS / "rqvae.pt")
    parser.add_argument("--titles", type=pathlib.Path, default=ARTIFACTS / "titles.parquet")
    parser.add_argument("--encoder-dir", type=pathlib.Path, default=HERE / ".cache" / "minilm-onnx")
    parser.add_argument("--limit", type=int, default=None, help="Check only the first N titles")
    args = parser.parse_args()

    model = load_torch_rqvae(args.checkpoint)
    npz = ARTIFACTS / "rqvae.npz"
    export_npz(model, npz)
    print(f"wrote {npz} ({npz.stat().st_size / 1e6:.1f} MB)")

    fetch_encoder(args.encoder_dir)
    runtime = SemanticIdEncoder(args.encoder_dir, npz)

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    titles = pd.read_parquet(args.titles).title.tolist()[: args.limit]

    torch_vectors = encoder.encode(titles, batch_size=256, convert_to_numpy=True, normalize_embeddings=True)
    with torch.no_grad():
        torch_codes = model.encode_ids(torch.from_numpy(torch_vectors).float()).numpy()

    runtime_vectors = np.concatenate([runtime.embed([title]) for title in titles])
    runtime_codes = runtime.quantize(runtime_vectors)

    agree = (torch_codes == runtime_codes).all(axis=1)
    report = {
        "titles": len(titles),
        "identical_ids": int(agree.sum()),
        "max_abs_embedding_diff": float(np.abs(torch_vectors - runtime_vectors).max()),
        "mismatches": [
            {"title": titles[i], "torch": torch_codes[i].tolist(), "runtime": runtime_codes[i].tolist()}
            for i in np.flatnonzero(~agree)[:20]
        ],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if agree.all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
