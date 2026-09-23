#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["torch", "numpy", "pandas", "pyarrow"]
# ///
"""Train the RQ-VAE over MicroLens title embeddings and write semantic IDs.

Reports the two numbers that decide whether the IDs are usable: reconstruction
error (are the codes keeping the content signal?) and codebook usage (are they
actually spread over the codebook, or has it collapsed onto a handful of codes?).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rqvae import RQVAE  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = HERE / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=pathlib.Path, default=ARTIFACTS / "title_embeddings.npy")
    parser.add_argument("--titles", type=pathlib.Path, default=ARTIFACTS / "titles.parquet")
    parser.add_argument("--out-dir", type=pathlib.Path, default=ARTIFACTS)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--num-levels", type=int, default=3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--holdout", type=int, default=1000, help="Rows held out to stand in for unseen videos")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    vectors = np.load(args.embeddings)
    titles = pd.read_parquet(args.titles)
    if len(titles) != len(vectors):
        raise SystemExit(f"{len(titles)} titles but {len(vectors)} embeddings")
    print(f"{len(vectors)} vectors of dim {vectors.shape[1]}")

    # A holdout standing in for videos the model never saw, so the cold-start claim is
    # measured on vectors that had no say in the codebooks.
    permutation = rng.permutation(len(vectors))
    holdout_idx = permutation[: args.holdout]
    train_idx = permutation[args.holdout :]
    train = torch.from_numpy(vectors[train_idx])
    holdout = torch.from_numpy(vectors[holdout_idx])

    model = RQVAE(
        input_dim=vectors.shape[1],
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        num_levels=args.num_levels,
        hidden_dims=tuple(args.hidden),
        commitment_weight=args.commitment_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(train.shape[0])
        epoch_recon = epoch_quant = 0.0
        batches = 0
        for start in range(0, train.shape[0], args.batch_size):
            batch = train[order[start : start + args.batch_size]]
            _, _, reconstruction_loss, quantization_loss = model(batch)
            loss = reconstruction_loss + quantization_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_recon += float(reconstruction_loss)
            epoch_quant += float(quantization_loss)
            batches += 1

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            revived = 0
            with torch.no_grad():
                latent = model.encoder(train[:4096])
                residual = latent
                for quantizer in model.quantizers:
                    revived += quantizer.revive_dead_codes(residual)
                    quantized, _, _ = quantizer(residual)
                    residual = residual - quantized
            print(
                f"epoch {epoch:4d} | recon {epoch_recon / batches:.5f} | "
                f"quant {epoch_quant / batches:.5f} | revived {revived}"
            )

    # -- Evaluation --------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        all_vectors = torch.from_numpy(vectors)
        reconstruction, codes, recon_loss, _ = model(all_vectors)
        holdout_recon = float(torch.nn.functional.mse_loss(model(holdout)[0], holdout))
        cosine = torch.nn.functional.cosine_similarity(reconstruction, all_vectors).mean()

    codes_np = codes.numpy()
    usage = [int(len(np.unique(codes_np[:, level]))) for level in range(args.num_levels)]
    unique_ids = len(np.unique(codes_np, axis=0))
    metrics = {
        "train_rows": int(len(train_idx)),
        "holdout_rows": int(len(holdout_idx)),
        "reconstruction_mse_all": float(recon_loss),
        "reconstruction_mse_holdout": holdout_recon,
        "reconstruction_cosine_all": float(cosine),
        "codebook_size": args.codebook_size,
        "codes_used_per_level": usage,
        "unique_semantic_ids": int(unique_ids),
        "collision_rate": float(1 - unique_ids / len(codes_np)),
    }
    print(json.dumps(metrics, indent=2))

    # -- Semantic IDs ------------------------------------------------------------
    # Videos whose content vectors quantize identically share a prefix; a per-collision
    # counter makes the full ID unique, exactly as TIGER's extra token does.
    frame = titles.copy()
    for level in range(args.num_levels):
        frame[f"semantic_id_{level}"] = codes_np[:, level]
    id_columns = [f"semantic_id_{level}" for level in range(args.num_levels)]
    frame["semantic_id_dedup"] = frame.groupby(id_columns).cumcount()
    frame["semantic_id"] = [
        "-".join(str(part) for part in row) for row in frame[id_columns + ["semantic_id_dedup"]].to_numpy()
    ]
    frame["is_holdout"] = frame.index.isin(holdout_idx)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out_dir / "semantic_ids.parquet", index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "input_dim": int(vectors.shape[1]),
                "latent_dim": args.latent_dim,
                "codebook_size": args.codebook_size,
                "num_levels": args.num_levels,
                "hidden_dims": list(args.hidden),
                "commitment_weight": args.commitment_weight,
            },
            "metrics": metrics,
        },
        args.out_dir / "rqvae.pt",
    )
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {args.out_dir / 'rqvae.pt'} and {args.out_dir / 'semantic_ids.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
