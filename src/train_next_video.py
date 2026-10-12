#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["torch", "numpy", "pandas", "pyarrow"]
# ///
"""Train the next-video transformer on MicroLens watch sequences, and measure it.

Split, per user (leave-one-out, as in SASRec): train on every watch but the last two,
pick the epoch on the second-to-last, report on the last.

The 1,000 videos the RQ-VAE held out stand in for **new uploads**: every watch of them
is removed from training, and they are left out of the training softmax, so the model
never learns they exist. They come back at test time, ranked against every other video.
Test cases whose next watch is one of them measure cold start directly.

    ./train_next_video.py --item-repr semantic      # video = sum of its semantic-ID tokens
    ./train_next_video.py --item-repr id            # video = its own learned vector
    ./train_next_video.py --item-repr hybrid        # both; unseen videos use tokens only
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from next_video_model import PAD, NextVideoTransformer, masked_cross_entropy, score_positions

HERE = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = HERE / "artifacts"
K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--item-repr", choices=["semantic", "id", "hybrid"], default="hybrid")
    parser.add_argument("--id-dropout", type=float, default=0.3, help="hybrid only: drop each video's own vector")
    parser.add_argument("--pairs", type=pathlib.Path, default=HERE / "data" / "MicroLens-100k_pairs.tsv")
    parser.add_argument("--ids", type=pathlib.Path, default=ARTIFACTS / "semantic_ids.parquet")
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4, help="Evaluations without a validation gain before stopping")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_sequences(path: pathlib.Path) -> list[list[int]]:
    sequences = []
    with path.open() as handle:
        for line in handle:
            _, _, watched = line.rstrip("\n").partition("\t")
            sequences.append([int(video) for video in watched.split()])
    return sequences


def left_pad(rows: list[list[int]], length: int) -> torch.Tensor:
    out = torch.zeros(len(rows), length, dtype=torch.long)
    for index, row in enumerate(rows):
        row = row[-length:]
        if row:
            out[index, length - len(row) :] = torch.tensor(row)
    return out


@torch.no_grad()
def rank_metrics(
    model: NextVideoTransformer,
    histories: list[list[int]],
    targets: list[int],
    max_len: int,
    cold: np.ndarray,
    batch_size: int = 2048,
) -> dict[str, float]:
    """Recall@K and NDCG@K over all videos, overall and split by whether the target is cold."""
    model.eval()
    ranks = np.empty(len(targets), dtype=np.int64)
    for start in range(0, len(targets), batch_size):
        batch_histories = histories[start : start + batch_size]
        history = left_pad(batch_histories, max_len)
        scores = model(history)
        scores[:, PAD] = float("-inf")
        # Nobody rewatches a video in this dataset, so already-watched ones are not candidates.
        for row, watched in enumerate(batch_histories):
            scores[row, watched] = float("-inf")
        target = torch.tensor(targets[start : start + batch_size])
        target_score = scores.gather(1, target[:, None])
        ranks[start : start + len(target)] = (scores > target_score).sum(dim=1).numpy()
    return summarize(ranks, np.array([cold[t] for t in targets]))


def summarize(ranks: np.ndarray, is_cold: np.ndarray) -> dict[str, float]:
    hit = ranks < K
    ndcg = np.where(hit, 1.0 / np.log2(ranks + 2.0), 0.0)
    out: dict[str, float] = {}
    for label, mask in (("all", np.ones_like(is_cold)), ("warm", ~is_cold), ("cold", is_cold)):
        if not mask.any():
            continue
        out[f"recall@{K}_{label}"] = float(hit[mask].mean())
        out[f"ndcg@{K}_{label}"] = float(ndcg[mask].mean())
        out[f"cases_{label}"] = int(mask.sum())
    return out


def popularity_metrics(
    train: list[list[int]], histories: list[list[int]], targets: list[int], num_videos: int, cold: np.ndarray
) -> dict[str, float]:
    counts = np.zeros(num_videos + 1)
    for sequence in train:
        np.add.at(counts, sequence, 1)
    order = np.argsort(-counts, kind="stable")
    position = np.empty_like(order)
    position[order] = np.arange(len(order))
    ranks = []
    for watched, target in zip(histories, targets, strict=True):
        # Rank among unwatched videos: subtract watched videos that outrank the target.
        ranks.append(position[target] - int((position[watched] < position[target]).sum()))
    return summarize(np.array(ranks), np.array([cold[t] for t in targets]))


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)

    ids = pd.read_parquet(args.ids).sort_values("video_id")
    num_videos = int(ids.video_id.max())
    if not (ids.video_id.to_numpy() == np.arange(1, num_videos + 1)).all():
        raise SystemExit("video ids must be 1..N so they can index the embedding table directly")
    level_columns = sorted(c for c in ids.columns if c.startswith("semantic_id_") and c[-1].isdigit())
    tokens = np.zeros((num_videos + 1, len(level_columns)), dtype=np.int64)
    tokens[1:] = ids[level_columns].to_numpy()
    cold = np.zeros(num_videos + 1, dtype=bool)
    cold[ids.video_id[ids.is_holdout].to_numpy()] = True

    sequences = load_sequences(args.pairs)
    train = [[v for v in seq[:-2] if not cold[v]] for seq in sequences]
    train = [seq for seq in train if len(seq) >= 2]
    valid_users = [seq for seq in sequences if not cold[seq[-2]]]
    valid_histories = [seq[:-2] for seq in valid_users]
    valid_targets = [seq[-2] for seq in valid_users]
    test_histories = [seq[:-1] for seq in sequences]
    test_targets = [seq[-1] for seq in sequences]
    print(
        f"{len(sequences)} users, {num_videos} videos ({int(cold.sum())} cold), "
        f"{sum(map(len, train))} training watches, {len(test_targets)} test cases "
        f"({int(cold[test_targets].sum())} with a cold target)"
    )

    # Training never sees a cold video, not even as a negative: it has not been uploaded yet.
    excluded = torch.from_numpy(cold.copy())
    excluded[PAD] = True

    trained_videos = np.zeros(num_videos + 1, dtype=bool)
    for sequence in train:
        trained_videos[sequence] = True
    model = NextVideoTransformer(
        num_videos=num_videos,
        item_repr=args.item_repr,
        semantic_tokens=torch.from_numpy(tokens) if args.item_repr != "id" else None,
        trained_videos=torch.from_numpy(trained_videos),
        id_dropout=args.id_dropout,
        codebook_size=int(tokens.max()) + 1,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        max_len=args.max_len,
        dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, item_repr={args.item_repr}")

    best_ndcg, best_state, best_epoch, stale = -1.0, None, -1, 0
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train)
        started, total_loss, steps = time.perf_counter(), 0.0, 0
        for start in range(0, len(train), args.batch_size):
            batch = train[start : start + args.batch_size]
            length = min(args.max_len, max(len(seq) - 1 for seq in batch))
            history = left_pad([seq[:-1] for seq in batch], length)
            target = left_pad([seq[1:] for seq in batch], length)
            keep = target != PAD
            logits = score_positions(model, history, keep)
            loss = masked_cross_entropy(logits, target[keep], excluded)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        valid = rank_metrics(model, valid_histories, valid_targets, args.max_len, cold)
        print(
            f"epoch {epoch:3d} | loss {total_loss / steps:.4f} | valid ndcg@{K} {valid[f'ndcg@{K}_all']:.4f} "
            f"recall@{K} {valid[f'recall@{K}_all']:.4f} | {time.perf_counter() - started:.0f}s",
            flush=True,
        )
        if valid[f"ndcg@{K}_all"] > best_ndcg:
            best_ndcg, best_epoch, stale = valid[f"ndcg@{K}_all"], epoch, 0
            best_state = {name: value.clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    test = rank_metrics(model, test_histories, test_targets, args.max_len, cold)
    popularity = popularity_metrics(train, test_histories, test_targets, num_videos, cold)
    report = {"item_repr": args.item_repr, "best_epoch": best_epoch, "test": test, "popularity_test": popularity}
    print(json.dumps(report, indent=2))

    config = {
        "num_videos": num_videos,
        "item_repr": args.item_repr,
        "codebook_size": int(tokens.max()) + 1,
        "dim": args.dim,
        "layers": args.layers,
        "heads": args.heads,
        "max_len": args.max_len,
        "dropout": args.dropout,
        "id_dropout": args.id_dropout,
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "semantic_tokens": tokens,
            "trained_videos": trained_videos,
            "metrics": report,
        },
        ARTIFACTS / f"next_video_{args.item_repr}.pt",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
