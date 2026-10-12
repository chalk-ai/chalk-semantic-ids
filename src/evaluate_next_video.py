#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["torch", "numpy", "pandas", "pyarrow"]
# ///
"""Compare the trained next-video models on two surfaces, and write `artifacts/next_video_metrics.json`.

- **Home feed**: rank each user's last watch among every video they have not watched.
- **New-uploads shelf**: for users whose last watch is one of the 1,000 cold videos, rank
  it among the cold videos only. This is the surface where cold start matters: a shelf of
  fresh uploads, where nothing has watch history and popularity cannot order them.

    ./evaluate_next_video.py                      # every artifacts/next_video_*.pt
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
from export_next_video import load_model
from next_video_model import PAD
from train_next_video import K, left_pad, load_sequences, popularity_metrics, summarize

HERE = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = HERE / "artifacts"


def shelf_metrics(ranks: np.ndarray) -> dict[str, float]:
    hit = ranks < K
    return {
        f"recall@{K}": float(hit.mean()),
        f"ndcg@{K}": float(np.where(hit, 1.0 / np.log2(ranks + 2.0), 0.0).mean()),
        "cases": len(ranks),
    }


@torch.no_grad()
def model_ranks(
    model: torch.nn.Module, histories: list[list[int]], targets: list[int], cold: np.ndarray, max_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rank of each target among all unwatched videos, and among unwatched cold videos (cold targets only)."""
    feed = np.empty(len(targets), dtype=np.int64)
    shelf = []
    cold_tensor = torch.from_numpy(cold)
    for start in range(0, len(targets), 2048):
        batch = histories[start : start + 2048]
        scores = model(left_pad(batch, max_len))
        scores[:, PAD] = float("-inf")
        for row, watched in enumerate(batch):
            scores[row, watched] = float("-inf")
        target = torch.tensor(targets[start : start + 2048])
        target_score = scores.gather(1, target[:, None])
        feed[start : start + len(target)] = (scores > target_score).sum(dim=1).numpy()
        for row in torch.nonzero(cold_tensor[target]).flatten().tolist():
            candidates = scores[row, cold_tensor]
            shelf.append(int((candidates > target_score[row]).sum()))
    return feed, np.array(shelf)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=pathlib.Path, nargs="*", default=None)
    parser.add_argument("--pairs", type=pathlib.Path, default=HERE / "data" / "MicroLens-100k_pairs.tsv")
    parser.add_argument("--ids", type=pathlib.Path, default=ARTIFACTS / "semantic_ids.parquet")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(32)

    ids = pd.read_parquet(args.ids)
    num_videos = int(ids.video_id.max())
    cold = np.zeros(num_videos + 1, dtype=bool)
    cold[ids.video_id[ids.is_holdout].to_numpy()] = True

    sequences = load_sequences(args.pairs)
    histories = [sequence[:-1] for sequence in sequences]
    targets = [sequence[-1] for sequence in sequences]
    is_cold = cold[targets]
    train = [[v for v in sequence[:-2] if not cold[v]] for sequence in sequences]

    # Random ranking: the target's rank is uniform over the candidates it competes with.
    rng = np.random.default_rng(args.seed)
    candidates = np.array([num_videos - len(set(h)) for h in histories])
    random_feed = (rng.random(len(targets)) * candidates).astype(np.int64)
    cold_candidates = np.array([cold.sum() - cold[h].sum() for h, c in zip(histories, is_cold, strict=True) if c])
    random_shelf = (rng.random(len(cold_candidates)) * cold_candidates).astype(np.int64)

    report: dict[str, dict] = {
        "random": {"feed": summarize(random_feed, is_cold), "new_uploads_shelf": shelf_metrics(random_shelf)},
        "popularity": {"feed": popularity_metrics(train, histories, targets, num_videos, cold)},
    }
    for path in args.checkpoints or sorted(ARTIFACTS.glob("next_video_*.pt")):
        model = load_model(path)
        feed, shelf = model_ranks(model, histories, targets, cold, model.max_len)
        report[model.item_repr] = {"feed": summarize(feed, is_cold), "new_uploads_shelf": shelf_metrics(shelf)}

    print(f"{'model':<11} {'feed R@10':>10} {'warm':>8} {'cold':>8} {'shelf R@10':>11} {'shelf NDCG':>11}")
    for name, metrics in report.items():
        feed, shelf = metrics["feed"], metrics.get("new_uploads_shelf", {})
        print(
            f"{name:<11} {feed[f'recall@{K}_all']:>10.4f} {feed[f'recall@{K}_warm']:>8.4f} "
            f"{feed[f'recall@{K}_cold']:>8.4f} {shelf.get(f'recall@{K}', float('nan')):>11.4f} "
            f"{shelf.get(f'ndcg@{K}', float('nan')):>11.4f}"
        )
    (ARTIFACTS / "next_video_metrics.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
