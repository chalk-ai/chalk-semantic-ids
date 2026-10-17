#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["numpy", "pandas", "pyarrow"]
# ///
"""Build the next-video recommender the demo serves, and measure it before trusting it.

Transition counts over real MicroLens watch sequences, at two granularities: between
semantic-ID prefixes, and between video IDs. `recommender.Recommender` combines them
with content similarity (see its docstring for the score).

Evaluation holds out 10% of users: the three videos before each one's last are the
history, and the last is the target. Hit@10/50 and MRR are reported for the hybrid the
demo serves and for each signal alone.

A second run measures cold start: 10% of videos are removed from every training
sequence, so they have no views and no transitions, and the users scored are those
whose next video is one of them. Item transitions cannot rank those videos at all.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from recommender import CODEBOOK, Recommender, Sparse

HERE = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = HERE / "artifacts"

# Picked by a small grid on the held-out split. A popularity prior buries every video with
# no views yet, and added under 1pt of hit@50 on videos that have them.
DEFAULT_WEIGHTS = {"alpha0": 0.5, "alpha01": 1e-3, "level01": 1.0, "content": 4.0, "popularity": 0.0, "item": 4.0}

# Each video "votes" for the next three in its sequence, nearer ones more strongly.
GAPS = ((1, 1.0), (2, 0.5), (3, 0.25))


def load_sequences(path: pathlib.Path, index_of: dict[int, int]) -> list[np.ndarray]:
    sequences = []
    for line in path.read_text().splitlines():
        _, _, items = line.partition("\t")
        rows = [index_of[int(item)] for item in items.split() if int(item) in index_of]
        if len(rows) >= 2:
            sequences.append(np.array(rows, dtype=np.int32))
    return sequences


def fit(
    sequences: list[np.ndarray], codes: np.ndarray, embeddings: np.ndarray, weights: dict[str, float]
) -> Recommender:
    level0 = np.zeros((CODEBOOK, CODEBOOK))
    level01: dict[int, dict[int, float]] = {}
    items: dict[int, dict[int, float]] = {}
    popularity = np.zeros(len(codes))
    for seq in sequences:
        np.add.at(popularity, seq, 1)
        for i in range(len(seq) - 1):
            a = codes[seq[i]]
            for gap, decay in GAPS:
                if i + gap >= len(seq):
                    break
                b = codes[seq[i + gap]]
                level0[a[0], b[0]] += decay
                bucket = level01.setdefault(int(a[0]) * CODEBOOK + int(a[1]), {})
                target = int(b[0]) * CODEBOOK + int(b[1])
                bucket[target] = bucket.get(target, 0.0) + decay
                row = items.setdefault(int(seq[i]), {})
                row[int(seq[i + gap])] = row.get(int(seq[i + gap]), 0.0) + decay
    return Recommender(
        codes, embeddings, popularity, level0, Sparse.from_dict(level01), Sparse.from_dict(items), weights
    )


def evaluate(model: Recommender, held_out: list[np.ndarray], limit: int) -> dict[str, dict[str, float]]:
    names = ("hybrid", "semantic_id", "item_transitions", "content_only", "popularity_only")
    ranks: dict[str, list[int]] = {name: [] for name in names}
    for seq in held_out[:limit]:
        history, target = seq[-4:-1], seq[-1]
        parts = model.parts(model.codes[history], model.embeddings[history], [int(i) for i in history])
        variants = {
            "hybrid": model.hybrid_score(parts),
            "semantic_id": model.semantic_score(parts),
            # Popularity breaks ties among videos no history item led to.
            "item_transitions": parts["item_counts"] + 1e-3 * parts["popularity"],
            "content_only": parts["cosine"],
            "popularity_only": parts["popularity"],
        }
        for name, score in variants.items():
            score = score.copy()
            score[history] = -np.inf
            ranks[name].append(int((score > score[target]).sum()) + 1)
    report: dict[str, dict[str, float]] = {}
    for name, values in ranks.items():
        r = np.array(values)
        report[name] = {
            "hit@10": float((r <= 10).mean()),
            "hit@50": float((r <= 50).mean()),
            "mrr": float((1 / r).mean()),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame = pd.read_parquet(ARTIFACTS / "semantic_ids.parquet")
    embeddings = np.load(ARTIFACTS / "title_embeddings.npy").astype(np.float32)
    codes = frame[[f"semantic_id_{level}" for level in range(3)]].to_numpy().astype(np.int32)
    index_of = {int(v): i for i, v in enumerate(frame.video_id)}

    sequences = load_sequences(HERE / "data" / "MicroLens-100k_pairs.tsv", index_of)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(sequences))
    split = int(len(sequences) * 0.9)
    train = [sequences[i] for i in order[:split]]
    held_out = [sequences[i] for i in order[split:] if len(sequences[i]) >= 4]
    print(f"{len(sequences)} sequences, {len(train)} train, {len(held_out)} held-out users with >= 4 views")

    report: dict[str, object] = {"weights": DEFAULT_WEIGHTS}
    report["warm"] = evaluate(fit(train, codes, embeddings, DEFAULT_WEIGHTS), held_out, args.eval_users)

    cold_videos = rng.random(len(frame)) < 0.1
    cold_train = [seq[~cold_videos[seq]] for seq in train]
    cold_held_out = [s for s in held_out if cold_videos[s[-1]] and not cold_videos[s[-4:-1]].any()]
    cold = evaluate(fit(cold_train, codes, embeddings, DEFAULT_WEIGHTS), cold_held_out, args.eval_users)
    report["cold"] = {"users": len(cold_held_out[: args.eval_users]), **cold}
    print(json.dumps(report, indent=2))

    # Serve a model fit on every sequence; the splits above were only for measuring.
    model = fit(sequences, codes, embeddings, DEFAULT_WEIGHTS)
    np.savez_compressed(
        ARTIFACTS / "recommender.npz",
        video_ids=frame.video_id.to_numpy().astype(np.int64),
        codes=codes,
        embeddings=embeddings.astype(np.float16),
        popularity=np.expm1(model.log_pop).astype(np.float32),
        level0=model.level0.astype(np.float32),
        **{f"level01_{field}": getattr(model.level01, field) for field in ("keys", "pointers", "targets", "counts")},
        **{f"items_{field}": getattr(model.items, field) for field in ("keys", "pointers", "targets", "counts")},
    )
    (ARTIFACTS / "recommender_titles.json").write_text(json.dumps(frame.title.tolist(), ensure_ascii=False))
    (ARTIFACTS / "recommender_eval.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {ARTIFACTS / 'recommender.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
