#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["numpy", "pandas", "pyarrow"]
# ///
"""Measure whether a shared semantic-ID prefix actually means "semantically similar".

Eyeballing prefix groups is misleading on this dataset: MicroLens titles include many
exact and near duplicates, so the largest groups look coherent for a trivial reason.
This compares intra-prefix cosine similarity against the corpus baseline, and repeats
the comparison after collapsing duplicate titles, which is the number that actually
supports the claim.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

ARTIFACTS = pathlib.Path(__file__).resolve().parent.parent / "artifacts"


def mean_pairwise_cosine(vectors: np.ndarray, rng: np.random.Generator, cap: int = 64) -> float:
    """Mean pairwise cosine within a set of unit-norm vectors, subsampled when large."""
    if len(vectors) < 2:
        return float("nan")
    if len(vectors) > cap:
        vectors = vectors[rng.choice(len(vectors), cap, replace=False)]
    similarity = vectors @ vectors.T
    upper = similarity[np.triu_indices(len(vectors), k=1)]
    return float(upper.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=pathlib.Path, default=ARTIFACTS / "semantic_ids.parquet")
    parser.add_argument("--embeddings", type=pathlib.Path, default=ARTIFACTS / "title_embeddings.npy")
    parser.add_argument("--prefix-levels", type=int, default=2)
    parser.add_argument("--min-group", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame = pd.read_parquet(args.ids)
    vectors = np.load(args.embeddings)
    rng = np.random.default_rng(args.seed)

    blank = frame.title.str.strip().eq("")
    duplicated = frame.title.str.strip().duplicated(keep=False) & ~blank
    print(f"{len(frame)} videos | {int(blank.sum())} blank titles | {int(duplicated.sum())} in duplicate-title sets")

    baseline_sample = vectors[rng.choice(len(vectors), 2000, replace=False)]
    baseline = mean_pairwise_cosine(baseline_sample, rng, cap=2000)

    prefix_columns = [f"semantic_id_{level}" for level in range(args.prefix_levels)]
    report: dict[str, object] = {"corpus_baseline_cosine": baseline}

    for label, mask in (("all_titles", pd.Series(True, index=frame.index)), ("distinct_titles", ~blank & ~duplicated)):
        subset = frame[mask]
        subset_vectors = vectors[subset.index.to_numpy()]
        position = {index: row for row, index in enumerate(subset.index)}
        scores: list[float] = []
        sizes: list[int] = []
        for _, group in subset.groupby(prefix_columns):
            if len(group) < args.min_group:
                continue
            rows = np.array([position[index] for index in group.index])
            score = mean_pairwise_cosine(subset_vectors[rows], rng)
            if not np.isnan(score):
                scores.append(score)
                sizes.append(len(group))
        report[label] = {
            "videos": int(len(subset)),
            "groups_scored": len(scores),
            "median_group_size": float(np.median(sizes)) if sizes else float("nan"),
            "mean_intra_prefix_cosine": float(np.mean(scores)) if scores else float("nan"),
            "lift_over_baseline": float(np.mean(scores) - baseline) if scores else float("nan"),
        }

    print(json.dumps(report, indent=2))
    (ARTIFACTS / "prefix_coherence.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
