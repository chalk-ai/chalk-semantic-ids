#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute", "pyyaml", "onnxruntime", "numpy", "pandas", "pyarrow"]
# ///
"""Check the deployed next-videos function against the local model, and show what it recommends.

Sends real MicroLens histories (each user's watches but the last, most recent first) to
the function and compares its ten videos with `NextVideoRecommender` on the same ONNX
file. A mismatch means the replica is serving a different model than the one in
`artifacts/`.
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import chalk_creds
from next_video_runtime import NextVideoRecommender

HERE = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = HERE / "artifacts"


def load_sequences(path: pathlib.Path) -> list[list[int]]:
    with path.open() as handle:
        return [[int(video) for video in line.rstrip("\n").partition("\t")[2].split()] for line in handle]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-id", default="chlkb5466073")
    parser.add_argument("--api-server", default=chalk_creds.DEFAULT_API_SERVER)
    parser.add_argument("--function", default="next-videos")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--show", type=int, default=2, help="Print titles for this many users")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    chalk_creds.apply_to_environ(args.api_server, args.env_id)
    from chalkcompute import RemoteFunction

    remote = RemoteFunction.from_name(args.function)
    local = NextVideoRecommender(ARTIFACTS / "next_video.onnx")
    titles = pd.read_parquet(ARTIFACTS / "titles.parquet").set_index("video_id").title

    sequences = random.Random(args.seed).sample(
        load_sequences(HERE / "data" / "MicroLens-100k_pairs.tsv"), args.samples
    )
    agreements = 0
    for index, sequence in enumerate(sequences):
        history = sequence[:-1]
        served = list(remote.remote(history[::-1]))
        expected = local.recommend([history], 10)[0]
        agreements += int(served == expected)
        if served != expected:
            print(f"MISMATCH served={served} local={expected}")
        if index < args.show:
            print("\nwatched, most recent first:")
            for video in history[::-1][:5]:
                print(f"  {video:>6}  {titles[video][:80]}")
            print(f"recommended (actual next watch: {sequence[-1]}, {'hit' if sequence[-1] in served else 'miss'}):")
            for video in served[:5]:
                print(f"  {video:>6}  {titles[video][:80]}")
    print(f"\nserving matches the local model on {agreements}/{len(sequences)} histories")
    return 0 if agreements == len(sequences) else 1


if __name__ == "__main__":
    raise SystemExit(main())
