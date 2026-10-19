#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute", "pandas", "pyarrow"]
# ///
"""Check the deployed function on videos the RQ-VAE never trained on.

Two things are worth proving, and they are different:

1. **Consistency** — the deployed function returns the same tokens the local model
   assigns. If it does not, the serving path has drifted from training (a different
   encoder, unseeded codebooks, a stale checkpoint) and every downstream ID is wrong.
2. **Cold start** — a title the model never saw lands in a prefix whose existing
   videos are recognisably related, which is the property that makes a brand-new
   video recommendable without retraining an ID embedding table.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

ARTIFACTS = pathlib.Path(__file__).resolve().parent.parent / "artifacts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--api-server", default=None)
    parser.add_argument("--function", default="semantic-id-embed")
    parser.add_argument("--ids", type=pathlib.Path, default=ARTIFACTS / "semantic_ids.parquet")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--neighbours", type=int, default=4, help="Existing videos to show per matched prefix")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.api_server:
        os.environ["CHALK_API_SERVER"] = args.api_server
    if args.env_id:
        os.environ["CHALK_ENVIRONMENT"] = args.env_id
    from chalkcompute import RemoteFunction

    remote = RemoteFunction.from_name(args.function)

    frame = pd.read_parquet(args.ids)
    level_columns = [column for column in frame.columns if column.startswith("semantic_id_") and column[-1].isdigit()]
    holdout = frame[frame.is_holdout & frame.title.str.strip().ne("")]
    sample = holdout.sample(min(args.samples, len(holdout)), random_state=args.seed)

    agreements = 0
    for _, row in sample.iterrows():
        served = list(remote.remote(row.title))
        local = [int(row[column]) for column in level_columns]
        match = served == local
        agreements += int(match)
        print(f"\n{'OK ' if match else 'MISMATCH'} served={served} local={local}")
        print(f"  unseen title: {row.title[:100]}")

        prefix_mask = (frame[level_columns[0]] == served[0]) & (frame[level_columns[1]] == served[1])
        neighbours = frame[prefix_mask & (frame.video_id != row.video_id)].head(args.neighbours)
        for _, neighbour in neighbours.iterrows():
            print(f"    shares prefix [{served[0]}, {served[1]}]: {neighbour.title[:90]}")

    print(f"\nserving matches local assignment on {agreements}/{len(sample)} held-out videos")
    return 0 if agreements == len(sample) else 1


if __name__ == "__main__":
    raise SystemExit(main())
