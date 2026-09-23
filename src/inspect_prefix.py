#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["pandas", "pyarrow"]
# ///
"""Show the titles sharing a semantic ID prefix — the check that the IDs mean something.

A prefix is only worth calling "semantic" if the videos under it look related to a
human. This prints the groups so that claim can be judged rather than asserted.
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd

ARTIFACTS = pathlib.Path(__file__).resolve().parent.parent / "artifacts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=pathlib.Path, default=ARTIFACTS / "semantic_ids.parquet")
    parser.add_argument("--prefix", nargs="*", type=int, help="e.g. --prefix 37 812 to filter on the first two tokens")
    parser.add_argument("--groups", type=int, default=8, help="How many prefix groups to show when none is given")
    parser.add_argument("--per-group", type=int, default=8)
    args = parser.parse_args()

    frame = pd.read_parquet(args.ids)
    level_columns = [column for column in frame.columns if column.startswith("semantic_id_") and column[-1].isdigit()]

    if args.prefix:
        columns = level_columns[: len(args.prefix)]
        mask = pd.Series(True, index=frame.index)
        for column, value in zip(columns, args.prefix):
            mask &= frame[column] == value
        selected = frame[mask]
        print(f"prefix {args.prefix} -> {len(selected)} videos")
        for _, row in selected.head(args.per_group * 4).iterrows():
            print(f"  [{row.semantic_id}] {row.title}")
        return 0

    # No prefix given: show the largest two-token groups, which are the ones big enough
    # for a viewer to judge whether the grouping is coherent.
    prefix_columns = level_columns[:2]
    sizes = frame.groupby(prefix_columns).size().sort_values(ascending=False)
    for prefix, count in sizes.head(args.groups).items():
        print(f"\n=== prefix {list(prefix)} — {count} videos")
        mask = pd.Series(True, index=frame.index)
        for column, value in zip(prefix_columns, prefix):
            mask &= frame[column] == value
        for title in frame[mask].title.head(args.per_group):
            print(f"  {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
