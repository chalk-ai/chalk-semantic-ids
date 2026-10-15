#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["sentence-transformers", "numpy", "pandas", "pyarrow"]
# ///
"""Historical utility for generating local MicroLens title-embedding artifacts.

The original experiment used a pretrained MiniLM without a recorded revision.
This utility preserves that behavior and the original row order. The current
how-to computes embeddings inside native Chalk tokenizer training using the
pinned encoder in semantic_ids/settings.py; it does not consume these files.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HERE = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_TITLES = HERE / "data" / "MicroLens-100k_title_en.csv"
DEFAULT_OUT = HERE / "artifacts"


def load_titles(path: pathlib.Path) -> pd.DataFrame:
    """Read the `<id>, <title>` CSV.

    Titles are unquoted and contain commas, so the split is on the first comma only;
    `read_csv` with a comma separator would shred a third of the rows into extra columns.
    """
    rows: list[tuple[int, str]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            raw_id, _, title = line.partition(",")
            if not raw_id.strip().isdigit():
                continue
            rows.append((int(raw_id), title.strip()))
    return pd.DataFrame(rows, columns=["video_id", "title"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--titles", type=pathlib.Path, default=DEFAULT_TITLES)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    titles = load_titles(args.titles)
    print(f"loaded {len(titles)} titles, {titles.title.str.len().mean():.1f} chars avg")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, device="cpu")
    vectors = model.encode(
        titles.title.tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        # Unit-norm vectors keep the RQ-VAE's reconstruction loss on a fixed scale and
        # make codebook distances comparable across levels.
        normalize_embeddings=True,
    ).astype(np.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "title_embeddings.npy", vectors)
    titles.to_parquet(args.out_dir / "titles.parquet", index=False)
    print(f"wrote {vectors.shape} embeddings to {args.out_dir / 'title_embeddings.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
