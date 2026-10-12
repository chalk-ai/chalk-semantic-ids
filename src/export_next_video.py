#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["torch", "onnx", "onnxruntime", "numpy", "pandas", "pyarrow"]
# ///
"""Export the trained next-video transformer to ONNX and prove the export ranks identically.

Writes `artifacts/next_video.onnx` (history of video ids in, a score per video out), then
recommends for a sample of real watch histories through both torch and
`NextVideoRecommender`, and exits non-zero unless every top-10 list matches up to
reordering of exactly tied videos.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from next_video_model import PAD, NextVideoTransformer
from next_video_runtime import NextVideoRecommender
from train_next_video import left_pad, load_sequences

HERE = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = HERE / "artifacts"


class Scorer(torch.nn.Module):
    """History -> scores, with the video table computed once instead of per call."""

    def __init__(self, model: NextVideoTransformer) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("table", model.item_table().detach())

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        hidden = self.model.encode(history, self.table)[:, -1]
        return hidden @ self.table.T


def load_model(path: pathlib.Path) -> NextVideoTransformer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = NextVideoTransformer(
        num_videos=config["num_videos"],
        item_repr=config["item_repr"],
        semantic_tokens=torch.from_numpy(checkpoint["semantic_tokens"]) if config["item_repr"] != "id" else None,
        trained_videos=torch.from_numpy(checkpoint["trained_videos"]) if "trained_videos" in checkpoint else None,
        id_dropout=config.get("id_dropout", 0.0),
        codebook_size=config["codebook_size"],
        dim=config["dim"],
        layers=config["layers"],
        heads=config["heads"],
        max_len=config["max_len"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=ARTIFACTS / "next_video_hybrid.pt")
    parser.add_argument("--out", type=pathlib.Path, default=ARTIFACTS / "next_video.onnx")
    parser.add_argument("--pairs", type=pathlib.Path, default=HERE / "data" / "MicroLens-100k_pairs.tsv")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    scorer = Scorer(model).eval()
    example = left_pad([[1958, 6346, 15223, 17700]], model.max_len)
    with torch.no_grad():
        torch.onnx.export(
            scorer,
            (example,),
            str(args.out),
            input_names=["history"],
            output_names=["scores"],
            dynamic_axes={"history": {0: "batch"}, "scores": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    runtime = NextVideoRecommender(args.out)
    histories = [sequence[:-1] for sequence in load_sequences(args.pairs)[: args.samples]]
    with torch.no_grad():
        torch_scores = scorer(left_pad(histories, model.max_len)).numpy()
    onnx_scores = runtime.scores(histories)

    torch_scores[:, PAD] = -np.inf
    identical = equivalent = 0
    onnx_top = runtime.recommend(histories, args.k)
    for row, history in enumerate(histories):
        torch_scores[row, history] = -np.inf
        top = np.argsort(-torch_scores[row], kind="stable")[: args.k]
        identical += int(top.tolist() == onnx_top[row])
        # Videos with colliding semantic IDs score exactly alike, so float noise can swap
        # tied videos. Equivalent = same torch score at every rank, to within 1e-5.
        equivalent += int(np.allclose(torch_scores[row, top], torch_scores[row, onnx_top[row]], rtol=0, atol=1e-5))
    finite = np.isfinite(torch_scores)
    report = {
        "histories": len(histories),
        "identical_top_k": identical,
        "equivalent_top_k_up_to_ties": equivalent,
        "max_abs_score_diff": float(np.abs(torch_scores[finite] - onnx_scores[finite]).max()),
    }
    print(json.dumps(report, indent=2))
    return 0 if equivalent == len(histories) else 1


if __name__ == "__main__":
    raise SystemExit(main())
