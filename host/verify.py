#!/usr/bin/env python3
"""Compare hosted tokenizer codes with its exact registered checkpoint.

Uses the saved FTQA project, immutable tokenizer version, its recorded held-out
video IDs, and its pinned title dataset/encoder revision. It downloads that
checkpoint and encoder and computes reference codes locally. It never deploys
or trains, and it does not use the historical checked-in semantic-ID artifacts.

This verifies tokenizer consistency on held-out videos. Generating codes for a
new title does not add a new video to the recommender's fixed catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples < 1 or args.seed < 0:
        parser.error("--samples must be positive and --seed must be nonnegative")

    # Keep --help independent of credentials, network and numerical dependencies.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semantic_ids.project import read_project
    from semantic_ids.settings import ENCODER_MODEL, ENCODER_REVISION, TOKENIZER_FUNCTION, TOKENIZER_MODEL

    project, target = read_project()
    version = project.get("training", {}).get("tokenizer", {}).get("model_version")
    if type(version) is not int or version < 1:
        raise ValueError("Complete native tokenizer training and registration first")
    titles_revision = project["datasets"]["titles"]["revision_id"]
    print(f"Verifying {TOKENIZER_MODEL} v{version} in {target.environment} | {target.api_server}", flush=True)

    import torch
    from chalkcompute import DatasetClient, RemoteFunction
    from sentence_transformers import SentenceTransformer
    from semantic_ids.rqvae import RQVAE

    client = target.client()
    with tempfile.TemporaryDirectory(prefix="semantic-tokenizer-verify-") as directory:
        downloaded = client.download_model_artifact(name=TOKENIZER_MODEL, version=version, download_dir=directory)
        files = set(downloaded.downloaded_model_files + downloaded.downloaded_additional_files)
        checkpoints = [Path(path) for path in files if Path(path).name == "rqvae.pt"]
        if len(checkpoints) != 1:
            raise ValueError("Registered tokenizer must contain exactly one rqvae.pt")
        checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
        metadata = checkpoint.get("metadata", {})
        if (metadata.get("completed") is not True or metadata.get("kind") != "tokenizer"
                or metadata.get("dataset_revision") != titles_revision
                or metadata.get("encoder_model") != ENCODER_MODEL
                or metadata.get("encoder_revision") != ENCODER_REVISION
                or checkpoint.get("quantizer_initialized") != [True, True, True]):
            raise ValueError("Registered checkpoint differs from the saved titles or hosted encoder configuration")
        rows = sorted(DatasetClient(chalk_client=client).read(titles_revision).to_pylist(), key=lambda row: row["id"])
        rows = [{"id": row["id"], "title": row["title"]} for row in rows]
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if digest != metadata.get("data_sha256"):
            raise ValueError("Pinned title data differs from the tokenizer's training data")
        holdout = metadata.get("holdout_video_ids")
        if (not isinstance(holdout, list) or not holdout
                or any(type(video) is not int or video < 0 for video in holdout)
                or len(set(holdout)) != len(holdout)):
            raise ValueError("Checkpoint has no valid held-out video IDs to verify")
        by_id = {row["id"]: row for row in rows}
        if len(by_id) != len(rows) or not set(holdout) <= set(by_id):
            raise ValueError("Held-out video IDs do not match the pinned title dataset")
        selected = random.Random(args.seed).sample(sorted(holdout), min(args.samples, len(holdout)))
        config = checkpoint["config"]
        if config.get("num_levels") != 3 or config.get("input_dim") != 384:
            raise ValueError("Expected a three-level tokenizer with 384-dimensional title embeddings")
        model = RQVAE(**config)
        model.load_state_dict(checkpoint["state_dict"])
        for quantizer in model.quantizers:
            quantizer._initialized = True
        model.eval()
        encoder = SentenceTransformer(ENCODER_MODEL, revision=ENCODER_REVISION, device="cpu")
        remote = RemoteFunction.from_name(TOKENIZER_FUNCTION)
        agreements = 0
        for video_id in selected:
            title = by_id[video_id]["title"]
            vector = encoder.encode([title], convert_to_numpy=True, normalize_embeddings=True)
            with torch.inference_mode():
                expected = model.encode_ids(torch.as_tensor(vector, dtype=torch.float32))[0].tolist()
            served = remote.remote(title, version)
            valid = isinstance(served, list) and len(served) == 3 and all(type(token) is int and token >= 0 for token in served)
            matched = valid and served == expected
            agreements += int(matched)
            print(f"{'OK' if matched else 'MISMATCH'} video={video_id} served={served} expected={expected}")
        print(f"Tokenizer consistency: {agreements}/{len(selected)} held-out videos match.")
        print("This does not insert new videos into the trained recommender's fixed catalog.")
        return 0 if selected and agreements == len(selected) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
