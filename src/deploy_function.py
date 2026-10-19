#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute"]
# ///
"""Deploy the semantic-ID function: an unseen video title in, its semantic ID out.

This is the cold-start half of the demo. A video the RQ-VAE never saw gets a
content vector from the same title encoder used in training, and the frozen
codebooks turn that vector into tokens — so the video is addressable by existing
recommenders immediately, with no ID-embedding table to retrain.

    ./deploy_function.py                       # deploy, then smoke-test it
    ./deploy_function.py --title "cat cooks dinner"

Both the title encoder and the RQ-VAE weights are baked into the image, so a
replica needs no model download at call time and no network egress to Hugging Face.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = HERE / "artifacts"
ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Where the baked artifacts land inside the image.
REMOTE_WEIGHTS = "/app/rqvae.pt"
REMOTE_MODULE = "/app/rqvae.py"
REMOTE_HF_HOME = "/opt/hf"


def semantic_id_for_title(title: str) -> list[int]:
    """Semantic ID tokens for a video title the model has never seen."""
    import sys

    import torch

    # Loaded once per replica and cached on the module. Doing this at import time
    # instead would also run it locally at deploy time, for no benefit.
    global _ENCODER, _RQVAE  # noqa: PLW0603
    try:
        encoder, model = _ENCODER, _RQVAE
    except NameError:
        sys.path.insert(0, "/app")
        from rqvae import RQVAE
        from sentence_transformers import SentenceTransformer

        checkpoint = torch.load("/app/rqvae.pt", map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = RQVAE(
            input_dim=config["input_dim"],
            latent_dim=config["latent_dim"],
            codebook_size=config["codebook_size"],
            num_levels=config["num_levels"],
            hidden_dims=tuple(config["hidden_dims"]),
            commitment_weight=config["commitment_weight"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        # The codebooks are seeded lazily during training; mark them initialized so
        # inference never reseeds them from whatever vector arrives first.
        for quantizer in model.quantizers:
            quantizer._initialized = True
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        _ENCODER, _RQVAE = encoder, model

    vector = encoder.encode([title], convert_to_numpy=True, normalize_embeddings=True)
    codes = model.encode_ids(torch.from_numpy(vector).float())
    return [int(code) for code in codes[0]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-id", default=None, help="Override the active Chalk environment")
    parser.add_argument("--api-server", default=None)
    parser.add_argument("--name", default="semantic-id-embed")
    parser.add_argument("--cpu", default="2")
    parser.add_argument("--memory", default="8Gi")
    parser.add_argument(
        "--title",
        action="append",
        default=None,
        help="Title to score after deploying (repeatable). Defaults to a few unseen ones.",
    )
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--ready-timeout", type=float, default=1800.0)
    return parser.parse_args()


DEFAULT_TITLES = [
    "grandma teaches me how to fold dumplings",
    "unboxing the new electric scooter and riding it downtown",
    "cat steals fish from the kitchen counter",
]


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    if args.api_server:
        os.environ["CHALK_API_SERVER"] = args.api_server
    if args.env_id:
        os.environ["CHALK_ENVIRONMENT"] = args.env_id

    weights = ARTIFACTS / "rqvae.pt"
    if not weights.is_file():
        raise SystemExit(f"{weights} is missing — run train_rqvae.py first")

    import chalkcompute
    from chalkcompute import Image

    image = (
        Image.debian_slim(python_version="3.12")
        .pip_install(["torch==2.9.1", "sentence-transformers==5.1.2", "numpy"])
        .env({"HF_HOME": REMOTE_HF_HOME, "SENTENCE_TRANSFORMERS_HOME": REMOTE_HF_HOME})
        # Bake the title encoder into the image. Downloading it on first call would
        # put a ~90MB Hugging Face fetch on the critical path of a cold replica, and
        # would make the function fail closed in an environment without egress.
        .run_commands(
            "python -c \"from sentence_transformers import SentenceTransformer; "
            f"SentenceTransformer('{ENCODER_MODEL}')\""
        )
        .add_local_file(str(weights), REMOTE_WEIGHTS)
        .add_local_file(str(HERE / "src" / "rqvae.py"), REMOTE_MODULE)
    )

    deploy = chalkcompute.function(
        image=image, name=args.name, cpu=args.cpu, memory=args.memory
    )
    try:
        deployed = deploy(semantic_id_for_title)
    except Exception as exc:  # noqa: BLE001 - only the readiness timeout is recoverable
        if "not ready after" not in str(exc):
            raise
        # The decorator's implicit deploy polls readiness for a fixed 300s with no knob
        # to raise it, and a ~3GB image does not pull that fast from cold. The rollout
        # continues server-side regardless, so re-attach and wait on our own budget.
        print(f"deploy poll gave up early ({exc}); re-attaching and waiting")
        from chalkcompute import RemoteFunction

        deployed = RemoteFunction.from_name(args.name)
        deployed.wait_ready(timeout=args.ready_timeout)

    print(f"deployed {args.name} to {args.env_id}")

    if args.skip_smoke_test:
        return 0
    # The image carries torch and the encoder; a cold replica has to pull ~3GB before it
    # can serve, so give readiness a lot more than the 120s default.
    deployed.wait_ready(timeout=args.ready_timeout)
    failures = 0
    for title in args.title or DEFAULT_TITLES:
        try:
            # `.remote()` is the wire call. Calling the decorated function directly runs
            # the body in this process instead, which fails here for an honest reason:
            # this script's own venv has no torch.
            codes = deployed.remote(title)
            print(f"  {codes}  {title}")
        except Exception as exc:  # noqa: BLE001 - one bad title should not hide the rest
            failures += 1
            print(f"  FAILED  {title}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
