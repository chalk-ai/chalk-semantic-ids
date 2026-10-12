#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute", "pyyaml", "numpy"]
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
from collections.abc import Callable

HERE = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = HERE / "artifacts"
ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Where the baked artifacts land inside the image.
REMOTE_WEIGHTS = "/app/rqvae.pt"
REMOTE_MODULE = "/app/rqvae.py"
REMOTE_HF_HOME = "/opt/hf"


def _load_torch() -> Callable[[str], list[int]]:
    """The training stack: sentence-transformers + the RQ-VAE checkpoint in torch."""
    import torch
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

    def run(title: str) -> list[int]:
        vector = encoder.encode([title], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return [int(code) for code in model.encode_ids(torch.from_numpy(vector).float())[0]]

    return run


def _load_onnx() -> Callable[[str], list[int]]:
    """Same weights through onnxruntime + numpy; no torch import on the cold path."""
    import pathlib

    from semantic_id_runtime import SemanticIdEncoder

    runtime = SemanticIdEncoder(pathlib.Path("/opt/minilm"), pathlib.Path("/app/rqvae.npz"))
    return lambda title: [int(code) for code in runtime.semantic_ids([title])[0]]


def _load_runtime() -> Callable[[str], list[int]]:
    """Import and load the backend the deploy chose, logging how long that took."""
    import time

    started = time.perf_counter()
    sys.path.insert(0, "/app")
    backend = os.environ.get("SEMANTIC_ID_RUNTIME", "torch")
    run = _load_onnx() if backend == "onnx" else _load_torch()
    print(f"semantic-id: {backend} runtime ready in {time.perf_counter() - started:.2f}s", flush=True)
    return run


def semantic_id_for_title(title: str) -> list[int]:
    """Semantic ID tokens for a video title the model has never seen."""
    # Loaded once per replica and cached on the module: at container start when the
    # deploy set SEMANTIC_ID_EAGER_LOAD=1, otherwise on the first call.
    global _RUNTIME
    try:
        run = _RUNTIME
    except NameError:
        run = _RUNTIME = _load_runtime()
    return run(title)


# The replica imports this module before its server starts listening, so loading here
# moves the import-and-load cost off the first call and into container start. Gated on
# an env var the deploy sets, because the deploy script imports this module locally too.
if os.environ.get("SEMANTIC_ID_EAGER_LOAD") == "1":
    _RUNTIME = _load_runtime()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-id", default="tmnmc9beyujew", help="Chalk environment id (default: the ftqa QA env)")
    parser.add_argument("--api-server", default="https://api.staging.chalk.dev")
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
    parser.add_argument(
        "--min-replicas",
        type=int,
        default=1,
        help="0 lets the function scale to zero after shutdown_delay_seconds idle (see measure_cold_start.py)",
    )
    parser.add_argument(
        "--idle-seconds",
        type=int,
        default=None,
        help="With --min-replicas 0: seconds without traffic before scaling to zero (server default 30)",
    )
    parser.add_argument(
        "--runtime",
        choices=["torch", "onnx"],
        default="torch",
        help="torch: the training stack. onnx: onnxruntime + numpy, ~8s less import on a cold replica, same IDs",
    )
    parser.add_argument(
        "--cpu-torch",
        action="store_true",
        help="Install the CPU-only torch wheel: drops the ~2.5GB of CUDA libraries a CPU replica never loads",
    )
    parser.add_argument(
        "--compute-class",
        choices=["k8s", "host"],
        default=None,
        help="Where replicas run: a Kubernetes pod, or a hypervisor-backed host container (server default if unset)",
    )
    parser.add_argument(
        "--eager-load",
        action="store_true",
        help="Import and load the models at container start instead of on the first call",
    )
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
    import chalk_creds

    chalk_creds.apply_to_environ(args.api_server, args.env_id)

    weights = ARTIFACTS / "rqvae.pt"
    if not weights.is_file():
        raise SystemExit(f"{weights} is missing — run train_rqvae.py first")

    import chalkcompute
    from chalkcompute import Image

    if args.runtime == "onnx":
        npz = ARTIFACTS / "rqvae.npz"
        if not npz.is_file():
            raise SystemExit(f"{npz} is missing — run export_numpy_runtime.py first")
        from semantic_id_runtime import ENCODER_FILES, encoder_file_url

        fetches = "; ".join(
            f"urllib.request.urlretrieve('{encoder_file_url(name)}', '/opt/minilm/{name}')" for name in ENCODER_FILES
        )
        image = (
            Image.debian_slim(python_version="3.12")
            .pip_install(["onnxruntime==1.31.0", "tokenizers==0.22.2", "numpy==2.5.4"])
            # Baked into the image at a pinned revision, for the same reasons as the
            # torch path below: no fetch on a cold replica's critical path, no egress.
            .run_commands("mkdir -p /opt/minilm/onnx", f'python -c "import urllib.request; {fetches}"')
            .add_local_file(str(npz), "/app/rqvae.npz")
            .add_local_file(str(HERE / "src" / "semantic_id_runtime.py"), "/app/semantic_id_runtime.py")
        )
    else:
        image = Image.debian_slim(python_version="3.12")
        if args.cpu_torch:
            # Installed first so sentence-transformers' torch requirement is already met
            # and pip does not pull the CUDA build from PyPI on top of it.
            image = image.run_commands(
                "pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.9.1"
            )
        image = (
            image.pip_install(["torch==2.9.1", "sentence-transformers==5.1.2", "numpy"])
            .env({"HF_HOME": REMOTE_HF_HOME, "SENTENCE_TRANSFORMERS_HOME": REMOTE_HF_HOME})
            # Bake the title encoder into the image. Downloading it on first call would
            # put a ~90MB Hugging Face fetch on the critical path of a cold replica, and
            # would make the function fail closed in an environment without egress.
            .run_commands(
                'python -c "from sentence_transformers import SentenceTransformer; '
                f"SentenceTransformer('{ENCODER_MODEL}')\""
            )
            .add_local_file(str(weights), REMOTE_WEIGHTS)
            .add_local_file(str(HERE / "src" / "rqvae.py"), REMOTE_MODULE)
        )

    scaling: dict[str, object] = {"min_replicas": args.min_replicas, "max_replicas": max(1, args.min_replicas)}
    if args.min_replicas == 0:
        # Hold the first call while a replica boots instead of answering 503, so a
        # caller sees one slow call rather than an error it has to retry.
        scaling["scale_from_zero_request_policy"] = chalkcompute.ScaleFromZeroRequestPolicy.HOLD
        if args.idle_seconds is not None:
            scaling["shutdown_delay_seconds"] = args.idle_seconds
    if args.compute_class is not None:
        scaling["compute_class"] = {"k8s": chalkcompute.ComputeClass.K8S, "host": chalkcompute.ComputeClass.HOST}[
            args.compute_class
        ]
    env = {"SEMANTIC_ID_RUNTIME": args.runtime}
    if args.eager_load:
        env["SEMANTIC_ID_EAGER_LOAD"] = "1"
    deploy = chalkcompute.function(image=image, name=args.name, cpu=args.cpu, memory=args.memory, env=env, **scaling)
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
