#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute", "pyyaml", "numpy"]
# ///
"""Deploy the next-video model as a Chalk function: a watch history in, 10 videos out.

    ./deploy_next_video.py                          # deploy to oracle-gpu, then smoke-test
    ./deploy_next_video.py --min-replicas 0         # scale to zero when idle

The function takes the user's watched video ids, **most recent first** (the order Chalk's
`max_by_n` aggregation produces), and returns the ten unwatched videos the transformer
ranks highest. It serves the ONNX export through
onnxruntime + numpy (no torch), and loads it at container start, which is what keeps a
scale-from-zero call to a few seconds (see the README's "Replica cold start").
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = HERE / "artifacts"
TOP_K = 10


def _load_recommender() -> object:
    import pathlib
    import sys
    import time

    started = time.perf_counter()
    sys.path.insert(0, "/app")
    from next_video_runtime import NextVideoRecommender

    recommender = NextVideoRecommender(pathlib.Path("/app/next_video.onnx"))
    print(f"next-videos: runtime ready in {time.perf_counter() - started:.2f}s", flush=True)
    return recommender


def next_videos(recent_video_ids: list[int]) -> list[int]:
    """The ten videos to show next, best first, given watched video ids most recent first."""
    global _RECOMMENDER
    try:
        recommender = _RECOMMENDER
    except NameError:
        recommender = _RECOMMENDER = _load_recommender()
    # The model reads histories oldest first.
    return recommender.recommend([list(reversed(recent_video_ids or []))], TOP_K)[0]


# The replica imports this module before its server starts listening; loading here keeps
# the first call off the model-load path. Gated on an env var the deploy sets, because
# the deploy script imports this module locally too.
if os.environ.get("NEXT_VIDEOS_EAGER_LOAD") == "1":
    _RECOMMENDER = _load_recommender()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-id", default="chlkb5466073", help="Chalk environment id (default: staging oracle-gpu)")
    parser.add_argument("--api-server", default="https://api.staging.chalk.dev")
    parser.add_argument("--name", default="next-videos")
    parser.add_argument("--cpu", default="1")
    parser.add_argument("--memory", default="2Gi")
    parser.add_argument("--min-replicas", type=int, default=1, help="0 scales to zero after ~30s without calls")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    return parser.parse_args()


# MicroLens users 1 and 2, most recent watch first, without their last watch.
SMOKE_HISTORIES = [
    [17700, 15223, 6346, 1958],
    [10949, 9500, 3775, 9127, 915, 16657],
]


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import chalk_creds

    chalk_creds.apply_to_environ(args.api_server, args.env_id)

    model = ARTIFACTS / "next_video.onnx"
    if not model.is_file():
        raise SystemExit(f"{model} is missing — run train_next_video.py and export_next_video.py first")

    import chalkcompute
    from chalkcompute import Image

    image = (
        Image.debian_slim(python_version="3.12")
        .pip_install(["onnxruntime==1.31.0", "numpy==2.5.4"])
        .add_local_file(str(model), "/app/next_video.onnx")
        .add_local_file(str(HERE / "src" / "next_video_runtime.py"), "/app/next_video_runtime.py")
    )
    scaling: dict[str, object] = {"min_replicas": args.min_replicas, "max_replicas": max(1, args.min_replicas)}
    if args.min_replicas == 0:
        # Hold a call that arrives at zero replicas until one is up, instead of a 503.
        scaling["scale_from_zero_request_policy"] = chalkcompute.ScaleFromZeroRequestPolicy.HOLD
    deploy = chalkcompute.function(
        image=image,
        name=args.name,
        cpu=args.cpu,
        memory=args.memory,
        env={"NEXT_VIDEOS_EAGER_LOAD": "1"},
        **scaling,
    )
    deployed = deploy(next_videos)
    print(f"deployed {args.name} to {args.env_id}")
    if args.skip_smoke_test:
        return 0
    deployed.wait_ready(timeout=args.ready_timeout)
    for history in SMOKE_HISTORIES:
        # `.remote()` is the wire call; calling `deployed(...)` would run it locally.
        print(f"  {history} -> {deployed.remote(history)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
