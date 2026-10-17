#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute", "pyyaml"]
# ///
"""Time a scale-from-zero call to the deployed semantic-ID function.

Deploy the function with `deploy_function.py --min-replicas 0` first. Each trial waits
until the scaling group reports `ScaledToZero`, then times one call — scheduling, image
pull, container start, torch import, model load and inference all land inside it —
followed by a few warm calls on the replica that call woke up.

    ./measure_cold_start.py --env-id chlkb5466073 --function semantic-id-embed-s2z --trials 5

The breakdown of a cold call (pull vs. boot vs. load) is not visible from here; read it
from the pod events and the replica's own log.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

TITLES = [
    "grandma teaches me how to fold dumplings",
    "unboxing the new electric scooter and riding it downtown",
    "cat steals fish from the kitchen counter",
    "the most relaxing lakeside camping trip of the summer",
]


def wait_for_zero(group: object, timeout: float, poll: float) -> float:
    """Block until the group has scaled to zero; return how long that took."""
    start = time.monotonic()
    while True:
        info = group.refresh()  # type: ignore[attr-defined]
        if info.status == "ScaledToZero" or (info.ready_replicas == 0 and info.available_replicas == 0):
            return time.monotonic() - start
        if time.monotonic() - start > timeout:
            raise SystemExit(f"group still {info.status} with {info.ready_replicas} ready after {timeout:.0f}s")
        time.sleep(poll)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-id", default="chlkb5466073", help="Chalk environment id (default: oracle-gpu)")
    parser.add_argument("--api-server", default="https://api.staging.chalk.dev")
    parser.add_argument("--function", default="semantic-id-embed-s2z", help="Scaling-group name of the function")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warm-calls", type=int, default=5)
    parser.add_argument("--zero-timeout", type=float, default=900.0)
    parser.add_argument("--out", type=pathlib.Path, default=None, help="Append one JSON line per trial here")
    args = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import chalk_creds

    chalk_creds.apply_to_environ(args.api_server, args.env_id)
    from chalkcompute import RemoteFunction, ScalingGroup

    remote = RemoteFunction.from_name(args.function)
    group = ScalingGroup.from_name(args.function)

    cold: list[float] = []
    warm: list[float] = []
    for trial in range(args.trials):
        waited = wait_for_zero(group, args.zero_timeout, poll=2.0)
        title = TITLES[trial % len(TITLES)]
        started = time.time()
        t0 = time.perf_counter()
        codes = remote.remote(title)
        cold_seconds = time.perf_counter() - t0

        warm_seconds: list[float] = []
        for index in range(args.warm_calls):
            t1 = time.perf_counter()
            remote.remote(TITLES[(trial + index + 1) % len(TITLES)])
            warm_seconds.append(time.perf_counter() - t1)

        cold.append(cold_seconds)
        warm.extend(warm_seconds)
        record = {
            "function": args.function,
            "trial": trial,
            "started_unix": started,
            "waited_for_zero_s": round(waited, 1),
            "cold_call_s": round(cold_seconds, 3),
            "warm_median_ms": round(1000 * statistics.median(warm_seconds), 1) if warm_seconds else None,
            "codes": list(codes),
        }
        print(json.dumps(record), flush=True)
        if args.out is not None:
            with args.out.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

    print(
        f"\n{args.function}: cold call median {statistics.median(cold):.1f}s "
        f"(min {min(cold):.1f}s, max {max(cold):.1f}s) over {len(cold)} trials"
    )
    if warm:
        print(f"warm call median {1000 * statistics.median(warm):.0f}ms over {len(warm)} calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
