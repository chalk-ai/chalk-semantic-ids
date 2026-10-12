#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute", "pyyaml", "numpy"]
# ///
"""Deploy the semantic-ID demo page as a Chalk scaling group and print its URL.

    ./demo/deploy_demo.py                      # oracle-gpu, name semantic-id-demo
    ./demo/deploy_demo.py --env-id <env> --name <dns-label>

Needs `artifacts/rqvae.npz` (`src/export_numpy_runtime.py`) and
`artifacts/recommender.npz` (`src/build_recommender.py`). The page is served on the
environment's gateway with no authentication; it only shows the public MicroLens titles.

The page uses Chalk's marketing fonts (Everett, Everett Mono) when `--fonts-dir` points at
a checkout's `marketing-frontend/public/fonts`. They are licensed, so they are baked into
the image at deploy time rather than kept in this repository; without them the page falls
back to system fonts.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = HERE / "artifacts"

# (local path, path inside the replica). The server reads all of these from /app.
FILES = [
    (HERE / "demo" / "server.py", "/app/server.py"),
    (HERE / "demo" / "index.html", "/app/index.html"),
    (HERE / "src" / "semantic_id_runtime.py", "/app/semantic_id_runtime.py"),
    (HERE / "src" / "recommender.py", "/app/recommender.py"),
    (ARTIFACTS / "rqvae.npz", "/app/rqvae.npz"),
    (ARTIFACTS / "recommender.npz", "/app/recommender.npz"),
    (ARTIFACTS / "recommender_titles.json", "/app/recommender_titles.json"),
    (ARTIFACTS / "recommender_eval.json", "/app/recommender_eval.json"),
]
FONT_FILES = [
    "TWKEverett-Regular-web.woff2",
    "TWKEverett-Medium-web.woff2",
    "TWKEverettMono-Regular.ttf",
    "TWKEverettMono-Medium.ttf",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-id", default="chlkb5466073", help="Chalk environment id (default: staging oracle-gpu)")
    parser.add_argument("--api-server", default="https://api.staging.chalk.dev")
    parser.add_argument("--name", default="semantic-id-demo")
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    parser.add_argument("--fonts-dir", type=pathlib.Path, default=None, help="marketing-frontend/public/fonts")
    args = parser.parse_args()

    files = list(FILES)
    if args.fonts_dir is not None:
        files += [(args.fonts_dir / name, f"/app/fonts/{name}") for name in FONT_FILES]
    for local, _ in files:
        if not local.is_file():
            raise SystemExit(f"{local} is missing; see this script's docstring")

    sys.path.insert(0, str(HERE / "src"))
    import chalk_creds
    from semantic_id_runtime import ENCODER_FILES, encoder_file_url

    chalk_creds.apply_to_environ(args.api_server, args.env_id)
    import chalkcompute

    fetches = "; ".join(
        f"urllib.request.urlretrieve('{encoder_file_url(name)}', '/opt/minilm/{name}')" for name in ENCODER_FILES
    )
    image = (
        chalkcompute.Image.debian_slim(python_version="3.12")
        .pip_install(["onnxruntime==1.31.0", "tokenizers==0.22.2", "numpy==2.5.4"])
        .run_commands("mkdir -p /opt/minilm/onnx", f'python -c "import urllib.request; {fetches}"')
    )
    for local, remote in files:
        image = image.add_local_file(str(local), remote)

    group = chalkcompute.ScalingGroup(
        image=image,
        name=args.name,
        entrypoint=["python", "/app/server.py", "--port", "8080"],
        port=8080,
        cpu="2",
        memory="4Gi",
        min_replicas=1,
        max_replicas=1,
        readiness_probe=chalkcompute.ReadinessProbe.http("/healthz"),
    )
    group.deploy()
    group.wait_ready(timeout=args.ready_timeout)
    url = group.web_url
    print(f"deployed {args.name} to {args.env_id}: {url}")

    # wait_ready() can report the previous revision on a redeploy; poll the page itself.
    deadline = time.monotonic() + args.ready_timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/meta", timeout=10) as response:
                if response.status == 200:
                    print(f"serving: {url}")
                    return 0
        except OSError:
            pass
        time.sleep(3)
    raise SystemExit(f"{url} did not answer within {args.ready_timeout:.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())
