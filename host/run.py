"""Deploy the two hosted Functions. Import only for an explicit deployment.

Run ``python host/run.py`` after training and registering the tokenizer.
The recommender's artifact loads lazily, so its Function can be deployed before
downstream training. All credentials come from the normal Chalk configuration.
"""

import argparse
import math
import os
from pathlib import Path
import sys


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if not math.isfinite(args.ready_timeout) or args.ready_timeout <= 0:
        parser.error("ready-timeout must be finite and positive")
    return args


# Parse the direct command before loading any deployment code. The second
# import gives the actual Functions a stable module name for SDK packaging.
if __name__ == "__main__":
    options = _arguments()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from semantic_ids.stages import load_stage

    raise SystemExit(load_stage("host").main(ready_timeout=options.ready_timeout))

# Local imports register Functions, so require the same saved target as the CLI.
# Inside a deployed worker, the SDK treats decorators as no-ops and this local
# receipt is neither needed nor shipped.
if not os.environ.get("CHALK_REMOTE_FUNCTION"):
    from semantic_ids.project import read_project

    _project, _target = read_project()
    if not _project["training"].get("tokenizer", {}).get("model_version"):
        raise ValueError("Finish tokenizer training before deploying inference")

import chalkcompute

from semantic_ids.settings import (
    CPU_TORCH_INSTALL, ENCODER_MODEL, ENCODER_REVISION, INFERENCE_DEPS,
    RECOMMENDER_FUNCTION, TOKENIZER_FUNCTION,
)

# Keep offline flags on serving workers, away from the encoder download.
# Image.env() is image-wide, not ordered relative to run_commands().
runtime_env = {
    **{key: os.environ[key] for key in ("CHALK_ENVIRONMENT", "CHALK_API_SERVER") if key in os.environ},
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


image = (
    chalkcompute.Image.debian_slim(python_version="3.12")
    .run_commands(CPU_TORCH_INSTALL)
    .pip_install(INFERENCE_DEPS)
    .add_local_python_source("semantic_ids")
    .run_commands(
        "python -c \"from sentence_transformers import SentenceTransformer; "
        f"SentenceTransformer('{ENCODER_MODEL}', revision='{ENCODER_REVISION}')"
        ".save('/opt/encoder')\""
    )
)


@chalkcompute.function(
    name=TOKENIZER_FUNCTION, image=image, cpu="2", memory="8Gi",
    min_replicas=0, max_replicas=2, chalk_identity=True, authenticated=True,
    target_queue_depth=1,
    call_timeout=1800, env=runtime_env,
    concurrency = 2,
)
def semantic_tokenize(title: str, model_version: int) -> list[int]:
    from semantic_ids.hosting import tokenize_title

    return tokenize_title(title, model_version)


@chalkcompute.function(
    name=RECOMMENDER_FUNCTION, image=image, cpu="2", memory="8Gi",
    min_replicas=0, max_replicas=2, chalk_identity=True, authenticated=True,
    target_queue_depth=1,
    call_timeout=1800, env=runtime_env,
)
def semantic_recommend(history_ids: list[int], model_version: int, top_k: int) -> str:
    from semantic_ids.hosting import recommend_videos

    return recommend_videos(history_ids, model_version, top_k)


def main(*, ready_timeout: float = 1800.0) -> int:
    timeout = ready_timeout
    semantic_tokenize.wait_ready(timeout=timeout)
    semantic_recommend.wait_ready(timeout=timeout)
    print(f"Ready: {TOKENIZER_FUNCTION} and {RECOMMENDER_FUNCTION}")
    return 0

