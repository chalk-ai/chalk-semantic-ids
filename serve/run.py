"""Request recommendations through the deployed Chalk feature."""

import json
from pathlib import Path
import sys
from uuid import uuid4


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from semantic_ids.cli import main

    raise SystemExit(main(sys.argv[1:], stage="recommend"))

from semantic_ids.project import read_project
from semantic_ids.import_cli import selected_recommender


def recommend(args):
    if len(args.history) > 500:
        raise ValueError("--history accepts at most 500 video IDs")
    project, target = read_project()
    model = selected_recommender(project, target, getattr(args, "model_source", "auto"))
    version = model["model_version"]
    result = target.client().query(
        input={"recommendation.id": uuid4().hex, "recommendation.history_ids": args.history,
               "recommendation.model_version": version, "recommendation.top_k": args.top_k},
        output=["recommendation.result"],
        branch=args.branch,
    )
    if result.errors:
        raise ValueError(f"Recommendation query failed: {result.errors}")
    print(json.dumps(json.loads(result.get_feature_value("recommendation.result")), indent=2))
