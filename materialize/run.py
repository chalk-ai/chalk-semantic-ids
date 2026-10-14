"""Compute video.codes with the trained hosted tokenizer and save a Chalk Dataset.

This stage runs inference. It does not retrain the tokenizer.
"""
# Parse --help before importing any SDK or starting remote work.
if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from semantic_ids.cli import main

    raise SystemExit(main(sys.argv[1:], stage="materialize"))


from time import perf_counter
from uuid import uuid4

from semantic_ids.project import read_project, write_project


def materialize(args):
    from chalkcompute import DatasetClient
    from semantic_ids.data import build_catalog

    project, target = read_project()
    if args.limit is None and "catalog" in project["datasets"]:
        print(f"Semantic feature dataset already saved: {project['datasets']['catalog']['revision_id']}")
        return
    version = project["training"].get("tokenizer", {}).get("model_version")
    if version is None:
        raise ValueError("Finish tokenizer training and registration first")
    client = target.client()
    read_started = perf_counter()
    table = DatasetClient(chalk_client=client).read(project["datasets"]["titles"]["revision_id"])
    # DatasetClient downloads the revision; slice inputs before invoking Chalk.
    titles = (table.slice(0, args.limit) if args.limit is not None else table).to_pylist()
    print(f"Title Dataset read: {perf_counter() - read_started:.2f}s | selected {len(titles):,} rows", flush=True)
    name = f"semantic-catalog-{project['snapshot']}"
    if args.limit is not None:
        name = f"semantic-catalog-sample-{project['snapshot']}-{uuid4().hex[:8]}"
        print("Sample query only; the saved training catalog will not change.", flush=True)
    print(f"Resolving video.codes for {len(titles):,} rows through the hosted tokenizer version {version} "
          f"(branch: {args.branch or 'default'})…", flush=True)
    query_started = perf_counter()
    try:
        dataset = client.offline_query(
            input={"video.id": [row["id"] for row in titles],
                   "video.title": [row["title"] for row in titles],
                   "video.tokenizer_version": [version] * len(titles)},
            output=["video.id", "video.title", "video.tokenizer_version", "video.codes"],
            required_output=["video.codes"], dataset_name=name, recompute_features=True,
            branch=args.branch,
            run_asynchronously=True,
        )
        frame = dataset.to_pandas(output_id=False, output_ts=False)
    finally:
        print(f"Chalk query elapsed: {perf_counter() - query_started:.2f}s "
              "(includes queueing, startup, inference and result download)", flush=True)
    if dataset.errors:
        raise ValueError(f"Semantic feature materialization failed: {dataset.errors}")
    rows = [{key.rsplit(".", 1)[-1]: value for key, value in row.items()}
            for row in frame.to_dict("records")]
    for row in rows:
        if hasattr(row["codes"], "tolist"):
            row["codes"] = row["codes"].tolist()
    catalog = build_catalog(rows)
    if {row["video_id"] for row in catalog} != {row["id"] for row in titles}:
        raise ValueError("Semantic feature materialization did not return every requested video")
    if set(frame["video.tokenizer_version"]) != {version}:
        raise ValueError("Semantic features came from a different tokenizer version")
    revision = str(dataset.revisions[-1].revision_id)
    if args.limit is not None:
        print(frame.to_string(index=False))
        print(f"Sample complete: {len(catalog):,} rows | revision {revision}")
        return
    project["datasets"]["catalog"] = {"name": name, "revision_id": revision, "rows": len(catalog)}
    write_project(project)
    print(f"Materialized {len(catalog):,} semantic-ID features | revision {revision}")
