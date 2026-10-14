"""Upload title content and ordered watch examples as pinned Chalk Datasets."""
# Parse --help before importing any SDK or starting remote work.
if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from semantic_ids.cli import main

    raise SystemExit(main(sys.argv[1:], stage="prepare"))


import hashlib
from uuid import uuid4

from semantic_ids import project as project_state
from semantic_ids.project import Target, new_project, read_project, write_project


def prepare(args):
    from chalkcompute import DatasetClient
    from semantic_ids.data import parse_titles, parse_watch_examples

    target = Target.selected(args.environment, args.api_server)
    target.activate()
    titles_raw = args.titles.read_bytes()
    watches_raw = args.watches.read_bytes()
    titles = parse_titles(titles_raw)
    examples = parse_watch_examples(watches_raw, catalog_ids={row["id"] for row in titles},
                                    max_users=args.max_users, seed=args.seed)
    sources = {"titles_sha256": hashlib.sha256(titles_raw).hexdigest(),
               "watches_sha256": hashlib.sha256(watches_raw).hexdigest(),
               "max_users": args.max_users, "seed": args.seed}
    if project_state.RECEIPT.exists():
        project, saved_target = read_project()
        if saved_target != target or project.get("sources") != sources:
            raise ValueError("This checkout already has different data or a different Chalk target. "
                             "Use a separate checkout for an independent example.")
        snapshot = project["snapshot"]
    else:
        snapshot = uuid4().hex[:12]
        project = new_project(target, snapshot)
        project["sources"] = sources
        write_project(project)
    print(f"Target: {target.environment} | {target.api_server}", flush=True)
    upload = DatasetClient(chalk_client=target.client())
    for key, rows in (("titles", titles), ("examples", examples)):
        if key in project["datasets"]:
            print(f"Reusing {key}: revision {project['datasets'][key]['revision_id']}", flush=True)
            continue
        name = f"semantic-{key}-{snapshot}"
        reference = upload.upload(name, rows)
        project["datasets"][key] = {"name": name, "revision_id": reference.revision_id,
                                    "rows": len(rows)}
        write_project(project)
        print(f"{key}: {len(rows):,} rows | dataset {name} | revision {reference.revision_id}", flush=True)
