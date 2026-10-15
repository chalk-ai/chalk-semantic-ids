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
    from semantic_ids.data import DEFAULT_SAMPLE_USERS, select_training_data

    target = Target.selected(args.environment, args.api_server)
    target.activate()
    project = None
    saved_sources = None
    if project_state.RECEIPT.exists():
        project, saved_target = read_project()
        saved_sources = project.get("sources")
        if saved_target != target or not isinstance(saved_sources, dict):
            raise ValueError("This checkout already has a different Chalk target or no saved data selection. "
                             "Use a separate checkout for an independent example.")
    previous = saved_sources or {}
    selection = previous.get("catalog_selection")
    if selection not in (None, "watched-v1", "full-v1"):
        raise ValueError(f"Unsupported saved catalog selection: {selection}. Preserve the receipt and use its original code.")
    # Missing selection identifies an existing full-catalog receipt. Preserve its
    # exact source identity instead of reinterpreting it under the new default.
    legacy = saved_sources is not None and selection is None
    max_users = getattr(args, "max_users", None)
    if max_users is None:
        max_users = previous.get("max_users", DEFAULT_SAMPLE_USERS)
    seed = getattr(args, "seed", None)
    if seed is None:
        seed = previous.get("seed", 0)
    full_catalog = getattr(args, "full_catalog", None)
    if full_catalog is None:
        full_catalog = legacy or selection == "full-v1"
    elif type(full_catalog) is not bool:
        raise ValueError("full_catalog must be a boolean")
    if legacy and not full_catalog:
        raise ValueError("This checkout's saved datasets use the full catalog. "
                         "Use a separate checkout for the smaller watched-video catalog.")

    titles_raw = args.titles.read_bytes()
    watches_raw = args.watches.read_bytes()
    sources = {"titles_sha256": hashlib.sha256(titles_raw).hexdigest(),
               "watches_sha256": hashlib.sha256(watches_raw).hexdigest(),
               "max_users": max_users, "seed": seed}
    if not legacy:
        sources["catalog_selection"] = "full-v1" if full_catalog else "watched-v1"
    if saved_sources is not None and saved_sources != sources:
        raise ValueError("This checkout already has different data or sampling options. "
                         "Omit selection flags to reuse its saved selection, or use a separate checkout for new data.")
    titles, examples = select_training_data(titles_raw, watches_raw, max_users=max_users,
                                            seed=seed, full_catalog=full_catalog)
    if project is not None:
        snapshot = project["snapshot"]
    else:
        snapshot = uuid4().hex[:12]
        project = new_project(target, snapshot)
        project["sources"] = sources
        write_project(project)
    print(f"Target: {target.environment} | {target.api_server}", flush=True)
    users = len({row["user_id"] for row in examples})
    scope = "full catalog" if full_catalog else "videos watched by the selected users"
    print(f"Selection: {users:,} users | {len(titles):,} titles ({scope}) | {len(examples):,} examples", flush=True)
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
