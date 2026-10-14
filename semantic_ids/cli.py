"""Explicit stages for the how-to; all remote work is initiated by a command."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from .project import ROOT, RECEIPT, read_project, write_project
from .settings import ENCODER_MODEL, ENCODER_REVISION, TOKENIZER_MODEL, RECOMMENDER_MODEL


def prepare(args):
    from .stages import load_stage

    return load_stage("prepare").prepare(args)


def _metadata(artifact):
    return {key: value.to_python() if hasattr(value, "to_python") else value
            for key, value in artifact.metadata.items()}


def _ensure_namespace(client, name):
    import grpc
    try:
        return client.get_model_namespace(name=name)
    except RuntimeError as exc:
        # ChalkPy wraps grpc.RpcError, retaining it as the exception context.
        cause = exc.__cause__ or exc.__context__
        if not isinstance(cause, grpc.RpcError) or cause.code() != grpc.StatusCode.NOT_FOUND:
            raise
    return client.register_model_namespace(name=name, description="MicroLens semantic-ID how-to")


def _register_model(client, project, record, kind, run_id, artifact, report, version=None, retry=False):
    name = TOKENIZER_MODEL if kind == "tokenizer" else RECOMMENDER_MODEL
    if version is not None:
        registered = client.get_model_version(name=name, version=version)
        if registered.model_artifact is None:
            raise ValueError("That registry version has no training artifact")
        metadata = _metadata(registered.model_artifact)
        # This public API exposes provenance, not the artifact ID. Require the
        # complete final report from this native run before attaching a version.
        if (registered.version != version or registered.model_name != name
                or metadata.get("completed") is not True or metadata.get("kind") != kind
                or json.loads(metadata.get("report_json", "{}")) != report):
            raise ValueError("That registry version does not match this run's completed training report")
        action = "Reconnected to"
    else:
        if record.get("registration_started") and not retry:
            raise ValueError("Registration was interrupted. Inspect the model registry and rerun "
                             f"train-{kind} --model-version <existing-version> to reconnect. "
                             "If no version was created, use --retry-registration.")
        _ensure_namespace(client, name)
        record["registration_started"] = True
        write_project(project)
        # Using run_id avoids ChalkPy 2.160's artifact-only registration bug.
        registered = client.promote_model_artifact(name=name, run_id=run_id)
        if registered.artifact.id != artifact.id:
            raise ValueError("The registry response identifies a different checkpoint")
        version = registered.model_version
        action = "Registered"
    record.update(model_version=version, model_name=name)
    write_project(project)
    print(f"{action} {name} version {version}")


def _restart_failed(project, kind, connection):
    """Replace a terminal startup failure without discarding training work."""
    from chalkcompute import TrainingRunHandle

    previous = project["training"][kind]
    run = TrainingRunHandle(previous["run_id"], connection)
    if run.status().name != "FAILED":
        raise ValueError("--restart-failed requires a confirmed FAILED native run")
    if run.latest_checkpoint() is not None:
        raise ValueError("The failed run has a checkpoint; inspect it before starting another run. "
                         "Its saved training progress was preserved.")
    record = {"config": dict(previous["config"]), "checkpoint_key": uuid4().hex[:16]}
    project.setdefault("training_history", {}).setdefault(kind, []).append(previous)
    project["training"][kind] = record
    write_project(project)
    print(f"Preserved failed run {previous['run_id']}; submitting current code with the saved datasets "
          "and original configuration.", flush=True)
    return record


def train(args, kind):
    from chalkcompute import ConnectClient, TrainingRunHandle

    project, target = read_project()
    key = "titles" if kind == "tokenizer" else "examples"
    if key not in project["datasets"]:
        raise ValueError(f"The {key} upload did not finish; inspect the saved receipt")
    if kind == "recommender" and "catalog" not in project["datasets"]:
        raise ValueError("Materialize the semantic-ID feature first")
    record = project["training"].get(kind)
    retry = getattr(args, "retry_submission", False)
    retry_registration = getattr(args, "retry_registration", False)
    restart = getattr(args, "restart_failed", False)
    version = getattr(args, "model_version", None)
    if restart and (retry or retry_registration or args.run_id or version is not None):
        raise ValueError("Use --restart-failed without retry or reconnection flags")
    if restart and (record is None or not record.get("run_id")
                    or any(record.get(field) for field in ("model_version", "registration_started", "artifact_id"))):
        raise ValueError("--restart-failed requires a saved native run without model publication")
    if retry and (args.run_id or version is not None or retry_registration):
        raise ValueError("Use --retry-submission only when no native run exists; otherwise reconnect to that run")
    if retry and (record is None or not record.get("submission_started") or record.get("run_id")):
        raise ValueError("--retry-submission requires an interrupted submission without a saved run ID")
    if version is not None and (record is None or not (record.get("run_id") or args.run_id)):
        raise ValueError("Reconnect to the original native run with --run-id before attaching a model version")
    if retry_registration and (version is not None or record is None or not record.get("registration_started")
                               or not record.get("run_id") or record.get("model_version")):
        raise ValueError("--retry-registration requires an interrupted registration without a saved model version")
    if record is None:
        config = {
            "dataset_revision": project["datasets"][key]["revision_id"],
            "epochs": args.epochs, "batch_size": args.batch_size,
            "seed": args.seed, "encoder_model": ENCODER_MODEL, "encoder_revision": ENCODER_REVISION,
            "checkpoint_every": 20 if kind == "tokenizer" else 1,
        }
        if kind == "tokenizer":
            title_count = project["datasets"]["titles"]["rows"]
            holdout = args.holdout if args.holdout is not None else min(1000, max(1, title_count // 10))
            if holdout >= title_count:
                raise ValueError("--holdout must leave at least one title for training")
            config["holdout"] = holdout
        else:
            config.update(catalog_revision=project["datasets"]["catalog"]["revision_id"],
                          tokenizer_version=project["training"]["tokenizer"]["model_version"])
        record = {"config": config, "checkpoint_key": uuid4().hex[:16]}
        project["training"][kind] = record
    elif record.get("model_version"):
        if version is not None and version != record["model_version"]:
            raise ValueError("--model-version differs from this stage's saved model version")
        if args.run_id and args.run_id != record["run_id"]:
            raise ValueError("--run-id differs from this stage's saved native training run")
        print(f"{kind} already registered as version {record['model_version']} (run {record['run_id']}).")
        return
    elif not restart:
        print("Resuming the saved run and its original training configuration.", flush=True)
    if args.run_id:
        if record.get("run_id") and record["run_id"] != args.run_id:
            raise ValueError("--run-id differs from this stage's saved native training run")
        # A user-supplied ID may be mistyped. Persist it only after the completed
        # artifact passes identity checks, so a wrong ID cannot trap this receipt.
    client = target.client()
    connection = ConnectClient(chalk_client=client)
    try:
        if restart:
            record = _restart_failed(project, kind, connection)
        os.environ["SEMANTIC_TRAINING_KEY"] = record["checkpoint_key"]
        os.environ["SEMANTIC_TOKENIZER_DATASET"] = project["datasets"]["titles"]["name"]
        os.environ["SEMANTIC_RECOMMENDER_DATASET"] = project["datasets"]["examples"]["name"]
        from .training import train_tokenizer, train_recommender

        if args.run_id or record.get("run_id"):
            run = TrainingRunHandle(args.run_id or record["run_id"], connection)
        else:
            if record.get("submission_started") and not retry:
                raise ValueError("A previous submission did not save a run ID. Inspect Chalk's Training page; "
                                 "reconnect with --run-id <id>. If no run exists, rerun with "
                                 "--retry-submission. No duplicate run was submitted.")
            if retry:
                print("Retrying submission using the saved configuration; you confirmed no prior run exists.", flush=True)
            record["submission_started"] = True
            write_project(project)
            function = train_tokenizer if kind == "tokenizer" else train_recommender
            run = function.run_training(config=record["config"], client=connection)
            record["run_id"] = run.run_id
            write_project(project)
        print(f"{kind} training in {target.environment} | native run {run.run_id}", flush=True)
        run.wait()
        artifact = run.latest_checkpoint()
        if artifact is None:
            raise ValueError("Completed training has no checkpoint artifact")
        metadata = _metadata(artifact)
        report = json.loads(metadata.get("report_json", "{}"))
        expected_config = {"threads": 2, "checkpoint_every": 1, **record["config"]}
        digest = hashlib.sha256(json.dumps(expected_config, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        if (metadata.get("completed") is not True or metadata.get("kind") != kind
                or report.get("run_id") != run.run_id or report.get("config_sha256") != digest
                or report.get("dataset_revision") != record["config"]["dataset_revision"]):
            raise ValueError("The latest checkpoint is not the expected complete training result")
        if kind == "recommender" and any(report.get(key) != record["config"][key]
                                         for key in ("catalog_revision", "tokenizer_version")):
            raise ValueError("Recommender checkpoint uses a different semantic feature catalog")
        record.update(run_id=run.run_id, artifact_id=artifact.id, report=report)
        write_project(project)
        _register_model(client, project, record, kind, run.run_id, artifact, report, version, retry_registration)
        print(json.dumps(report.get("metrics", {}), indent=2))
    finally:
        connection.close()


def materialize(args):
    from .stages import load_stage

    return load_stage("materialize").materialize(args)


def recommend(args):
    from .stages import load_stage

    return load_stage("serve").recommend(args)


def status(args):
    """Show local receipts without contacting Chalk or importing deployment code."""
    if not RECEIPT.exists():
        print("No saved project. Start with prepare --environment <id> --api-server <url>.")
        return
    project, target = read_project()
    print(f"Saved progress | {target.environment} | {target.api_server}")
    print("Local records only; this command does not query live Chalk status.")
    for key in ("titles", "examples", "catalog"):
        saved = project["datasets"].get(key)
        detail = f"revision {saved['revision_id']}" if saved else "not recorded"
        print(f"Dataset {key}: {detail}")
    for kind in ("tokenizer", "recommender"):
        saved = project["training"].get(kind, {})
        if saved.get("model_version"):
            detail = f"registered version {saved['model_version']} | run {saved.get('run_id', 'unknown')}"
        elif saved.get("registration_started"):
            detail = f"registration interrupted | run {saved['run_id']} | reconnect with --model-version"
        elif saved.get("run_id"):
            detail = f"run {saved['run_id']} | rerun train-{kind} to inspect or wait"
        elif saved.get("submission_started"):
            detail = "submission interrupted | inspect Chalk before retrying"
        else:
            detail = "not submitted"
        print(f"Model {kind}: {detail}")
    from .import_cli import read_import

    imported = read_import(target)
    if imported:
        detail = (f"version {imported['model_version']} | {imported['catalog_count']:,} videos"
                  if imported.get("model_version") else "registration interrupted; inspect registry before retrying")
        print(f"Imported recommender: {detail} | source data: {imported['source_kind']}")


def _nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _positive(value):
    number = _nonnegative(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser(stage=None):
    result = argparse.ArgumentParser(description="Train and serve semantic-ID recommendations in Chalk")
    commands = result.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare", help="Upload raw MicroLens titles and ordered examples as Chalk datasets")
    data.add_argument("--environment", required=True, help="Chalk environment ID for this repository")
    data.add_argument("--api-server", help="API origin; defaults to this repository's Chalk authentication")
    data.add_argument("--titles", type=Path, default=ROOT / "data/MicroLens-100k_title_en.csv")
    data.add_argument("--watches", type=Path, default=ROOT / "data/MicroLens-100k_pairs.tsv")
    data.add_argument("--max-users", type=_positive,
                      help="Sample this many complete user histories; defaults to 100 for new projects or the saved selection")
    data.add_argument("--seed", type=_nonnegative,
                      help="Sampling seed; defaults to 0 for new projects or the saved seed")
    data.add_argument("--full-catalog", action="store_true", default=None,
                      help="Upload all source titles instead of only videos watched by the sampled users")
    for kind, epochs, batch in (("tokenizer", 400, 1024), ("recommender", 10, 256)):
        training = commands.add_parser(f"train-{kind}", help=f"Run native Chalk {kind} training")
        training.add_argument("--epochs", type=_positive, default=epochs)
        training.add_argument("--batch-size", type=_positive, default=batch)
        training.add_argument("--seed", type=_nonnegative, default=0)
        training.add_argument("--run-id", help="Reconnect to an existing native training run after an interrupted submission")
        training.add_argument("--model-version", type=_positive,
                              help="Reconnect to an existing registry version from this completed native run")
        training.add_argument("--retry-submission", action="store_true",
                              help="Retry an interrupted submission ONLY after confirming no native run exists in Chalk")
        training.add_argument("--retry-registration", action="store_true",
                              help="Retry interrupted registration ONLY after confirming no registry version was created")
        training.add_argument("--restart-failed", action="store_true",
                              help="Submit current code for a FAILED run with no checkpoint; preserve its record and reuse datasets")
        if kind == "tokenizer":
            training.add_argument("--holdout", type=_positive,
                                  help="Held-out titles; defaults to 10%% of the catalog, capped at 1,000")
    transfer = commands.add_parser("import-recommender", help="Import compatible pretrained weights and their original catalog")
    transfer.add_argument("--source-environment", required=True)
    transfer.add_argument("--source-api-server", required=True)
    transfer.add_argument("--source-model", required=True)
    transfer.add_argument("--source-version", type=_positive, required=True)
    transfer.add_argument("--dry-run", action="store_true", help="Download and validate locally without registering or changing model receipts")
    transfer.add_argument("--model-version", type=_positive, help="Reconnect an interrupted import to its verified destination version")
    transfer.add_argument("--retry-registration", action="store_true", help="Retry ONLY after confirming no destination version was created")
    commands.add_parser("deploy", help="Deploy both inference Functions to Chalk Scaling Groups")
    materialization = commands.add_parser("materialize", help="Compute the catalog's semantic-ID feature in Chalk")
    materialization.add_argument("--branch", help="Chalk feature branch to query; omitted uses the default deployment")
    materialization.add_argument("--limit", type=_positive,
                                 help="Time a sample of the first N title rows without saving it as the training catalog")
    commands.add_parser("status", help="Show saved local progress without contacting Chalk")
    serving = commands.add_parser("recommend", help="Query the Chalk recommendation feature")
    serving.add_argument("--model-source", choices=("auto", "imported", "trained"), default="auto",
                         help="auto prefers a completed native recommender, then an imported model")
    serving.add_argument("--history", nargs="+", type=_nonnegative, required=True)
    serving.add_argument("--top-k", type=_positive, choices=range(1, 21), default=6)
    serving.add_argument("--branch", help="Chalk feature branch to query; omitted uses the default deployment")
    if stage is not None:
        selected = commands.choices[stage]
        selected.prog = result.prog
        selected.set_defaults(command=stage)
        return selected
    return result


def main(argv=None, *, stage=None):
    args = parser(stage).parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command.startswith("train-"):
            train(args, args.command.removeprefix("train-"))
        elif args.command == "import-recommender":
            from .import_cli import import_recommender

            import_recommender(args)
        elif args.command == "deploy":
            project, target = read_project()
            if not project["training"].get("tokenizer", {}).get("model_version"):
                raise ValueError("Finish tokenizer training before deploying inference")
            # Importing this dedicated module executes the Function decorators.
            # It is deliberately absent from feature discovery and normal imports.
            from . import deploy
            deploy.main()
        elif args.command == "materialize":
            materialize(args)
        elif args.command == "status":
            status(args)
        else:
            recommend(args)
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
