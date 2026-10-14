"""Explicit artifact import; keeps imported lineage separate from native training."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

from .project import ROOT, Target, read_project
from .settings import RECOMMENDER_MODEL

IMPORT_RECEIPT = ROOT / ".semantic-ids" / "imported-recommender.json"


def read_import(target: Target):
    if not IMPORT_RECEIPT.exists():
        return None
    record = json.loads(IMPORT_RECEIPT.read_text())
    if record.get("format_version") != 1 or record.get("target") != asdict(target):
        raise ValueError("Imported recommender receipt belongs to a different Chalk target or format")
    if record.get("model_name") != RECOMMENDER_MODEL:
        raise ValueError("Imported recommender receipt has a different model namespace")
    return record


def selected_recommender(project, target, source="auto"):
    trained = project["training"].get("recommender", {})
    if source in ("auto", "trained") and trained.get("model_version"):
        return trained
    if source in ("auto", "imported"):
        imported = read_import(target)
        if imported and imported.get("model_version"):
            return imported
    if source == "trained":
        raise ValueError("Train the recommender first, or select --model-source imported")
    if source == "imported":
        raise ValueError("Import the recommender first")
    raise ValueError("Train or import the recommender first")


def _save(record):
    IMPORT_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    temporary = IMPORT_RECEIPT.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(IMPORT_RECEIPT)


@contextmanager
def _import_lock():
    import fcntl

    IMPORT_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    with IMPORT_RECEIPT.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another recommender import is running in this checkout") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _manifest(downloaded):
    rows = []
    for role, paths in (("model", downloaded.downloaded_model_files),
                        ("additional", downloaded.downloaded_additional_files)):
        for path in paths:
            path = Path(path)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            rows.append({"role": role, "name": path.name, "sha256": digest.hexdigest()})
    return sorted(rows, key=lambda row: (row["role"], row["name"]))


def _print_ready(record):
    print(f"Imported {record['model_name']} version {record['model_version']} | "
          f"{record['catalog_count']:,} videos | source data: {record['source_kind']}")
    history = " ".join(str(video) for video in record["sample_history_ids"])
    print(f"Sample history: {history}")
    print("Use recommend --model-source imported with the appropriate --branch.")


def import_recommender(args):
    # A separate receipt cannot be overwritten by an in-flight materialization.
    with _import_lock():
        _import_recommender(args)


def _import_recommender(args):
    from .cli import _ensure_namespace, _metadata
    from .importing import source_client, validate_artifact, registration_kwargs

    _, target = read_project()
    source = {"environment": args.source_environment,
              "api_server": args.source_api_server.rstrip("/"),
              "model_name": args.source_model, "model_version": args.source_version}
    # Validate the source origin without activating it or replacing destination auth.
    Target(source["environment"], source["api_server"])
    record = read_import(target)
    if record and record.get("source") != source:
        raise ValueError("This checkout already records another import; use a separate checkout")
    if args.dry_run and (args.model_version is not None or args.retry_registration):
        raise ValueError("--dry-run cannot reconnect or retry registration")
    if args.model_version is not None and args.retry_registration:
        raise ValueError("Choose --model-version or --retry-registration")
    if record and record.get("model_version"):
        if args.model_version is not None and args.model_version != record["model_version"]:
            raise ValueError("--model-version differs from the saved imported version")
        if not args.dry_run:
            if args.retry_registration:
                raise ValueError("The imported model is already registered")
            _print_ready(record)
            return
    pending = bool(record and record.get("registration_started") and not record.get("model_version"))
    if args.retry_registration and not pending:
        raise ValueError("--retry-registration requires an interrupted import")
    if pending and not (args.model_version is not None or args.retry_registration or args.dry_run):
        raise ValueError("Import registration was interrupted. Inspect the destination registry and reconnect "
                         "with --model-version <existing-version>. Only if none was created, use --retry-registration.")

    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="import-recommender-", dir=work) as directory:
        client = source_client(source["environment"], source["api_server"])
        print(f"Downloading {source['model_name']} v{source['model_version']} "
              f"from {source['environment']}…", flush=True)
        try:
            downloaded = client.download_model_artifact(
                name=source["model_name"], version=source["model_version"], download_dir=directory)
        except Exception as exc:
            # SDK errors can contain signed storage URLs; do not display them.
            raise RuntimeError(f"Source download failed ({type(exc).__name__}); check source access") from None
        summary = validate_artifact(downloaded)
        kwargs = registration_kwargs(downloaded, source_environment=source["environment"],
                                     source_api_server=source["api_server"], source_model_name=source["model_name"],
                                     source_model_version=source["model_version"], summary=summary)
        identity = {"source": source, "target": asdict(target), "files": _manifest(downloaded)}
        import_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if record and record.get("import_id") != import_id:
            raise ValueError("Source artifact bytes differ from the saved import")
        kwargs["metadata"]["semantic_ids_import_id"] = import_id
        print(f"Validated {summary['catalog_count']:,} catalog videos | {summary['catalog_fingerprint']}", flush=True)
        if args.dry_run:
            print("Dry run passed; no destination model was registered and no model receipt was changed.")
            print("Sample history: " + " ".join(map(str, summary["sample_history_ids"])))
            return

        if record is None:
            record = {"format_version": 1, "target": asdict(target), "source": source,
                      "model_name": RECOMMENDER_MODEL, "import_id": import_id,
                      "catalog_count": summary["catalog_count"],
                      "catalog_fingerprint": summary["catalog_fingerprint"],
                      "sample_history_ids": summary["sample_history_ids"],
                      "source_kind": summary["source_kind"]}
        destination = target.client()
        if args.model_version is not None:
            registered = destination.get_model_version(name=RECOMMENDER_MODEL, version=args.model_version)
            if (registered.model_name != RECOMMENDER_MODEL or registered.version != args.model_version
                    or registered.model_artifact is None
                    or _metadata(registered.model_artifact).get("semantic_ids_import_id") != import_id):
                raise ValueError("Destination version does not match this exact imported artifact")
            record["model_version"] = args.model_version
        else:
            _ensure_namespace(destination, RECOMMENDER_MODEL)
            record["registration_started"] = True
            _save(record)
            print(f"Registering {RECOMMENDER_MODEL} in {target.environment}…", flush=True)
            try:
                registered = destination.register_model_version(name=RECOMMENDER_MODEL, **kwargs)
            except Exception as exc:
                raise RuntimeError(f"Import registration interrupted ({type(exc).__name__}). Inspect the destination "
                                   "registry before retrying; reconnect with --model-version if created.") from None
            if registered.model_name != RECOMMENDER_MODEL:
                raise ValueError("Destination registration returned a different model namespace")
            record["model_version"] = registered.model_version
            record["artifact_id"] = registered.artifact.id
        _save(record)
        _print_ready(record)
