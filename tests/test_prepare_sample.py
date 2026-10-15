"""Preparation selection and receipt recovery with fake Dataset uploads only."""
from argparse import Namespace
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import pytest

from semantic_ids import project
from semantic_ids.project import Target
from semantic_ids.stages import load_stage


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    import chalkcompute

    receipt = tmp_path / "project.json"
    monkeypatch.setattr(project, "RECEIPT", receipt)
    for key in ("CHALK_ENVIRONMENT", "CHALK_ENVIRONMENT_ID", "CHALK_API_SERVER"):
        monkeypatch.delenv(key, raising=False)
    titles = tmp_path / "titles.csv"
    watches = tmp_path / "watches.tsv"
    titles.write_text("".join(f"{video},title {video}\n" for video in range(1, 10)))
    watches.write_text("a\t1 2 3 4\nb\t5 6 7 8\n")
    target = Target("fixture-environment", "https://api.example.test")
    monkeypatch.setattr(Target, "selected", lambda *args: target)
    monkeypatch.setattr(Target, "client", lambda self: object())
    uploads = []

    def upload(name, rows):
        uploads.append((name, rows))
        return SimpleNamespace(revision_id=name + "-revision")

    monkeypatch.setattr(chalkcompute, "DatasetClient", lambda **kwargs: SimpleNamespace(upload=upload))
    args = Namespace(environment=target.environment, api_server=target.api_server,
                     titles=titles, watches=watches, max_users=None, seed=None, full_catalog=None)
    return SimpleNamespace(receipt=receipt, target=target, args=args, uploads=uploads,
                           prepare=load_stage("prepare").prepare)


def source_identity(args, *, max_users, seed):
    return {"titles_sha256": hashlib.sha256(args.titles.read_bytes()).hexdigest(),
            "watches_sha256": hashlib.sha256(args.watches.read_bytes()).hexdigest(),
            "max_users": max_users, "seed": seed}


def test_new_defaults_upload_watched_catalog_and_reuse_it(prepared_inputs):
    case = prepared_inputs
    case.prepare(case.args)
    saved = json.loads(case.receipt.read_text())
    assert saved["sources"] == {**source_identity(case.args, max_users=100, seed=0),
                                "catalog_selection": "watched-v1"}
    assert [row["id"] for row in case.uploads[0][1]] == list(range(1, 9))
    assert saved["datasets"]["titles"]["rows"] == 8
    assert saved["datasets"]["examples"]["rows"] == 6
    before = case.receipt.read_bytes()
    case.prepare(case.args)
    assert len(case.uploads) == 2
    assert case.receipt.read_bytes() == before


def test_full_catalog_is_explicit_and_reused_when_flag_is_omitted(prepared_inputs):
    case = prepared_inputs
    case.args.full_catalog = True
    case.args.max_users = 1
    case.args.seed = 7
    case.prepare(case.args)
    saved = json.loads(case.receipt.read_text())
    assert saved["sources"]["catalog_selection"] == "full-v1"
    assert saved["datasets"]["titles"]["rows"] == 9
    assert saved["datasets"]["examples"]["rows"] == 3
    before = case.receipt.read_bytes()
    case.args.full_catalog = case.args.max_users = case.args.seed = None
    case.prepare(case.args)
    assert len(case.uploads) == 2
    assert case.receipt.read_bytes() == before


def test_legacy_partial_upload_retains_full_catalog_and_saved_sampling(prepared_inputs):
    case = prepared_inputs
    sources = source_identity(case.args, max_users=1, seed=7)
    saved_title = {"name": "legacy-titles", "revision_id": "legacy-title-revision", "rows": 9}
    project.write_project({"format_version": 1, "target": asdict(case.target), "sources": sources,
                           "snapshot": "legacy", "datasets": {"titles": saved_title}, "training": {}})
    case.prepare(case.args)
    saved = json.loads(case.receipt.read_text())
    assert saved["sources"] == sources
    assert "catalog_selection" not in saved["sources"]
    assert saved["datasets"]["titles"] == saved_title
    assert len(case.uploads) == 1 and "examples" in case.uploads[0][0]
    assert len(case.uploads[0][1]) == 3
    assert {row["user_id"] for row in case.uploads[0][1]} == {"b"}


def test_legacy_receipt_without_uploads_still_uploads_full_title_catalog(prepared_inputs):
    case = prepared_inputs
    sources = source_identity(case.args, max_users=1, seed=7)
    project.write_project({"format_version": 1, "target": asdict(case.target), "sources": sources,
                           "snapshot": "legacy", "datasets": {}, "training": {}})
    # Older callers need not construct the newly introduced argument.
    del case.args.full_catalog
    case.prepare(case.args)
    saved = json.loads(case.receipt.read_text())
    assert saved["sources"] == sources
    assert saved["datasets"]["titles"]["rows"] == 9
    assert [row["id"] for row in case.uploads[0][1]] == list(range(1, 10))


@pytest.mark.parametrize("option,value", [("max_users", 1), ("seed", 7), ("full_catalog", True)])
def test_changed_explicit_options_do_not_mutate_saved_receipt(prepared_inputs, option, value):
    case = prepared_inputs
    case.prepare(case.args)
    before = case.receipt.read_bytes()
    setattr(case.args, option, value)
    with pytest.raises(ValueError, match="different data or sampling options"):
        case.prepare(case.args)
    assert case.receipt.read_bytes() == before
    assert len(case.uploads) == 2


def test_legacy_catalog_cannot_silently_become_a_watched_subset(prepared_inputs):
    case = prepared_inputs
    project.write_project({"format_version": 1, "target": asdict(case.target),
                           "sources": source_identity(case.args, max_users=100, seed=0),
                           "snapshot": "legacy", "datasets": {}, "training": {}})
    before = case.receipt.read_bytes()
    case.args.full_catalog = False
    with pytest.raises(ValueError, match="saved datasets use the full catalog"):
        case.prepare(case.args)
    assert case.receipt.read_bytes() == before
    assert not case.uploads


def test_sample_partial_upload_resumes_its_recorded_selection(prepared_inputs, monkeypatch):
    import chalkcompute

    case = prepared_inputs
    attempt = []

    def upload(name, rows):
        attempt.append(name)
        if len(attempt) == 2:
            raise RuntimeError("Interrupted example upload")
        return SimpleNamespace(revision_id=name + "-revision")

    monkeypatch.setattr(chalkcompute, "DatasetClient", lambda **kwargs: SimpleNamespace(upload=upload))
    case.args.max_users, case.args.seed = 1, 7
    with pytest.raises(RuntimeError, match="Interrupted"):
        case.prepare(case.args)
    first = json.loads(case.receipt.read_text())
    case.args.max_users = case.args.seed = None
    case.prepare(case.args)
    final = json.loads(case.receipt.read_text())
    assert final["sources"] == first["sources"]
    assert final["datasets"]["titles"] == first["datasets"]["titles"]
    assert final["datasets"]["titles"]["rows"] == 4
    assert final["datasets"]["examples"]["rows"] == 3
    assert len([name for name in attempt if "titles" in name]) == 1
