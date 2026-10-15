"""Small catalogs must reach native submission with a valid held-out split."""
from types import SimpleNamespace
import sys

import pytest

from semantic_ids import cli, project
from semantic_ids.project import Target


@pytest.mark.parametrize("title_count,override,expected", [
    (4, None, 1), (500, None, 50), (19738, None, 1000), (500, 100, 100),
])
def test_tokenizer_submission_chooses_valid_holdout(tmp_path, monkeypatch, title_count, override, expected):
    import chalkcompute

    monkeypatch.setattr(project, "RECEIPT", tmp_path / "project.json")
    for key in ("CHALK_ENVIRONMENT", "CHALK_ENVIRONMENT_ID", "CHALK_API_SERVER"):
        monkeypatch.delenv(key, raising=False)
    project.write_project({
        "format_version": 1,
        "target": {"environment": "test", "api_server": "https://api.example.test"},
        "snapshot": "test", "training": {},
        "datasets": {
            "titles": {"name": "titles-test", "revision_id": "titles-revision", "rows": title_count},
            "examples": {"name": "examples-test", "revision_id": "examples-revision"},
        },
    })
    submitted = []

    def submit(**kwargs):
        submitted.append(kwargs["config"])
        raise RuntimeError("stop at mocked submission")

    function = SimpleNamespace(run_training=submit)
    monkeypatch.setitem(sys.modules, "semantic_ids.training", SimpleNamespace(
        train_tokenizer=function, train_recommender=function))
    monkeypatch.setattr(Target, "client", lambda self: object())
    monkeypatch.setattr(chalkcompute, "ConnectClient", lambda **kwargs: SimpleNamespace(close=lambda: None))
    args = cli.parser().parse_args(["train-tokenizer", *(["--holdout", str(override)] if override else [])])
    with pytest.raises(RuntimeError, match="mocked submission"):
        cli.train(args, "tokenizer")
    assert submitted[0]["holdout"] == expected
    assert 0 < expected < title_count


def test_prepare_parser_leaves_omitted_selection_available_for_receipt_recovery():
    args = cli.parser().parse_args(["prepare", "--environment", "test"])
    assert args.max_users is None
    assert args.seed is None
    assert args.full_catalog is None
    full = cli.parser().parse_args(["prepare", "--environment", "test", "--max-users", "5000", "--full-catalog"])
    assert full.max_users == 5000
    assert full.full_catalog is True
