"""Exercise native worker imports and the staged source image without networking."""
import importlib
from pathlib import Path
import sys

import pytest


STAGES = {
    "train_tokenizer": ("train", "semantic_ids_train"),
    "train_recommender": ("train-recommender", "semantic_ids_train_recommender"),
}


@pytest.fixture
def isolated_training_imports(forbid_network):
    import chalkcompute
    import semantic_ids

    # Preserve both import caches and Python's parent-package bindings. Without
    # this isolation, worker-mode wrappers could leak into the CLI tests.
    missing = object()
    names = (
        "semantic_ids.training",
        "semantic_ids_train",
        "semantic_ids_train_recommender",
        "chalkcompute.training.entrypoint",
        "chalkcompute.training",
    )
    modules = {name: sys.modules.get(name, missing) for name in names}
    bindings = [(package, vars(package).get("training", missing))
                for package in (chalkcompute, semantic_ids)]
    for name in names:
        sys.modules.pop(name, None)
    for package, _ in bindings:
        vars(package).pop("training", None)
    try:
        yield
    finally:
        for name, previous in modules.items():
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        for package, previous in bindings:
            if previous is missing:
                vars(package).pop("training", None)
            else:
                vars(package)["training"] = previous


@pytest.mark.parametrize("name", STAGES)
def test_worker_entrypoint_imports_then_invokes_training_body(
    isolated_training_imports, monkeypatch, name,
):
    from semantic_ids.stages import load_stage

    monkeypatch.setenv("CHALK_TRAINING_RUN_ID", "import-test-run")
    entrypoint = importlib.import_module("chalkcompute.training.entrypoint")
    stage_name, module_name = STAGES[name]
    stage = load_stage(stage_name)
    frame, config, result = object(), {"example": "config"}, object()
    calls = []

    def body(df, supplied_config):
        calls.append((df, supplied_config))
        return result

    monkeypatch.setattr(stage, "run_body", body)
    function = entrypoint._discover_function(f"{module_name}.{name}")

    assert callable(function)
    assert function(frame, config) is result
    assert calls == [(frame, config)]
    assert not hasattr(function, "run_training")
    assert entrypoint._discover_function(f"semantic_ids.training.{name}") is function


@pytest.mark.parametrize("name", STAGES)
def test_client_import_retains_native_submission_after_entrypoint_import(
    isolated_training_imports, monkeypatch, name,
):
    monkeypatch.delenv("CHALK_TRAINING_RUN_ID", raising=False)
    importlib.import_module("chalkcompute.training.entrypoint")
    module = importlib.import_module("semantic_ids.training")
    from chalkcompute import TrainingFunction

    function = getattr(module, name)
    assert isinstance(function, TrainingFunction)
    assert callable(function.run_training)
    assert function._func.__module__ == STAGES[name][1]


@pytest.mark.parametrize("name", STAGES)
def test_native_image_ships_shared_package_and_standalone_stage_source(
    isolated_training_imports, monkeypatch, name,
):
    monkeypatch.delenv("CHALK_TRAINING_RUN_ID", raising=False)
    module = importlib.import_module("semantic_ids.training")
    function = getattr(module, name)
    image = function._build_deploy_image()
    sources = {item.dest: Path(item.src) for item in image.lazy_local_files}
    stage_name, module_name = STAGES[name]
    root = Path(__file__).resolve().parents[1]

    assert sources[f"/app/{module_name}.py"] == root / stage_name / "run.py"
    for filename in ("__init__.py", "training_runtime.py", "rqvae.py", "model.py", "settings.py", "data.py"):
        assert sources[f"/app/semantic_ids/{filename}"] == root / "semantic_ids" / filename
    assert not any(".semantic-ids" in str(path) for path in sources.values())
