"""Keep encoder-download settings separate from the serving environment."""
import importlib.util
from pathlib import Path

import chalkcompute

from semantic_ids import project


def test_encoder_build_can_download_while_both_workers_stay_offline(monkeypatch):
    declarations = []

    def capture_function(**kwargs):
        declarations.append(kwargs)
        return lambda function: function

    # Evaluate the actual module's image and decorator arguments without
    # registering Functions, reading credentials, or downloading an encoder.
    monkeypatch.setattr(chalkcompute, "function", capture_function)
    monkeypatch.delenv("CHALK_REMOTE_FUNCTION", raising=False)
    monkeypatch.setenv("CHALK_ENVIRONMENT", "test-environment")
    monkeypatch.setenv("CHALK_API_SERVER", "https://api.example.test")
    monkeypatch.setattr(project, "read_project", lambda: (
        {"training": {"tokenizer": {"model_version": 1}}}, None))
    path = Path(__file__).resolve().parents[1] / "host" / "run.py"
    spec = importlib.util.spec_from_file_location("semantic_ids_host_image_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert len(declarations) == 2
    for declaration in declarations:
        assert declaration["authenticated"] is True
        assert declaration["chalk_identity"] is True
        # Features enqueue calls: queued work must be able to start a worker
        # even when no synchronous request reaches the Function's route.
        assert declaration["min_replicas"] == 0
        assert declaration["target_queue_depth"] == 1
        build_env = dict(declaration["image"].to_proto().env)
        assert build_env.get("HF_HUB_OFFLINE", "0") == "0"
        assert build_env.get("TRANSFORMERS_OFFLINE", "0") == "0"
        assert declaration["env"]["HF_HUB_OFFLINE"] == "1"
        assert declaration["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert declaration["env"]["CHALK_ENVIRONMENT"] == "test-environment"
        destinations = {file.dest for file in declaration["image"].lazy_local_files}
        assert "/app/semantic_ids/hosting.py" in destinations
        assert "/app/semantic_ids/model.py" in destinations
