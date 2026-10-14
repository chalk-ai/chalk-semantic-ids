"""Inference bodies for Chalk Functions; importing this module performs no I/O."""

import json
from pathlib import Path
import tempfile
import threading

from .settings import ENCODER_MODEL, ENCODER_REVISION, MAX_TITLE_LENGTH, RECOMMENDER_MODEL, TOKENIZER_MODEL


_lock = threading.RLock()
_client = None
_artifacts = {}
_models = {}
_encoder = None


def _positive_integer(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _video_id(value):
    if type(value) is not int or value < 0:
        raise ValueError("Video IDs must be nonnegative integers")
    return value


def _checkpoint_files(name, version, filenames):
    """Keep each immutable registry download alive for this worker's lifetime."""
    global _client
    _positive_integer(version, "Model version")
    key = (name, version)
    with _lock:
        if key not in _artifacts:
            from chalk.client import ChalkClient

            if _client is None:
                _client = ChalkClient()
            directory = tempfile.TemporaryDirectory(prefix="semantic-id-model-")
            try:
                result = _client.download_model_artifact(
                    name=name, version=version, download_dir=directory.name
                )
                paths = set(result.downloaded_model_files + result.downloaded_additional_files)
                selected = {}
                for filename in filenames:
                    matches = [Path(path) for path in paths if Path(path).name == filename]
                    if len(matches) != 1 or not matches[0].is_file():
                        raise ValueError(f"{name} v{version} must contain exactly one {filename}")
                    selected[filename] = matches[0]
                _artifacts[key] = (directory, selected)
            except BaseException:
                directory.cleanup()
                raise
        return _artifacts[key][1]


def _tokenizer(version):
    import torch
    from .rqvae import RQVAE

    key = (TOKENIZER_MODEL, version)
    with _lock:
        if key not in _models:
            path = _checkpoint_files(TOKENIZER_MODEL, version, ["rqvae.pt"])["rqvae.pt"]
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            metadata = checkpoint.get("metadata", {})
            if (metadata.get("encoder_model") != ENCODER_MODEL
                    or metadata.get("encoder_revision") != ENCODER_REVISION):
                raise ValueError("Tokenizer checkpoint and hosted title encoder differ")
            if metadata.get("completed") is not True or checkpoint.get("quantizer_initialized") != [True, True, True]:
                raise ValueError("Tokenizer must be a completed checkpoint with initialized codebooks")
            config = checkpoint["config"]
            if config["num_levels"] != 3 or config["input_dim"] != 384:
                raise ValueError("Expected three semantic codes from 384-dimensional title embeddings")
            model = RQVAE(**config)
            model.load_state_dict(checkpoint["state_dict"])
            for quantizer in model.quantizers:
                quantizer._initialized = True
            _models[key] = model.eval()
        return _models[key]


def tokenize_title(title: str, model_version: int) -> list[int]:
    """Encode a title with a pinned RQ-VAE; the catalog adds collision suffixes."""
    global _encoder
    _positive_integer(model_version, "Tokenizer model version")
    # MicroLens includes empty titles. Preserve them for train/serve consistency.
    if not isinstance(title, str) or len(title) > MAX_TITLE_LENGTH:
        raise ValueError(f"Title must be a string of at most {MAX_TITLE_LENGTH:,} characters")
    import torch

    with _lock:
        model = _tokenizer(model_version)
        if _encoder is None:
            from sentence_transformers import SentenceTransformer

            _encoder = SentenceTransformer("/opt/encoder", device="cpu", local_files_only=True)
        vector = _encoder.encode([title], convert_to_numpy=True, normalize_embeddings=True)
        with torch.inference_mode():
            tokens = model.encode_ids(torch.as_tensor(vector, dtype=torch.float32))[0].tolist()
    if len(tokens) != 3 or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Tokenizer returned invalid semantic codes")
    return tokens


def _recommender(version):
    from .model import load_model

    key = (RECOMMENDER_MODEL, version)
    with _lock:
        if key not in _models:
            files = _checkpoint_files(RECOMMENDER_MODEL, version, ["recommender.pt", "catalog.json"])
            rows = json.loads(files["catalog.json"].read_text())
            if not isinstance(rows, list) or not rows:
                raise ValueError("Recommender artifact must contain a nonempty catalog")
            details = {}
            catalog = {}
            for row in rows:
                video_id = _video_id(row["video_id"])
                codes = row["semantic_tokens"]
                if video_id in catalog or not isinstance(row["title"], str):
                    raise ValueError("Catalog requires unique video IDs and string titles")
                if (not isinstance(codes, list) or len(codes) != 4
                        or any(type(token) is not int or token < 0 for token in codes)):
                    raise ValueError("Catalog entries require four nonnegative integer semantic tokens")
                catalog[video_id] = tuple(codes)
                details[video_id] = row["title"]
            if len(set(catalog.values())) != len(catalog):
                raise ValueError("Catalog semantic tokens must uniquely identify each video")
            # load_model uses weights_only=True and checks the catalog fingerprint.
            model = load_model(files["recommender.pt"], catalog=catalog)
            _models[key] = (model, catalog, details)
        return _models[key]


def recommend_videos(history_ids: list[int], model_version: int, top_k: int) -> str:
    """Serve next-video predictions from one immutable model and its catalog."""
    _positive_integer(model_version, "Recommender model version")
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("top_k must be an integer between 1 and 20")
    if (not isinstance(history_ids, list) or len(history_ids) > 500
            or any(type(video) is not int or video < 0 for video in history_ids)):
        raise ValueError("History must contain at most 500 nonnegative integer video IDs")
    from .model import recommend

    model, catalog, titles = _recommender(model_version)
    if any(video not in catalog for video in history_ids):
        raise ValueError("History contains a video outside the trained model's catalog")
    rows = recommend(model, catalog, history_ids, top_k=top_k)
    return json.dumps({
        "model_version": model_version,
        "execution": "chalk",
        "items": [{**row, "title": titles[row["video_id"]]} for row in rows],
    }, allow_nan=False)
