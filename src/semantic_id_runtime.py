"""Torch-free serving path: title -> semantic ID with onnxruntime and numpy.

Serving only needs a forward pass, and importing torch plus sentence-transformers
costs ~9s on a cold replica, longer than everything else in a scale-from-zero call
combined. This runs the same MiniLM weights through the ONNX export published in the
model's own Hugging Face repo, then the RQ-VAE encoder and codebooks in numpy.
`export_numpy_runtime.py` checks that it assigns the same tokens as the torch path.
"""

from __future__ import annotations

import pathlib

import numpy as np

# The ONNX export and tokenizer come from this exact revision of the encoder used in
# training; a different revision would assign different IDs without any error.
ENCODER_REPO = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ENCODER_FILES = ("onnx/model.onnx", "tokenizer.json")
# sentence-transformers truncates this model's input at 256 tokens.
MAX_SEQ_LENGTH = 256


def encoder_file_url(filename: str) -> str:
    return f"https://huggingface.co/{ENCODER_REPO}/resolve/{ENCODER_REVISION}/{filename}"


class SemanticIdEncoder:
    """Title encoder + RQ-VAE quantizer, loaded once and reused for every call."""

    def __init__(self, encoder_dir: pathlib.Path, rqvae_npz: pathlib.Path) -> None:
        import onnxruntime
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(encoder_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = onnxruntime.InferenceSession(
            str(encoder_dir / "onnx" / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {node.name for node in self._session.get_inputs()}

        weights = np.load(rqvae_npz)
        layer_count = int(weights["encoder_layers"])
        self._layers = [(weights[f"encoder_{i}_weight"], weights[f"encoder_{i}_bias"]) for i in range(layer_count)]
        self._codebooks = [weights[f"codebook_{level}"] for level in range(int(weights["num_levels"]))]

    def embed(self, titles: list[str]) -> np.ndarray:
        """Unit-norm 384-d title vectors, as `SentenceTransformer.encode(normalize_embeddings=True)`."""
        batch = self._tokenizer.encode_batch(titles)
        ids = np.array([item.ids for item in batch], dtype=np.int64)
        mask = np.array([item.attention_mask for item in batch], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        tokens = self._session.run(None, feeds)[0]
        # Mean pooling over real tokens only, then L2 normalization: the same two
        # modules sentence-transformers stacks after the transformer for this model.
        weights = mask[..., None].astype(np.float32)
        pooled = (tokens * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        return pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)

    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        """Semantic ID tokens for content vectors, as `RQVAE.encode_ids`."""
        hidden = vectors.astype(np.float32)
        for index, (weight, bias) in enumerate(self._layers):
            hidden = hidden @ weight.T + bias
            if index < len(self._layers) - 1:
                hidden = np.maximum(hidden, 0.0)
        residual = hidden
        codes = []
        for codebook in self._codebooks:
            # argmin of squared distance picks the same code as argmin of distance.
            distances = (
                (residual**2).sum(axis=1, keepdims=True) - 2.0 * residual @ codebook.T + (codebook**2).sum(axis=1)
            )
            level_codes = distances.argmin(axis=1)
            codes.append(level_codes)
            residual = residual - codebook[level_codes]
        return np.stack(codes, axis=1)

    def semantic_ids(self, titles: list[str]) -> np.ndarray:
        return self.quantize(self.embed(titles))
