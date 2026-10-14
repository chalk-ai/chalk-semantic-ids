"""Versioned dependencies shared by native training and hosted inference."""

ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_TITLE_LENGTH = 2000
TOKENIZER_MODEL = "semantic-id-tokenizer"
RECOMMENDER_MODEL = "semantic-id-recommender"
TOKENIZER_FUNCTION = "semantic-id-tokenize"
RECOMMENDER_FUNCTION = "semantic-id-recommend"

# Chalk workers use CPUs. Install this wheel before the other dependencies so
# pip does not download PyTorch's default Linux CUDA stack.
CPU_TORCH = "torch==2.9.1+cpu"
CPU_TORCH_INSTALL = (
    f"python -m pip install --no-cache-dir {CPU_TORCH} "
    "--index-url https://download.pytorch.org/whl/cpu"
)

MODEL_DEPS = [
    CPU_TORCH, "sentence-transformers==5.1.2", "transformers==4.57.6",
    "numpy==2.2.6", "pandas==2.2.3", "pyarrow==20.0.0", "safetensors==0.6.2",
]
INFERENCE_DEPS = ["chalkcompute==2.13.1", "chalkpy[runtime]==2.160.0", *MODEL_DEPS]
TRAINING_DEPS = ["chalkcompute==2.13.1", "chalkpy[runtime,training]==2.160.0", *MODEL_DEPS]
