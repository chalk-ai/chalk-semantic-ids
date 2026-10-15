"""Compatibility imports for the original tokenizer experiment.

The maintained implementation lives in ``semantic_ids.rqvae``. Both the legacy
``from rqvae import RQVAE`` path and ``src.rqvae`` expose the same classes, so the
historical state-dictionary checkpoints retain their architecture and parameter
names without maintaining a second model implementation.
"""

from pathlib import Path
import sys

# Legacy scripts may place only src/ on sys.path when run by filename.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from semantic_ids.rqvae import RQVAE, VectorQuantizer  # noqa: E402

__all__ = ["RQVAE", "VectorQuantizer"]
