"""Load the checkout's stage scripts under stable, worker-importable names.

The scripts are deliberately standalone so the visible directory can be named
``train-recommender``. Chalk packages each training/hosting script alongside the
shared ``semantic_ids`` source explicitly included in its image.
"""
from importlib.util import module_from_spec, spec_from_file_location
import sys

from .project import ROOT


_SCRIPTS = {
    "prepare": "train/prepare.py",
    "train": "train/run.py",
    "host": "host/run.py",
    "materialize": "materialize/run.py",
    "train-recommender": "train-recommender/run.py",
    "serve": "serve/run.py",
}


def load_stage(stage):
    """Load only the requested stage; loading ``host`` registers Functions."""
    if stage not in _SCRIPTS:
        raise ValueError(f"Unknown semantic-ID stage: {stage}")
    name = f"semantic_ids_{stage.replace('-', '_')}"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / _SCRIPTS[stage]
    if not path.is_file():
        raise RuntimeError(f"Missing stage script {path}. Run this example from its repository checkout.")
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
