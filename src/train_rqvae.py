#!/usr/bin/env python3
"""Compatibility launcher for ``python -m semantic_ids.cli train-tokenizer``.

Tokenizer training now runs as a native Chalk training job using this
repository's saved dataset revision and explicit FTQA target. Run ``--help``
for the new options; the historical local-training flags are not supported.
"""

from pathlib import Path
import sys


def main():
    print("Use: python -m semantic_ids.cli train-tokenizer (native Chalk training)", file=sys.stderr)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semantic_ids.cli import main as cli_main

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], "train-tokenizer", *original_argv[1:]]
        return cli_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
