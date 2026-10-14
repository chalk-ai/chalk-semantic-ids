#!/usr/bin/env python3
"""Compatibility launcher for ``python -m semantic_ids.cli deploy``.

Deployment uses the explicitly saved FTQA target and registered tokenizer from
the canonical how-to. Run ``--help`` without deploying. Historical target and
smoke-test flags are not supported; this script has no default environment.
"""

from pathlib import Path
import sys


def main():
    print("Use: python -m semantic_ids.cli deploy (saved FTQA target)", file=sys.stderr)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semantic_ids.cli import main as cli_main

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], "deploy", *original_argv[1:]]
        return cli_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
