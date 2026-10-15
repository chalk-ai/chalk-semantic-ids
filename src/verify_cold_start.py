#!/usr/bin/env python3
"""Compatibility entrypoint for the tokenizer consistency check in host/verify.py."""
from pathlib import Path
import runpy


def main():
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "host" / "verify.py"), run_name="__main__")


if __name__ == "__main__":
    main()
