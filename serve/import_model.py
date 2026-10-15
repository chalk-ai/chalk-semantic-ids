"""Import compatible pretrained weights and their original catalog."""

if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from semantic_ids.cli import main

    raise SystemExit(main(sys.argv[1:], stage="import-recommender"))
