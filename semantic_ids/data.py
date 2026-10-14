"""Pure MicroLens parsing and catalog helpers; no network or model execution.

The source provides watch order, not timestamps. Every example contains only
the history preceding its target. Each user's last two targets are held out.
"""

from collections import defaultdict
import hashlib
from numbers import Integral
import random
from typing import Iterable, Mapping

from .settings import MAX_TITLE_LENGTH


def _integer(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def parse_titles(raw: bytes) -> list[dict]:
    """Read unquoted ``video_id, title`` rows, preserving commas and empty titles."""
    rows = {}
    for number, line in enumerate(raw.decode("utf-8", errors="strict").splitlines(), 1):
        if not line.strip():
            continue
        identifier, separator, title = line.partition(",")
        if not separator or not identifier.strip().isdigit():
            raise ValueError(f"Malformed title row {number}")
        video_id = int(identifier)
        if video_id in rows:
            raise ValueError(f"Duplicate video ID {video_id}")
        if len(title.strip()) > MAX_TITLE_LENGTH:
            raise ValueError(f"Title row {number} exceeds the {MAX_TITLE_LENGTH:,}-character serving limit")
        rows[video_id] = {"id": video_id, "title": title.strip()}
    if not rows:
        raise ValueError("Title source is empty")
    return [rows[video_id] for video_id in sorted(rows)]


def parse_watch_examples(
    raw: bytes, catalog_ids: set[int], *, max_users: int = 5000, seed: int = 0
) -> list[dict]:
    """Create reproducible per-user train, validation and test next-video examples.

    Users need at least four watches: one history item, one training target,
    one validation target and one test target. Validation's observed target can
    appear in the later test history, but neither held-out target position is
    included in training. Video identities may appear across splits; the split
    measures next-watch prediction rather than unseen-item generalization.
    """
    if isinstance(max_users, bool) or not isinstance(max_users, Integral) or max_users < 1:
        raise ValueError("max_users must be a positive integer")
    seed = _integer(seed, "seed")
    catalog_ids = {_integer(video, "Catalog video ID") for video in catalog_ids}
    if not catalog_ids:
        raise ValueError("Catalog must contain at least one video")
    sequences = {}
    for number, line in enumerate(raw.decode("utf-8", errors="strict").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0].strip():
            raise ValueError(f"Malformed watch row {number}")
        user_id = fields[0].strip()
        if user_id in sequences:
            raise ValueError(f"Duplicate watch user {user_id}")
        tokens = fields[1].split()
        if not tokens or any(not token.isdigit() for token in tokens):
            raise ValueError(f"Watch row {number} requires nonnegative integer video IDs")
        sequence = [int(token) for token in tokens]
        if any(video not in catalog_ids for video in sequence):
            raise ValueError(f"Watch user {user_id} references an unknown catalog video")
        sequences[user_id] = sequence
    eligible = sorted(user for user, sequence in sequences.items() if len(sequence) >= 4)
    if not eligible:
        raise ValueError("At least one user must have four ordered watches")
    selected = sorted(random.Random(seed).sample(eligible, min(int(max_users), len(eligible))))
    source_id = hashlib.sha256(raw).hexdigest()[:16]
    rows = []
    for user_id in selected:
        sequence = sequences[user_id]
        for index in range(1, len(sequence)):
            split = "test" if index == len(sequence) - 1 else "validation" if index == len(sequence) - 2 else "train"
            rows.append({
                "id": f"microlens:{source_id}:{user_id}:{index}",
                "user_id": user_id,
                "source_index": index,
                "history_ids": sequence[:index],
                "target_id": sequence[index],
                "split": split,
            })
    return rows


def build_catalog(rows: Iterable[Mapping]) -> list[dict]:
    """Add a stable collision suffix to three Chalk-computed semantic codes.

    Assignment is stable for a fixed complete catalog, independent of row order.
    Changing catalog membership may change suffixes; freeze the catalog alongside
    each trained recommender and verify its fingerprint when loading the model.
    """
    checked = {}
    for row in rows:
        video_id = _integer(row["id"], "Video ID")
        if video_id in checked:
            raise ValueError(f"Duplicate video ID {video_id}")
        if not isinstance(row["title"], str):
            raise ValueError("Catalog titles must be strings")
        codes = row["codes"]
        if not isinstance(codes, (list, tuple)) or len(codes) != 3:
            raise ValueError("Each catalog video requires exactly three semantic codes")
        checked[video_id] = (row["title"], tuple(_integer(token, "Semantic code") for token in codes))
    if not checked:
        raise ValueError("Catalog must contain at least one video")
    counts = defaultdict(int)
    catalog = []
    for video_id, (title, codes) in sorted(checked.items()):
        catalog.append({"video_id": video_id, "title": title, "semantic_tokens": [*codes, counts[codes]]})
        counts[codes] += 1
    return catalog
