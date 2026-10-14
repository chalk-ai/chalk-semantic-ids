"""Pure data tests; no Chalk client, credentials, downloads or model execution."""

import pytest

from semantic_ids.data import build_catalog, parse_titles, parse_watch_examples, select_training_data
from semantic_ids.settings import MAX_TITLE_LENGTH


def test_titles_preserve_commas_empty_titles_and_sort_ids():
    assert parse_titles(b"3, title, with commas\n1,\n2, other\n") == [
        {"id": 1, "title": ""}, {"id": 2, "title": "other"}, {"id": 3, "title": "title, with commas"},
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        parse_titles(b"1, one\n1, again\n")
    with pytest.raises(ValueError, match="Malformed"):
        parse_titles(b"not-a-video\n")
    with pytest.raises(ValueError, match="empty"):
        parse_titles(b"\n")


def test_watch_splits_are_complete_prefixes_and_hold_out_last_two_targets():
    sequences = {"a": [1, 2, 3, 4, 5, 6], "b": [7, 8, 9, 10]}
    raw = b"a\t1 2 3 4 5 6\nb\t7 8 9 10\nshort\t1 2 3\n"
    rows = parse_watch_examples(raw, set(range(1, 11)))
    assert len({row["id"] for row in rows}) == len(rows)
    assert {row["user_id"] for row in rows} == set(sequences)
    for user, sequence in sequences.items():
        examples = [row for row in rows if row["user_id"] == user]
        assert [row["split"] for row in examples] == ["train"] * (len(sequence) - 3) + ["validation", "test"]
        for row in examples:
            assert row["history_ids"] == sequence[:row["source_index"]]
            assert row["target_id"] == sequence[row["source_index"]]
            assert set(row) == {"id", "user_id", "source_index", "history_ids", "target_id", "split"}
        training = [row for row in examples if row["split"] == "train"]
        assert all(row["source_index"] < examples[-2]["source_index"] for row in training)
        assert not set(sequence[-2:]) & {video for row in training for video in row["history_ids"] + [row["target_id"]]}
        # Test history may contain the previously observed validation watch.
        assert examples[-1]["history_ids"][-1] == examples[-2]["target_id"]


def test_title_length_matches_serving_limit_before_upload():
    title = "x" * MAX_TITLE_LENGTH
    assert parse_titles(f"1,{title}\n".encode())[0]["title"] == title
    with pytest.raises(ValueError, match="row 2 exceeds"):
        parse_titles(f"1,ok\n2,{title}x\n".encode())


def test_watch_sampling_is_reproducible_without_changing_split_membership():
    raw = b"".join(f"user-{index}\t1 2 3 4 5\n".encode() for index in range(12))
    first = parse_watch_examples(raw, {1, 2, 3, 4, 5}, max_users=3, seed=7)
    assert first == parse_watch_examples(raw, {1, 2, 3, 4, 5}, max_users=3, seed=7)
    assert len({row["user_id"] for row in first}) == 3
    all_rows = parse_watch_examples(raw, {1, 2, 3, 4, 5}, max_users=20, seed=7)
    assert {row["id"] for row in first} <= {row["id"] for row in all_rows}


def test_small_catalog_keeps_every_selected_history_and_held_out_target():
    titles = b"".join(f"{video},title {video}\n".encode() for video in range(1, 12))
    watches = b"a\t1 2 3 4 5\nb\t6 7 8 9 10\n"
    catalog, examples = select_training_data(titles, watches, max_users=1, seed=0)
    expected = parse_watch_examples(watches, set(range(1, 12)), max_users=1, seed=0)
    used = {video for row in expected for video in row["history_ids"] + [row["target_id"]]}
    assert examples == expected
    assert {row["id"] for row in catalog} == used
    assert len(catalog) == 5
    assert [row["split"] for row in examples] == ["train", "train", "validation", "test"]
    assert (catalog, examples) == select_training_data(titles, watches, max_users=1, seed=0)
    full, same_examples = select_training_data(titles, watches, max_users=1, seed=0, full_catalog=True)
    assert full == parse_titles(titles)
    assert same_examples == examples


def test_default_selection_limits_users_and_their_catalog():
    titles = b"".join(f"{video},title {video}\n".encode() for video in range(480))
    watches = b"".join(f"user-{user}\t{' '.join(str(user * 4 + offset) for offset in range(4))}\n".encode()
                       for user in range(120))
    catalog, examples = select_training_data(titles, watches)
    assert len({row["user_id"] for row in examples}) == 100
    assert len(catalog) == 400
    assert len(examples) == 300


def test_small_selection_validates_unknown_items_even_in_unselected_users():
    # Seed zero selects b. Invalid a must still fail before the sample is taken.
    with pytest.raises(ValueError, match="unknown catalog video"):
        select_training_data(b"1,one\n2,two\n3,three\n4,four\n",
                             b"a\t1 2 3 999\nb\t1 2 3 4\n", max_users=1, seed=0)


@pytest.mark.parametrize("raw,error", [
    (b"a\t1 2 3 99\n", "unknown"),
    (b"a\t1 2 3 4\na\t1 2 3 4\n", "Duplicate"),
    (b"a\t1 2 -3 4\n", "nonnegative"),
    (b"a\t1 2 3\n", "four ordered"),
    (b"a 1 2 3 4\n", "Malformed"),
])
def test_invalid_watch_sources_fail(raw, error):
    with pytest.raises(ValueError, match=error):
        parse_watch_examples(raw, {1, 2, 3, 4})


def test_catalog_suffixes_are_deterministic_unique_and_separate_from_semantic_codes():
    rows = [{"id": 20, "title": "second", "codes": [2, 3, 4]},
            {"id": 10, "title": "first", "codes": [2, 3, 4]},
            {"id": 30, "title": "", "codes": [2, 3, 5]}]
    catalog = build_catalog(rows)
    assert catalog == build_catalog(reversed(rows))
    assert [row["video_id"] for row in catalog] == [10, 20, 30]
    assert [row["semantic_tokens"] for row in catalog] == [[2, 3, 4, 0], [2, 3, 4, 1], [2, 3, 5, 0]]
    assert len({tuple(row["semantic_tokens"]) for row in catalog}) == len(rows)
    with pytest.raises(ValueError, match="Duplicate"):
        build_catalog(rows + [rows[0]])


@pytest.mark.parametrize("codes", [[1, 2], [1, 2, 3, 4], [1, -1, 2], [1, True, 2], [1, 2.5, 3]])
def test_invalid_catalog_codes_fail(codes):
    with pytest.raises(ValueError):
        build_catalog([{"id": 1, "title": "one", "codes": codes}])
