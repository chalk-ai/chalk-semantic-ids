"""Small offline numerical tests for generation, metrics and exact recovery."""

import math

import pytest
import torch

from semantic_ids.model import (
    SemanticRecommender, evaluate_model, load_model, recommend, save_model, train_model,
)


@pytest.fixture(autouse=True)
def small_cpu_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def catalog():
    return {11: (0, 0, 0, 0), 12: (0, 0, 0, 1), 13: (0, 1, 1, 0),
            14: (1, 0, 1, 0), 15: (1, 1, 0, 0)}


@pytest.fixture
def model(catalog):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        return SemanticRecommender(catalog, hidden_dim=12, max_history=3)


@pytest.fixture
def examples():
    return [{"history": [11, 13], "target": 14, "event_id": str(index)} if index % 2 == 0
            else {"history": [15, 14], "target": 12, "event_id": str(index)} for index in range(64)]


def test_beam_matches_exhaustive_autoregressive_likelihood(model, catalog):
    results = recommend(model, catalog, [11], top_k=5, beam_width=50)
    assert {item["video_id"] for item in results} == set(catalog) - {11}
    assert [item["score"] for item in results] == sorted((item["score"] for item in results), reverse=True)
    with torch.inference_mode():
        for item in results:
            exact = -model.target_nll([[11]], torch.tensor([catalog[item["video_id"]]])).item()
            assert item["score"] == pytest.approx(exact, abs=1e-6)
            assert sum(step["log_probability"] for step in item["token_trace"]) == pytest.approx(exact, abs=1e-6)
            assert item["semantic_tokens"] == list(catalog[item["video_id"]])


def test_collisions_remain_distinct_and_seen_items_do_not_consume_beams(model, catalog):
    results = recommend(model, catalog, [], top_k=10)
    assert {item["video_id"] for item in results} == set(catalog)
    assert len({tuple(item["semantic_tokens"]) for item in results}) == len(catalog)
    # Filtering considers the full history, including items outside the GRU window.
    assert recommend(model, catalog, [12, 11, 13, 14, 15]) == []
    assert [item["video_id"] for item in recommend(model, catalog, [11, 13, 14, 15], top_k=1, beam_width=1)] == [12]


def test_history_order_matters_and_unknown_history_is_ignored(model):
    with torch.inference_mode():
        hidden = model.encode_histories([[11, 14], [14, 11], [9999, 11, 14], []])
    assert not torch.allclose(hidden[0], hidden[1])
    assert torch.allclose(hidden[0], hidden[2])
    assert torch.isfinite(hidden).all()


def test_safe_checkpoint_round_trip_rejects_a_changed_catalog(model, catalog, tmp_path):
    path = tmp_path / "recommender.pt"
    save_model(model, path, catalog=catalog, metadata={"tokenizer_version": 3})
    raw = torch.load(path, weights_only=True)
    assert raw["metadata"] == {"tokenizer_version": 3}
    restored = load_model(path, catalog=dict(reversed(list(catalog.items()))))
    assert recommend(restored, catalog, [11]) == recommend(model, catalog, [11])
    changed = dict(catalog)
    changed[11], changed[12] = changed[12], changed[11]
    with pytest.raises(ValueError, match="fingerprint"):
        load_model(path, catalog=changed)
    with pytest.raises(ValueError, match="fingerprint"):
        recommend(model, changed, [])


def test_training_learns_repeatable_conditioned_targets(catalog, examples):
    settings = dict(epochs=16, hidden_dim=12, seed=19, batch_size=32, learning_rate=0.02)
    model, metrics = train_model(catalog, examples, **settings)
    twin, twin_metrics = train_model(catalog, examples, **settings)
    assert metrics == twin_metrics
    assert metrics["train_token_nll"] < metrics["initial_token_nll"] * 0.3
    assert all(torch.equal(value, twin.state_dict()[key]) for key, value in model.state_dict().items())
    # This is a learnability fixture, not a claim about held-out MicroLens quality.
    ranking = evaluate_model(model, catalog, examples[:2], top_k=1)
    assert ranking["recall_at_k"] == ranking["ndcg_at_k"] == 1.0


def test_metrics_match_manual_ranks_and_keep_seen_targets_in_denominator(model, catalog):
    ranking = recommend(model, catalog, [11], top_k=5)
    examples = [{"history": [11], "target": ranking[0]["video_id"]},
                {"history": [11], "target": ranking[1]["video_id"]}, {"history": [11], "target": 11}]
    metrics = evaluate_model(model, catalog, examples, top_k=2)
    assert metrics["examples"] == 3
    assert metrics["recall_at_k"] == pytest.approx(2 / 3)
    assert metrics["ndcg_at_k"] == pytest.approx((1 + 1 / math.log2(3)) / 3)
    assert math.isfinite(metrics["token_nll"])


def test_interrupted_resume_matches_uninterrupted_optimizer_rng_and_weights(catalog, examples, tmp_path):
    settings = dict(epochs=4, hidden_dim=12, seed=19, batch_size=16, learning_rate=0.02)
    full, full_metrics = train_model(catalog, examples, **settings)
    path = tmp_path / "recovery.pt"
    class Interrupted(RuntimeError):
        pass
    def save_and_interrupt(model, state, metrics):
        assert metrics["loss"] == state["epoch_losses"][-1]
        if state["epoch"] == 2:
            torch.save(state, path)
            raise Interrupted()
        torch.rand(3)  # A callback must not perturb subsequent training randomness.
    with pytest.raises(Interrupted):
        train_model(catalog, examples, on_epoch=save_and_interrupt, **settings)
    saved = torch.load(path, weights_only=True)
    assert saved["epoch"] == 2 and saved["optimizer_state"]["state"]
    torch.manual_seed(987)
    resumed_epochs = []
    resumed, resumed_metrics = train_model(catalog, examples, resume_state=saved,
        on_epoch=lambda model, state, metrics: resumed_epochs.append(state["epoch"]), **settings)
    assert resumed_epochs == [3, 4]
    assert resumed_metrics == full_metrics
    assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in full.state_dict().items())
    changed = [dict(examples[0], target=12), *examples[1:]]
    with pytest.raises(ValueError, match="configuration and data"):
        train_model(catalog, changed, resume_state=saved, **settings)


def test_invalid_catalog_and_unknown_targets_fail(catalog):
    for invalid in ({1: (0, 0, 0)}, {1: (0, 0, 0, 0), 2: (0, 0, 0, 0)}, {1: (0, 0, 1.5, 0)}):
        with pytest.raises(ValueError):
            SemanticRecommender(invalid)
    with pytest.raises(ValueError, match="at least one example"):
        train_model(catalog, [])
    with pytest.raises(ValueError, match="absent"):
        train_model(catalog, [{"history": [11], "target": 9999}])
