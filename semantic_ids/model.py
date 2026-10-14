"""CPU model implementation used by Chalk training jobs and inference resolvers.

A GRU reads watch history and an autoregressive decoder generates four tokens:
three RQ-VAE codes and a collision suffix that identifies one catalog video.
A catalog trie constrains decoding, and previously seen items are filtered.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

Catalog = Mapping[int, tuple[int, ...]]
CHECKPOINT_VERSION = 2


def _catalog_rows(catalog: Catalog) -> dict[int, tuple[int, ...]]:
    for video, codes in catalog.items():
        if isinstance(video, bool) or not isinstance(video, Integral) or video < 0:
            raise ValueError("Catalog video IDs must be nonnegative integers")
        if any(isinstance(token, bool) or not isinstance(token, Integral) for token in codes):
            raise ValueError("Semantic-ID tokens must be integers")
    rows = {int(video): tuple(int(token) for token in codes) for video, codes in catalog.items()}
    if not rows:
        raise ValueError("The catalog must contain at least one video")
    lengths = {len(codes) for codes in rows.values()}
    if lengths != {4} or any(code < 0 for codes in rows.values() for code in codes):
        raise ValueError("Each video requires four nonnegative semantic-ID tokens")
    if len(set(rows.values())) != len(rows):
        raise ValueError("Full item token sequences must be unique; semantic IDs require the fourth dedup token")
    return dict(sorted(rows.items()))


def catalog_fingerprint(catalog: Catalog) -> str:
    payload = json.dumps(list(_catalog_rows(catalog).items()), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SemanticRecommender(nn.Module):
    def __init__(self, catalog: Catalog, *, hidden_dim: int = 64, max_history: int = 20):
        super().__init__()
        if hidden_dim < 1 or max_history < 1:
            raise ValueError("hidden_dim and max_history must be positive")
        self.token_count = 4
        self.catalog = _catalog_rows(catalog)
        self.catalog_fingerprint = catalog_fingerprint(self.catalog)
        self.hidden_dim = hidden_dim
        self.max_history = max_history
        self.vocab_sizes = [max(codes[level] for codes in self.catalog.values()) + 1 for level in range(self.token_count)]
        self.token_embeddings = nn.ModuleList(
            nn.Embedding(size + 1, hidden_dim, padding_idx=size) for size in self.vocab_sizes
        )
        self.history_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRUCell(hidden_dim, hidden_dim)
        self.token_heads = nn.ModuleList(nn.Linear(hidden_dim, size) for size in self.vocab_sizes)
        self.bos = nn.Parameter(torch.zeros(hidden_dim))
        self.empty_history = nn.Parameter(torch.zeros(hidden_dim))
        self.metadata: dict = {}
        # Indexes are derived from the verified catalog rather than pickled.
        self.prefix_children: dict[tuple[int, ...], tuple[int, ...]] = {}
        child_sets: dict[tuple[int, ...], set[int]] = {}
        self.prefix_videos: dict[tuple[int, ...], set[int]] = {}
        self.id_to_video = {codes: video for video, codes in self.catalog.items()}
        for video, codes in self.catalog.items():
            for level, code in enumerate(codes):
                child_sets.setdefault(codes[:level], set()).add(code)
                self.prefix_videos.setdefault(codes[: level + 1], set()).add(video)
        self.prefix_children = {prefix: tuple(sorted(children)) for prefix, children in child_sets.items()}

    def encode_histories(self, histories: Sequence[Sequence[int]]) -> torch.Tensor:
        """Unknown videos are ignored; an empty history uses a learned prior."""
        known = [[self.catalog[int(v)] for v in history if int(v) in self.catalog][-self.max_history :]
                 for history in histories]
        lengths = torch.tensor([len(row) for row in known], dtype=torch.long)
        width = max(1, int(lengths.max().item()))
        tokens = torch.tensor(self.vocab_sizes, dtype=torch.long).repeat(len(known), width, 1)
        for row, codes in enumerate(known):
            if codes:
                tokens[row, : len(codes)] = torch.tensor(codes, dtype=torch.long)
        vectors = sum(embedding(tokens[:, :, level]) for level, embedding in enumerate(self.token_embeddings)) / math.sqrt(self.token_count)
        packed = pack_padded_sequence(vectors, lengths.clamp_min(1), batch_first=True, enforce_sorted=False)
        _, hidden = self.history_encoder(packed)
        return torch.where(lengths[:, None] > 0, hidden[0], self.empty_history[None, :])

    def decode_step(self, hidden: torch.Tensor, level: int, previous: torch.Tensor | None = None):
        inputs = self.bos.expand(hidden.shape[0], -1) if level == 0 else self.token_embeddings[level - 1](previous)
        next_hidden = self.decoder(inputs, hidden)
        return self.token_heads[level](next_hidden), next_hidden

    def target_nll(self, histories: Sequence[Sequence[int]], targets: torch.Tensor) -> torch.Tensor:
        """Per-example summed negative log likelihood using teacher forcing."""
        hidden = self.encode_histories(histories)
        losses = torch.zeros(len(histories))
        for level in range(self.token_count):
            logits, hidden = self.decode_step(hidden, level, None if level == 0 else targets[:, level - 1])
            losses = losses + F.cross_entropy(logits, targets[:, level], reduction="none")
        return losses


def _verify_catalog(model: SemanticRecommender, catalog: Catalog) -> None:
    if catalog_fingerprint(catalog) != model.catalog_fingerprint:
        raise ValueError("Catalog fingerprint differs from the model checkpoint; retrain for the new token mapping")


def _prepare_examples(model: SemanticRecommender, examples: Sequence[dict]):
    histories = [list(example["history"]) for example in examples]
    try:
        targets = torch.tensor([model.catalog[int(example["target"])] for example in examples], dtype=torch.long)
    except KeyError as exc:
        raise ValueError(f"Example target {exc.args[0]} is absent from the catalog") from exc
    return histories, targets


def _mean_nll(model: SemanticRecommender, examples: Sequence[dict], batch_size: int = 256) -> float:
    if not examples:
        return 0.0
    total = 0.0
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(examples), batch_size):
            histories, targets = _prepare_examples(model, examples[offset : offset + batch_size])
            total += float(model.target_nll(histories, targets).sum().item())
    return total / (model.token_count * len(examples))


def train_model(
    catalog: Catalog,
    examples: list[dict],
    *,
    initial_checkpoint: Path | None = None,
    epochs: int = 3,
    hidden_dim: int = 64,
    seed: int = 0,
    batch_size: int = 256,
    learning_rate: float = 0.003,
    max_history: int = 20,
    resume_state: dict | None = None,
    on_epoch=None,
) -> tuple[SemanticRecommender, dict]:
    """Train on CPU; callers own the train/validation/test split.

    ``on_epoch(model, state, metrics)`` receives an optimizer/RNG checkpoint after
    each completed epoch. Serialize that state synchronously (or deepcopy it)
    before returning: its parameter tensors share storage with the live model.
    Pass the saved state back as ``resume_state`` with identical data/config.
    """
    if not examples:
        raise ValueError("Training requires at least one example")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    # fork_rng makes initialization repeatable without changing the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = load_model(initial_checkpoint, catalog=catalog) if initial_checkpoint else SemanticRecommender(
            catalog, hidden_dim=hidden_dim, max_history=max_history
        )
    histories, targets = _prepare_examples(model, examples)
    signature = {"catalog_fingerprint": model.catalog_fingerprint,
                 "epochs": epochs, "hidden_dim": model.hidden_dim, "max_history": model.max_history,
                 "seed": seed, "batch_size": batch_size, "learning_rate": learning_rate,
                 "examples_sha256": hashlib.sha256(json.dumps(examples, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = torch.Generator().manual_seed(seed)
    epoch_losses, start_epoch = [], 0
    if resume_state is not None:
        if resume_state.get("signature") != signature:
            raise ValueError("Recovery checkpoint does not match recommender configuration and data")
        start_epoch = resume_state.get("epoch")
        epoch_losses = resume_state.get("epoch_losses")
        if (type(start_epoch) is not int or not 0 <= start_epoch <= epochs
                or not isinstance(epoch_losses, list) or len(epoch_losses) != start_epoch):
            raise ValueError("Recovery checkpoint has invalid epoch progress")
        epoch_losses = list(epoch_losses)
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        rng.set_state(resume_state["shuffle_rng_state"])
        torch.set_rng_state(resume_state["torch_rng_state"])
        initial_nll = resume_state["initial_token_nll"]
    else:
        initial_nll = _mean_nll(model, examples, batch_size)
    for epoch in range(start_epoch, epochs):
        model.train()
        order = torch.randperm(len(examples), generator=rng)
        running_loss = 0.0
        for indices in order.split(batch_size):
            batch_histories = [histories[index] for index in indices.tolist()]
            losses = model.target_nll(batch_histories, targets[indices])
            loss = losses.mean() / model.token_count
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(losses.detach().sum().item())
        epoch_losses.append(running_loss / (model.token_count * len(examples)))
        if on_epoch is not None:
            state = {"signature": signature, "epoch": epoch + 1,
                     "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                     "shuffle_rng_state": rng.get_state(), "torch_rng_state": torch.get_rng_state(),
                     "initial_token_nll": initial_nll, "epoch_losses": list(epoch_losses)}
            # Serialization/telemetry must not perturb the next epoch's RNG.
            try:
                on_epoch(model, state, {"loss": epoch_losses[-1], "learning_rate": learning_rate})
            finally:
                torch.set_rng_state(state["torch_rng_state"])
                rng.set_state(state["shuffle_rng_state"])
    final_nll = _mean_nll(model, examples, batch_size)
    return model, {
        "token_count": model.token_count,
        "initial_token_nll": initial_nll,
        "train_token_nll": final_nll,
        "loss_by_epoch": epoch_losses,
        "epochs": epochs,
        "training_examples": len(examples),
    }


def save_model(model: SemanticRecommender, path: Path, *, catalog: Catalog, metadata: dict) -> None:
    _verify_catalog(model, catalog)
    # Enforce primitive metadata so torch.load(weights_only=True) remains sufficient.
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": CHECKPOINT_VERSION,
        "token_count": model.token_count,
        "catalog_fingerprint": model.catalog_fingerprint,
        "hidden_dim": model.hidden_dim,
        "max_history": model.max_history,
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }, path)
    model.metadata = metadata


def load_model(path: Path, *, catalog: Catalog) -> SemanticRecommender:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") not in {1, CHECKPOINT_VERSION}:
        raise ValueError("Unsupported semantic recommender checkpoint version")
    if checkpoint["catalog_fingerprint"] != catalog_fingerprint(catalog):
        raise ValueError("Catalog fingerprint differs from the model checkpoint; retrain for the new token mapping")
    if checkpoint.get("representation", "semantic_id") != "semantic_id":
        raise ValueError("Checkpoint is not a semantic-ID recommender")
    model = SemanticRecommender(catalog, hidden_dim=checkpoint["hidden_dim"], max_history=checkpoint["max_history"])
    if checkpoint.get("token_count", model.token_count) != model.token_count:
        raise ValueError("Checkpoint requires four semantic tokens")
    model.load_state_dict(checkpoint["state_dict"])
    model.metadata = checkpoint.get("metadata", {})
    model.eval()
    return model


def _recommend_batch(
    model: SemanticRecommender, histories: Sequence[Sequence[int]], *, top_k: int, beam_width: int
) -> list[list[dict]]:
    if top_k < 1 or beam_width < 1:
        raise ValueError("top_k and beam_width must be positive")
    if not histories:
        return []
    # Keep enough beams for top_k; candidate branches containing only seen items
    # are excluded before pruning, so seen items do not consume the beam budget.
    width = max(top_k, beam_width)
    seen = [{int(video) for video in history} for history in histories]
    model.eval()
    with torch.inference_mode():
        encoded = model.encode_histories(histories)
        beams = [[((), 0.0, encoded[row], ())] for row in range(len(histories))]
        for level in range(model.token_count):
            flat = [(row, beam) for row, group in enumerate(beams) for beam in group]
            if not flat:
                break
            states = torch.stack([beam[2] for _, beam in flat])
            previous = None if level == 0 else torch.tensor([beam[0][-1] for _, beam in flat], dtype=torch.long)
            logits, next_states = model.decode_step(states, level, previous)
            probabilities = F.log_softmax(logits, dim=-1).tolist()
            candidates: list[list[tuple]] = [[] for _ in histories]
            for index, (row, (prefix, score, _, trace)) in enumerate(flat):
                for token in model.prefix_children.get(prefix, ()):
                    child = prefix + (token,)
                    if model.prefix_videos[child].issubset(seen[row]):
                        continue
                    token_log_probability = probabilities[index][token]
                    candidates[row].append((child, score + token_log_probability, next_states[index],
                                            trace + (token_log_probability,)))
            # Prefix provides a deterministic tie break without comparing tensors.
            beams = [heapq.nsmallest(width, group, key=lambda beam: (-beam[1], beam[0])) for group in candidates]
        results = []
        for group in beams:
            result = []
            for codes, score, _, trace in group[:top_k]:
                if len(codes) != model.token_count:
                    continue
                result.append({
                    "video_id": model.id_to_video[codes],
                    "semantic_tokens": list(codes),
                    "score": score,
                    "token_trace": [
                        {"position": level, "token": token, "log_probability": trace[level]}
                        for level, token in enumerate(codes)
                    ],
                })
            results.append(result)
        return results


def recommend(
    model: SemanticRecommender,
    catalog: Catalog,
    history: list[int],
    *,
    top_k: int = 10,
    beam_width: int = 32,
) -> list[dict]:
    _verify_catalog(model, catalog)
    return _recommend_batch(model, [history], top_k=top_k, beam_width=beam_width)[0]


def evaluate_model(
    model: SemanticRecommender,
    catalog: Catalog,
    examples: list[dict],
    *,
    top_k: int = 10,
    beam_width: int = 32,
) -> dict:
    """Evaluate supplied examples only; callers own leakage-free splitting.

    With one target per example, Recall@k is the fraction whose target is in the
    top k, and NDCG@k discounts a hit by log2(rank + 1). Previously seen targets
    are still in the denominator and cannot be recommended.
    """
    _verify_catalog(model, catalog)
    if top_k < 1 or beam_width < 1:
        raise ValueError("top_k and beam_width must be positive")
    nll = _mean_nll(model, examples)
    hits = 0
    discounted_hits = 0.0
    for offset in range(0, len(examples), 64):
        batch = examples[offset : offset + 64]
        results = _recommend_batch(model, [example["history"] for example in batch], top_k=top_k, beam_width=beam_width)
        for example, recommendations in zip(batch, results):
            for rank, item in enumerate(recommendations, start=1):
                if item["video_id"] == int(example["target"]):
                    hits += 1
                    discounted_hits += 1.0 / math.log2(rank + 1)
                    break
    return {
        "recall_at_k": hits / len(examples) if examples else 0.0,
        "ndcg_at_k": discounted_hits / len(examples) if examples else 0.0,
        "token_nll": nll,
        "token_count": model.token_count,
        "examples": len(examples),
        "top_k": top_k,
    }
