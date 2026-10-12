"""A small causal transformer that reads a watch history and scores every video as the next one.

The shape is SASRec (Kang & McAuley, *Self-Attentive Sequential Recommendation*): each
position attends only to earlier watches, and the hidden state at the last position is
dotted against every video's embedding to rank candidates.

What a video's embedding *is* is the interesting choice:

- ``id``: one learned vector per video, the classic setup. A video nobody watched during
  training keeps its random initial vector, so it can never be recommended.
- ``semantic``: the sum of one learned vector per semantic-ID token (3 levels x 256
  codes). A brand-new video already has tokens from its title, so it lands near the
  videos it shares a prefix with and is recommendable the moment it is uploaded.
- ``hybrid``: both. The per-video vector is used only for videos seen in training, so a
  new upload falls back to its semantic tokens. Training drops each video's own vector
  with probability ``id_dropout``, which forces the tokens to carry signal on their own.

Attention is written out by hand rather than with ``nn.TransformerEncoder``, whose fused
eval-mode fast path does not export to ONNX.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

PAD = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, blocked: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        head_dim = dim // self.heads
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, head_dim).permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores.masked_fill(blocked[:, None, :, :], float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        mixed = (weights @ v).transpose(1, 2).reshape(batch, length, dim)
        return self.out(mixed)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = CausalSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, blocked: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), blocked))
        return x + self.dropout(self.mlp(self.norm2(x)))


class NextVideoTransformer(nn.Module):
    def __init__(
        self,
        num_videos: int,
        item_repr: str,
        semantic_tokens: torch.Tensor | None,
        trained_videos: torch.Tensor | None,
        id_dropout: float,
        codebook_size: int,
        dim: int,
        layers: int,
        heads: int,
        max_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if item_repr not in ("semantic", "id", "hybrid"):
            raise ValueError(f"unknown item_repr {item_repr!r}")
        self.item_repr = item_repr
        self.max_len = max_len
        self.id_dropout = id_dropout
        if item_repr in ("semantic", "hybrid"):
            if semantic_tokens is None:
                raise ValueError(f"item_repr={item_repr!r} needs semantic_tokens")
            # Row v holds video v's tokens; row 0 is padding and is zeroed in item_table().
            self.register_buffer("semantic_tokens", semantic_tokens.long())
            self.token_embeddings = nn.ModuleList(
                nn.Embedding(codebook_size, dim) for _ in range(semantic_tokens.shape[1])
            )
        if item_repr in ("id", "hybrid"):
            self.video_embedding = nn.Embedding(num_videos + 1, dim, padding_idx=PAD)
        if item_repr == "hybrid":
            if trained_videos is None:
                raise ValueError("item_repr='hybrid' needs trained_videos")
            self.register_buffer("trained_videos", trained_videos.bool())
        self.position = nn.Embedding(max_len, dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(dim, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def item_table(self) -> torch.Tensor:
        """One embedding per video id, [num_videos + 1, dim], with the padding row zeroed."""
        table = 0
        if self.item_repr in ("semantic", "hybrid"):
            table = sum(
                embedding(self.semantic_tokens[:, level]) for level, embedding in enumerate(self.token_embeddings)
            )
        if self.item_repr == "id":
            table = self.video_embedding.weight
        elif self.item_repr == "hybrid":
            # A video's own vector counts only if training ever saw it; otherwise it is
            # untrained noise, and the semantic tokens alone place the video.
            keep_id = self.trained_videos[:, None].float()
            if self.training and self.id_dropout > 0:
                keep_id = keep_id * (torch.rand_like(keep_id) >= self.id_dropout).float()
            table = table + keep_id * self.video_embedding.weight
        mask = torch.ones(table.shape[0], 1, device=table.device)
        mask[PAD] = 0.0
        return table * mask

    def encode(self, history: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """Hidden state per position for left-padded histories [batch, length] of video ids."""
        length = history.shape[1]
        positions = torch.arange(length, device=history.device)
        x = table[history] + self.position(positions)[None]
        x = self.dropout(x)
        # A position may attend to itself and to earlier non-padding positions. Always
        # allowing the diagonal keeps padding rows from being fully masked (NaN softmax).
        future = positions[None, :] > positions[:, None]
        diagonal = positions[None, :] == positions[:, None]
        padding = (history == PAD)[:, None, :] & ~diagonal[None]
        blocked = future[None] | padding
        for block in self.blocks:
            x = block(x, blocked)
        return self.norm(x)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Scores for every video as the next watch after each history: [batch, num_videos + 1]."""
        table = self.item_table()
        hidden = self.encode(history, table)[:, -1]
        return hidden @ table.T


def score_positions(model: NextVideoTransformer, history: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Scores at every non-padding position selected by ``keep``: [kept, num_videos + 1]."""
    table = model.item_table()
    hidden = model.encode(history, table)
    return hidden[keep] @ table.T


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, excluded: torch.Tensor) -> torch.Tensor:
    """Full-softmax loss over videos, ignoring ``excluded`` ones (padding, not-yet-uploaded)."""
    logits = logits.masked_fill(excluded[None, :], float("-inf"))
    return F.cross_entropy(logits, targets)
