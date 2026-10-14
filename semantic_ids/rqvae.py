"""Residual-quantized VAE that turns a content vector into semantic ID tokens.

The shape follows TIGER (Rajput et al., "Recommender Systems with Generative
Retrieval"): an MLP encoder into a small latent, then L levels of residual vector
quantization, then an MLP decoder. Level l quantizes what levels 0..l-1 could not
represent, so the first token is the coarsest bucket and later tokens refine it —
which is what makes a shared prefix mean "semantically similar".
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    """One codebook level, with straight-through gradients and dead-code revival."""

    def __init__(self, codebook_size: int, latent_dim: int, commitment_weight: float) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self._initialized = False
        # Usage counts drive dead-code revival; kept as a buffer so they survive
        # checkpointing and move with the module's device.
        self.register_buffer("usage", torch.zeros(codebook_size))

    @torch.no_grad()
    def initialize_from(self, residual: torch.Tensor) -> None:
        """Seed the codebook with k-means++-ish samples drawn from real residuals.

        Random init leaves most codes far from the data, and those codes never win a
        nearest-neighbour lookup, so they stay dead for the whole run. Seeding from the
        residuals this level actually sees is what gets usage off the floor.
        """
        if self._initialized:
            return
        sample_count = min(self.codebook_size, residual.shape[0])
        picks = torch.randperm(residual.shape[0], device=residual.device)[:sample_count]
        self.embedding.weight[:sample_count] = residual[picks]
        if sample_count < self.codebook_size:
            # Fewer residuals than codes: jitter copies rather than leaving them at init.
            extra = self.codebook_size - sample_count
            repeat = residual[torch.randint(0, residual.shape[0], (extra,), device=residual.device)]
            self.embedding.weight[sample_count:] = repeat + 0.01 * torch.randn_like(repeat)
        self._initialized = True

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = torch.cdist(residual, self.embedding.weight)
        codes = distances.argmin(dim=1)
        quantized = self.embedding(codes)

        codebook_loss = F.mse_loss(quantized, residual.detach())
        commitment_loss = F.mse_loss(residual, quantized.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through: the decoder sees the quantized vector, the encoder gets the
        # gradient as if quantization were the identity.
        quantized = residual + (quantized - residual).detach()

        if self.training:
            with torch.no_grad():
                self.usage.mul_(0.99).index_add_(0, codes, torch.ones_like(codes, dtype=self.usage.dtype))
        return quantized, codes, loss

    @torch.no_grad()
    def revive_dead_codes(self, residual: torch.Tensor, threshold: float = 1e-3) -> int:
        """Move unused codes onto real residuals. Returns how many were moved."""
        dead = (self.usage < threshold).nonzero(as_tuple=True)[0]
        if dead.numel() == 0:
            return 0
        picks = torch.randint(0, residual.shape[0], (dead.numel(),), device=residual.device)
        self.embedding.weight[dead] = residual[picks]
        self.usage[dead] = 1.0
        return int(dead.numel())


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        codebook_size: int,
        num_levels: int,
        hidden_dims: tuple[int, ...],
        commitment_weight: float,
    ) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            encoder_layers += [nn.Linear(previous, hidden), nn.ReLU()]
            previous = hidden
        encoder_layers.append(nn.Linear(previous, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        previous = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers += [nn.Linear(previous, hidden), nn.ReLU()]
            previous = hidden
        decoder_layers.append(nn.Linear(previous, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.quantizers = nn.ModuleList(
            VectorQuantizer(codebook_size, latent_dim, commitment_weight) for _ in range(num_levels)
        )

    def quantize(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = latent
        quantized_total = torch.zeros_like(latent)
        codes: list[torch.Tensor] = []
        quantization_loss = latent.new_zeros(())
        for quantizer in self.quantizers:
            quantizer.initialize_from(residual.detach())
            quantized, level_codes, level_loss = quantizer(residual)
            quantized_total = quantized_total + quantized
            # The next level only has to explain what this one missed.
            residual = residual - quantized.detach()
            codes.append(level_codes)
            quantization_loss = quantization_loss + level_loss
        return quantized_total, torch.stack(codes, dim=1), quantization_loss

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(batch)
        quantized, codes, quantization_loss = self.quantize(latent)
        reconstruction = self.decoder(quantized)
        reconstruction_loss = F.mse_loss(reconstruction, batch)
        return reconstruction, codes, reconstruction_loss, quantization_loss

    @torch.no_grad()
    def encode_ids(self, batch: torch.Tensor) -> torch.Tensor:
        """Semantic ID tokens for a batch of content vectors."""
        self.eval()
        _, codes, _ = self.quantize(self.encoder(batch))
        return codes
