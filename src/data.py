"""Synthetic data generation utilities for the Week 1 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import torch


@dataclass(frozen=True)
class SyntheticStreamConfig:
    """Configuration for the drifting synthetic classification stream."""

    input_dim: int = 10
    batch_size: int = 64
    total_steps: int = 2000
    swap_interval: int = 500
    noise_std: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class SyntheticRule:
    """Linear rule used to generate labels for a given regime."""

    weights: torch.Tensor
    bias: torch.Tensor


def _make_rule(input_dim: int, generator: torch.Generator) -> SyntheticRule:
    weights = torch.randn(input_dim, generator=generator)
    weights = weights / weights.norm().clamp_min(1e-6)
    bias = torch.randn((), generator=generator) * 0.25
    return SyntheticRule(weights=weights, bias=bias)


def _make_rules(input_dim: int, num_rules: int, seed: int) -> list[SyntheticRule]:
    generator = torch.Generator().manual_seed(seed)
    return [_make_rule(input_dim, generator) for _ in range(num_rules)]


def sample_batch(
    *,
    rule: SyntheticRule,
    batch_size: int,
    input_dim: int,
    noise_std: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch from the active synthetic rule."""

    features = torch.randn(batch_size, input_dim, generator=generator)
    if noise_std:
        noise = torch.randn(features.shape, generator=generator, dtype=features.dtype)
        features = features + noise * noise_std
    logits = features @ rule.weights + rule.bias
    labels = (logits > 0).long()
    return features, labels


def generate_stream(config: SyntheticStreamConfig) -> Iterator[tuple[int, int, torch.Tensor, torch.Tensor]]:
    """Yield `(step, regime_index, x, y)` batches for a drifting task."""

    num_regimes = max(2, config.total_steps // config.swap_interval + 1)
    rules = _make_rules(config.input_dim, num_regimes, config.seed)
    generator = torch.Generator().manual_seed(config.seed + 1)

    for step in range(config.total_steps):
        regime = step // config.swap_interval
        x, y = sample_batch(
            rule=rules[regime],
            batch_size=config.batch_size,
            input_dim=config.input_dim,
            noise_std=config.noise_std,
            generator=generator,
        )
        yield step, regime, x, y


def make_evaluation_set(
    *,
    config: SyntheticStreamConfig,
    regime: int,
    size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a fixed evaluation batch for a particular regime."""

    rules = _make_rules(
        config.input_dim,
        max(2, config.total_steps // config.swap_interval + 1),
        config.seed,
    )
    generator = torch.Generator().manual_seed(config.seed + 7_500 + regime)
    return sample_batch(
        rule=rules[regime],
        batch_size=size,
        input_dim=config.input_dim,
        noise_std=config.noise_std,
        generator=generator,
    )
