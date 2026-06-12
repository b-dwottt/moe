"""Metrics used by the synthetic drift and MoE experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return (predictions == targets).float().mean().item()


def moving_average(values: Iterable[float], window: int = 25) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return array
    window = max(1, min(window, array.size))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(array, kernel, mode="valid")


def normalized_entropy(weights: torch.Tensor, eps: float = 1e-8) -> float:
    """Return mean routing entropy normalized to [0, 1]."""

    if weights.ndim != 2 or weights.shape[-1] <= 1:
        return 0.0
    safe_weights = weights.clamp_min(eps)
    entropy = -(safe_weights * safe_weights.log()).sum(dim=-1)
    normalizer = torch.log(torch.tensor(float(weights.shape[-1]), device=weights.device))
    return (entropy / normalizer.clamp_min(eps)).mean().item()


def adaptation_speed(
    metrics: pd.DataFrame,
    *,
    swap_interval: int,
    threshold: float = 0.85,
    window: int = 25,
) -> float:
    """Mean steps needed after each drift boundary to recover above a target accuracy."""

    if metrics.empty or "accuracy" not in metrics:
        return float("nan")

    max_step = int(metrics["step"].max())
    recovery_steps: list[float] = []
    for boundary in range(swap_interval, max_step + 1, swap_interval):
        segment = metrics[(metrics["step"] >= boundary) & (metrics["step"] < boundary + swap_interval)].copy()
        if segment.empty:
            continue
        smoothed = moving_average(segment["accuracy"], window=min(window, len(segment)))
        if smoothed.size == 0:
            continue
        recovered = np.flatnonzero(smoothed >= threshold)
        if recovered.size:
            recovery_steps.append(float(recovered[0]))
        else:
            recovery_steps.append(float(len(segment)))

    if not recovery_steps:
        return float("nan")
    return float(np.mean(recovery_steps))


def final_window_mean(metrics: pd.DataFrame, column: str, window: int = 100) -> float:
    if metrics.empty or column not in metrics:
        return float("nan")
    return float(metrics[column].tail(min(window, len(metrics))).mean())


def regime_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate accuracy/loss/routing diagnostics by model, seed, and regime."""

    group_cols = [col for col in ("model", "seed", "regime") if col in metrics.columns]
    if not group_cols:
        return pd.DataFrame()
    value_cols = [col for col in ("accuracy", "loss", "expert_entropy") if col in metrics.columns]
    return metrics.groupby(group_cols, as_index=False)[value_cols].mean()
