"""Model definitions for dense, MoE, and memory-augmented MoE experiments."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


RoutingMode = Literal["soft", "top1"]


class DenseMLP(nn.Module):
    """Simple dense baseline for the drifting synthetic task."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ExpertMLP(nn.Module):
    """One expert in a small synthetic-task MoE."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RouterMLP(nn.Module):
    """Router network that emits one logit per expert."""

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _routing_weights(router_logits: torch.Tensor, routing: RoutingMode, temperature: float) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    soft_weights = torch.softmax(router_logits / temperature, dim=-1)
    if routing == "soft":
        return soft_weights

    hard_weights = F.one_hot(soft_weights.argmax(dim=-1), num_classes=soft_weights.shape[-1]).to(soft_weights.dtype)
    return hard_weights + soft_weights - soft_weights.detach()


def per_expert_cross_entropy(expert_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return cross-entropy loss for each sample-expert pair."""

    batch_size, num_experts, output_dim = expert_logits.shape
    expanded_targets = targets[:, None].expand(batch_size, num_experts).reshape(-1)
    flat_logits = expert_logits.reshape(batch_size * num_experts, output_dim)
    return F.cross_entropy(flat_logits, expanded_targets, reduction="none").reshape(batch_size, num_experts)


class MixtureOfExperts(nn.Module):
    """Small MoE with optional memory bonus and load-balancing auxiliary loss."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_experts: int = 2,
        router_hidden_dim: int = 32,
        routing: RoutingMode = "soft",
        temperature: float = 1.0,
        memory_alpha: float = 0.0,
        load_balance_strength: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.routing = routing
        self.temperature = temperature
        self.memory_alpha = memory_alpha
        self.load_balance_strength = load_balance_strength
        self.experts = nn.ModuleList(
            [ExpertMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim) for _ in range(num_experts)]
        )
        self.router = RouterMLP(input_dim=input_dim, hidden_dim=router_hidden_dim, num_experts=num_experts)

    def compute_memory_bonus(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return None, None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)
        raw_router_logits = self.router(x)
        memory_bonus, context = self.compute_memory_bonus(x)
        router_logits = raw_router_logits
        if memory_bonus is not None:
            router_logits = router_logits + self.memory_alpha * memory_bonus

        router_weights = _routing_weights(router_logits, routing=self.routing, temperature=self.temperature)
        logits = torch.einsum("be,beo->bo", router_weights, expert_logits)
        selected_expert = router_weights.argmax(dim=-1)

        mean_usage = router_weights.mean(dim=0)
        target_usage = torch.full_like(mean_usage, 1.0 / self.num_experts)
        load_balance_loss = (mean_usage - target_usage).pow(2).sum()

        return logits, {
            "expert_logits": expert_logits,
            "router_logits": router_logits,
            "raw_router_logits": raw_router_logits,
            "router_weights": router_weights,
            "selected_expert": selected_expert,
            "load_balance_loss": load_balance_loss,
            "aux_loss": load_balance_loss * self.load_balance_strength,
            "memory_bonus": memory_bonus if memory_bonus is not None else torch.zeros_like(router_logits),
            "context": context if context is not None else torch.full_like(selected_expert, -1),
        }

    @torch.no_grad()
    def update_memory(self, x: torch.Tensor, targets: torch.Tensor, info: dict[str, torch.Tensor]) -> None:
        """No-op for non-memory MoE variants."""

    def memory_state(self) -> dict[str, list[float]]:
        return {}


class RandomRoutingMoE(nn.Module):
    """MoE baseline that assigns samples to experts uniformly at random."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_experts: int = 2,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [ExpertMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)
        selected_expert = torch.randint(self.num_experts, (x.shape[0],), device=x.device)
        router_weights = F.one_hot(selected_expert, num_classes=self.num_experts).to(expert_logits.dtype)
        logits = torch.einsum("be,beo->bo", router_weights, expert_logits)
        router_logits = torch.zeros(x.shape[0], self.num_experts, device=x.device, dtype=expert_logits.dtype)
        return logits, {
            "expert_logits": expert_logits,
            "router_logits": router_logits,
            "raw_router_logits": router_logits,
            "router_weights": router_weights,
            "selected_expert": selected_expert,
            "load_balance_loss": torch.zeros((), device=x.device, dtype=expert_logits.dtype),
            "aux_loss": torch.zeros((), device=x.device, dtype=expert_logits.dtype),
            "memory_bonus": torch.zeros_like(router_logits),
            "context": torch.full_like(selected_expert, -1),
        }

    @torch.no_grad()
    def update_memory(self, x: torch.Tensor, targets: torch.Tensor, info: dict[str, torch.Tensor]) -> None:
        return None

    def memory_state(self) -> dict[str, list[float]]:
        return {}


class ScoreboardMemoryMoE(MixtureOfExperts):
    """MoE whose router is biased by expert-level EMA reward memory."""

    def __init__(self, *args: object, memory_decay: float = 0.95, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.memory_decay = memory_decay
        self.register_buffer("expert_memory", torch.zeros(self.num_experts))

    def compute_memory_bonus(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.expert_memory[None, :].expand(x.shape[0], -1), None

    @torch.no_grad()
    def update_memory(self, x: torch.Tensor, targets: torch.Tensor, info: dict[str, torch.Tensor]) -> None:
        losses = per_expert_cross_entropy(info["expert_logits"].detach(), targets)
        reward = -losses.mean(dim=0)
        self.expert_memory.mul_(self.memory_decay).add_(reward * (1.0 - self.memory_decay))

    def memory_state(self) -> dict[str, list[float]]:
        return {"memory": [float(v) for v in self.expert_memory.detach().cpu()]}


class QueueMemoryMoE(MixtureOfExperts):
    """MoE whose router uses a rolling window of expert loss mean and stability."""

    def __init__(
        self,
        *args: object,
        memory_decay: float = 0.95,
        queue_size: int = 50,
        stability_weight: float = 0.25,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.memory_decay = memory_decay
        self.queue_size = queue_size
        self.stability_weight = stability_weight
        self.register_buffer("loss_queue", torch.full((self.num_experts, queue_size), float("nan")))
        self.register_buffer("queue_pointer", torch.zeros((), dtype=torch.long))

    def _queue_bonus(self) -> torch.Tensor:
        valid = torch.isfinite(self.loss_queue)
        safe_losses = torch.nan_to_num(self.loss_queue, nan=0.0)
        counts = valid.sum(dim=1).clamp_min(1)
        mean_loss = safe_losses.sum(dim=1) / counts
        centered = torch.where(valid, self.loss_queue - mean_loss[:, None], torch.zeros_like(self.loss_queue))
        std_loss = torch.sqrt(centered.pow(2).sum(dim=1) / counts)
        has_history = valid.any(dim=1)
        bonus = -(mean_loss + self.stability_weight * std_loss)
        return torch.where(has_history, bonus, torch.zeros_like(bonus))

    def compute_memory_bonus(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._queue_bonus()[None, :].expand(x.shape[0], -1), None

    @torch.no_grad()
    def update_memory(self, x: torch.Tensor, targets: torch.Tensor, info: dict[str, torch.Tensor]) -> None:
        losses = per_expert_cross_entropy(info["expert_logits"].detach(), targets).mean(dim=0)
        pointer = int(self.queue_pointer.item())
        previous = self.loss_queue[:, pointer]
        updated = losses if not torch.isfinite(previous).all() else self.memory_decay * previous + (1.0 - self.memory_decay) * losses
        self.loss_queue[:, pointer] = updated
        self.queue_pointer.fill_((pointer + 1) % self.queue_size)

    def memory_state(self) -> dict[str, list[float]]:
        return {"queue_bonus": [float(v) for v in self._queue_bonus().detach().cpu()]}


class ContextualMemoryMoE(MixtureOfExperts):
    """MoE with expert memory conditioned on a deterministic input context."""

    def __init__(
        self,
        *args: object,
        input_dim: int = 10,
        num_contexts: int = 4,
        memory_decay: float = 0.95,
        context_seed: int = 13,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, input_dim=input_dim, **kwargs)
        self.num_contexts = num_contexts
        self.memory_decay = memory_decay
        generator = torch.Generator().manual_seed(context_seed)
        prototypes = torch.randn(input_dim, num_contexts, generator=generator)
        prototypes = prototypes / prototypes.norm(dim=0, keepdim=True).clamp_min(1e-6)
        self.register_buffer("context_prototypes", prototypes)
        self.register_buffer("expert_context_memory", torch.zeros(self.num_experts, num_contexts))

    def assign_context(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.context_prototypes).argmax(dim=-1)

    def compute_memory_bonus(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.assign_context(x)
        bonus = self.expert_context_memory[:, context].T
        return bonus, context

    @torch.no_grad()
    def update_memory(self, x: torch.Tensor, targets: torch.Tensor, info: dict[str, torch.Tensor]) -> None:
        losses = per_expert_cross_entropy(info["expert_logits"].detach(), targets)
        context = info["context"]
        for context_id in context.unique():
            mask = context == context_id
            if not bool(mask.any()):
                continue
            reward = -losses[mask].mean(dim=0)
            column = int(context_id.item())
            self.expert_context_memory[:, column].mul_(self.memory_decay).add_(reward * (1.0 - self.memory_decay))

    def memory_state(self) -> dict[str, list[float]]:
        matrix = self.expert_context_memory.detach().cpu()
        return {f"context_{idx}_memory": [float(v) for v in matrix[:, idx]] for idx in range(self.num_contexts)}
