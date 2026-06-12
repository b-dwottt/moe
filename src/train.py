"""Train dense, MoE, and memory-augmented MoE variants on synthetic drift."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

_PLOT_CACHE_ROOT = Path("results").resolve()
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE_ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE_ROOT / ".cache"))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from src.data import SyntheticStreamConfig, generate_stream, make_evaluation_set
from src.metrics import accuracy, adaptation_speed, final_window_mean, moving_average, normalized_entropy, regime_summary
from src.models import (
    ContextualMemoryMoE,
    DenseMLP,
    MixtureOfExperts,
    QueueMemoryMoE,
    RandomRoutingMoE,
    ScoreboardMemoryMoE,
)


MODEL_CHOICES = (
    "dense",
    "random_moe",
    "standard_moe",
    "load_balanced_moe",
    "memory_scoreboard",
    "memory_queue",
    "memory_contextual",
)

SUITE_MODELS = (
    "dense",
    "random_moe",
    "standard_moe",
    "load_balanced_moe",
    "memory_scoreboard",
    "memory_queue",
    "memory_contextual",
)

DISPLAY_NAMES = {
    "dense": "Dense MLP",
    "random_moe": "Random MoE",
    "standard_moe": "Standard MoE",
    "load_balanced_moe": "Load-balanced MoE",
    "memory_scoreboard": "Scoreboard Memory MoE",
    "memory_queue": "Queue Memory MoE",
    "memory_contextual": "Contextual Memory MoE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train synthetic drift baselines and MoE variants.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="dense")
    parser.add_argument("--suite", action="store_true", help="Run the full comparison suite.")
    parser.add_argument("--models", type=str, default=",".join(SUITE_MODELS), help="Comma-separated model list for --suite.")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated seeds for --suite.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--swap-interval", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--router-hidden-dim", type=int, default=32)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--routing", choices=("soft", "top1"), default="soft")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--load-balance-strength", type=float, default=0.05)
    parser.add_argument("--memory-alpha", type=float, default=0.5)
    parser.add_argument("--memory-decay", type=float, default=0.95)
    parser.add_argument("--queue-size", type=int, default=50)
    parser.add_argument("--stability-weight", type=float, default=0.25)
    parser.add_argument("--num-contexts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--experiment-name", type=str, default=None)
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seed_list(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, args: argparse.Namespace, config: SyntheticStreamConfig) -> nn.Module:
    if model_name == "dense":
        return DenseMLP(input_dim=config.input_dim, hidden_dim=args.hidden_dim)

    if model_name == "random_moe":
        return RandomRoutingMoE(
            input_dim=config.input_dim,
            hidden_dim=args.hidden_dim,
            num_experts=args.num_experts,
        )

    common_kwargs = {
        "input_dim": config.input_dim,
        "hidden_dim": args.hidden_dim,
        "num_experts": args.num_experts,
        "router_hidden_dim": args.router_hidden_dim,
        "routing": args.routing,
        "temperature": args.temperature,
    }

    if model_name == "standard_moe":
        return MixtureOfExperts(**common_kwargs)
    if model_name == "load_balanced_moe":
        return MixtureOfExperts(**common_kwargs, load_balance_strength=args.load_balance_strength)
    if model_name == "memory_scoreboard":
        return ScoreboardMemoryMoE(
            **common_kwargs,
            memory_alpha=args.memory_alpha,
            memory_decay=args.memory_decay,
        )
    if model_name == "memory_queue":
        return QueueMemoryMoE(
            **common_kwargs,
            memory_alpha=args.memory_alpha,
            memory_decay=args.memory_decay,
            queue_size=args.queue_size,
            stability_weight=args.stability_weight,
        )
    if model_name == "memory_contextual":
        return ContextualMemoryMoE(
            **common_kwargs,
            memory_alpha=args.memory_alpha,
            memory_decay=args.memory_decay,
            num_contexts=args.num_contexts,
            context_seed=args.seed,
        )

    raise ValueError(f"Unknown model: {model_name}")


def forward_model(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = model(x)
    if isinstance(output, tuple):
        return output
    return output, {}


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits, info = forward_model(model, x)
        loss = criterion(logits, y).item()
        acc = accuracy(logits, y)
        entropy = normalized_entropy(info["router_weights"]) if "router_weights" in info else float("nan")
    model.train()
    return {"eval_loss": loss, "eval_accuracy": acc, "eval_expert_entropy": entropy}


def add_routing_columns(row: dict[str, Any], info: dict[str, torch.Tensor], num_experts: int) -> None:
    if "router_weights" not in info:
        row["expert_entropy"] = float("nan")
        for expert_idx in range(num_experts):
            row[f"expert_{expert_idx}_usage"] = float("nan")
        return

    weights = info["router_weights"].detach()
    selected = info["selected_expert"].detach()
    row["expert_entropy"] = normalized_entropy(weights)
    for expert_idx in range(num_experts):
        row[f"expert_{expert_idx}_usage"] = float((selected == expert_idx).float().mean().item())


def add_memory_columns(row: dict[str, Any], model: nn.Module) -> None:
    if not hasattr(model, "memory_state"):
        return
    state = model.memory_state()
    for key, values in state.items():
        for idx, value in enumerate(values):
            row[f"{key}_{idx}"] = value


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def run_single_experiment(args: argparse.Namespace, model_name: str, seed: int) -> dict[str, Any]:
    set_seed(seed)
    config = SyntheticStreamConfig(
        batch_size=args.batch_size,
        total_steps=args.steps,
        swap_interval=args.swap_interval,
        seed=seed,
    )

    model = build_model(model_name, args, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    rows: list[dict[str, Any]] = []
    for step, regime, x, y in generate_stream(config):
        logits, info = forward_model(model, x)
        task_loss = criterion(logits, y)
        aux_loss = info.get("aux_loss", torch.zeros((), dtype=task_loss.dtype))
        loss = task_loss + aux_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if hasattr(model, "update_memory"):
            model.update_memory(x, y, info)

        row: dict[str, Any] = {
            "model": model_name,
            "seed": int(seed),
            "step": int(step),
            "regime": int(regime),
            "loss": float(task_loss.item()),
            "total_loss": float(loss.item()),
            "aux_loss": float(aux_loss.detach().item()),
            "accuracy": float(accuracy(logits, y)),
        }
        add_routing_columns(row, info, args.num_experts)
        add_memory_columns(row, model)
        rows.append(row)

    metrics = pd.DataFrame(rows)
    eval_x, eval_y = make_evaluation_set(
        config=config,
        regime=int(metrics["regime"].iloc[-1]),
        size=args.eval_size,
    )
    evaluation = evaluate(model, eval_x, eval_y, criterion)
    evaluation.update(
        {
            "model": model_name,
            "seed": int(seed),
            "train_accuracy_mean": float(metrics["accuracy"].mean()),
            "train_loss_mean": float(metrics["loss"].mean()),
            "final_100_accuracy": final_window_mean(metrics, "accuracy", window=100),
            "final_100_loss": final_window_mean(metrics, "loss", window=100),
            "adaptation_speed_steps": adaptation_speed(metrics, swap_interval=args.swap_interval),
            "expert_entropy_mean": float(metrics["expert_entropy"].mean()) if "expert_entropy" in metrics else float("nan"),
            "num_params": parameter_count(model),
            "config": asdict(config),
        }
    )
    add_memory_columns(evaluation, model)
    return {"metrics": metrics, "evaluation": evaluation}


def save_comparison_plot(metrics: pd.DataFrame, output_path: Path, swap_interval: int) -> None:
    window = min(35, max(5, int(metrics["step"].max()) // 30))
    fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for model_name in MODEL_CHOICES:
        model_metrics = metrics[metrics["model"] == model_name]
        if model_metrics.empty:
            continue
        per_step = model_metrics.groupby("step", as_index=False)[["accuracy", "loss"]].mean()
        acc_ma = moving_average(per_step["accuracy"], window=window)
        loss_ma = moving_average(per_step["loss"], window=window)
        x_values = per_step["step"].iloc[window - 1 :]
        label = DISPLAY_NAMES.get(model_name, model_name)
        ax_acc.plot(x_values, acc_ma, linewidth=2, label=label)
        ax_loss.plot(x_values, loss_ma, linewidth=2, label=label)

    for swap_step in range(swap_interval, int(metrics["step"].max()) + 1, swap_interval):
        ax_acc.axvline(swap_step, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)
        ax_loss.axvline(swap_step, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)

    ax_acc.set_ylabel("Accuracy")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_xlabel("Training step")
    ax_acc.set_title("Synthetic drift comparison")
    ax_acc.legend(loc="lower right", ncols=2, fontsize=8)
    ax_loss.legend(loc="upper right", ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_routing_plot(metrics: pd.DataFrame, output_path: Path, swap_interval: int) -> None:
    moe_metrics = metrics[metrics["expert_entropy"].notna()]
    if moe_metrics.empty:
        return

    window = min(35, max(5, int(metrics["step"].max()) // 30))
    usage_cols = [col for col in metrics.columns if col.startswith("expert_") and col.endswith("_usage")]
    fig, (ax_entropy, ax_usage) = plt.subplots(2, 1, figsize=(12, 8))

    for model_name in MODEL_CHOICES:
        model_metrics = moe_metrics[moe_metrics["model"] == model_name]
        if model_metrics.empty:
            continue
        per_step = model_metrics.groupby("step", as_index=False)["expert_entropy"].mean()
        entropy_ma = moving_average(per_step["expert_entropy"], window=window)
        x_values = per_step["step"].iloc[window - 1 :]
        ax_entropy.plot(x_values, entropy_ma, linewidth=2, label=DISPLAY_NAMES.get(model_name, model_name))

    for swap_step in range(swap_interval, int(metrics["step"].max()) + 1, swap_interval):
        ax_entropy.axvline(swap_step, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)

    usage = moe_metrics.groupby("model")[usage_cols].mean().reindex([name for name in MODEL_CHOICES if name in moe_metrics["model"].unique()])
    bottoms = None
    x_positions = range(len(usage))
    for usage_col in usage_cols:
        values = usage[usage_col].fillna(0.0).to_numpy()
        ax_usage.bar(x_positions, values, bottom=bottoms, label=usage_col.replace("_usage", "").replace("_", " "))
        bottoms = values if bottoms is None else bottoms + values

    ax_entropy.set_ylabel("Normalized routing entropy")
    ax_entropy.set_title("Routing diagnostics")
    ax_entropy.legend(loc="lower right", ncols=2, fontsize=8)
    ax_usage.set_ylabel("Mean selected expert share")
    ax_usage.set_xticks(list(x_positions))
    ax_usage.set_xticklabels([DISPLAY_NAMES.get(idx, idx) for idx in usage.index], rotation=25, ha="right")
    ax_usage.legend(loc="upper right", ncols=max(1, len(usage_cols) // 2), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_memory_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    memory_cols = [col for col in metrics.columns if "memory_" in col or "queue_bonus_" in col]
    if not memory_cols:
        return

    memory_metrics = metrics[metrics["model"].str.startswith("memory_")]
    if memory_metrics.empty:
        return

    first_seed = int(memory_metrics["seed"].min())
    memory_metrics = memory_metrics[memory_metrics["seed"] == first_seed]
    models = [name for name in SUITE_MODELS if name in memory_metrics["model"].unique()]
    fig, axes = plt.subplots(len(models), 1, figsize=(12, max(3, 3 * len(models))), sharex=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        model_metrics = memory_metrics[memory_metrics["model"] == model_name]
        active_cols = [col for col in memory_cols if col in model_metrics and model_metrics[col].notna().any()]
        for col in active_cols[:8]:
            ax.plot(model_metrics["step"], model_metrics[col], linewidth=1.5, label=col)
        ax.set_ylabel("Memory value")
        ax.set_title(DISPLAY_NAMES.get(model_name, model_name))
        ax.legend(loc="best", fontsize=7, ncols=2)

    axes[-1].set_xlabel("Training step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [col for col in summary.select_dtypes(include="number").columns if col != "seed"]
    mean_df = summary.groupby("model")[numeric_cols].mean().add_suffix("_mean")
    std_df = summary.groupby("model")[numeric_cols].std(ddof=0).add_suffix("_std")
    return pd.concat([mean_df, std_df], axis=1).reset_index()


def default_experiment_name(args: argparse.Namespace) -> str:
    if args.experiment_name:
        return args.experiment_name
    if args.suite:
        return "synthetic_moe_comparison"
    if args.model == "dense":
        return "synthetic_baseline"
    return f"synthetic_{args.model}"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    model_names = parse_csv_list(args.models) if args.suite else [args.model]
    unknown_models = sorted(set(model_names) - set(MODEL_CHOICES))
    if unknown_models:
        raise ValueError(f"Unknown models in suite: {unknown_models}")
    seeds = parse_seed_list(args.seeds) if args.suite else [args.seed]

    metric_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for model_name in model_names:
        for seed in seeds:
            print(f"Running {DISPLAY_NAMES.get(model_name, model_name)} with seed {seed}...")
            result = run_single_experiment(args, model_name, seed)
            metric_frames.append(result["metrics"])
            summaries.append(result["evaluation"])

    metrics = pd.concat(metric_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    aggregate = aggregate_summary(summary)
    regime = regime_summary(metrics)

    prefix = default_experiment_name(args)
    metrics_path = tables_dir / f"{prefix}_metrics.csv"
    summary_path = tables_dir / f"{prefix}_summary.csv"
    aggregate_path = tables_dir / f"{prefix}_aggregate.csv"
    regime_path = tables_dir / f"{prefix}_regime_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    regime.to_csv(regime_path, index=False)

    comparison_plot = figures_dir / f"{prefix}_loss_accuracy.png"
    routing_plot = figures_dir / f"{prefix}_routing_diagnostics.png"
    memory_plot = figures_dir / f"{prefix}_memory_traces.png"
    save_comparison_plot(metrics, comparison_plot, args.swap_interval)
    save_routing_plot(metrics, routing_plot, args.swap_interval)
    save_memory_plot(metrics, memory_plot)

    print("\nFinished experiment run.")
    for _, row in aggregate.sort_values("eval_accuracy_mean", ascending=False).iterrows():
        print(
            f"{DISPLAY_NAMES.get(row['model'], row['model'])}: "
            f"eval_acc={row['eval_accuracy_mean']:.3f}±{row['eval_accuracy_std']:.3f}, "
            f"adapt_steps={row['adaptation_speed_steps_mean']:.1f}"
        )
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved aggregate summary to {aggregate_path}")
    print(f"Saved regime summary to {regime_path}")
    print(f"Saved comparison figure to {comparison_plot}")
    if routing_plot.exists():
        print(f"Saved routing diagnostics to {routing_plot}")
    if memory_plot.exists():
        print(f"Saved memory traces to {memory_plot}")


if __name__ == "__main__":
    main()
