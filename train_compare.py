"""PyTorch neural network comparison experiment.

This script builds a synthetic 2D non-linear binary classification dataset,
then compares a small baseline MLP with an improved MLP.

Run:
    python train_compare.py

Useful options:
    python train_compare.py --epochs 120 --output-dir results --assets-dir assets
    python train_compare.py --cpu
    python train_compare.py --no-plots
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42


@dataclass
class Metrics:
    train_loss: float
    test_loss: float
    test_accuracy: float


@dataclass
class Evaluation:
    metrics: Metrics
    predictions: torch.Tensor
    confusion_matrix: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and improved PyTorch MLP models.")
    parser.add_argument("--samples", type=int, default=1200, help="Total number of generated samples.")
    parser.add_argument("--test-ratio", type=float, default=0.25, help="Test set ratio.")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs for each model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--baseline-lr", type=float, default=0.05, help="Learning rate for the SGD baseline.")
    parser.add_argument("--improved-lr", type=float, default=0.01, help="Learning rate for the Adam improved model.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--num-threads", type=int, default=1, help="Number of CPU threads used by PyTorch.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution even when CUDA is available.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for result files.")
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"), help="Directory for generated figures.")
    parser.add_argument("--no-plots", action="store_true", help="Skip optional figure generation.")
    return parser.parse_args()


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED, num_threads: int = 1) -> None:
    """Fix random seeds to make the experiment easier to reproduce."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if num_threads > 0:
        torch.set_num_threads(num_threads)


def make_dataset(
    n_samples: int = 1200,
    test_ratio: float = 0.25,
    seed: int = SEED,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a 2D non-linear binary classification dataset.

    The label is determined by a radius term plus sinusoidal perturbations,
    so the decision boundary is intentionally non-linear.
    """
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    if n_samples < 20:
        raise ValueError("n_samples should be at least 20")

    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, 2, generator=generator)

    radius = x[:, 0] ** 2 + x[:, 1] ** 2
    wave = 0.35 * torch.sin(3.0 * x[:, 0]) + 0.25 * torch.cos(4.0 * x[:, 1])
    score = radius + wave
    y = (score > 1.25).long()

    indices = torch.randperm(n_samples, generator=generator)
    x = x[indices]
    y = y[indices]

    test_size = int(n_samples * test_ratio)
    x_train = x[:-test_size]
    y_train = y[:-test_size]
    x_test = x[-test_size:]
    y_test = y[-test_size:]
    return x_train, y_train, x_test, y_test


def standardize_train_test(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize using training-set statistics to avoid test data leakage."""
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


class BaselineNet(nn.Module):
    """Small baseline model: one hidden layer and ReLU activation."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImprovedNet(nn.Module):
    """Improved model: deeper/wider MLP and LeakyReLU activation."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(32, 32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int = SEED,
) -> DataLoader:
    dataset = TensorDataset(x, y)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    train_loss: float = 0.0,
) -> Evaluation:
    model.eval()
    x = x.to(device)
    y = y.to(device)
    logits = model(x)
    loss = criterion(logits, y).item()
    pred = logits.argmax(dim=1)
    accuracy = (pred == y).float().mean().item()

    matrix = torch.zeros((2, 2), dtype=torch.int64)
    for true_label, pred_label in zip(y.cpu(), pred.cpu()):
        matrix[int(true_label), int(pred_label)] += 1

    return Evaluation(
        metrics=Metrics(train_loss=train_loss, test_loss=loss, test_accuracy=accuracy),
        predictions=pred.cpu(),
        confusion_matrix=matrix,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
) -> Tuple[Metrics, Evaluation, List[Dict[str, float]]]:
    history: List[Dict[str, float]] = []
    last_train_loss = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            total_count += batch_x.size(0)

        last_train_loss = total_loss / total_count
        eval_result = evaluate_model(model, x_test, y_test, criterion, device, train_loss=last_train_loss)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(last_train_loss),
                "test_loss": float(eval_result.metrics.test_loss),
                "test_accuracy": float(eval_result.metrics.test_accuracy),
            }
        )

    final_eval = evaluate_model(model, x_test, y_test, criterion, device, train_loss=last_train_loss)
    return final_eval.metrics, final_eval, history


def run_experiment(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    optimizer_name: str,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, Metrics, Evaluation, List[Dict[str, float]]]:
    model = model.to(device)
    train_loader = make_loader(x_train, y_train, batch_size=batch_size, seed=seed)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    metrics, eval_result, history = train_model(model, train_loader, x_test, y_test, optimizer, criterion, epochs, device)
    return model, metrics, eval_result, history


def format_result(name: str, metrics: Metrics, parameters: int) -> str:
    return (
        f"{name}: "
        f"parameters={parameters}, "
        f"train_loss={metrics.train_loss:.4f}, "
        f"test_loss={metrics.test_loss:.4f}, "
        f"test_accuracy={metrics.test_accuracy * 100:.2f}%"
    )


def build_results_text(
    baseline: Metrics,
    improved: Metrics,
    baseline_parameters: int,
    improved_parameters: int,
    device: torch.device,
    config: Dict[str, object],
) -> str:
    lines = [
        "PyTorch 2D Non-linear Binary Classification Experiment",
        f"Device: {device}",
        f"Config: {json.dumps(config, ensure_ascii=False)}",
        "",
        format_result("Baseline", baseline, baseline_parameters),
        format_result("Improved", improved, improved_parameters),
        "",
        "Markdown table:",
        "",
        "| Model | Parameters | Train loss | Test loss | Test accuracy | Method |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        (
            "| Baseline | "
            f"{baseline_parameters} | {baseline.train_loss:.4f} | {baseline.test_loss:.4f} | "
            f"{baseline.test_accuracy * 100:.2f}% | Raw features + 1 hidden layer + ReLU + SGD |"
        ),
        (
            "| Improved | "
            f"{improved_parameters} | {improved.train_loss:.4f} | {improved.test_loss:.4f} | "
            f"{improved.test_accuracy * 100:.2f}% | Standardization + 2 hidden layers + LeakyReLU + Adam |"
        ),
    ]
    return "\n".join(lines)


def save_history_csv(output_dir: Path, baseline_history: Iterable[Dict[str, float]], improved_history: Iterable[Dict[str, float]]) -> None:
    path = output_dir / "history.csv"
    rows: List[Dict[str, object]] = []
    for row in baseline_history:
        rows.append({"model": "Baseline", **row})
    for row in improved_history:
        rows.append({"model": "Improved", **row})

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "epoch", "train_loss", "test_loss", "test_accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(
    output_dir: Path,
    baseline: Metrics,
    improved: Metrics,
    baseline_parameters: int,
    improved_parameters: int,
    baseline_confusion: torch.Tensor,
    improved_confusion: torch.Tensor,
    device: torch.device,
    config: Dict[str, object],
) -> None:
    data = {
        "device": str(device),
        "config": config,
        "baseline": {
            **asdict(baseline),
            "parameters": baseline_parameters,
            "confusion_matrix": baseline_confusion.tolist(),
        },
        "improved": {
            **asdict(improved),
            "parameters": improved_parameters,
            "confusion_matrix": improved_confusion.tolist(),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_plots(
    assets_dir: Path,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    baseline_history: List[Dict[str, float]],
    improved_history: List[Dict[str, float]],
    baseline_model: nn.Module,
    improved_model: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    baseline_confusion: torch.Tensor,
    improved_confusion: torch.Tensor,
    device: torch.device,
) -> None:
    """Save figures used by the README and experiment report."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[WARN] matplotlib is unavailable, skip plots: {exc}")
        return

    assets_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in baseline_history]

    # Dataset preview.
    plt.figure(figsize=(6, 5))
    plt.scatter(x_train[:, 0].numpy(), x_train[:, 1].numpy(), c=y_train.numpy(), s=14, alpha=0.75)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Synthetic Dataset Preview")
    plt.tight_layout()
    plt.savefig(assets_dir / "dataset_preview.png", dpi=180)
    plt.close()

    # Loss curve.
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["test_loss"] for row in baseline_history], label="Baseline test loss")
    plt.plot(epochs, [row["test_loss"] for row in improved_history], label="Improved test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Test Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(assets_dir / "loss_curve.png", dpi=180)
    plt.close()

    # Accuracy curve.
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["test_accuracy"] * 100 for row in baseline_history], label="Baseline accuracy")
    plt.plot(epochs, [row["test_accuracy"] * 100 for row in improved_history], label="Improved accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(assets_dir / "accuracy_curve.png", dpi=180)
    plt.close()

    # Decision boundaries.
    x_min, x_max = -3.0, 3.0
    y_min, y_max = -3.0, 3.0
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 240), np.linspace(y_min, y_max, 240))
    points = torch.tensor(np.c_[grid_x.ravel(), grid_y.ravel()], dtype=torch.float32)

    def plot_boundary(model: nn.Module, transformed_points: torch.Tensor, filename: str, title: str) -> None:
        model.eval()
        with torch.no_grad():
            logits = model(transformed_points.to(device))
            pred = logits.argmax(dim=1).cpu().numpy().reshape(grid_x.shape)
        plt.figure(figsize=(6, 6))
        plt.contourf(grid_x, grid_y, pred, alpha=0.35)
        plt.scatter(x_train[:, 0].numpy(), x_train[:, 1].numpy(), c=y_train.numpy(), s=10, alpha=0.65)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(assets_dir / filename, dpi=180)
        plt.close()

    plot_boundary(baseline_model, points, "decision_boundary_baseline.png", "Baseline Decision Boundary")
    plot_boundary(improved_model, (points - mean) / std, "decision_boundary_improved.png", "Improved Decision Boundary")

    # Confusion matrix comparison.
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, matrix, title in [
        (axes[0], baseline_confusion.numpy(), "Baseline"),
        (axes[1], improved_confusion.numpy(), "Improved"),
    ]:
        im = ax.imshow(matrix)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    plt.savefig(assets_dir / "confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed, num_threads=args.num_threads)
    device = get_device(force_cpu=args.cpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = make_dataset(args.samples, args.test_ratio, seed=args.seed)

    baseline_model, baseline_metrics, baseline_eval, baseline_history = run_experiment(
        model=BaselineNet(),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        optimizer_name="sgd",
        lr=args.baseline_lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )

    x_train_std, x_test_std, mean, std = standardize_train_test(x_train, x_test)
    improved_model, improved_metrics, improved_eval, improved_history = run_experiment(
        model=ImprovedNet(),
        x_train=x_train_std,
        y_train=y_train,
        x_test=x_test_std,
        y_test=y_test,
        optimizer_name="adam",
        lr=args.improved_lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )

    baseline_parameters = count_parameters(baseline_model)
    improved_parameters = count_parameters(improved_model)
    config: Dict[str, object] = {
        "samples": args.samples,
        "test_ratio": args.test_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "baseline_lr": args.baseline_lr,
        "improved_lr": args.improved_lr,
        "seed": args.seed,
        "num_threads": args.num_threads,
    }

    results_text = build_results_text(
        baseline_metrics,
        improved_metrics,
        baseline_parameters,
        improved_parameters,
        device,
        config,
    )
    print(results_text)
    (args.output_dir / "results.txt").write_text(results_text + "\n", encoding="utf-8")
    save_history_csv(args.output_dir, baseline_history, improved_history)
    save_summary_json(
        args.output_dir,
        baseline_metrics,
        improved_metrics,
        baseline_parameters,
        improved_parameters,
        baseline_eval.confusion_matrix,
        improved_eval.confusion_matrix,
        device,
        config,
    )

    if not args.no_plots:
        save_plots(
            args.assets_dir,
            x_train,
            y_train,
            baseline_history,
            improved_history,
            baseline_model,
            improved_model,
            mean,
            std,
            baseline_eval.confusion_matrix,
            improved_eval.confusion_matrix,
            device,
        )


if __name__ == "__main__":
    main()
