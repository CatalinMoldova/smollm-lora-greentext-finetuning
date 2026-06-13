"""Perplexity evaluation and plotting."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt


def calculate_perplexity(log_loss: float) -> float:
    return math.exp(log_loss)


def ppl_at_step(steps, ppls, target):
    if not steps:
        return None
    idx = min(range(len(steps)), key=lambda i: abs(steps[i] - target))
    return ppls[idx]


def loss_curve(trainer):
    steps, losses = [], []
    for entry in trainer.state.log_history:
        if "loss" in entry and "eval_loss" not in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])
    return steps, losses


def plot_perplexity(steps, losses, title, output_path, color="#3d5a80"):
    ppls = [calculate_perplexity(x) for x in losses]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, ppls, color=color, linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Perplexity")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ppls


def fetch_wandb_loss(run_path: str, key: str = "train/loss"):
    import wandb

    api = wandb.Api()
    run = api.run(run_path)
    hist = run.history(keys=[key])
    steps = hist["_step"].tolist()
    losses = hist[key].tolist()
    return steps, losses
