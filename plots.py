from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def plot_reward_curve(rewards: Iterable[float], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(list(rewards), label="Episode Reward")
    plt.title("MergeMind Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_eval_metrics(results: dict[str, dict[str, float]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = list(results.keys())
    collision_rates = [results[label]["collision_rate"] for label in labels]
    throughput = [results[label]["throughput"] for label in labels]
    courtesy = [results[label]["courtesy_score"] for label in labels]
    reward = [results[label]["reward_score"] for label in labels]

    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    axs[0, 0].bar(labels, collision_rates, color="#f87171")
    axs[0, 0].set_title("Collision Rate ↓")
    axs[0, 1].bar(labels, throughput, color="#34d399")
    axs[0, 1].set_title("Throughput ↑")
    axs[1, 0].bar(labels, courtesy, color="#60a5fa")
    axs[1, 0].set_title("Courtesy Score ↑")
    axs[1, 1].bar(labels, reward, color="#fbbf24")
    axs[1, 1].set_title("Reward Score ↑")
    for ax in axs.flat:
        ax.set_ylim(0, max(ax.get_ylim()[1], 1))
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
