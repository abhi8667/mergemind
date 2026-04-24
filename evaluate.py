from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from env.mergemind_env import MergeMindEnv
from env.utils import load_qtable, qtable_policy, random_policy, rule_based_policy
from plots import plot_eval_metrics

PolicyFn = Callable[[dict], str]


def run_policy(env: MergeMindEnv, policy: PolicyFn, episodes: int) -> dict[str, float]:
    collision_total = 0
    steps_total = 0
    throughput_total = 0
    courtesy_total = 0
    reward_total = 0.0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        last_step = 0
        while not done:
            actions = {car_id: policy(observation) for car_id, observation in obs.items()}
            obs, rewards, done, info = env.step(actions)
            episode_reward += sum(rewards.values())
            last_step = info.get("step", last_step)
        collision_total += info.get("total_collisions", 0)
        steps_total += last_step
        throughput_total += info.get("throughput", 0)
        courtesy_total += info.get("courtesy_events", 0)
        reward_total += episode_reward
    episodes = max(1, episodes)
    return {
        "collision_rate": collision_total / episodes,
        "average_merge_time": steps_total / episodes,
        "throughput": throughput_total / episodes,
        "courtesy_score": courtesy_total / episodes,
        "reward_score": reward_total / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MergeMind policies.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--policy-path", type=str, default="outputs/policy.json")
    args = parser.parse_args()

    env = MergeMindEnv()
    results: dict[str, dict[str, float]] = {}
    results["random"] = run_policy(env, random_policy, args.episodes)
    results["rule_based"] = run_policy(env, rule_based_policy, args.episodes)

    policy_path = Path(args.policy_path)
    if policy_path.exists():
        qtable = load_qtable(str(policy_path))

        def trained_policy(obs: dict) -> str:
            return qtable_policy(obs, qtable)

        results["trained"] = run_policy(env, trained_policy, args.episodes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    plot_eval_metrics(results, str(output_dir / "eval_metrics.png"))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
