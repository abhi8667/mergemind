from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from env.mergemind_env import MergeMindEnv
from env.utils import ACTIONS, key_to_str, observation_to_key, save_qtable
from plots import plot_reward_curve


def train_qlearning(args: argparse.Namespace) -> dict[str, Any]:
    env = MergeMindEnv(
        lane_length=args.lane_length,
        merge_point=args.merge_point,
        cars_per_lane=args.cars_per_lane,
        max_speed=args.max_speed,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        seed=args.seed,
    )
    qtable: dict[str, list[float]] = {}
    reward_history: list[float] = []
    collision_history: list[int] = []
    throughput_history: list[int] = []

    epsilon = args.epsilon
    for episode in range(args.episodes):
        obs = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            actions: dict[str, str] = {}
            for car_id, observation in obs.items():
                key = key_to_str(observation_to_key(observation))
                if key not in qtable:
                    qtable[key] = [0.0 for _ in ACTIONS]
                if np.random.random() < epsilon:
                    action = np.random.choice(ACTIONS)
                else:
                    action = ACTIONS[int(np.argmax(qtable[key]))]
                actions[car_id] = action
            next_obs, rewards, done, info = env.step(actions)
            for car_id, observation in obs.items():
                key = key_to_str(observation_to_key(observation))
                reward = rewards.get(car_id, 0.0)
                episode_reward += reward
                next_key = key_to_str(
                    observation_to_key(next_obs.get(car_id, observation))
                )
                if next_key not in qtable:
                    qtable[next_key] = [0.0 for _ in ACTIONS]
                action_index = ACTIONS.index(actions[car_id])
                best_next = max(qtable[next_key])
                td_target = reward + args.gamma * best_next
                current_q = qtable[key][action_index]
                updated_q = (1 - args.alpha) * current_q + args.alpha * td_target
                qtable[key][action_index] = updated_q
            obs = next_obs
        reward_history.append(episode_reward)
        collision_history.append(info.get("total_collisions", 0))
        throughput_history.append(info.get("throughput", 0))
        epsilon = max(args.min_epsilon, epsilon * args.epsilon_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "policy.json"
    save_qtable(str(policy_path), qtable)
    plot_reward_curve(reward_history, str(output_dir / "reward_curve.png"))
    metrics = {
        "episodes": args.episodes,
        "avg_reward": float(np.mean(reward_history)) if reward_history else 0.0,
        "avg_collisions": float(np.mean(collision_history)) if collision_history else 0.0,
        "avg_throughput": float(np.mean(throughput_history)) if throughput_history else 0.0,
        "reward_history": reward_history,
    }
    with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def train_with_trl(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import PPOConfig, PPOTrainer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "TRL/Transformers not installed. Install with `pip install trl transformers torch`."
        ) from exc

    env = MergeMindEnv(
        lane_length=args.lane_length,
        merge_point=args.merge_point,
        cars_per_lane=args.cars_per_lane,
        max_speed=args.max_speed,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)

    config = PPOConfig(
        batch_size=args.ppo_batch_size,
        mini_batch_size=args.ppo_mini_batch_size,
        learning_rate=args.ppo_lr,
        log_with=None,
    )
    trainer = PPOTrainer(config, model, tokenizer)

    reward_history: list[float] = []
    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            query_texts = []
            action_map: dict[str, str] = {}
            for car_id, observation in obs.items():
                prompt = f"Observation: {observation}\nAction:"
                query_texts.append(prompt)
                action_map[car_id] = prompt
            query_tensors = tokenizer(query_texts, return_tensors="pt", padding=True).input_ids.to(args.device)
            response_tensors = trainer.generate(query_tensors, max_new_tokens=3)
            responses = tokenizer.batch_decode(response_tensors[:, query_tensors.shape[1] :], skip_special_tokens=True)
            actions = {}
            for car_id, response in zip(action_map.keys(), responses):
                response = response.strip().lower()
                action = next((act for act in ACTIONS if act in response), "hold_speed")
                actions[car_id] = action
            next_obs, rewards, done, _info = env.step(actions)
            reward_list = [rewards.get(car_id, 0.0) for car_id in action_map.keys()]
            episode_reward += float(np.sum(reward_list))
            reward_tensor = torch.tensor(reward_list, dtype=torch.float32, device=query_tensors.device)
            trainer.step(query_tensors, response_tensors, reward_tensor)
            obs = next_obs
        reward_history.append(episode_reward)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_reward_curve(reward_history, str(output_dir / "reward_curve.png"))
    metrics = {"episodes": args.episodes, "avg_reward": float(np.mean(reward_history))}
    with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    trainer.model.save_pretrained(output_dir / "trl_model")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MergeMind policies.")
    parser.add_argument("--mode", choices=["qlearn", "trl"], default="qlearn")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--lane-length", type=int, default=12)
    parser.add_argument("--merge-point", type=int, default=6)
    parser.add_argument("--cars-per-lane", type=int, default=3)
    parser.add_argument("--max-speed", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="outputs")

    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.8)
    parser.add_argument("--min-epsilon", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=float, default=0.95)

    parser.add_argument("--model-name", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ppo-batch-size", type=int, default=4)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=2)
    parser.add_argument("--ppo-lr", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "trl":
        metrics = train_with_trl(args)
    else:
        metrics = train_qlearning(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
