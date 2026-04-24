from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from env.mergemind_env import MergeMindEnv
from env.utils import render_grid


@dataclass
class ReplayResult:
    frames: list[str]
    metrics: dict[str, float]


def run_episode(
    env: MergeMindEnv, policy: Callable[[dict], str], max_steps: int | None = None
) -> ReplayResult:
    frames: list[str] = []
    obs = env.reset()
    done = False
    step = 0
    while not done:
        actions = {car_id: policy(observation) for car_id, observation in obs.items()}
        obs, rewards, done, info = env.step(actions)
        car_positions = {
            car_id: (car.lane, car.position) for car_id, car in env.cars.items() if not car.done
        }
        frame = render_grid(
            car_positions, lane_length=env.lane_length, merge_point=env.merge_point
        )
        frames.append(f"Step {step}\n{frame}")
        step += 1
        if max_steps and step >= max_steps:
            break
    metrics = {
        "reward": sum(rewards.values()),
        "collisions": info.get("total_collisions", 0),
        "throughput": info.get("throughput", 0),
        "courtesy_events": info.get("courtesy_events", 0),
        "steps": info.get("step", step),
    }
    return ReplayResult(frames=frames, metrics=metrics)
