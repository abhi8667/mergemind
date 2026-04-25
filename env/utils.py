from __future__ import annotations

import json
import random
from typing import Callable, Iterable

import numpy as np

ACTIONS: list[str] = [
    "accelerate",
    "brake",
    "hold_speed",
    "merge_left",
    "merge_right",
    "yield",
    "block",
    "signal_merge",
]

PolicyFn = Callable[[dict], str]


def seed_everything(seed: int | None) -> np.random.Generator:
    if seed is None:
        seed = random.randint(0, 1_000_000)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def observation_to_key(
    obs: dict,
    *,
    max_gap: int = 3,
    max_merge: int = 3,
    max_behind: int = 3,
    max_speed: int = 3,
) -> tuple[int, int, int, int, int, int, int]:
    lane = 1 if obs.get("lane") == "left" else 0
    speed = min(int(obs.get("speed", 0)), max_speed)
    front_gap = min(int(obs.get("front_gap", 0)), max_gap)
    merge_distance = min(int(obs.get("merge_distance", 0)), max_merge)
    right_car_waiting = 1 if obs.get("right_car_waiting") else 0
    can_merge = 1 if obs.get("can_merge") else 0
    cars_behind = min(int(obs.get("cars_behind", 0)), max_behind)
    return (lane, speed, front_gap, merge_distance, right_car_waiting, can_merge, cars_behind)


def key_to_str(key: Iterable[int]) -> str:
    return ",".join(str(v) for v in key)


def load_qtable(path: str) -> dict[str, list[float]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {k: list(map(float, v)) for k, v in data.items()}


def save_qtable(path: str, table: dict[str, list[float]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2, sort_keys=True)


def random_policy(_: dict) -> str:
    return random.choice(ACTIONS)


def rule_based_policy(obs: dict) -> str:
    if obs.get("lane") == "right" and obs.get("can_merge"):
        return "merge_left"
    if obs.get("lane") == "left" and obs.get("right_car_waiting"):
        if obs.get("front_gap", 0) > 1:
            return "yield"
    if obs.get("speed", 0) < 2:
        return "accelerate"
    return "hold_speed"


def qtable_policy(obs: dict, qtable: dict[str, list[float]]) -> str:
    key = key_to_str(observation_to_key(obs))
    if key not in qtable:
        return rule_based_policy(obs)
    values = qtable[key]
    if not values:
        return rule_based_policy(obs)
    best_index = int(np.argmax(values))
    return ACTIONS[best_index]


def render_grid(
    car_positions: dict[str, tuple[str, int]],
    *,
    lane_length: int,
    merge_point: int,
) -> str:
    left_lane = ["·"] * lane_length
    right_lane = ["·"] * lane_length
    merged_lane = ["·"] * lane_length
    for car_id, (lane, pos) in car_positions.items():
        icon = "🚗" if "L" in car_id else "🚙"
        if pos >= lane_length:
            continue
        if pos >= merge_point:
            merged_lane[pos] = icon
        elif lane == "left":
            left_lane[pos] = icon
        else:
            right_lane[pos] = icon
    merge_marker = [" "] * lane_length
    if merge_point < lane_length:
        merge_marker[merge_point] = "🚧"
    output = [
        "Lane L: " + "".join(left_lane),
        "Lane R: " + "".join(right_lane),
        "Merge : " + "".join(merge_marker),
        "Merged: " + "".join(merged_lane),
    ]
    return "\n".join(output)
