from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .rewards import RewardConfig, RewardBreakdown, compute_reward
from .utils import ACTIONS, seed_everything


@dataclass
class CarState:
    car_id: str
    lane: str
    position: int
    speed: int
    done: bool = False
    collided: bool = False
    brake_steps: int = 0
    no_progress_steps: int = 0
    last_action: str = "hold_speed"


class MergeMindEnv:
    def __init__(
        self,
        *,
        lane_length: int = 12,
        merge_point: int = 6,
        cars_per_lane: int = 3,
        max_speed: int = 3,
        max_steps: int = 40,
        deterministic: bool = False,
        seed: int | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.lane_length = lane_length
        self.merge_point = merge_point
        self.cars_per_lane = cars_per_lane
        self.max_speed = max_speed
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.reward_config = reward_config or RewardConfig()
        self.rng = seed_everything(seed)
        self.step_count = 0
        self.cars: dict[str, CarState] = {}
        self.collision_count = 0
        self.courtesy_events = 0

    def seed(self, seed: int | None) -> None:
        self.rng = seed_everything(seed)

    def reset(self) -> dict[str, dict]:
        self.step_count = 0
        self.collision_count = 0
        self.courtesy_events = 0
        self.cars = {}
        spacing = 2
        for idx in range(self.cars_per_lane):
            left_id = f"L{idx}"
            right_id = f"R{idx}"
            left_pos = idx * spacing
            right_pos = idx * spacing
            left_speed = 1 if self.deterministic else int(self.rng.integers(1, 3))
            right_speed = 1 if self.deterministic else int(self.rng.integers(1, 3))
            self.cars[left_id] = CarState(
                car_id=left_id, lane="left", position=left_pos, speed=left_speed
            )
            self.cars[right_id] = CarState(
                car_id=right_id, lane="right", position=right_pos, speed=right_speed
            )
        return self.state()

    def state(self) -> dict[str, dict]:
        return {car_id: self._observe(car) for car_id, car in self.cars.items() if not car.done}

    def step(self, actions: dict[str, str]) -> tuple[dict[str, dict], dict[str, float], bool, dict]:
        self.step_count += 1
        observations = {car_id: self._observe(car) for car_id, car in self.cars.items() if not car.done}
        rewards: dict[str, RewardBreakdown] = {}
        lane_updates: dict[str, tuple[str, str, int, int, str]] = {}
        collision_ids: set[str] = set()

        for car_id, car in self.cars.items():
            if car.done:
                continue
            action = actions.get(car_id, "hold_speed")
            if action not in ACTIONS:
                action = "hold_speed"
            original_position = car.position
            original_lane = car.lane
            if action == "accelerate":
                car.speed = min(self.max_speed, car.speed + 1)
            elif action in {"brake", "yield"}:
                car.speed = max(0, car.speed - 1)
            elif action == "block":
                car.speed = min(self.max_speed, car.speed)
            if action == "brake":
                car.brake_steps += 1
            else:
                car.brake_steps = 0

            can_merge_left, can_merge_right = self._merge_options(car)
            if action == "merge_left" and can_merge_left:
                car.lane = "left"
            elif action == "merge_right" and can_merge_right:
                car.lane = "right"

            new_position = car.position + car.speed
            lane_updates[car_id] = (original_lane, car.lane, original_position, new_position, action)

        occupancy: dict[tuple[str, int], str] = {}
        for car_id, (original_lane, lane, original_position, new_position, action) in lane_updates.items():
            car = self.cars[car_id]
            if car.done:
                continue
            if lane == "right" and new_position > self.merge_point:
                collision_ids.add(car_id)
                continue
            if new_position >= self.lane_length:
                car.done = True
                car.position = new_position
                continue
            lane_key = "merged" if new_position >= self.merge_point else lane
            slot = (lane_key, new_position)
            if slot in occupancy:
                collision_ids.add(car_id)
                collision_ids.add(occupancy[slot])
            else:
                occupancy[slot] = car_id
                car.position = new_position

        for car_id, car in self.cars.items():
            if car_id in collision_ids:
                car.done = True
                car.collided = True

        self.collision_count += len(collision_ids)
        global_clear = self._all_cleared() and self.collision_count == 0
        rewards_out: dict[str, float] = {}
        for car_id, car in self.cars.items():
            if car.done and not car.collided and car.position >= self.lane_length:
                pass
            if car_id not in lane_updates:
                continue
            original_lane, lane, old_position, new_position, action = lane_updates[car_id]
            obs = observations.get(car_id, {})
            progress = max(0, new_position - old_position)
            merged = action.startswith("merge") and original_lane == "right" and lane == "left"
            courtesy = action == "yield" and obs.get("right_car_waiting")
            block = action == "block" and obs.get("right_car_waiting") and obs.get("lane") == "left"
            tailgating = obs.get("front_gap", 2) <= 1 and car.speed > 0
            brake_spam = car.brake_steps >= 2 and action == "brake"
            stalled = progress == 0
            if stalled:
                car.no_progress_steps += 1
            else:
                car.no_progress_steps = 0
            stalled = car.no_progress_steps >= 3
            if courtesy:
                self.courtesy_events += 1
            reward = compute_reward(
                config=self.reward_config,
                collision=car.collided,
                progress=progress,
                merged=merged,
                courtesy=courtesy,
                block=block,
                tailgating=tailgating,
                brake_spam=brake_spam,
                stalled=stalled,
                global_clear=global_clear,
            )
            rewards[car_id] = reward
            rewards_out[car_id] = reward.total

        done = self._all_done() or self.step_count >= self.max_steps
        info = {
            "collisions": len(collision_ids),
            "total_collisions": self.collision_count,
            "throughput": self._cleared_count(),
            "courtesy_events": self.courtesy_events,
            "global_clear": global_clear,
            "step": self.step_count,
            "reward_breakdown": {car_id: breakdown.as_dict() for car_id, breakdown in rewards.items()},
        }
        return self.state(), rewards_out, done, info

    def _cleared_count(self) -> int:
        return sum(1 for car in self.cars.values() if car.position >= self.lane_length and not car.collided)

    def _all_cleared(self) -> bool:
        return self._cleared_count() == len(self.cars)

    def _all_done(self) -> bool:
        return all(car.done for car in self.cars.values())

    def _merge_options(self, car: CarState) -> tuple[bool, bool]:
        if car.position > self.merge_point:
            return False, False
        if car.lane == "right":
            return self._lane_free("left", car.position), False
        return False, self._lane_free("right", car.position)

    def _lane_free(self, lane: str, position: int) -> bool:
        for other in self.cars.values():
            if other.done:
                continue
            if other.lane == lane and other.position == position:
                return False
        return True

    def _observe(self, car: CarState) -> dict[str, Any]:
        front_gap = self._front_gap(car)
        merge_distance = max(self.merge_point - car.position, 0)
        right_car_waiting = False
        if car.lane == "left":
            for other in self.cars.values():
                if other.done or other.lane != "right":
                    continue
                if other.position <= car.position and car.position - other.position <= 1:
                    if other.speed <= 1:
                        right_car_waiting = True
                        break
        cars_behind = sum(
            1
            for other in self.cars.values()
            if not other.done and other.lane == car.lane and other.position < car.position
        )
        can_merge_left, can_merge_right = self._merge_options(car)
        can_merge = can_merge_left or can_merge_right
        return {
            "lane": car.lane,
            "speed": car.speed,
            "front_gap": front_gap,
            "merge_distance": merge_distance,
            "right_car_waiting": right_car_waiting,
            "cars_behind": cars_behind,
            "can_merge": can_merge,
        }

    def _front_gap(self, car: CarState) -> int:
        gaps = []
        for other in self.cars.values():
            if other.done or other.car_id == car.car_id:
                continue
            same_lane = other.lane == car.lane
            if car.position < self.merge_point and other.position >= self.merge_point:
                same_lane = True
            if not same_lane:
                continue
            if other.position > car.position:
                gaps.append(other.position - car.position)
        return min(gaps) if gaps else self.lane_length - car.position
