from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable

try:  # pragma: no cover - optional dependency
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from policy.agent import parse_action
from policy.prompt_builder import build_llm_prompt

from .mesh import MeshLayer
from .rewards import RewardConfig, RewardBreakdown, compute_reward
from .utils import ACTIONS, seed_everything

LlmPolicy = Callable[[str], str]


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
    last_action: str = "MAINTAIN"


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
        llm_policy: LlmPolicy | None = None,
        scenario: str | None = None,
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
        self.altruism_events = 0
        self.pending_altruism: dict[str, set[str]] = {}
        self.parse_failure_total = 0
        self.parse_failure_steps = 0
        self.mesh_layer = MeshLayer()
        self.llm_policy = llm_policy or self._default_llm_policy
        self.scenario = scenario or "Two lanes merge into a single bottleneck."

    def seed(self, seed: int | None) -> None:
        self.rng = seed_everything(seed)

    def reset(self) -> dict[str, dict]:
        self.step_count = 0
        self.collision_count = 0
        self.altruism_events = 0
        self.pending_altruism = {}
        self.parse_failure_total = 0
        self.parse_failure_steps = 0
        self.mesh_layer.reset()
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

    def step(
        self,
        actions: dict[str, str] | None = None,
    ) -> tuple[dict[str, dict], dict[str, float], bool, dict]:
        self.step_count += 1
        observations = {car_id: self._observe(car) for car_id, car in self.cars.items() if not car.done}
        rewards: dict[str, RewardBreakdown] = {}
        lane_updates: dict[str, tuple[str, str, int, int, str]] = {}
        collision_ids: set[str] = set()
        reasoning_by_car_id: dict[str, str] = {}
        parse_failures: dict[str, bool] = {}

        if actions is None:
            actions = {}
            self.mesh_layer.reset()
            active_cars = {car_id: car for car_id, car in self.cars.items() if not car.done}
            resolution_order = self.mesh_layer.get_resolution_order(active_cars, self.merge_point)
            for car_id in resolution_order:
                observation = observations.get(car_id, {})
                prompt = build_llm_prompt(observation, self.mesh_layer.broadcast_dict, self.scenario)
                response = self.llm_policy(prompt)
                action, reasoning, parse_failure = self._extract_action_and_reasoning(response)
                if action not in ACTIONS:
                    action = "MAINTAIN"
                actions[car_id] = action
                self.mesh_layer.update_broadcast(car_id, action, reasoning)
                reasoning_by_car_id[car_id] = reasoning
                parse_failures[car_id] = parse_failure
        else:
            for car_id in observations:
                reasoning_by_car_id[car_id] = ""
                parse_failures[car_id] = False

        if parse_failures:
            parse_failure_count = sum(1 for failed in parse_failures.values() if failed)
            parse_failure_rate = parse_failure_count / max(1, len(parse_failures))
            self.parse_failure_total += parse_failure_count
            self.parse_failure_steps += len(parse_failures)
            if wandb and getattr(wandb, "run", None):
                wandb.log(
                    {
                        "parse_failure_rate": parse_failure_rate,
                        "parse_failure_count": parse_failure_count,
                        "step": self.step_count,
                    }
                )
        else:
            parse_failure_count = 0
            parse_failure_rate = 0.0

        for car_id, car in self.cars.items():
            if car.done:
                continue
            action = actions.get(car_id, "MAINTAIN")
            if action not in ACTIONS:
                action = "MAINTAIN"
            original_position = car.position
            original_lane = car.lane
            obs = observations.get(car_id, {})
            if action == "ACCELERATE":
                car.speed = min(self.max_speed, car.speed + 1)
            elif action == "DECELERATE":
                car.speed = max(0, car.speed - 1)
            elif action == "EMERGENCY_BRAKE":
                car.speed = max(0, car.speed - 2)
            if action in {"DECELERATE", "EMERGENCY_BRAKE"}:
                car.brake_steps += 1
            else:
                car.brake_steps = 0

            if (
                action in {"DECELERATE", "EMERGENCY_BRAKE"}
                and obs.get("right_car_waiting")
                and obs.get("lane") == "left"
            ):
                beneficiary_id = self._find_altruism_beneficiary(car)
                if beneficiary_id:
                    self.pending_altruism.setdefault(car_id, set()).add(beneficiary_id)

            can_merge_left, can_merge_right = self._merge_options(car)
            if action == "LANE_LEFT" and can_merge_left:
                car.lane = "left"
            elif action == "LANE_RIGHT" and can_merge_right:
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
        active_speeds = [car.speed for car in self.cars.values() if not car.done]
        avg_speed = float(sum(active_speeds) / len(active_speeds)) if active_speeds else 0.0
        rewards_out: dict[str, float] = {}
        for car_id, car in self.cars.items():
            if car_id not in lane_updates:
                continue
            original_lane, lane, old_position, new_position, action = lane_updates[car_id]
            obs = observations.get(car_id, {})
            front_gap = obs.get("front_gap", self.lane_length - car.position)
            decel_max = max(self.reward_config.deceleration_max, 0.1)
            safe_distance = (car.speed**2) / (2 * decel_max)
            unsafe_gap = front_gap < safe_distance
            survived = new_position >= self.lane_length and not car.collided
            altruism = self._resolve_altruism_bonus(car_id)
            if altruism:
                self.altruism_events += 1
            reasoning_quality = self._reasoning_quality(
                reasoning_by_car_id.get(car_id, ""), car_id
            )
            parse_failure = parse_failures.get(car_id, False)
            reward = compute_reward(
                config=self.reward_config,
                collision=car.collided,
                unsafe_gap=unsafe_gap,
                survived=survived,
                avg_speed=avg_speed,
                max_speed=self.max_speed,
                altruism=altruism,
                reasoning_quality=reasoning_quality,
                parse_failure=parse_failure,
            )
            rewards[car_id] = reward
            rewards_out[car_id] = reward.total

        done = self._all_done() or self.step_count >= self.max_steps
        info = {
            "collisions": len(collision_ids),
            "total_collisions": self.collision_count,
            "throughput": self._cleared_count(),
            "altruism_events": self.altruism_events,
            "courtesy_events": self.altruism_events,
            "global_clear": global_clear,
            "avg_speed": avg_speed,
            "step": self.step_count,
            "reward_breakdown": {car_id: breakdown.as_dict() for car_id, breakdown in rewards.items()},
            "parse_failure_rate": parse_failure_rate,
            "parse_failure_count": parse_failure_count,
        }
        return self.state(), rewards_out, done, info

    def _default_llm_policy(self, _prompt: str) -> str:
        return "REASONING: Maintain safe spacing. ACTION: MAINTAIN"

    def _extract_action_and_reasoning(self, response: str) -> tuple[str, str, bool]:
        action, parse_failure = parse_action(response)
        if not response:
            return action, "", parse_failure
        response_text = response.strip()
        response_lower = response_text.lower()
        reasoning = ""
        reasoning_token = "reasoning:"
        action_token = "action:"
        if reasoning_token in response_lower:
            start = response_lower.index(reasoning_token) + len(reasoning_token)
            reasoning_section = response_text[start:]
            reasoning_lower = reasoning_section.lower()
            if action_token in reasoning_lower:
                cut = reasoning_lower.index(action_token)
                reasoning = reasoning_section[:cut].strip()
            else:
                reasoning = reasoning_section.strip()
        return action, reasoning, parse_failure

    def _find_altruism_beneficiary(self, car: CarState) -> str | None:
        candidates = [
            other
            for other in self.cars.values()
            if not other.done and other.lane == "right" and other.position <= car.position
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda other: other.position, reverse=True)
        return candidates[0].car_id

    def _resolve_altruism_bonus(self, car_id: str) -> bool:
        pending = self.pending_altruism.get(car_id)
        if not pending:
            return False
        awarded = False
        for beneficiary_id in list(pending):
            beneficiary = self.cars.get(beneficiary_id)
            if beneficiary is None:
                pending.discard(beneficiary_id)
                continue
            if beneficiary.done:
                if not beneficiary.collided:
                    awarded = True
                pending.discard(beneficiary_id)
        if not pending:
            self.pending_altruism.pop(car_id, None)
        return awarded

    def _reasoning_quality(self, reasoning: str, car_id: str) -> bool:
        if not reasoning:
            return False
        words = re.findall(r"\b\w+\b", reasoning)
        if len(words) <= 20:
            return False
        reasoning_lower = reasoning.lower()
        if "mesh" in reasoning_lower or "broadcast" in reasoning_lower:
            return True
        for other_id in self.cars:
            if other_id == car_id:
                continue
            if other_id.lower() in reasoning_lower:
                return True
        return False

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
