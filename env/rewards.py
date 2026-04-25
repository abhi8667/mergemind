from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardWeights:
    survival: float = 0.4
    throughput: float = 0.4
    altruism: float = 0.15
    reasoning: float = 0.05


@dataclass
class RewardConfig:
    weights: RewardWeights = field(default_factory=RewardWeights)
    collision_penalty: float = -1.0
    unsafe_gap_penalty: float = -0.5
    survival_bonus: float = 1.0
    parse_failure_penalty: float = -0.1
    deceleration_max: float = 2.0


@dataclass
class RewardBreakdown:
    survival: float = 0.0
    throughput: float = 0.0
    altruism: float = 0.0
    reasoning_quality: float = 0.0
    parse_failure_penalty: float = 0.0
    total: float = field(init=False, default=0.0)

    def finalize(self) -> float:
        self.total = sum(
            [
                self.survival,
                self.throughput,
                self.altruism,
                self.reasoning_quality,
                self.parse_failure_penalty,
            ]
        )
        return self.total

    def as_dict(self) -> dict[str, float]:
        self.finalize()
        return {
            "survival": self.survival,
            "throughput": self.throughput,
            "altruism": self.altruism,
            "reasoning_quality": self.reasoning_quality,
            "parse_failure_penalty": self.parse_failure_penalty,
            "total": self.total,
        }


def compute_reward(
    *,
    config: RewardConfig,
    collision: bool,
    unsafe_gap: bool,
    survived: bool,
    avg_speed: float,
    max_speed: int,
    altruism: bool,
    reasoning_quality: bool,
    parse_failure: bool,
) -> RewardBreakdown:
    reward = RewardBreakdown()
    survival_metric = 0.0
    if collision:
        survival_metric += config.collision_penalty
    if unsafe_gap:
        survival_metric += config.unsafe_gap_penalty
    if survived:
        survival_metric += config.survival_bonus
    reward.survival = config.weights.survival * survival_metric
    throughput_metric = avg_speed / max_speed if max_speed > 0 else 0.0
    reward.throughput = config.weights.throughput * throughput_metric
    reward.altruism = config.weights.altruism if altruism else 0.0
    reward.reasoning_quality = config.weights.reasoning if reasoning_quality else 0.0
    reward.parse_failure_penalty = (
        config.parse_failure_penalty if parse_failure else 0.0
    )
    reward.finalize()
    return reward
