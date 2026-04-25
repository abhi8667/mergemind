from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    collision: float = -10.0
    progress: float = 1.0
    successful_merge: float = 2.0
    courtesy: float = 3.0
    block_penalty: float = -2.0
    tailgating_penalty: float = -1.0
    brake_spam_penalty: float = -1.0
    stall_penalty: float = -0.5
    global_clear_bonus: float = 5.0


@dataclass
class RewardBreakdown:
    collision: float = 0.0
    progress: float = 0.0
    successful_merge: float = 0.0
    courtesy: float = 0.0
    block_penalty: float = 0.0
    tailgating_penalty: float = 0.0
    brake_spam_penalty: float = 0.0
    stall_penalty: float = 0.0
    global_clear_bonus: float = 0.0
    total: float = field(init=False, default=0.0)

    def finalize(self) -> float:
        self.total = sum(
            [
                self.collision,
                self.progress,
                self.successful_merge,
                self.courtesy,
                self.block_penalty,
                self.tailgating_penalty,
                self.brake_spam_penalty,
                self.stall_penalty,
                self.global_clear_bonus,
            ]
        )
        return self.total

    def as_dict(self) -> dict[str, float]:
        self.finalize()
        return {
            "collision": self.collision,
            "progress": self.progress,
            "successful_merge": self.successful_merge,
            "courtesy": self.courtesy,
            "block_penalty": self.block_penalty,
            "tailgating_penalty": self.tailgating_penalty,
            "brake_spam_penalty": self.brake_spam_penalty,
            "stall_penalty": self.stall_penalty,
            "global_clear_bonus": self.global_clear_bonus,
            "total": self.total,
        }


def compute_reward(
    *,
    config: RewardConfig,
    collision: bool,
    progress: int,
    merged: bool,
    courtesy: bool,
    block: bool,
    tailgating: bool,
    brake_spam: bool,
    stalled: bool,
    global_clear: bool,
) -> RewardBreakdown:
    reward = RewardBreakdown()
    if collision:
        reward.collision = config.collision
    reward.progress = min(progress, 1) * config.progress
    if merged:
        reward.successful_merge = config.successful_merge
    if courtesy:
        reward.courtesy = config.courtesy
    if block:
        reward.block_penalty = config.block_penalty
    if tailgating:
        reward.tailgating_penalty = config.tailgating_penalty
    if brake_spam:
        reward.brake_spam_penalty = config.brake_spam_penalty
    if stalled:
        reward.stall_penalty = config.stall_penalty
    if global_clear:
        reward.global_clear_bonus = config.global_clear_bonus
    reward.finalize()
    return reward
