from __future__ import annotations

from typing import Any

from .agent import ALLOWED_ACTIONS

SYSTEM_PROMPT = (
    "You are an autonomous vehicle agent coordinating a safe, efficient merge."
)


def build_llm_prompt(
    vehicle_state: dict[str, Any],
    mesh_broadcasts: dict[str, dict[str, str]],
    scenario: str,
) -> str:
    lane = vehicle_state.get("lane", "unknown")
    speed = vehicle_state.get("speed", 0)
    front_gap = vehicle_state.get("front_gap", "unknown")
    merge_distance = vehicle_state.get("merge_distance", "unknown")
    cars_behind = vehicle_state.get("cars_behind", "unknown")
    right_car_waiting = vehicle_state.get("right_car_waiting", False)
    can_merge = vehicle_state.get("can_merge", False)

    lines = [
        SYSTEM_PROMPT,
        f"Scenario: {scenario}",
        "Vehicle State:",
        f"- Lane: {lane}",
        f"- Speed: {speed}",
        f"- Front gap: {front_gap}",
        f"- Merge distance: {merge_distance}",
        f"- Cars behind: {cars_behind}",
        f"- Right car waiting: {right_car_waiting}",
        f"- Can merge: {can_merge}",
    ]

    if mesh_broadcasts:
        lines.append("Upstream broadcasts:")
        for agent_id, broadcast in mesh_broadcasts.items():
            action = broadcast.get("action", "")
            reasoning = broadcast.get("reasoning", "")
            lines.append(f"- {agent_id}: ACTION={action} REASONING={reasoning}")
    else:
        lines.append("Upstream broadcasts: none")

    action_hint = ", ".join(ALLOWED_ACTIONS)
    lines.append(f"Allowed actions: {action_hint}")
    lines.append("Format: REASONING:  ACTION: ")
    return "\n".join(lines)
