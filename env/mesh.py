from __future__ import annotations

from typing import Iterable


class MeshLayer:
    def __init__(self) -> None:
        self.broadcast_dict: dict[str, dict[str, str]] = {}

    def reset(self) -> None:
        self.broadcast_dict.clear()

    def update_broadcast(self, agent_id: str, action: str, reasoning: str) -> None:
        self.broadcast_dict[agent_id] = {"action": action, "reasoning": reasoning}

    def get_resolution_order(self, agents: dict[str, object] | Iterable[object], hazard_pos: int) -> list[str]:
        if isinstance(agents, dict):
            items = list(agents.items())
        else:
            items = [(agent.car_id, agent) for agent in agents]  # type: ignore[attr-defined]
        items.sort(key=lambda item: abs(getattr(item[1], "position", 0) - hazard_pos))
        return [agent_id for agent_id, _agent in items]
