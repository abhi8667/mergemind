from __future__ import annotations

import re

ALLOWED_ACTIONS: list[str] = [
    "ACCELERATE",
    "DECELERATE",
    "LANE_LEFT",
    "LANE_RIGHT",
    "MAINTAIN",
    "EMERGENCY_BRAKE",
]

ACTION_REGEX = re.compile(r"ACTION:\s*([A-Z_]+)")


def parse_action(raw_text: str | None) -> tuple[str, bool]:
    if not raw_text:
        return "MAINTAIN", True
    match = ACTION_REGEX.search(raw_text)
    if match:
        candidate = match.group(1).strip()
        if candidate in ALLOWED_ACTIONS:
            return candidate, False
    upper_text = raw_text.upper()
    for action in ALLOWED_ACTIONS:
        if action in upper_text:
            return action, True
    return "MAINTAIN", True
