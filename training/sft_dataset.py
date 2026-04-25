from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from policy.agent import ALLOWED_ACTIONS, parse_action
from policy.prompt_builder import build_llm_prompt

SCENARIOS = [
    "Highway Merge",
    "Emergency Vehicle Corridor",
    "Chain Reaction Avoidance",
]


def _random_vehicle_state(rng: random.Random) -> dict[str, Any]:
    lane = rng.choice(["left", "right"])
    speed = rng.randint(0, 3)
    front_gap = rng.randint(1, 6)
    merge_distance = rng.randint(0, 6)
    cars_behind = rng.randint(0, 3)
    right_car_waiting = lane == "left" and rng.choice([True, False])
    can_merge = lane == "right" and rng.choice([True, False])
    return {
        "lane": lane,
        "speed": speed,
        "front_gap": front_gap,
        "merge_distance": merge_distance,
        "cars_behind": cars_behind,
        "right_car_waiting": right_car_waiting,
        "can_merge": can_merge,
    }


def _random_mesh_broadcasts(rng: random.Random) -> dict[str, dict[str, str]]:
    broadcasts: dict[str, dict[str, str]] = {}
    for idx in range(rng.randint(1, 3)):
        agent_id = rng.choice([f"L{idx}", f"R{idx}", f"L{idx + 1}", f"R{idx + 1}"])
        position = rng.randint(0, 6)
        action = rng.choice(ALLOWED_ACTIONS)
        reasoning = (
            f"MESH BROADCAST id={agent_id} pos={position} intent={action} "
            f"gap={rng.randint(1, 4)}"
        )
        broadcasts[agent_id] = {"action": action, "reasoning": reasoning}
    return broadcasts


def _call_openai(prompt: str, model: str, api_key: str, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        status = exc.code
        if status in {429, 503}:
            message = "OpenAI API rate limit exceeded or service unavailable."
        elif status == 401:
            message = "OpenAI API authentication failed."
        else:
            message = f"OpenAI API error (status {status})."
        raise RuntimeError(message) from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI connection failed: {exc.reason}") from exc
    choices = body.get("choices", [])
    if not choices or "message" not in choices[0]:
        raise RuntimeError(f"Unexpected OpenAI response: {body}")
    return choices[0]["message"]["content"]


def _fallback_reasoning(vehicle_state: dict[str, Any], scenario: str) -> str:
    lane = vehicle_state.get("lane", "unknown")
    speed = vehicle_state.get("speed", 0)
    front_gap = vehicle_state.get("front_gap", "unknown")
    merge_distance = vehicle_state.get("merge_distance", "unknown")
    return (
        f"Maintain a {front_gap} gap in the {lane} lane at speed {speed} "
        f"while approaching merge distance {merge_distance} in {scenario}."
    )


def _requires_reformat(raw_response: str) -> bool:
    return "REASONING:" not in raw_response or "ACTION:" not in raw_response


def _generate_response(
    prompt: str,
    model: str,
    api_key: str,
    max_tokens: int,
    vehicle_state: dict[str, Any],
    scenario: str,
) -> str:
    raw_response = _call_openai(prompt, model, api_key, max_tokens=max_tokens)
    raw_response = raw_response.strip()
    action, parse_failure = parse_action(raw_response)
    if _requires_reformat(raw_response):
        parse_failure = True
    if parse_failure:
        fallback = _fallback_reasoning(vehicle_state, scenario)
        return f"REASONING: {fallback} ACTION: {action}"
    return raw_response


def generate_dataset(
    *,
    num_examples: int,
    output_path: Path,
    model: str,
    api_key: str,
    sleep_s: float,
    seed: int,
    max_tokens: int,
) -> None:
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for _ in range(num_examples):
            scenario = rng.choice(SCENARIOS)
            vehicle_state = _random_vehicle_state(rng)
            mesh_broadcasts = _random_mesh_broadcasts(rng)
            prompt = build_llm_prompt(vehicle_state, mesh_broadcasts, scenario)
            response = _generate_response(
                prompt, model, api_key, max_tokens, vehicle_state, scenario
            )
            record = {"prompt": prompt, "response": response}
            handle.write(json.dumps(record) + "\n")
            if sleep_s > 0:
                time.sleep(sleep_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SFT dataset for MergeMind.")
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/sft_dataset.jsonl")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--sleep-s", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable {args.api_key_env}.")
    generate_dataset(
        num_examples=args.num_examples,
        output_path=Path(args.output),
        model=args.model,
        api_key=api_key,
        sleep_s=args.sleep_s,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
