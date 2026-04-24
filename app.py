from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from env.mergemind_env import MergeMindEnv
from env.utils import load_qtable, qtable_policy, random_policy, rule_based_policy
from evaluate import run_policy
from ui.dashboard import build_comparison_chart, build_metric_cards
from ui.replay import run_episode


def get_policy(policy_name: str):
    if policy_name == "Random":
        return random_policy
    if policy_name == "Rule-based":
        return rule_based_policy
    policy_path = Path("outputs/policy.json")
    if policy_path.exists():
        qtable = load_qtable(str(policy_path))
        return lambda obs: qtable_policy(obs, qtable)
    return rule_based_policy


def render_replay(policy_name: str) -> tuple[str, str]:
    env = MergeMindEnv()
    policy = get_policy(policy_name)
    result = run_episode(env, policy, max_steps=env.max_steps)
    frames_text = "\n\n".join(result.frames[-10:])
    metrics_text = build_metric_cards(result.metrics)
    return frames_text, metrics_text


def compare_policies() -> tuple[gr.Plot, str]:
    env = MergeMindEnv()
    results = {
        "random": run_policy(env, random_policy, episodes=10),
        "rule_based": run_policy(env, rule_based_policy, episodes=10),
    }
    policy_path = Path("outputs/policy.json")
    if policy_path.exists():
        qtable = load_qtable(str(policy_path))
        results["trained"] = run_policy(
            env, lambda obs: qtable_policy(obs, qtable), episodes=10
        )
    chart = build_comparison_chart(results)
    table = json.dumps(results, indent=2)
    return chart, table


def load_about() -> str:
    return (
        "### MergeMind 🚗🚙🚧\n"
        "MergeMind is a semantic multi-agent benchmark where LLM agents learn cooperative "
        "zipper merges on a two-lane highway bottleneck. The goal is safe, courteous, "
        "and high-throughput merges with minimal collisions.\n\n"
        "- **Environment**: Two lanes merge into one with explicit merge distance.\n"
        "- **Agents**: Each car chooses actions like merge, yield, or block.\n"
        "- **Rewards**: Safety + efficiency + courtesy with anti-toxicity penalties.\n"
    )


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# MergeMind: Multi-Agent Traffic Merge Benchmark")
    gr.Markdown("LLM-driven cooperative behavior for zipper merges.")

    with gr.Tab("Live Replay"):
        policy_choice = gr.Dropdown(
            ["Random", "Rule-based", "Trained"], value="Rule-based", label="Policy"
        )
        replay_button = gr.Button("Run Episode")
        replay_output = gr.Textbox(label="Replay (last 10 steps)", lines=12)
        replay_metrics = gr.Markdown()
        replay_button.click(render_replay, inputs=[policy_choice], outputs=[replay_output, replay_metrics])

    with gr.Tab("Before vs After Training"):
        compare_button = gr.Button("Compare Policies")
        compare_chart = gr.Plot()
        compare_table = gr.Textbox(label="Metrics JSON", lines=10)
        compare_button.click(compare_policies, outputs=[compare_chart, compare_table])

    with gr.Tab("Metrics Dashboard"):
        gr.Markdown("Load the latest evaluation results from `outputs/eval_results.json`.")
        dashboard_output = gr.Textbox(label="Evaluation Results", lines=12)

        def load_dashboard() -> str:
            path = Path("outputs/eval_results.json")
            if not path.exists():
                return "Run `python evaluate.py` to generate metrics."
            return path.read_text(encoding="utf-8")

        gr.Button("Refresh Metrics").click(load_dashboard, outputs=[dashboard_output])

    with gr.Tab("About MergeMind"):
        gr.Markdown(load_about())


if __name__ == "__main__":
    demo.launch()
