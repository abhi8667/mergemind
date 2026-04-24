from __future__ import annotations

import plotly.graph_objects as go


def build_metric_cards(metrics: dict[str, float]) -> str:
    return (
        f"**Reward:** {metrics.get('reward', 0):.2f}  \n"
        f"**Collisions:** {metrics.get('collisions', 0)}  \n"
        f"**Throughput:** {metrics.get('throughput', 0)}  \n"
        f"**Courtesy Events:** {metrics.get('courtesy_events', 0)}  \n"
        f"**Steps:** {metrics.get('steps', 0)}"
    )


def build_comparison_chart(results: dict[str, dict[str, float]]) -> go.Figure:
    labels = list(results.keys())
    reward_scores = [results[label]["reward_score"] for label in labels]
    collision_scores = [results[label]["collision_rate"] for label in labels]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Reward", x=labels, y=reward_scores, marker_color="#34d399"))
    fig.add_trace(
        go.Bar(
            name="Collision Rate",
            x=labels,
            y=collision_scores,
            marker_color="#f87171",
        )
    )
    fig.update_layout(
        barmode="group",
        title="MergeMind Policy Comparison",
        yaxis_title="Score",
        template="plotly_white",
    )
    return fig
