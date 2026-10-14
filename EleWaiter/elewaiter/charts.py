"""Plotly charts used by the Streamlit UI."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def comparison_bar(df: pd.DataFrame, metric: str, title: str, ylabel: str = "seconds"):
    d = df.dropna(subset=[metric])
    fig = px.bar(d, x="algorithm", y=metric, color="algorithm", title=title, text_auto=".1f")
    fig.update_layout(showlegend=False, yaxis_title=ylabel, xaxis_title="", height=360)
    return fig


def wait_histogram(results, em: bool = False, labels: dict[str, str] | None = None):
    fig = go.Figure()
    labels = labels or {}
    for r in results:
        vals = r.em_wait_times if em else r.wait_times
        if vals:
            fig.add_trace(go.Histogram(x=vals, name=labels.get(r.algorithm, r.algorithm),
                                       opacity=0.6, xbins=dict(size=2)))
    fig.update_layout(barmode="overlay", height=340, xaxis_title="waiting time (s)",
                      yaxis_title="passengers", title="Emergency passenger waits" if em else "Waiting-time distribution")
    return fig


def lift_activity(timeline: list, title: str = "Lift activity (door openings)"):
    if not timeline:
        return go.Figure()
    d = pd.DataFrame(timeline)
    d["lift"] = "Lift " + d["lift"].astype(str)
    fig = px.scatter(d, x="time", y="floor", color="lift", size=(d["load"] + 2), title=title,
                     symbol=d["lane"].map({True: "emergency lane", False: "normal"}))
    fig.update_layout(height=380, xaxis_title="time (s)", yaxis_title="floor")
    return fig


def training_curve(histories: dict):
    fig = go.Figure()
    for name, h in histories.items():
        fig.add_trace(go.Scatter(y=h, mode="lines+markers", name=name))
    fig.update_layout(height=340, xaxis_title="generation / iteration", yaxis_title="best fitness (lower is better)",
                      title="Optimizer progress on the fixed training set")
    return fig
