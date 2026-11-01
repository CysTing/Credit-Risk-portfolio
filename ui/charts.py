"""Consistent, accessible Plotly figures; all axes retain their units."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TEAL = "#087F8C"
NAVY = "#234765"
AMBER = "#B7751D"
CORAL = "#BD4856"
PALETTE = [TEAL, NAVY, AMBER, CORAL]


def polish(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=20, t=25, b=10),
        font=dict(family="Arial, sans-serif", color="#36516b", size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE, legend=dict(orientation="h", y=1.15, x=0),
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#E7EDF3", zeroline=False)
    return fig


def probability_histogram(y: pd.Series, probabilities: np.ndarray, threshold: float) -> go.Figure:
    fig = go.Figure()
    for value, name, color in [(0, "Observed: no default", TEAL), (1, "Observed: default", CORAL)]:
        fig.add_histogram(x=probabilities[np.asarray(y) == value], name=name,
                         xbins=dict(start=0, end=1, size=.05), marker_color=color, opacity=.7)
    fig.add_vline(x=threshold, line_dash="dash", line_color=AMBER,
                  annotation_text="Teaching threshold", annotation_position="top right")
    fig.update_layout(barmode="overlay", xaxis_title="Calibrated estimated probability",
                      yaxis_title="Test observations")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    return polish(fig)


def binary_matrix(y: pd.Series, probabilities: np.ndarray, threshold: float) -> go.Figure:
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y, probabilities >= threshold, labels=[0, 1])
    fig = go.Figure(go.Heatmap(
        z=matrix, x=["Predicted: no default", "Predicted: default"],
        y=["Observed: no default", "Observed: default"],
        text=matrix, texttemplate="%{text:,}", colorscale=[[0, "#edf7f7"], [1, TEAL]],
        showscale=False, hovertemplate="%{y}<br>%{x}<br>Count: %{z:,}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return polish(fig, 300)


def cohort_rates(data: pd.DataFrame) -> go.Figure:
    grouped = data.groupby("PAY_0")["default"].agg(["mean", "size"]).reset_index()
    fig = go.Figure(go.Bar(
        x=grouped["PAY_0"].astype(str), y=grouped["mean"], marker_color=TEAL,
        customdata=grouped[["size"]],
        hovertemplate="Status code %{x}<br>Observed default: %{y:.1%}<br>n=%{customdata[0]:,}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="September repayment status code", yaxis_title="Observed next-month default rate")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return polish(fig)


def horizontal_effects(data: pd.DataFrame, value: str, title: str) -> go.Figure:
    view = data.reindex(data[value].abs().sort_values(ascending=False).index).head(10)
    view = view.sort_values(value)
    fig = go.Figure(go.Bar(
        x=view[value], y=view["feature"], orientation="h",
        marker_color=[CORAL if x > 0 else TEAL for x in view[value]],
        hovertemplate="%{y}<br>%{x:.3f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title=title, yaxis_title=None)
    return polish(fig, 390)
