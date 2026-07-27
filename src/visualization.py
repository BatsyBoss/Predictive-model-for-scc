"""
visualization.py
------------------
All Plotly chart construction lives here so app.py stays focused on layout
and control flow. Every function returns a `go.Figure` — nothing is rendered
directly, so the same figures can be st.plotly_chart()'d AND exported to PNG
for the download buttons.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import FEATURE_LABELS

TEMPLATE = "plotly_white"
COLOR_SEQUENCE = ["#2563EB", "#F97316", "#10B981", "#8B5CF6", "#EF4444"]


# ---------------------------------------------------------------------------
# USER MODE charts
# ---------------------------------------------------------------------------
def plot_mix_comparison_bar(designs: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing the generated mix designs. Superplasticizer
    is plotted on a secondary row since it's an order of magnitude smaller
    than the other components and would otherwise be invisible.
    """
    big_components = ["Cement", "Water", "Fine_Aggregate", "Coarse_Aggregate"]
    small_components = ["Superplasticizer"]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.75, 0.25],
        subplot_titles=("Mix Components (kg/m³)", "Superplasticizer (kg/m³)"),
    )

    for i, design_type in enumerate(designs["Design_Type"]):
        row = designs[designs["Design_Type"] == design_type].iloc[0]
        color = COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]

        fig.add_trace(
            go.Bar(
                name=design_type,
                x=[FEATURE_LABELS[c].split(" (")[0] for c in big_components],
                y=[row[c] for c in big_components],
                marker_color=color,
                legendgroup=design_type,
                showlegend=True,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                name=design_type,
                x=["Superplasticizer"],
                y=[row[c] for c in small_components],
                marker_color=color,
                legendgroup=design_type,
                showlegend=False,
            ),
            row=1, col=2,
        )

    fig.update_layout(
        barmode="group",
        template=TEMPLATE,
        title="Generated Mix Designs — Component Comparison",
        legend_title="Design",
        height=450,
    )
    return fig


def plot_predicted_vs_target(designs: pd.DataFrame, target_strength: float) -> go.Figure:
    """Bar chart of each design's predicted strength vs. the target line."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=designs["Design_Type"],
        y=designs["Predicted_Strength"],
        marker_color=[COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i in range(len(designs))],
        text=designs["Predicted_Strength"].astype(str) + " MPa",
        textposition="outside",
        name="Predicted Strength",
    ))
    fig.add_hline(
        y=target_strength,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Target: {target_strength} MPa",
        annotation_position="top left",
    )
    fig.update_layout(
        template=TEMPLATE,
        title="Predicted vs. Target Compressive Strength",
        yaxis_title="Compressive Strength (MPa)",
        xaxis_title="Design",
        height=420,
        showlegend=False,
    )
    return fig


def plot_strength_gauge(predicted: float, low: float = 15, high: float = 80) -> go.Figure:
    """Single-value gauge for the Predict Strength mode."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted,
        number={"suffix": " MPa"},
        gauge={
            "axis": {"range": [low, high]},
            "bar": {"color": "#2563EB"},
            "steps": [
                {"range": [low, 30], "color": "#FEE2E2"},
                {"range": [30, 50], "color": "#FEF3C7"},
                {"range": [50, high], "color": "#D1FAE5"},
            ],
        },
        title={"text": "Predicted Compressive Strength"},
    ))
    fig.update_layout(template=TEMPLATE, height=320)
    return fig


# ---------------------------------------------------------------------------
# ADMIN MODE charts
# ---------------------------------------------------------------------------
def plot_actual_vs_predicted(y_test: pd.Series, y_pred: np.ndarray) -> go.Figure:
    """Scatter of actual vs. predicted strength on the held-out test set."""
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    lo = float(min(y_test.min(), y_pred.min()))
    hi = float(max(y_test.max(), y_pred.max()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode="markers",
        marker=dict(color="#2563EB", size=8, opacity=0.65),
        name="Test samples",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="#EF4444", dash="dash"),
        name="Perfect prediction (y = x)",
    ))
    fig.update_layout(
        template=TEMPLATE,
        title="Actual vs. Predicted Compressive Strength (Test Set)",
        xaxis_title="Actual Strength (MPa)",
        yaxis_title="Predicted Strength (MPa)",
        height=450,
    )
    return fig


def plot_feature_importance(fi_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of model feature importances."""
    df = fi_df.sort_values("Importance", ascending=True)
    labels = [FEATURE_LABELS.get(f, f) for f in df["Feature"]]

    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=labels,
        orientation="h",
        marker_color="#8B5CF6",
    ))
    fig.update_layout(
        template=TEMPLATE,
        title="Feature Importance",
        xaxis_title="Importance",
        height=400,
    )
    return fig


def plot_distribution(df: pd.DataFrame, column: str, title: str, color: str = "#2563EB") -> go.Figure:
    """Generic histogram used for strength / cement content distributions."""
    fig = go.Figure(go.Histogram(
        x=df[column],
        marker_color=color,
        nbinsx=30,
    ))
    fig.update_layout(
        template=TEMPLATE,
        title=title,
        xaxis_title=FEATURE_LABELS.get(column, column),
        yaxis_title="Count",
        height=380,
        bargap=0.02,
    )
    return fig


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------
def fig_to_png_bytes(fig: go.Figure) -> bytes | None:
    """
    Convert a Plotly figure to PNG bytes for st.download_button.
    Returns None (rather than raising) if the Kaleido export engine /
    headless Chrome isn't available in the current environment, so the UI
    can show a clear warning instead of crashing.
    """
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None
