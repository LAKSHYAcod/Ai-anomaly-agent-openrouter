"""Charts contrasting the raw dataset with the anomaly-free dataset.

Outputs
-------
* ``raw_vs_clean_timeseries.png``  - one panel per metric: raw line, cleaned
  line, and the flagged points highlighted (static, high-resolution).
* ``anomaly_summary.png``          - anomalies per metric, split by severity.
* ``raw_vs_clean_interactive.html``- the same time-series as a zoomable,
  hoverable Plotly page.

Matplotlib's object-oriented ``Figure`` API is used (not ``pyplot``) so
rendering is thread-safe and needs no display backend - important when the
code runs under Streamlit, cron or Docker.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure

from src.analysis.anomaly_flagger import AnomalyFlags

logger = logging.getLogger(__name__)

# Colour-blind-safe palette (Okabe-Ito)
RAW_COLOUR = "#0072B2"
CLEAN_COLOUR = "#009E73"
FLAG_COLOUR = "#D55E00"
SEVERITY_COLOURS = {"Warning": "#E69F00", "Critical": "#D55E00"}


class VisualisationError(RuntimeError):
    """Raised when a chart cannot be produced."""


def _prepare(raw: pd.DataFrame, clean: pd.DataFrame, flags: AnomalyFlags):
    raw_dates = pd.to_datetime(raw["Date"], errors="coerce")
    clean_dates = pd.to_datetime(clean["Date"], errors="coerce")
    if raw_dates.isna().all():
        raise VisualisationError("The 'Date' column could not be parsed; cannot plot a time series.")
    return raw_dates, clean_dates


def _save(fig: Figure, path: Path, dpi: int) -> Path:
    FigureCanvasAgg(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


def plot_raw_vs_clean(
    raw: pd.DataFrame, clean: pd.DataFrame, flags: AnomalyFlags, out_path: Path, dpi: int = 300
) -> Path:
    """Static small-multiples chart (PNG). Returns the written path."""
    raw_dates, clean_dates = _prepare(raw, clean, flags)
    metrics = flags.metrics
    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig = Figure(figsize=(14, 3.4 * nrows), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel()

    for ax, col in zip(axes, metrics):
        ax.plot(raw_dates, raw[col], color=RAW_COLOUR, lw=1.6, alpha=0.55, label="Raw data")
        ax.plot(clean_dates, clean[col], color=CLEAN_COLOUR, lw=2.0, ls="--", label="Anomaly-free data")
        hit = flags.mask[col].to_numpy()
        if hit.any():
            ax.scatter(raw_dates[hit], raw.loc[hit, col], s=70, color=FLAG_COLOUR, marker="X",
                       zorder=5, edgecolor="white", linewidth=0.8, label="Flagged anomaly")
        ax.set_title(col, fontsize=12, fontweight="bold", loc="left")
        ax.set_ylabel(col)
        ax.set_xlabel("Date")
        loc = AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    # Legend must list the flagged marker even if the first metric has none.
    if "Flagged anomaly" not in labels:
        for ax in axes[: len(metrics)]:
            h, l = ax.get_legend_handles_labels()
            if "Flagged anomaly" in l:
                handles.append(h[l.index("Flagged anomaly")])
                labels.append("Flagged anomaly")
                break
    fig.legend(handles, labels, loc="outside upper right", ncols=3, frameon=False)
    fig.suptitle("Raw vs anomaly-free business metrics", fontsize=15, fontweight="bold", x=0.01, ha="left")
    return _save(fig, out_path, dpi)


def plot_anomaly_summary(flags: AnomalyFlags, out_path: Path, dpi: int = 300) -> Path:
    """Bar chart of flagged values per metric, stacked by severity (PNG)."""
    fig = Figure(figsize=(9, 5), layout="constrained")
    ax = fig.subplots()
    if flags.details.empty:
        ax.text(0.5, 0.5, "No anomalies detected", ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_axis_off()
    else:
        counts = (
            flags.details.groupby(["Metric", "Severity"]).size().unstack(fill_value=0)
            .reindex(index=flags.metrics, columns=["Warning", "Critical"], fill_value=0)
        )
        bottom = pd.Series(0, index=counts.index)
        for sev in counts.columns:
            ax.bar(counts.index, counts[sev], bottom=bottom, label=sev, color=SEVERITY_COLOURS[sev])
            bottom += counts[sev]
        ax.set_ylim(0, bottom.max() * 1.25 + 0.5)  # headroom above the tallest bar
        ax.set_ylabel("Number of flagged values")
        ax.set_xlabel("Metric")
        ax.set_title("Anomalies detected per metric", fontweight="bold", loc="left")
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.legend(title="Severity", frameon=False)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out_path, dpi)


def plot_interactive_html(
    raw: pd.DataFrame, clean: pd.DataFrame, flags: AnomalyFlags, out_path: Path,
    plotly_js: str | bool = "cdn",
) -> Path:
    """Interactive Plotly page. ``plotly_js=True`` embeds the library for fully offline use."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    raw_dates, clean_dates = _prepare(raw, clean, flags)
    metrics = flags.metrics
    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=metrics, vertical_spacing=0.10)

    for i, col in enumerate(metrics):
        r, c = divmod(i, ncols)
        first = i == 0
        fig.add_trace(go.Scatter(x=raw_dates, y=raw[col], name="Raw data", mode="lines",
                                 line=dict(color=RAW_COLOUR, width=2), opacity=0.6,
                                 legendgroup="raw", showlegend=first), row=r + 1, col=c + 1)
        fig.add_trace(go.Scatter(x=clean_dates, y=clean[col], name="Anomaly-free data", mode="lines",
                                 line=dict(color=CLEAN_COLOUR, width=2, dash="dash"),
                                 legendgroup="clean", showlegend=first), row=r + 1, col=c + 1)
        hit = flags.mask[col].to_numpy()
        fig.add_trace(go.Scatter(
            x=raw_dates[hit], y=raw.loc[hit, col], name="Flagged anomaly", mode="markers",
            marker=dict(color=FLAG_COLOUR, size=11, symbol="x", line=dict(width=1, color="white")),
            legendgroup="flag", showlegend=first,
            hovertemplate=f"{col}<br>%{{x|%Y-%m-%d}}<br>Value: %{{y:,.2f}}<extra>Flagged</extra>",
        ), row=r + 1, col=c + 1)
        fig.update_yaxes(title_text=col, row=r + 1, col=c + 1)
        fig.update_xaxes(title_text="Date", row=r + 1, col=c + 1)

    fig.update_layout(
        title=dict(text="Raw vs anomaly-free business metrics", x=0.01),
        height=330 * nrows + 120, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=plotly_js, full_html=True)
    return out_path
