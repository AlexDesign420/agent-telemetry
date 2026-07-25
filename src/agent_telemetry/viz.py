"""Charts for the notebooks and the report.

One style, applied once, so that figures from different notebooks can sit next to
each other without looking like they came from different projects. matplotlib is
an optional dependency: everything else in the package works without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "bar",
    "cache_curve",
    "distribution",
    "save",
    "use_style",
]

INK = "#1c1c22"
MUTED = "#8a8a96"
ACCENT = "#3b7dd8"
SECONDARY = "#d8763b"
GRID = "#e3e3e9"


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("plotting needs matplotlib: pip install 'agent-telemetry[viz]'") from exc
    return plt


def use_style() -> None:
    """Apply the project style to the current matplotlib session."""
    plt = _pyplot()
    plt.rcParams.update(
        {
            "figure.figsize": (9, 4.5),
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "medium",
            "axes.titlelocation": "left",
            "axes.labelcolor": MUTED,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
        }
    )


def save(figure: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    return path


def bar(
    frame: pd.DataFrame,
    label_column: str,
    value_column: str,
    title: str = "",
    horizontal: bool = True,
    highlight: int = 0,
) -> Any:
    """Bar chart, horizontal by default because tool names are long.

    ``highlight`` colours the first n bars differently, which is how the top few
    entries are marked without a legend nobody reads.
    """
    plt = _pyplot()
    use_style()

    labels = frame[label_column].astype(str).tolist()
    values = frame[value_column].tolist()
    colors = [ACCENT if index < highlight else MUTED for index in range(len(values))]

    height = max(2.4, 0.34 * len(values) + 1.2)
    figure, axes = plt.subplots(figsize=(9, height) if horizontal else None)

    if horizontal:
        axes.barh(labels[::-1], values[::-1], color=colors[::-1])
        axes.grid(axis="y", visible=False)
    else:
        axes.bar(labels, values, color=colors)
        axes.grid(axis="x", visible=False)

    if title:
        axes.set_title(title)
    return figure


def distribution(
    series: pd.Series,
    title: str = "",
    bins: int = 40,
    log_x: bool = True,
    percentiles: tuple[float, ...] = (0.5, 0.9),
) -> Any:
    """Histogram with percentile markers.

    The log scale is the default because every distribution in this dataset is
    heavily skewed; on a linear axis they all collapse into a single bar at the
    left edge.
    """
    plt = _pyplot()
    use_style()

    values = series[series > 0] if log_x else series
    figure, axes = plt.subplots()

    if log_x and len(values):
        import numpy as np

        edges = np.logspace(np.log10(max(values.min(), 1e-3)), np.log10(values.max()), bins)
        axes.hist(values, bins=edges, color=ACCENT, alpha=0.85)
        axes.set_xscale("log")
    else:
        axes.hist(values, bins=bins, color=ACCENT, alpha=0.85)

    for percentile in percentiles:
        marker = float(values.quantile(percentile))
        axes.axvline(marker, color=SECONDARY, linewidth=1.2, linestyle="--")
        axes.annotate(
            f"p{int(percentile * 100)} = {marker:,.0f}",
            xy=(marker, axes.get_ylim()[1] * 0.92),
            xytext=(4, 0),
            textcoords="offset points",
            color=SECONDARY,
            fontsize=9,
        )

    if title:
        axes.set_title(title)
    axes.grid(axis="x", visible=False)
    return figure


def cache_curve(frame: pd.DataFrame, title: str = "") -> Any:
    """Cache hit rate against session length, with the session count behind it."""
    plt = _pyplot()
    use_style()

    figure, axes = plt.subplots()
    x = frame["median_events"]

    counts = axes.twinx()
    counts.bar(range(len(frame)), frame["sessions"], color=GRID, width=0.6)
    counts.set_ylabel("sessions", color=MUTED)
    counts.grid(visible=False)
    counts.set_ylim(0, frame["sessions"].max() * 3)

    axes.set_zorder(counts.get_zorder() + 1)
    axes.patch.set_visible(False)
    axes.plot(range(len(frame)), frame["cache_hit_rate"], color=ACCENT, marker="o", linewidth=2)
    axes.set_xticks(range(len(frame)))
    axes.set_xticklabels([f"{int(value)}" for value in x])
    axes.set_xlabel("median events per session")
    axes.set_ylabel("cache hit rate")
    axes.set_ylim(0, 1.02)

    if title:
        axes.set_title(title)
    return figure
