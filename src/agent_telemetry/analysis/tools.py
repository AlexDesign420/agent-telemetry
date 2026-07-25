"""What the agent actually does.

A session log is mostly tool calls. Which tools, how often they fail, and what
follows what, describes the working style of an agent more directly than any
token count does.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["error_rates", "tool_frequency", "tool_transitions", "tools_per_session"]


def tool_frequency(events: pd.DataFrame, top: int | None = None) -> pd.DataFrame:
    """Calls, errors and error rate per tool."""
    calling = events[events["tool_name"] != ""]
    if calling.empty:
        return pd.DataFrame(columns=["tool_name", "calls", "sessions", "share"])

    grouped = calling.groupby("tool_name", observed=True).agg(
        calls=("tool_calls", "sum"),
        events=("tool_name", "count"),
        sessions=("session", "nunique"),
    )
    grouped["share"] = (grouped["calls"] / grouped["calls"].sum()).round(4)
    grouped = grouped.sort_values("calls", ascending=False).reset_index()
    return grouped.head(top) if top else grouped


def error_rates(events: pd.DataFrame) -> pd.DataFrame:
    """Tool errors as a share of tool results.

    Errors are attached to the result rather than to the call, so this groups on
    the events that carry results. A tool whose result never comes back as an
    error is not necessarily reliable; it may simply report failure in its output
    text, which this dataset deliberately cannot see.
    """
    returning = events[events["tool_results"] > 0]
    if returning.empty:
        return pd.DataFrame(columns=["results", "errors", "error_rate"])

    total_results = int(returning["tool_results"].sum())
    total_errors = int(returning["tool_errors"].sum())

    by_session = returning.groupby("session", observed=True).agg(
        results=("tool_results", "sum"),
        errors=("tool_errors", "sum"),
    )
    by_session["error_rate"] = (by_session["errors"] / by_session["results"]).round(4)

    return pd.DataFrame(
        {
            "results": [total_results],
            "errors": [total_errors],
            "error_rate": [round(total_errors / total_results, 4) if total_results else 0.0],
            "median_session_error_rate": [round(float(by_session["error_rate"].median()), 4)],
            "sessions_without_errors": [int((by_session["errors"] == 0).sum())],
            "sessions": [len(by_session)],
        }
    )


def tools_per_session(sessions: pd.DataFrame) -> pd.DataFrame:
    """Distribution of tool usage across sessions."""
    return pd.DataFrame(
        {
            "metric": [
                "sessions",
                "sessions with tool calls",
                "median calls per session",
                "median distinct tools per session",
                "max calls in one session",
            ],
            "value": [
                len(sessions),
                int((sessions["tool_calls"] > 0).sum()),
                float(sessions["tool_calls"].median()),
                float(sessions["distinct_tools"].median()),
                int(sessions["tool_calls"].max()) if len(sessions) else 0,
            ],
        }
    )


def tool_transitions(events: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Most common tool pairs, where one call directly follows another.

    Only consecutive calls inside the same session count. The pairs are what
    workflows look like from the outside: read then edit, edit then run.
    """
    calling = events[events["tool_name"] != ""].sort_values(["session", "sequence"])
    if calling.empty:
        return pd.DataFrame(columns=["from_tool", "to_tool", "count"])

    calling = calling.assign(
        next_tool=calling.groupby("session", observed=True)["tool_name"].shift(-1)
    )
    pairs = calling.dropna(subset=["next_tool"])

    counted = (
        pairs.groupby(["tool_name", "next_tool"], observed=True)
        .size()
        .reset_index(name="count")
        .rename(columns={"tool_name": "from_tool", "next_tool": "to_tool"})
        .sort_values("count", ascending=False)
    )
    return counted.head(top).reset_index(drop=True)
