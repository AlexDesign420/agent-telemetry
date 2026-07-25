"""The shape of sessions themselves.

Length, duration, how much of the work was delegated to subagents, and how the
mix changed as the client was updated.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "delegating_sessions",
    "length_distribution",
    "session_overview",
    "subagent_usage",
    "version_timeline",
]


def session_overview(sessions: pd.DataFrame) -> pd.DataFrame:
    """The headline numbers, as a table rather than as prose in a notebook."""
    if sessions.empty:
        return pd.DataFrame(columns=["metric", "value"])

    tokens = sessions[
        ["input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens"]
    ]
    metrics = {
        "sessions": len(sessions),
        "events": int(sessions["events"].sum()),
        "median events per session": float(sessions["events"].median()),
        "median duration (minutes)": float(sessions["duration_minutes"].median()),
        "total tokens (millions)": round(float(tokens.to_numpy().sum()) / 1e6, 1),
        "tool calls": int(sessions["tool_calls"].sum()),
        "tool errors": int(sessions["tool_errors"].sum()),
        "subagent sessions": int(sessions["is_sidechain"].sum()),
        "main sessions": int((~sessions["is_sidechain"]).sum()),
        "distinct projects": int(sessions["project"].nunique()),
    }
    return pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())})


def length_distribution(sessions: pd.DataFrame) -> pd.DataFrame:
    """Percentiles of session length.

    The distribution is heavily skewed: most sessions are a question and an
    answer, and a small number run for hours. The mean is therefore the wrong
    summary and is included only to show how far it sits from the median.
    """
    if sessions.empty:
        return pd.DataFrame()

    percentiles = [0.5, 0.75, 0.9, 0.95, 0.99]
    rows = []
    for column in ("events", "duration_minutes", "tool_calls", "total_tokens"):
        if column not in sessions.columns:
            continue
        series = sessions[column]
        row = {"metric": column, "mean": round(float(series.mean()), 1)}
        for percentile in percentiles:
            row[f"p{int(percentile * 100)}"] = round(float(series.quantile(percentile)), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def subagent_usage(sessions: pd.DataFrame) -> pd.DataFrame:
    """How much work ran in subagents rather than in the main loop.

    A subagent gets its own log file, so ``is_sidechain`` is a property of the
    whole session and not of individual events inside a larger one. Counting
    events with the flag set therefore counts subagent sessions, not main
    sessions that delegated. The two questions are separated here because
    conflating them roughly quadruples the apparent delegation rate.
    """
    if sessions.empty:
        return pd.DataFrame(columns=["metric", "value"])

    subagents = sessions[sessions["is_sidechain"]]
    main = sessions[~sessions["is_sidechain"]]
    total_events = int(sessions["events"].sum())

    metrics = {
        "subagent sessions": len(subagents),
        "share of all sessions": round(len(subagents) / len(sessions), 4),
        "events inside subagent sessions": int(subagents["events"].sum()),
        "share of all events": round(int(subagents["events"].sum()) / total_events, 4)
        if total_events
        else 0.0,
        "median events, subagent session": float(subagents["events"].median())
        if len(subagents)
        else 0.0,
        "median events, main session": float(main["events"].median()) if len(main) else 0.0,
        "median tool calls, subagent session": float(subagents["tool_calls"].median())
        if len(subagents)
        else 0.0,
        "median tool calls, main session": float(main["tool_calls"].median()) if len(main) else 0.0,
    }
    return pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())})


def delegating_sessions(sessions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Main sessions that started a subagent, identified by the Agent tool call.

    This is the counterpart to ``subagent_usage``: it counts the sessions that
    delegated rather than the sessions that were delegated to.
    """
    if sessions.empty or events.empty:
        return pd.DataFrame(columns=["metric", "value"])

    main = sessions[~sessions["is_sidechain"]]
    delegating = set(events[events["tool_name"].isin(["Agent", "Task"])]["session"].unique())
    delegating_main = main[main["session"].isin(delegating)]

    metrics = {
        "main sessions": len(main),
        "main sessions that spawned an agent": len(delegating_main),
        "share": round(len(delegating_main) / len(main), 4) if len(main) else 0.0,
        "median events when delegating": float(delegating_main["events"].median())
        if len(delegating_main)
        else 0.0,
        "median events otherwise": float(main[~main["session"].isin(delegating)]["events"].median())
        if len(main)
        else 0.0,
    }
    return pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())})


def version_timeline(sessions: pd.DataFrame) -> pd.DataFrame:
    """Sessions per client version, in version order.

    Useful mostly as a data quality check: a field that changes meaning between
    releases will show up as a discontinuity here before it shows up as a strange
    result somewhere else.
    """
    if sessions.empty or "client_version" not in sessions.columns:
        return pd.DataFrame(columns=["client_version", "sessions"])

    grouped = (
        sessions[sessions["client_version"] != ""]
        .groupby("client_version", observed=True)
        .agg(
            sessions=("session", "count"),
            median_events=("events", "median"),
            mean_cache_hit_rate=("cache_hit_rate", "mean"),
        )
    )
    grouped["mean_cache_hit_rate"] = grouped["mean_cache_hit_rate"].round(4)

    def sort_key(version: str) -> tuple[int, ...]:
        try:
            return tuple(int(part) for part in version.split("."))
        except ValueError:
            return (0,)

    return grouped.reindex(sorted(grouped.index, key=sort_key)).reset_index()
