"""Comparing models on the work they were actually given.

The obvious comparison, average tokens per session by model, is close to
meaningless: models are not assigned to sessions at random. A model that is
picked for hard problems will look expensive whatever it does.

The functions here do what can be done honestly with observational data. They
report the confounder alongside the comparison instead of pretending it is not
there, and the paired comparison restricts itself to sessions where more than one
model was actually used, which is the closest this data comes to a controlled
comparison.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["effort_distribution", "model_profile", "paired_sessions", "provider_of"]


def provider_of(model: str) -> str:
    """Group a model name by where it runs.

    The logs mix hosted Anthropic models with third party APIs and locally served
    models, and those are not comparable on cost at all.
    """
    if not model or model == "other":
        return "unknown"
    name = model.lower()
    if name.startswith("<"):
        return "synthetic"
    if "claude" in name:
        return "anthropic"
    if "deepseek" in name:
        return "deepseek"
    if "/" in name:
        return "local"
    return "other"


def model_profile(sessions: pd.DataFrame) -> pd.DataFrame:
    """Per model: how many sessions, how long, how many tools, how much output.

    Read this as a description of what each model was used for, not as a
    performance ranking.
    """
    working = sessions.copy()
    working["provider"] = working["primary_model"].map(provider_of)

    grouped = working.groupby(["provider", "primary_model"], observed=True).agg(
        sessions=("session", "count"),
        median_events=("events", "median"),
        median_duration=("duration_minutes", "median"),
        median_tool_calls=("tool_calls", "median"),
        median_output_tokens=("output_tokens", "median"),
        mean_cache_hit_rate=("cache_hit_rate", "mean"),
        mean_tool_error_rate=("tool_error_rate", "mean"),
    )
    grouped["mean_cache_hit_rate"] = grouped["mean_cache_hit_rate"].round(4)
    grouped["mean_tool_error_rate"] = grouped["mean_tool_error_rate"].round(4)
    return grouped.sort_values("sessions", ascending=False).reset_index()


def paired_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """Sessions where more than one model ran.

    These are sessions where the model was switched mid-task, so both models saw
    related work. It is a small subset and the direction of the switch is itself
    informative: switching usually happens because the first model was not doing
    well, which biases the comparison against whichever model came second.
    """
    multi = sessions[sessions["models"].str.contains(r"\+", regex=True, na=False)]
    if multi.empty:
        return pd.DataFrame(columns=["models", "sessions", "median_events", "median_tool_calls"])

    grouped = multi.groupby("models", observed=True).agg(
        sessions=("session", "count"),
        median_events=("events", "median"),
        median_tool_calls=("tool_calls", "median"),
        median_duration=("duration_minutes", "median"),
    )
    return grouped.sort_values("sessions", ascending=False).reset_index()


def effort_distribution(events: pd.DataFrame) -> pd.DataFrame:
    """How reasoning effort was distributed, per model.

    Effort is set by the caller, not by the model, so this describes usage habits
    rather than model behaviour.
    """
    working = events[(events["effort"] != "") & (events["model"] != "")]
    if working.empty:
        return pd.DataFrame(columns=["model", "effort", "events", "share"])

    counted = (
        working.groupby(["model", "effort"], observed=True).size().reset_index(name="events")
    )
    totals = counted.groupby("model", observed=True)["events"].transform("sum")
    counted["share"] = (counted["events"] / totals).round(4)
    return counted.sort_values(["model", "events"], ascending=[True, False]).reset_index(drop=True)
