"""Where the tokens go.

The headline number in any agent session is not the output length. It is the
input side, and specifically how much of it is served from the prompt cache. A
long session re-sends its entire context on every turn; whether that context is
billed at the full rate or the cache read rate decides the cost of the session by
an order of magnitude.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "PRICING",
    "cache_efficiency_by_length",
    "estimate_cost",
    "token_mix",
    "tokens_by_model",
]

TOKEN_COLUMNS = [
    "input_tokens",
    "output_tokens",
    "cache_creation_tokens",
    "cache_read_tokens",
]

# Dollars per million tokens. Only used for the relative comparisons in the
# notebooks; absolute figures depend on a plan this dataset knows nothing about,
# and subscription sessions are not billed per token at all.
PRICING: dict[str, dict[str, float]] = {
    "default": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
}


def token_mix(sessions: pd.DataFrame) -> pd.Series:
    """Total tokens by kind, over every session."""
    return sessions[TOKEN_COLUMNS].sum()


def cache_hit_rate(sessions: pd.DataFrame) -> float:
    """Share of input side tokens served from the cache, across the whole dataset."""
    totals = token_mix(sessions)
    input_side = (
        totals["input_tokens"] + totals["cache_creation_tokens"] + totals["cache_read_tokens"]
    )
    return float(totals["cache_read_tokens"] / input_side) if input_side else 0.0


def cache_efficiency_by_length(sessions: pd.DataFrame, bins: int = 6) -> pd.DataFrame:
    """Cache hit rate against session length.

    The expected shape is a curve that rises steeply and then flattens: a cache
    entry has to be written before it can be read, so a session short enough to
    end before the second turn pays full price for everything.
    """
    working = sessions[sessions["events"] > 0].copy()
    if working.empty:
        return pd.DataFrame()

    working["length_bucket"] = pd.qcut(
        working["events"], q=bins, duplicates="drop", precision=0
    )

    grouped = working.groupby("length_bucket", observed=True).agg(
        sessions=("session", "count"),
        median_events=("events", "median"),
        cache_read=("cache_read_tokens", "sum"),
        cache_write=("cache_creation_tokens", "sum"),
        fresh_input=("input_tokens", "sum"),
        output=("output_tokens", "sum"),
    )

    input_side = grouped["cache_read"] + grouped["cache_write"] + grouped["fresh_input"]
    grouped["cache_hit_rate"] = (grouped["cache_read"] / input_side).round(4)
    return grouped.reset_index()


def tokens_by_model(sessions: pd.DataFrame) -> pd.DataFrame:
    """Token totals and cache behaviour per primary model."""
    grouped = sessions.groupby("primary_model", observed=True).agg(
        sessions=("session", "count"),
        median_events=("events", "median"),
        input_tokens=("input_tokens", "sum"),
        output_tokens=("output_tokens", "sum"),
        cache_creation_tokens=("cache_creation_tokens", "sum"),
        cache_read_tokens=("cache_read_tokens", "sum"),
    )

    input_side = (
        grouped["input_tokens"] + grouped["cache_creation_tokens"] + grouped["cache_read_tokens"]
    )
    grouped["cache_hit_rate"] = (grouped["cache_read_tokens"] / input_side).round(4)
    grouped["output_per_session"] = (grouped["output_tokens"] / grouped["sessions"]).round(0)
    return grouped.sort_values("sessions", ascending=False).reset_index()


def estimate_cost(sessions: pd.DataFrame, rates: dict[str, float] | None = None) -> pd.Series:
    """Cost per session at the given rates, and what it would have cost without the cache.

    Both numbers are hypothetical. They exist to show the size of the cache effect,
    not to reconcile against a bill.
    """
    rates = rates or PRICING["default"]
    per_million = 1_000_000

    with_cache = (
        sessions["input_tokens"] * rates["input"]
        + sessions["cache_creation_tokens"] * rates["cache_write"]
        + sessions["cache_read_tokens"] * rates["cache_read"]
        + sessions["output_tokens"] * rates["output"]
    ) / per_million

    without_cache = (
        (sessions["input_tokens"] + sessions["cache_creation_tokens"] + sessions["cache_read_tokens"])
        * rates["input"]
        + sessions["output_tokens"] * rates["output"]
    ) / per_million

    return pd.Series(
        {
            "with_cache": round(float(with_cache.sum()), 2),
            "without_cache": round(float(without_cache.sum()), 2),
            "saved": round(float((without_cache - with_cache).sum()), 2),
            "saved_share": round(float(1 - with_cache.sum() / without_cache.sum()), 4)
            if without_cache.sum()
            else 0.0,
        }
    )
