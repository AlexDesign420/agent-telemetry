"""Roll events up into sessions and write both tables.

The event table answers questions about turns; the session table answers questions
about work. Most of what is interesting lives at the session level, because that
is the unit a person actually experiences.

Both tables pass through k-anonymity and the privacy scan on the way out. The scan
runs last, on exactly the frame that is about to be written, so nothing can be
added after the check.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .privacy.kanon import DEFAULT_K, Suppression, apply_k_anonymity
from .privacy.scan import enforce
from .schema import Event, SessionSummary, as_row, event_columns, session_columns

__all__ = [
    "events_to_frame",
    "sessions_to_frame",
    "summarize_session",
    "summarize_sessions",
    "write_dataset",
]


def _parse(timestamp: str) -> datetime | None:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def summarize_session(events: Sequence[Event]) -> SessionSummary:
    """Aggregate the events of one session.

    Duration comes from ``offset_seconds`` rather than from the published
    timestamps, which are truncated to the hour. It measures wall clock time
    between the first and the last event, so a session left open overnight reports
    the whole night. The data card says so; it is the most misreadable column here.
    """
    if not events:
        raise ValueError("cannot summarize an empty session")

    stamps = [parsed for parsed in (_parse(event.timestamp) for event in events) if parsed]
    started = min(stamps) if stamps else None
    ended = max(stamps) if stamps else None
    duration = max((event.offset_seconds for event in events), default=0.0) / 60

    tool_counter: Counter[str] = Counter()
    for event in events:
        if event.tool_name:
            tool_counter[event.tool_name] += event.tool_calls or 1

    models = Counter(event.model for event in events if event.model)
    efforts = sorted({event.effort for event in events if event.effort})
    versions = Counter(event.client_version for event in events if event.client_version)

    return SessionSummary(
        session=events[0].session,
        project=events[0].project,
        branch=events[0].branch,
        started_at=started.isoformat() if started else "",
        ended_at=ended.isoformat() if ended else "",
        duration_minutes=round(duration, 1),
        events=len(events),
        user_turns=sum(1 for event in events if event.type == "user"),
        assistant_turns=sum(1 for event in events if event.type == "assistant"),
        input_tokens=sum(event.input_tokens for event in events),
        output_tokens=sum(event.output_tokens for event in events),
        cache_creation_tokens=sum(event.cache_creation_tokens for event in events),
        cache_read_tokens=sum(event.cache_read_tokens for event in events),
        tool_calls=sum(event.tool_calls for event in events),
        tool_errors=sum(event.tool_errors for event in events),
        distinct_tools=len(tool_counter),
        top_tool=tool_counter.most_common(1)[0][0] if tool_counter else "",
        models="+".join(sorted(models)),
        primary_model=models.most_common(1)[0][0] if models else "",
        client_version=versions.most_common(1)[0][0] if versions else "",
        efforts="+".join(efforts),
        is_sidechain=any(event.is_sidechain for event in events),
        subagent_events=sum(1 for event in events if event.is_sidechain),
        tool_names=dict(tool_counter),
    )


def summarize_sessions(events: Iterable[Event]) -> list[SessionSummary]:
    """Group events by session and summarize each one."""
    grouped: dict[str, list[Event]] = {}
    for event in events:
        grouped.setdefault(event.session, []).append(event)

    return [summarize_session(group) for group in grouped.values() if group]


def events_to_frame(events: Iterable[Event]) -> pd.DataFrame:
    columns = event_columns()
    return pd.DataFrame([as_row(event, columns) for event in events], columns=columns)


def sessions_to_frame(sessions: Iterable[SessionSummary]) -> pd.DataFrame:
    columns = session_columns()
    frame = pd.DataFrame([as_row(session, columns) for session in sessions], columns=columns)
    if frame.empty:
        return frame

    frame["cache_hit_rate"] = (
        frame["cache_read_tokens"]
        / (
            frame["input_tokens"] + frame["cache_creation_tokens"] + frame["cache_read_tokens"]
        ).replace(0, pd.NA)
    ).fillna(0.0).round(4)

    frame["tool_error_rate"] = (
        frame["tool_errors"] / frame["tool_calls"].replace(0, pd.NA)
    ).fillna(0.0).round(4)

    frame["total_tokens"] = (
        frame["input_tokens"]
        + frame["output_tokens"]
        + frame["cache_creation_tokens"]
        + frame["cache_read_tokens"]
    )
    return frame


def tool_usage_frame(sessions: Iterable[SessionSummary]) -> pd.DataFrame:
    """Long format table of tool calls per session, for the tool analyses."""
    rows: list[dict[str, Any]] = []
    for session in sessions:
        for tool, count in session.tool_names.items():
            rows.append({"session": session.session, "tool": tool, "calls": count})
    return pd.DataFrame(rows, columns=["session", "tool", "calls"])


def write_dataset(
    events: Sequence[Event],
    output_dir: Path,
    k: int = DEFAULT_K,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Suppression]]:
    """Aggregate, anonymize, verify and write both tables.

    Order matters and is not negotiable: anonymize, then scan, then write. The
    scan sees the exact frame that reaches disk.
    """
    sessions = summarize_sessions(events)

    event_frame = events_to_frame(events)
    session_frame = sessions_to_frame(sessions)

    event_frame, event_suppressions = apply_k_anonymity(event_frame, list(Event.CATEGORICAL), k)
    session_frame, session_suppressions = apply_k_anonymity(
        session_frame, list(SessionSummary.CATEGORICAL), k
    )

    enforce(event_frame)
    enforce(session_frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    event_frame.to_parquet(output_dir / "events.parquet", index=False)
    session_frame.to_parquet(output_dir / "sessions.parquet", index=False)

    return event_frame, session_frame, event_suppressions + session_suppressions
