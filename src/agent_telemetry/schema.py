"""The shape of the data at every stage.

Two levels: one row per event as it appeared in the log, and one row per session
after aggregation. Both are declared here rather than being implied by whatever
columns the extractor happened to produce, so that a change to the log format
shows up as a schema mismatch instead of as a new column in a published dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

__all__ = ["Event", "SessionSummary", "event_columns", "session_columns"]


@dataclass(slots=True)
class Event:
    """One line of a session log, reduced to what can be published.

    Every field here is a number, a bounded category, or an irreversible hash.
    There is no field that can hold free text; that is the property the privacy
    scan verifies afterwards.
    """

    session: str
    """Pseudonymized session identifier."""

    project: str
    """Pseudonymized working directory."""

    branch: str
    """Pseudonymized git branch, or an empty string outside a repository."""

    sequence: int
    """Position of this event within its session, starting at zero."""

    timestamp: str
    """ISO timestamp truncated to the hour.

    Truncated because an exact timestamp combined with a project label is close to
    a fingerprint. Anything that needs finer resolution uses ``offset_seconds``.
    """

    offset_seconds: float = 0.0
    """Seconds since the first event of this session.

    Computed from the untruncated timestamps and kept because a duration is not
    identifying while an exact wall clock time is.
    """

    type: str = ""
    role: str = ""
    model: str = ""
    effort: str = ""
    permission_mode: str = ""
    client_version: str = ""
    entrypoint: str = ""

    is_sidechain: bool = False
    """True when the event belongs to a subagent rather than the main loop."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    stop_reason: str = ""
    service_tier: str = ""

    tool_name: str = ""
    """Name of the tool invoked by this event, if it invoked one."""

    tool_calls: int = 0
    """Number of tool calls in this event, since one turn can issue several."""

    tool_results: int = 0
    tool_errors: int = 0

    thinking_blocks: int = 0
    text_blocks: int = 0
    content_length: int = 0
    """Character count of the text content. The length only; never the text."""

    CATEGORICAL: ClassVar[tuple[str, ...]] = (
        "type",
        "role",
        "model",
        "effort",
        "permission_mode",
        "client_version",
        "entrypoint",
        "stop_reason",
        "service_tier",
        "tool_name",
    )

    @property
    def billable_input_tokens(self) -> int:
        """Fresh input plus cache writes. Cache reads are billed separately and cheaper."""
        return self.input_tokens + self.cache_creation_tokens

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
        )


@dataclass(slots=True)
class SessionSummary:
    """One session, aggregated from its events."""

    session: str
    project: str
    branch: str

    started_at: str
    ended_at: str
    duration_minutes: float

    events: int
    user_turns: int
    assistant_turns: int

    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int

    tool_calls: int
    tool_errors: int
    distinct_tools: int
    top_tool: str

    models: str
    """Models used, sorted and joined with a plus sign."""

    primary_model: str
    client_version: str
    efforts: str

    is_sidechain: bool
    subagent_events: int

    tool_names: dict[str, int] = field(default_factory=dict, repr=False)
    """Per-tool call counts. Dropped before the dataset is written."""

    CATEGORICAL: ClassVar[tuple[str, ...]] = (
        "primary_model",
        "client_version",
        "top_tool",
        "models",
        "efforts",
    )

    @property
    def cache_hit_rate(self) -> float:
        """Share of all input side tokens that were served from the cache."""
        total = self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens
        return self.cache_read_tokens / total if total else 0.0

    @property
    def tool_error_rate(self) -> float:
        return self.tool_errors / self.tool_calls if self.tool_calls else 0.0


def event_columns() -> list[str]:
    return [f.name for f in fields(Event)]


def session_columns() -> list[str]:
    """Columns of the published session table, in order. ``tool_names`` is internal."""
    return [f.name for f in fields(SessionSummary) if f.name != "tool_names"]


def as_row(item: Event | SessionSummary, columns: list[str]) -> dict[str, Any]:
    return {name: getattr(item, name) for name in columns}
