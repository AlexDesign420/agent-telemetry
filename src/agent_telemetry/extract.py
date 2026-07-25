"""Read session logs and emit events.

The rule this module follows: a value either passes through the allowlist, or it
is counted and discarded, or it never gets looked at. Nothing is copied into an
Event because it happened to be there.

Session logs are JSON Lines, one event per line, and they are written while a
session runs. A file whose last line is half written is normal rather than
exceptional, so a malformed line is counted and skipped instead of aborting a run
over the tail of a file that is still being appended to.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .privacy.allowlist import (
    ALLOWED_USAGE_FIELDS,
    filter_mapping,
    unknown_fields,
)
from .privacy.pseudonymize import Pseudonymizer
from .schema import Event

__all__ = ["ContentCounts", "ExtractionStats", "extract_file", "extract_tree", "iter_session_files"]


@dataclass(slots=True)
class ExtractionStats:
    """What the extractor saw, including what it refused to read."""

    files: int = 0
    lines: int = 0
    events: int = 0
    malformed_lines: int = 0
    empty_files: int = 0
    unknown_field_names: dict[str, int] = field(default_factory=dict)

    def note_unknown(self, names: set[str]) -> None:
        for name in names:
            self.unknown_field_names[name] = self.unknown_field_names.get(name, 0) + 1

    def summary(self) -> str:
        lines = [
            f"files          {self.files} ({self.empty_files} empty)",
            f"lines          {self.lines}",
            f"events         {self.events}",
            f"malformed      {self.malformed_lines}",
        ]
        if self.unknown_field_names:
            top = sorted(self.unknown_field_names.items(), key=lambda kv: -kv[1])[:10]
            lines.append("unknown fields " + ", ".join(f"{name} x{count}" for name, count in top))
        return "\n".join(lines)


def iter_session_files(root: Path) -> Iterator[Path]:
    """Yield every session log below ``root`` in a stable order."""
    yield from sorted(root.rglob("*.jsonl"))


def _truncate_to_hour(timestamp: str | None) -> str:
    """Keep the hour, drop minutes and seconds.

    An exact timestamp plus a project label is close to a fingerprint. The hour
    is enough for every question this dataset is meant to answer.
    """
    if not timestamp or len(timestamp) < 13:
        return ""
    return timestamp[:13] + ":00:00Z"


@dataclass(slots=True)
class ContentCounts:
    """The shape of a message's content, with none of its text."""

    tool_calls: int = 0
    tool_results: int = 0
    tool_errors: int = 0
    thinking_blocks: int = 0
    text_blocks: int = 0
    content_length: int = 0
    tool_name: str = ""


def _count_content(blocks: Any) -> ContentCounts:
    """Count block kinds and measure text length without reading the text."""
    counts = ContentCounts()

    if isinstance(blocks, str):
        counts.text_blocks = 1
        counts.content_length = len(blocks)
        return counts

    if not isinstance(blocks, list):
        return counts

    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "tool_use":
            counts.tool_calls += 1
            if not counts.tool_name:
                counts.tool_name = str(block.get("name") or "")
        elif kind == "tool_result":
            counts.tool_results += 1
            if block.get("is_error"):
                counts.tool_errors += 1
        elif kind == "thinking":
            counts.thinking_blocks += 1
            counts.content_length += len(str(block.get("thinking") or ""))
        elif kind == "text":
            counts.text_blocks += 1
            counts.content_length += len(str(block.get("text") or ""))

    return counts


def _event_from_record(
    record: dict[str, Any],
    sequence: int,
    pseudonymizer: Pseudonymizer,
) -> Event:
    message = record.get("message")
    message = message if isinstance(message, dict) else {}
    usage = filter_mapping(message.get("usage") or {}, ALLOWED_USAGE_FIELDS)
    content = _count_content(message.get("content"))

    return Event(
        session=pseudonymizer.session(record.get("sessionId")),
        project=pseudonymizer.project(record.get("cwd")),
        branch=pseudonymizer.branch(record.get("gitBranch")),
        sequence=sequence,
        timestamp=_truncate_to_hour(record.get("timestamp")),
        type=str(record.get("type") or ""),
        role=str(message.get("role") or ""),
        model=str(message.get("model") or ""),
        effort=str(record.get("effort") or ""),
        permission_mode=str(record.get("permissionMode") or ""),
        client_version=str(record.get("version") or ""),
        entrypoint=str(record.get("entrypoint") or ""),
        is_sidechain=bool(record.get("isSidechain")),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_creation_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
        stop_reason=str(message.get("stop_reason") or ""),
        service_tier=str(usage.get("service_tier") or ""),
        tool_name=content.tool_name,
        tool_calls=content.tool_calls,
        tool_results=content.tool_results,
        tool_errors=content.tool_errors,
        thinking_blocks=content.thinking_blocks,
        text_blocks=content.text_blocks,
        content_length=content.content_length,
    )


def extract_file(
    path: Path,
    pseudonymizer: Pseudonymizer,
    stats: ExtractionStats | None = None,
    track_unknown: bool = False,
) -> list[Event]:
    """Extract every event from one session log."""
    stats = stats if stats is not None else ExtractionStats()
    stats.files += 1

    events: list[Event] = []
    sequence = 0

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            stats.lines += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats.malformed_lines += 1
                continue

            if not isinstance(record, dict):
                stats.malformed_lines += 1
                continue

            if track_unknown:
                stats.note_unknown(unknown_fields(record))

            events.append(_event_from_record(record, sequence, pseudonymizer))
            sequence += 1
            stats.events += 1

    if not events:
        stats.empty_files += 1
    return events


def extract_tree(
    root: Path,
    pseudonymizer: Pseudonymizer | None = None,
    track_unknown: bool = False,
) -> tuple[list[Event], ExtractionStats]:
    """Extract every session log below ``root``."""
    pseudonymizer = pseudonymizer or Pseudonymizer()
    stats = ExtractionStats()
    events: list[Event] = []

    for path in iter_session_files(root):
        events.extend(extract_file(path, pseudonymizer, stats, track_unknown))

    return events, stats
