from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agent_telemetry.aggregate import (
    events_to_frame,
    sessions_to_frame,
    summarize_session,
    summarize_sessions,
    tool_usage_frame,
    write_dataset,
)
from agent_telemetry.extract import extract_file, extract_tree
from agent_telemetry.privacy.pseudonymize import Pseudonymizer
from agent_telemetry.privacy.scan import PrivacyError, scan_frame
from agent_telemetry.schema import Event, SessionSummary, session_columns


def event(**overrides: object) -> Event:
    base = {
        "session": "session-000000000001",
        "project": "project-000000000001",
        "branch": "",
        "sequence": 0,
        "timestamp": "2026-07-19T14:00:00Z",
        "type": "assistant",
    }
    return Event(**{**base, **overrides})  # type: ignore[arg-type]


class TestSummarize:
    def test_an_empty_session_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty session"):
            summarize_session([])

    def test_tokens_are_summed(self) -> None:
        summary = summarize_session(
            [event(input_tokens=10, output_tokens=1), event(input_tokens=5, output_tokens=2)]
        )
        assert summary.input_tokens == 15
        assert summary.output_tokens == 3

    def test_turns_are_counted_by_type(self) -> None:
        summary = summarize_session(
            [event(type="user"), event(type="assistant"), event(type="assistant")]
        )
        assert summary.user_turns == 1
        assert summary.assistant_turns == 2

    def test_duration_comes_from_the_offsets(self) -> None:
        summary = summarize_session([event(offset_seconds=0.0), event(offset_seconds=180.0)])
        assert summary.duration_minutes == 3.0

    def test_duration_is_zero_without_offsets(self) -> None:
        assert summarize_session([event(), event()]).duration_minutes == 0.0

    def test_the_most_used_tool_wins(self) -> None:
        summary = summarize_session(
            [
                event(tool_name="Bash", tool_calls=3),
                event(tool_name="Read", tool_calls=1),
            ]
        )
        assert summary.top_tool == "Bash"
        assert summary.distinct_tools == 2
        assert summary.tool_calls == 4

    def test_models_are_listed_and_the_primary_one_identified(self) -> None:
        summary = summarize_session(
            [
                event(model="claude-sonnet-5"),
                event(model="claude-sonnet-5"),
                event(model="claude-opus-5"),
            ]
        )
        assert summary.models == "claude-opus-5+claude-sonnet-5"
        assert summary.primary_model == "claude-sonnet-5"

    def test_a_subagent_session_is_flagged(self) -> None:
        summary = summarize_session([event(is_sidechain=True), event(is_sidechain=True)])
        assert summary.is_sidechain
        assert summary.subagent_events == 2

    def test_cache_hit_rate_is_the_read_share_of_the_input_side(self) -> None:
        summary = summarize_session(
            [event(input_tokens=10, cache_creation_tokens=10, cache_read_tokens=80)]
        )
        assert summary.cache_hit_rate == pytest.approx(0.8)

    def test_cache_hit_rate_of_a_session_without_tokens_is_zero(self) -> None:
        assert summarize_session([event()]).cache_hit_rate == 0.0

    def test_tool_error_rate_handles_no_calls(self) -> None:
        assert summarize_session([event()]).tool_error_rate == 0.0

    def test_sessions_are_grouped_by_identifier(self) -> None:
        summaries = summarize_sessions(
            [
                event(session="session-000000000001"),
                event(session="session-000000000001"),
                event(session="session-000000000002"),
            ]
        )
        assert sorted(summary.events for summary in summaries) == [1, 2]


class TestFrames:
    def test_event_frame_has_the_declared_columns(self, session_file: Path) -> None:
        events = extract_file(session_file, Pseudonymizer(b"salt"))
        frame = events_to_frame(events)
        assert list(frame.columns) == [field for field in frame.columns]
        assert len(frame) == len(events)

    def test_session_frame_adds_derived_columns(self) -> None:
        frame = sessions_to_frame(
            summarize_sessions([event(input_tokens=10, cache_read_tokens=90)])
        )
        assert {"cache_hit_rate", "tool_error_rate", "total_tokens"} <= set(frame.columns)

    def test_session_frame_never_exposes_the_internal_tool_map(self) -> None:
        frame = sessions_to_frame(summarize_sessions([event(tool_name="Bash", tool_calls=1)]))
        assert "tool_names" not in frame.columns
        assert "tool_names" not in session_columns()

    def test_an_empty_input_gives_an_empty_frame(self) -> None:
        assert sessions_to_frame([]).empty

    def test_division_by_zero_becomes_zero_not_nan(self) -> None:
        frame = sessions_to_frame(summarize_sessions([event()]))
        assert frame["cache_hit_rate"].iloc[0] == 0.0
        assert frame["tool_error_rate"].iloc[0] == 0.0

    def test_tool_usage_is_available_in_long_format(self) -> None:
        summaries = summarize_sessions(
            [event(tool_name="Bash", tool_calls=2), event(tool_name="Read", tool_calls=1)]
        )
        frame = tool_usage_frame(summaries)
        assert set(frame["tool"]) == {"Bash", "Read"}


class TestWriteDataset:
    def test_both_tables_are_written(self, log_tree: Path, tmp_path: Path) -> None:
        events, _ = extract_tree(log_tree, Pseudonymizer(b"salt"))
        output = tmp_path / "out"
        write_dataset(events, output)
        assert (output / "events.parquet").exists()
        assert (output / "sessions.parquet").exists()

    def test_the_written_data_is_clean(self, log_tree: Path, tmp_path: Path) -> None:
        events, _ = extract_tree(log_tree, Pseudonymizer(b"salt"))
        event_frame, session_frame, _ = write_dataset(events, tmp_path / "out")
        assert scan_frame(event_frame) == []
        assert scan_frame(session_frame) == []

    def test_rare_categories_are_folded_before_writing(
        self, log_tree: Path, tmp_path: Path
    ) -> None:
        events, _ = extract_tree(log_tree, Pseudonymizer(b"salt"))
        _, session_frame, suppressions = write_dataset(events, tmp_path / "out")
        assert "9.9.9" not in set(session_frame["client_version"])
        assert any(suppression.column == "client_version" for suppression in suppressions)

    def test_a_leak_stops_the_write(self, log_tree: Path, tmp_path: Path) -> None:
        events, _ = extract_tree(log_tree, Pseudonymizer(b"salt"))
        # Simulate a hole in the allowlist by putting content back into an event.
        events[0].tool_name = "contact someone@example.com for the key"
        output = tmp_path / "out"
        with pytest.raises(PrivacyError):
            write_dataset(events, output)
        assert not (output / "events.parquet").exists()

    def test_parquet_round_trips(self, log_tree: Path, tmp_path: Path) -> None:
        events, _ = extract_tree(log_tree, Pseudonymizer(b"salt"))
        output = tmp_path / "out"
        _, session_frame, _ = write_dataset(events, output)
        reloaded = pd.read_parquet(output / "sessions.parquet")
        assert list(reloaded.columns) == list(session_frame.columns)
        assert len(reloaded) == len(session_frame)

    def test_k_is_configurable(self, log_tree: Path, tmp_path: Path) -> None:
        events, _ = extract_tree(log_tree, Pseudonymizer(b"salt"))
        _, frame_k2, _ = write_dataset(events, tmp_path / "k2", k=2)
        _, frame_k20, _ = write_dataset(events, tmp_path / "k20", k=20)
        assert frame_k2["primary_model"].nunique() >= frame_k20["primary_model"].nunique()


class TestSessionSummaryProperties:
    def test_categorical_columns_all_exist_in_the_output(self) -> None:
        columns = set(session_columns())
        assert set(SessionSummary.CATEGORICAL) <= columns

    def test_event_categorical_columns_all_exist(self, session_file: Path) -> None:
        frame = events_to_frame(extract_file(session_file, Pseudonymizer(b"salt")))
        assert set(Event.CATEGORICAL) <= set(frame.columns)

    def test_a_rare_leak_is_not_hidden_by_k_anonymity(self, log_tree: Path, tmp_path: Path) -> None:
        """A leak occurring once must not be suppressed into 'other' unnoticed."""
        events, _ = extract_tree(log_tree, Pseudonymizer(b"salt"))
        events[0].tool_name = "someone@example.com"
        with pytest.raises(PrivacyError, match="tool_name"):
            write_dataset(events, tmp_path / "out")

    def test_a_repeated_leak_is_also_caught(self, log_tree: Path, tmp_path: Path) -> None:
        events, _ = extract_tree(log_tree, Pseudonymizer(b"salt"))
        for item in events[:20]:
            item.tool_name = "someone@example.com"
        with pytest.raises(PrivacyError):
            write_dataset(events, tmp_path / "out")
