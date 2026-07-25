from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_telemetry.extract import extract_file, extract_tree, iter_session_files
from agent_telemetry.privacy.pseudonymize import Pseudonymizer
from agent_telemetry.privacy.scan import scan_rows
from agent_telemetry.schema import Event, as_row, event_columns

from .conftest import TOXIC_TEXT, session_records, write_session


class TestNoContentEscapes:
    """The extractor's one hard requirement."""

    def test_no_field_of_any_event_contains_the_prompt_text(
        self, session_file: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        events = extract_file(session_file, pseudonymizer)
        for event in events:
            for value in as_row(event, event_columns()).values():
                assert TOXIC_TEXT not in str(value)

    def test_the_extracted_events_survive_the_privacy_scan(
        self, session_file: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        events = extract_file(session_file, pseudonymizer)
        rows = [as_row(event, event_columns()) for event in events]
        assert scan_rows(rows) == []

    def test_the_working_directory_never_appears(
        self, session_file: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        events = extract_file(session_file, pseudonymizer)
        assert all("someone" not in event.project for event in events)
        assert all(event.project.startswith("project-") for event in events)

    def test_the_branch_name_never_appears(
        self, session_file: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        events = extract_file(session_file, pseudonymizer)
        assert all("private" not in event.branch for event in events)

    def test_text_is_measured_not_kept(
        self, session_file: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        events = extract_file(session_file, pseudonymizer)
        assert any(event.content_length > 100 for event in events)


class TestFieldMapping:
    def test_usage_fields_are_carried_over(
        self, session_file: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        events = extract_file(session_file, pseudonymizer)
        assistant = next(event for event in events if event.model)
        assert assistant.input_tokens == 12
        assert assistant.output_tokens == 340
        assert assistant.cache_creation_tokens == 4000
        assert assistant.cache_read_tokens == 18000
        assert assistant.service_tier == "standard"

    def test_categories_are_carried_over(
        self, session_file: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        events = extract_file(session_file, pseudonymizer)
        assistant = next(event for event in events if event.model)
        assert assistant.model == "claude-sonnet-5"
        assert assistant.effort == "high"
        assert assistant.client_version == "2.1.214"
        assert assistant.stop_reason == "tool_use"

    def test_content_blocks_are_counted(
        self, session_file: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        events = extract_file(session_file, pseudonymizer)
        assistant = next(event for event in events if event.tool_calls)
        assert assistant.tool_calls == 1
        assert assistant.tool_name == "Read"
        assert assistant.thinking_blocks == 1
        assert assistant.text_blocks == 1

    def test_tool_errors_are_counted(
        self, session_file: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        events = extract_file(session_file, pseudonymizer)
        assert sum(event.tool_errors for event in events) == 1
        assert sum(event.tool_results for event in events) == 1

    def test_string_content_is_measured(
        self, session_file: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        events = extract_file(session_file, pseudonymizer)
        user = next(event for event in events if event.type == "user" and event.text_blocks)
        assert user.content_length == len(TOXIC_TEXT)


class TestTimestamps:
    def test_published_timestamps_are_truncated_to_the_hour(
        self, session_file: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        events = extract_file(session_file, pseudonymizer)
        stamps = {event.timestamp for event in events if event.timestamp}
        assert stamps == {"2026-07-19T14:00:00Z"}

    def test_offsets_keep_the_resolution_the_timestamps_lost(
        self, session_file: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        events = extract_file(session_file, pseudonymizer)
        offsets = [event.offset_seconds for event in events if event.offset_seconds]
        assert max(offsets) == pytest.approx(337.9, abs=0.5)

    def test_a_missing_timestamp_is_empty_not_invented(
        self, tmp_path: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        path = write_session(tmp_path / "s.jsonl", [{"type": "mode"}])
        assert extract_file(path, pseudonymizer)[0].timestamp == ""

    def test_an_unparseable_timestamp_does_not_raise(
        self, tmp_path: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        path = write_session(tmp_path / "s.jsonl", [{"type": "user", "timestamp": "not a date"}])
        assert extract_file(path, pseudonymizer)[0].offset_seconds == 0.0


class TestSessionIdentity:
    def test_the_file_defines_the_session(
        self, session_file: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        events = extract_file(session_file, pseudonymizer)
        assert len({event.session for event in events}) == 1

    def test_bookkeeping_lines_join_their_own_session(
        self, session_file: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        events = extract_file(session_file, pseudonymizer)
        modes = [event for event in events if event.type == "mode"]
        assert modes
        assert modes[0].session == events[0].session

    def test_sequence_numbers_are_dense_and_ordered(
        self, session_file: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        events = extract_file(session_file, pseudonymizer)
        assert [event.sequence for event in events] == list(range(len(events)))

    def test_separate_files_are_separate_sessions(
        self, log_tree: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        events, _ = extract_tree(log_tree, pseudonymizer)
        assert len({event.session for event in events}) == 13


class TestMalformedInput:
    def test_a_broken_line_is_counted_and_skipped(
        self, tmp_path: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        path = tmp_path / "broken.jsonl"
        path.write_text(json.dumps({"type": "user"}) + "\n" + '{"type": "assis\n', encoding="utf-8")
        events = extract_file(path, pseudonymizer)
        assert len(events) == 1

    def test_a_truncated_tail_does_not_lose_the_rest(
        self, tmp_path: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        records = session_records()
        path = tmp_path / "partial.jsonl"
        text = "\n".join(json.dumps(record) for record in records)
        path.write_text(text + '\n{"type": "assist', encoding="utf-8")

        from agent_telemetry.extract import ExtractionStats

        stats = ExtractionStats()
        events = extract_file(path, pseudonymizer, stats)
        assert len(events) == len(records)
        assert stats.malformed_lines == 1

    def test_a_json_array_line_is_not_treated_as_an_event(
        self, tmp_path: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        path = tmp_path / "array.jsonl"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")
        assert extract_file(path, pseudonymizer) == []

    def test_blank_lines_are_ignored(self, tmp_path: Path, pseudonymizer: Pseudonymizer) -> None:
        path = tmp_path / "blanks.jsonl"
        path.write_text('\n\n{"type": "user"}\n\n', encoding="utf-8")
        assert len(extract_file(path, pseudonymizer)) == 1

    def test_an_empty_file_is_reported(self, tmp_path: Path, pseudonymizer: Pseudonymizer) -> None:
        from agent_telemetry.extract import ExtractionStats

        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        stats = ExtractionStats()
        extract_file(path, pseudonymizer, stats)
        assert stats.empty_files == 1


class TestTree:
    def test_every_session_file_is_found(self, log_tree: Path) -> None:
        assert len(list(iter_session_files(log_tree))) == 13

    def test_traversal_order_is_stable(self, log_tree: Path) -> None:
        assert list(iter_session_files(log_tree)) == list(iter_session_files(log_tree))

    def test_stats_add_up(self, log_tree: Path, pseudonymizer: Pseudonymizer) -> None:
        events, stats = extract_tree(log_tree, pseudonymizer)
        assert stats.files == 13
        assert stats.events == len(events)
        assert stats.malformed_lines == 0

    def test_unknown_fields_are_reported_when_asked(
        self, log_tree: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        _, stats = extract_tree(log_tree, pseudonymizer, track_unknown=True)
        assert "aiTitle" in stats.unknown_field_names

    def test_unknown_fields_are_not_tracked_by_default(
        self, log_tree: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        _, stats = extract_tree(log_tree, pseudonymizer)
        assert stats.unknown_field_names == {}

    def test_summary_mentions_what_it_saw(
        self, log_tree: Path, pseudonymizer: Pseudonymizer
    ) -> None:
        _, stats = extract_tree(log_tree, pseudonymizer, track_unknown=True)
        text = stats.summary()
        assert "files" in text
        assert "unknown fields" in text


class TestEventProperties:
    def test_billable_input_excludes_cache_reads(self) -> None:
        event = Event(
            session="s",
            project="p",
            branch="",
            sequence=0,
            timestamp="",
            input_tokens=10,
            cache_creation_tokens=100,
            cache_read_tokens=1000,
        )
        assert event.billable_input_tokens == 110

    def test_total_tokens_sums_every_kind(self) -> None:
        event = Event(
            session="s",
            project="p",
            branch="",
            sequence=0,
            timestamp="",
            input_tokens=1,
            output_tokens=2,
            cache_creation_tokens=4,
            cache_read_tokens=8,
        )
        assert event.total_tokens == 15
