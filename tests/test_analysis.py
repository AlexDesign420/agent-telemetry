from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agent_telemetry.aggregate import events_to_frame, sessions_to_frame, summarize_sessions
from agent_telemetry.analysis import economics, models, sessions, tools
from agent_telemetry.extract import extract_tree
from agent_telemetry.privacy.pseudonymize import Pseudonymizer


@pytest.fixture
def frames(log_tree: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    events, _ = extract_tree(log_tree, Pseudonymizer(b"salt"))
    return events_to_frame(events), sessions_to_frame(summarize_sessions(events))


class TestEconomics:
    def test_token_mix_covers_every_kind(self, frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        mix = economics.token_mix(frames[1])
        assert set(mix.index) == set(economics.TOKEN_COLUMNS)
        assert mix.sum() > 0

    def test_cache_hit_rate_is_a_share(self, frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        assert 0.0 <= economics.cache_hit_rate(frames[1]) <= 1.0

    def test_cache_hit_rate_of_empty_data_is_zero(self) -> None:
        empty = pd.DataFrame({column: [] for column in economics.TOKEN_COLUMNS})
        assert economics.cache_hit_rate(empty) == 0.0

    def test_cache_by_length_keeps_every_session(
        self, frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        grouped = economics.cache_efficiency_by_length(frames[1], bins=2)
        assert grouped["sessions"].sum() == len(frames[1])

    def test_cost_estimate_shows_the_cache_effect(
        self, frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        cost = economics.estimate_cost(frames[1])
        assert cost["with_cache"] < cost["without_cache"]
        assert 0.0 <= cost["saved_share"] <= 1.0

    def test_cost_without_cached_tokens_is_unchanged(self) -> None:
        frame = pd.DataFrame(
            {
                "input_tokens": [1_000_000],
                "output_tokens": [0],
                "cache_creation_tokens": [0],
                "cache_read_tokens": [0],
            }
        )
        cost = economics.estimate_cost(frame)
        assert cost["with_cache"] == cost["without_cache"]
        assert cost["saved"] == 0.0

    def test_tokens_by_model_is_sorted_by_usage(
        self, frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        by_model = economics.tokens_by_model(frames[1])
        assert by_model["sessions"].is_monotonic_decreasing


class TestTools:
    def test_frequency_shares_sum_to_one(self, frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        frequency = tools.tool_frequency(frames[0])
        assert frequency["share"].sum() == pytest.approx(1.0, abs=0.01)

    def test_frequency_is_sorted(self, frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        assert tools.tool_frequency(frames[0])["calls"].is_monotonic_decreasing

    def test_top_limits_the_result(self, frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        assert len(tools.tool_frequency(frames[0], top=1)) == 1

    def test_frequency_of_a_toolless_dataset_is_empty(self) -> None:
        frame = pd.DataFrame({"tool_name": ["", ""], "tool_calls": [0, 0], "session": ["a", "b"]})
        assert tools.tool_frequency(frame).empty

    def test_error_rate_matches_the_fixture(
        self, frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        rates = tools.error_rates(frames[0])
        # Every synthetic session has exactly one failing tool result.
        assert rates["error_rate"].iloc[0] == 1.0

    def test_error_rate_without_results_is_empty(self) -> None:
        frame = pd.DataFrame({"tool_results": [0], "tool_errors": [0], "session": ["a"]})
        assert tools.error_rates(frame).empty

    def test_transitions_stay_inside_a_session(self) -> None:
        frame = pd.DataFrame(
            {
                "session": ["a", "a", "b"],
                "sequence": [0, 1, 0],
                "tool_name": ["Read", "Edit", "Bash"],
            }
        )
        transitions = tools.tool_transitions(frame)
        assert len(transitions) == 1
        assert transitions.iloc[0]["from_tool"] == "Read"
        assert transitions.iloc[0]["to_tool"] == "Edit"

    def test_transitions_of_a_single_call_are_empty(self) -> None:
        frame = pd.DataFrame({"session": ["a"], "sequence": [0], "tool_name": ["Read"]})
        assert tools.tool_transitions(frame).empty

    def test_per_session_summary_counts_sessions(
        self, frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        summary = tools.tools_per_session(frames[1])
        assert summary.loc[summary["metric"] == "sessions", "value"].iloc[0] == len(frames[1])


class TestModels:
    @pytest.mark.parametrize(
        ("model", "provider"),
        [
            ("claude-sonnet-5", "anthropic"),
            ("deepseek-v4-flash", "deepseek"),
            ("qwen/qwen3-coder-30b", "local"),
            ("<synthetic>", "synthetic"),
            ("", "unknown"),
            ("other", "unknown"),
        ],
    )
    def test_provider_grouping(self, model: str, provider: str) -> None:
        assert models.provider_of(model) == provider

    def test_profile_lists_every_model(self, frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        profile = models.model_profile(frames[1])
        assert set(profile["primary_model"]) == set(frames[1]["primary_model"])

    def test_profile_is_sorted_by_session_count(
        self, frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        assert models.model_profile(frames[1])["sessions"].is_monotonic_decreasing

    def test_paired_sessions_only_include_multi_model_runs(self) -> None:
        frame = pd.DataFrame(
            {
                "session": ["a", "b"],
                "models": ["claude-opus-5+claude-sonnet-5", "claude-sonnet-5"],
                "events": [10, 20],
                "tool_calls": [1, 2],
                "duration_minutes": [1.0, 2.0],
            }
        )
        paired = models.paired_sessions(frame)
        assert len(paired) == 1
        assert paired.iloc[0]["models"] == "claude-opus-5+claude-sonnet-5"

    def test_paired_sessions_can_be_empty(self) -> None:
        frame = pd.DataFrame(
            {
                "session": ["a"],
                "models": ["claude-sonnet-5"],
                "events": [1],
                "tool_calls": [0],
                "duration_minutes": [0.0],
            }
        )
        assert models.paired_sessions(frame).empty

    def test_effort_shares_sum_to_one_per_model(
        self, frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        distribution = models.effort_distribution(frames[0])
        if distribution.empty:
            pytest.skip("fixture has no effort values")
        totals = distribution.groupby("model")["share"].sum()
        assert all(total == pytest.approx(1.0, abs=0.01) for total in totals)


class TestSessions:
    def test_overview_counts_sessions_and_events(
        self, frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        overview = sessions.session_overview(frames[1])
        assert overview.loc[overview["metric"] == "sessions", "value"].iloc[0] == len(frames[1])

    def test_overview_of_empty_data_is_empty(self) -> None:
        assert sessions.session_overview(pd.DataFrame()).empty

    def test_percentiles_are_ordered(self, frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        distribution = sessions.length_distribution(frames[1])
        row = distribution[distribution["metric"] == "events"].iloc[0]
        assert row["p50"] <= row["p90"] <= row["p99"]

    def test_subagent_sessions_are_counted_not_their_events(
        self, frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        usage = sessions.subagent_usage(frames[1])
        count = usage.loc[usage["metric"] == "subagent sessions", "value"].iloc[0]
        assert count == int(frames[1]["is_sidechain"].sum()) == 6

    def test_delegating_sessions_look_at_the_agent_tool(
        self, frames: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        # The fixture only calls Read, so nothing delegated.
        delegation = sessions.delegating_sessions(frames[1], frames[0])
        assert (
            delegation.loc[
                delegation["metric"] == "main sessions that spawned an agent", "value"
            ].iloc[0]
            == 0
        )

    def test_version_timeline_is_in_version_order(self) -> None:
        frame = pd.DataFrame(
            {
                "session": ["a", "b", "c"],
                "client_version": ["2.1.10", "2.1.9", "2.1.100"],
                "events": [1, 2, 3],
                "cache_hit_rate": [0.5, 0.5, 0.5],
            }
        )
        timeline = sessions.version_timeline(frame)
        assert list(timeline["client_version"]) == ["2.1.9", "2.1.10", "2.1.100"]

    def test_version_timeline_survives_an_odd_version_string(self) -> None:
        frame = pd.DataFrame(
            {
                "session": ["a", "b"],
                "client_version": ["2.1.9", "nightly"],
                "events": [1, 2],
                "cache_hit_rate": [0.5, 0.5],
            }
        )
        assert len(sessions.version_timeline(frame)) == 2
