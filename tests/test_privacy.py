"""The tests that matter most.

Everything else in this package is an analysis that can be wrong in a way that
costs someone an afternoon. These check the property that cannot be wrong once.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agent_telemetry.privacy.allowlist import (
    ALLOWED_TOP_LEVEL_FIELDS,
    ALLOWED_USAGE_FIELDS,
    PSEUDONYMIZED_FIELDS,
    TEXT_BEARING_FIELDS,
    filter_mapping,
    unknown_fields,
)
from agent_telemetry.privacy.kanon import (
    OTHER,
    apply_k_anonymity,
    rare_values,
    verify_k_anonymity,
)
from agent_telemetry.privacy.pseudonymize import Pseudonymizer, load_or_create_salt
from agent_telemetry.privacy.scan import PrivacyError, enforce, scan_frame, scan_rows

from .conftest import TOXIC_TEXT


class TestAllowlist:
    def test_unlisted_keys_are_dropped(self) -> None:
        raw = {"type": "user", "content": "secret", "cwd": "/Users/someone"}
        assert filter_mapping(raw, ALLOWED_TOP_LEVEL_FIELDS) == {"type": "user"}

    def test_text_fields_are_never_allowed(self) -> None:
        assert not (TEXT_BEARING_FIELDS & ALLOWED_TOP_LEVEL_FIELDS)
        assert not (TEXT_BEARING_FIELDS & ALLOWED_USAGE_FIELDS)

    def test_identifiers_are_not_readable_directly(self) -> None:
        assert not (PSEUDONYMIZED_FIELDS & ALLOWED_TOP_LEVEL_FIELDS)

    def test_a_new_log_field_is_reported_not_absorbed(self) -> None:
        raw = {"type": "user", "brandNewField": "whatever this is"}
        assert unknown_fields(raw) == {"brandNewField"}

    def test_known_fields_are_not_reported_as_unknown(self) -> None:
        raw = {"type": "user", "cwd": "/x", "content": "text", "uuid": "abc"}
        assert unknown_fields(raw) == set()

    def test_usage_allowlist_drops_unlisted_metrics(self) -> None:
        usage = {"input_tokens": 5, "inference_geo": "somewhere", "speed": 12}
        assert filter_mapping(usage, ALLOWED_USAGE_FIELDS) == {"input_tokens": 5}


class TestPseudonymizer:
    def test_same_input_gives_the_same_label(self, pseudonymizer: Pseudonymizer) -> None:
        assert pseudonymizer.session("abc") == pseudonymizer.session("abc")

    def test_different_inputs_give_different_labels(self, pseudonymizer: Pseudonymizer) -> None:
        assert pseudonymizer.session("abc") != pseudonymizer.session("abd")

    def test_a_different_salt_gives_a_different_label(self) -> None:
        assert Pseudonymizer(b"one").session("abc") != Pseudonymizer(b"two").session("abc")

    def test_label_does_not_contain_the_input(self, pseudonymizer: Pseudonymizer) -> None:
        secret = "/Users/someone/projects/secret-thing"
        label = pseudonymizer.project(secret)
        assert "someone" not in label
        assert "secret" not in label
        assert label.startswith("project-")

    def test_empty_input_stays_empty(self, pseudonymizer: Pseudonymizer) -> None:
        assert pseudonymizer.branch("") == ""
        assert pseudonymizer.branch(None) == ""

    def test_salt_from_the_environment_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_TELEMETRY_SALT", "from-env")
        assert load_or_create_salt(Path("/nonexistent/salt")) == b"from-env"

    def test_generated_salt_is_written_once_and_reused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AGENT_TELEMETRY_SALT", raising=False)
        path = tmp_path / "nested" / "salt"
        first = load_or_create_salt(path)
        assert path.exists()
        assert load_or_create_salt(path) == first

    def test_generated_salt_is_not_world_readable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AGENT_TELEMETRY_SALT", raising=False)
        path = tmp_path / "salt"
        load_or_create_salt(path)
        assert path.stat().st_mode & 0o077 == 0


class TestScan:
    @pytest.mark.parametrize(
        ("rule", "value"),
        [
            ("anthropic api key", "sk-ant-api03-AAAABBBBCCCCDDDDEEEE"),
            ("openai api key", "sk-" + "A" * 40),
            ("github token", "ghp_" + "b" * 30),
            ("aws access key", "AKIAIOSFODNN7EXAMPLE"),
            ("slack token", "xoxb-123456789012-abcdefghijkl"),
            ("private key block", "-----BEGIN RSA PRIVATE KEY-----"),
            ("email address", "someone@example.com"),
            ("absolute posix path", "/Users/someone/Documents/notes.md"),
            ("url with host", "https://internal.example.com/dashboard"),
            ("ipv4 address", "192.168.1.42"),
        ],
    )
    def test_each_rule_fires(self, rule: str, value: str) -> None:
        findings = list(scan_rows([{"note": value}]))
        assert findings, f"{rule} was not detected"
        assert any(finding.rule == rule for finding in findings)

    def test_prompt_text_is_caught(self) -> None:
        assert scan_rows([{"note": TOXIC_TEXT}])

    def test_a_finding_does_not_reproduce_the_secret(self) -> None:
        secret = "sk-ant-api03-AAAABBBBCCCCDDDDEEEE"
        finding = scan_rows([{"note": secret}])[0]
        assert secret not in finding.excerpt
        assert secret not in str(finding)

    def test_clean_metrics_produce_nothing(self) -> None:
        rows = [
            {
                "session": "session-0123456789ab",
                "project": "project-0123456789ab",
                "branch": "branch-0123456789ab",
                "model": "claude-sonnet-5",
                "tool_name": "Bash",
                "input_tokens": 1234,
                "cache_hit_rate": 0.98,
                "is_sidechain": True,
            }
        ]
        assert scan_rows(rows) == []

    def test_numbers_are_not_scanned_as_text(self) -> None:
        assert scan_rows([{"tokens": 19216811, "rate": 0.4}]) == []

    def test_an_unhashed_identifier_column_is_a_finding(self) -> None:
        findings = scan_rows([{"session": "11111111-2222-3333-4444-555555555555"}])
        assert findings
        assert "pseudonymized" in findings[0].rule

    def test_a_correctly_shaped_label_passes(self) -> None:
        assert scan_rows([{"project": "project-abcdef012345"}]) == []

    def test_enforce_raises_and_names_the_column(self) -> None:
        frame = pd.DataFrame({"note": ["someone@example.com"], "tokens": [5]})
        with pytest.raises(PrivacyError, match="note"):
            enforce(frame)

    def test_enforce_is_silent_on_clean_data(self) -> None:
        enforce(pd.DataFrame({"tokens": [1, 2, 3], "model": ["claude-sonnet-5"] * 3}))

    def test_scan_frame_can_be_limited_while_iterating(self) -> None:
        frame = pd.DataFrame({"note": ["fine"] * 10 + ["someone@example.com"]})
        assert scan_frame(frame, sample=5) == []
        assert scan_frame(frame) != []


class TestKAnonymity:
    def test_values_below_k_are_identified(self) -> None:
        values = ["a"] * 6 + ["b"] * 2
        assert rare_values(values, k=5) == {"b"}

    def test_empty_values_are_left_alone(self) -> None:
        assert rare_values(["", "", "a", "a", "a", "a", "a"], k=5) == set()

    def test_rare_categories_are_folded(self) -> None:
        frame = pd.DataFrame({"version": ["2.1.1"] * 6 + ["9.9.9"]})
        result, suppressions = apply_k_anonymity(frame, ["version"], k=5)
        assert set(result["version"]) == {"2.1.1", OTHER}
        assert suppressions[0].rows_affected == 1

    def test_common_categories_survive(self) -> None:
        frame = pd.DataFrame({"model": ["a"] * 5 + ["b"] * 5})
        result, suppressions = apply_k_anonymity(frame, ["model"], k=5)
        assert set(result["model"]) == {"a", "b"}
        assert suppressions == []

    def test_the_result_verifies_clean(self) -> None:
        frame = pd.DataFrame({"model": ["a"] * 6 + ["b", "c", "d"]})
        result, _ = apply_k_anonymity(frame, ["model"], k=5)
        assert verify_k_anonymity(result, ["model"], k=5) == []

    def test_verification_catches_an_unprocessed_frame(self) -> None:
        frame = pd.DataFrame({"model": ["a"] * 6 + ["b"]})
        assert verify_k_anonymity(frame, ["model"], k=5)

    def test_missing_columns_are_ignored(self) -> None:
        frame = pd.DataFrame({"model": ["a"] * 6})
        result, suppressions = apply_k_anonymity(frame, ["model", "absent"], k=5)
        assert len(result) == 6
        assert suppressions == []

    def test_suppression_reports_what_it_did(self) -> None:
        frame = pd.DataFrame({"tool": ["Bash"] * 6 + ["Rare1", "Rare2"]})
        _, suppressions = apply_k_anonymity(frame, ["tool"], k=5)
        text = str(suppressions[0])
        assert "tool" in text
        assert "2" in text
