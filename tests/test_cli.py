from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agent_telemetry.cli import EXIT_FINDINGS, EXIT_OK, EXIT_USAGE, main


@pytest.fixture(autouse=True)
def deterministic_salt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the tests touch the real salt file in the home directory."""
    monkeypatch.setenv("AGENT_TELEMETRY_SALT", "test-salt")


@pytest.fixture
def dataset(log_tree: Path, tmp_path: Path) -> Path:
    output = tmp_path / "data"
    assert main(["extract", "--source", str(log_tree), "--output", str(output)]) == EXIT_OK
    return output


class TestExtract:
    def test_extraction_writes_both_tables(self, dataset: Path) -> None:
        assert (dataset / "events.parquet").exists()
        assert (dataset / "sessions.parquet").exists()

    def test_extraction_reports_what_it_read(
        self, log_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["extract", "--source", str(log_tree), "--output", str(tmp_path / "out")])
        output = capsys.readouterr().out
        assert "files" in output
        assert "events" in output

    def test_extraction_names_unknown_log_fields(
        self, log_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["extract", "--source", str(log_tree), "--output", str(tmp_path / "out")])
        assert "aiTitle" in capsys.readouterr().out

    def test_extraction_reports_suppressions(
        self, log_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["extract", "--source", str(log_tree), "--output", str(tmp_path / "out")])
        assert "k-anonymity" in capsys.readouterr().out

    def test_a_missing_source_is_a_usage_error(self, tmp_path: Path) -> None:
        assert main(["extract", "--source", str(tmp_path / "absent")]) == EXIT_USAGE

    def test_an_empty_source_is_a_usage_error(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert (
            main(["extract", "--source", str(empty), "--output", str(tmp_path / "o")]) == EXIT_USAGE
        )


class TestScan:
    def test_a_clean_table_passes(self, dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["scan", str(dataset / "sessions.parquet")]) == EXIT_OK
        assert "no findings" in capsys.readouterr().out

    def test_the_event_table_passes_too(self, dataset: Path) -> None:
        assert main(["scan", str(dataset / "events.parquet")]) == EXIT_OK

    def test_a_dirty_table_fails(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = tmp_path / "dirty.parquet"
        pd.DataFrame({"note": ["reach me at someone@example.com"]}).to_parquet(path)
        assert main(["scan", str(path)]) == EXIT_FINDINGS
        assert "email address" in capsys.readouterr().out

    def test_a_table_below_k_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "rare.parquet"
        pd.DataFrame({"primary_model": ["a"] * 6 + ["rare"], "top_tool": ["Bash"] * 7}).to_parquet(
            path
        )
        assert main(["scan", str(path)]) == EXIT_FINDINGS
        assert "below k" in capsys.readouterr().out

    def test_a_missing_file_is_a_usage_error(self, tmp_path: Path) -> None:
        assert main(["scan", str(tmp_path / "absent.parquet")]) == EXIT_USAGE


class TestSummary:
    def test_summary_prints_every_block(
        self, dataset: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["summary", "--data", str(dataset)]) == EXIT_OK
        output = capsys.readouterr().out
        for block in ("sessions", "token mix", "models", "tools", "subagents", "delegation"):
            assert block in output

    def test_summary_without_a_dataset_is_a_usage_error(self, tmp_path: Path) -> None:
        assert main(["summary", "--data", str(tmp_path)]) == EXIT_USAGE


class TestFigures:
    def test_figures_are_written(self, dataset: Path, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        output = tmp_path / "figures"
        assert main(["figures", "--data", str(dataset), "--output", str(output)]) == EXIT_OK
        assert len(list(output.glob("*.png"))) == 4

    def test_figures_without_a_dataset_is_a_usage_error(self, tmp_path: Path) -> None:
        assert main(["figures", "--data", str(tmp_path), "--output", str(tmp_path)]) == EXIT_USAGE


class TestParser:
    def test_version_exits_cleanly(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0

    def test_no_subcommand_is_a_usage_error(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2
