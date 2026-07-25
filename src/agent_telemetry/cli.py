"""Command line interface.

extract    read session logs and write the anonymized tables
scan       check an existing table for anything identifying
summary    print the headline numbers
figures    render the charts used in the report
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from . import __version__
from .aggregate import write_dataset
from .analysis import economics, models, sessions, tools
from .extract import extract_tree
from .privacy.kanon import DEFAULT_K, verify_k_anonymity
from .privacy.pseudonymize import Pseudonymizer
from .privacy.scan import PrivacyError, scan_frame
from .schema import Event, SessionSummary

__all__ = ["main"]

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FINDINGS = 3

DEFAULT_LOG_ROOT = Path.home() / ".claude" / "projects"


def cmd_extract(args: argparse.Namespace) -> int:
    root = Path(args.source)
    if not root.is_dir():
        print(f"source directory {root} does not exist", file=sys.stderr)
        return EXIT_USAGE

    print(f"reading {root}")
    events, stats = extract_tree(root, Pseudonymizer(), track_unknown=True)
    print(stats.summary())

    if not events:
        print("no events found", file=sys.stderr)
        return EXIT_USAGE

    try:
        event_frame, session_frame, suppressions = write_dataset(events, Path(args.output), args.k)
    except PrivacyError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return EXIT_FINDINGS

    print(f"\nwrote {args.output}/events.parquet   {len(event_frame):,} rows")
    print(f"wrote {args.output}/sessions.parquet {len(session_frame):,} rows")

    if suppressions:
        print(f"\nk-anonymity at k={args.k}:")
        for suppression in suppressions:
            print(f"  {suppression}")

    if stats.unknown_field_names:
        print(
            "\nFields in the logs that the allowlist does not know. They were not read; "
            "decide what they are before the next extraction:"
        )
        for name, count in sorted(stats.unknown_field_names.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {name} (in {count} records)")

    return EXIT_OK


def cmd_scan(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"{path} does not exist", file=sys.stderr)
        return EXIT_USAGE

    frame = pd.read_parquet(path)
    findings = scan_frame(frame)

    categorical = (
        list(SessionSummary.CATEGORICAL)
        if "primary_model" in frame.columns
        else list(Event.CATEGORICAL)
    )
    k_failures = verify_k_anonymity(frame, categorical, args.k)

    print(f"{path}: {len(frame):,} rows, {len(frame.columns)} columns")

    if findings:
        print(f"\n{len(findings)} privacy findings:")
        for finding in findings[:20]:
            print(f"  {finding}")
        if len(findings) > 20:
            print(f"  and {len(findings) - 20} more")

    if k_failures:
        print(f"\ncategories below k={args.k}:")
        for failure in k_failures:
            print(f"  {failure}")

    if findings or k_failures:
        return EXIT_FINDINGS

    print(f"\nno findings, every category at or above k={args.k}")
    return EXIT_OK


def cmd_summary(args: argparse.Namespace) -> int:
    data_dir = Path(args.data)
    session_path = data_dir / "sessions.parquet"
    event_path = data_dir / "events.parquet"

    if not session_path.exists():
        print(f"{session_path} not found; run 'agent-telemetry extract' first", file=sys.stderr)
        return EXIT_USAGE

    session_frame = pd.read_parquet(session_path)
    event_frame = pd.read_parquet(event_path) if event_path.exists() else pd.DataFrame()

    def block(title: str, frame: pd.DataFrame) -> None:
        print(f"\n{title}")
        print("-" * len(title))
        if frame.empty:
            print("  no data")
        else:
            print(frame.to_string(index=False))

    block("sessions", sessions.session_overview(session_frame))
    block("length distribution", sessions.length_distribution(session_frame))
    block("subagents", sessions.subagent_usage(session_frame))
    block("token mix", economics.token_mix(session_frame).to_frame("tokens").reset_index())
    block("cache by session length", economics.cache_efficiency_by_length(session_frame))
    block("models", models.model_profile(session_frame))

    if not event_frame.empty:
        block("tools", tools.tool_frequency(event_frame, top=15))
        block("tool errors", tools.error_rates(event_frame))
        block("tool sequences", tools.tool_transitions(event_frame, top=10))
        block("delegation", sessions.delegating_sessions(session_frame, event_frame))

    return EXIT_OK


def cmd_figures(args: argparse.Namespace) -> int:
    data_dir = Path(args.data)
    output_dir = Path(args.output)

    session_path = data_dir / "sessions.parquet"
    if not session_path.exists():
        print(f"{session_path} not found; run 'agent-telemetry extract' first", file=sys.stderr)
        return EXIT_USAGE

    try:
        from . import viz
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    session_frame = pd.read_parquet(session_path)
    event_frame = pd.read_parquet(data_dir / "events.parquet")

    written = [
        viz.save(
            viz.cache_curve(
                economics.cache_efficiency_by_length(session_frame),
                "Cache hit rate rises with session length",
            ),
            output_dir / "cache-by-length.png",
        ),
        viz.save(
            viz.bar(
                tools.tool_frequency(event_frame, top=12),
                "tool_name",
                "calls",
                "Tool calls across all sessions",
                highlight=2,
            ),
            output_dir / "tool-frequency.png",
        ),
        viz.save(
            viz.distribution(
                session_frame["events"], "Session length is heavily skewed", log_x=True
            ),
            output_dir / "session-length.png",
        ),
        viz.save(
            viz.bar(
                models.model_profile(session_frame).head(8),
                "primary_model",
                "sessions",
                "Sessions per model",
                highlight=1,
            ),
            output_dir / "model-usage.png",
        ),
    ]

    for path in written:
        print(f"wrote {path}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-telemetry",
        description="Extract and analyze metrics from Claude Code session logs",
    )
    parser.add_argument("--version", action="version", version=f"agent-telemetry {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract", help="read session logs and write the anonymized tables"
    )
    extract_parser.add_argument("--source", default=str(DEFAULT_LOG_ROOT), help="log directory")
    extract_parser.add_argument("--output", default="data", help="where to write the parquet files")
    extract_parser.add_argument(
        "-k", type=int, default=DEFAULT_K, help="k-anonymity threshold for categories"
    )
    extract_parser.set_defaults(func=cmd_extract)

    scan_parser = subparsers.add_parser("scan", help="check a table for anything identifying")
    scan_parser.add_argument("path", help="parquet file to scan")
    scan_parser.add_argument("-k", type=int, default=DEFAULT_K)
    scan_parser.set_defaults(func=cmd_scan)

    summary_parser = subparsers.add_parser("summary", help="print the headline numbers")
    summary_parser.add_argument(
        "--data", default="data", help="directory holding the parquet files"
    )
    summary_parser.set_defaults(func=cmd_summary)

    figures_parser = subparsers.add_parser("figures", help="render the charts")
    figures_parser.add_argument("--data", default="data")
    figures_parser.add_argument("--output", default="reports/figures")
    figures_parser.set_defaults(func=cmd_figures)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except PrivacyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FINDINGS
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
