"""Shared fixtures.

The synthetic session deliberately contains everything the pipeline must not
publish: prompt text, absolute paths, an email address, and an API key shaped
string. If a change to the extractor starts letting content through, these
fixtures are what notices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_telemetry.privacy.pseudonymize import Pseudonymizer

SALT = b"deterministic-test-salt"

TOXIC_TEXT = (
    "Please read /Users/someone/Documents/private/notes.md and email the result to "
    "someone@example.com using the key sk-ant-api03-AAAABBBBCCCCDDDDEEEEFFFF"
)


def session_records(
    session_id: str = "11111111-2222-3333-4444-555555555555",
) -> list[dict[str, Any]]:
    """A short session with the field shapes that appear in real logs."""
    return [
        {
            "type": "user",
            "sessionId": session_id,
            "cwd": "/Users/someone/projects/secret-thing",
            "gitBranch": "feature/private-branch-name",
            "timestamp": "2026-07-19T14:03:22.100Z",
            "version": "2.1.214",
            "entrypoint": "cli",
            "isSidechain": False,
            "userType": "external",
            "message": {"role": "user", "content": TOXIC_TEXT},
        },
        {
            "type": "assistant",
            "sessionId": session_id,
            "cwd": "/Users/someone/projects/secret-thing",
            "gitBranch": "feature/private-branch-name",
            "timestamp": "2026-07-19T14:05:00.000Z",
            "version": "2.1.214",
            "entrypoint": "cli",
            "effort": "high",
            "isSidechain": False,
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-5",
                "stop_reason": "tool_use",
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 340,
                    "cache_creation_input_tokens": 4000,
                    "cache_read_input_tokens": 18000,
                    "service_tier": "standard",
                },
                "content": [
                    {"type": "thinking", "thinking": TOXIC_TEXT},
                    {"type": "text", "text": "I will read the file."},
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "input": {"file_path": "/Users/someone/Documents/private/notes.md"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "sessionId": session_id,
            "cwd": "/Users/someone/projects/secret-thing",
            "timestamp": "2026-07-19T14:06:30.000Z",
            "version": "2.1.214",
            "isSidechain": False,
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "is_error": True,
                        "content": "Error: ENOENT /Users/someone/Documents/private/notes.md",
                    }
                ],
            },
        },
        {
            "type": "assistant",
            "sessionId": session_id,
            "cwd": "/Users/someone/projects/secret-thing",
            "timestamp": "2026-07-19T14:09:00.000Z",
            "version": "2.1.214",
            "isSidechain": False,
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-5",
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 120,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 22000,
                    "service_tier": "standard",
                },
                "content": [{"type": "text", "text": "The file does not exist."}],
            },
        },
        # Bookkeeping lines carry no message and, in real logs, often no sessionId.
        {"type": "mode", "mode": "plan"},
        {"type": "ai-title", "aiTitle": "Reading a private file that does not exist"},
    ]


def write_session(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", "utf-8")
    return path


@pytest.fixture
def pseudonymizer() -> Pseudonymizer:
    return Pseudonymizer(SALT)


@pytest.fixture
def session_file(tmp_path: Path) -> Path:
    return write_session(tmp_path / "project-a" / "session-one.jsonl", session_records())


@pytest.fixture
def log_tree(tmp_path: Path) -> Path:
    """Several sessions across two projects, including a subagent session."""
    root = tmp_path / "logs"

    for index in range(6):
        write_session(
            root / "project-a" / f"main-{index}.jsonl",
            session_records(f"aaaaaaaa-0000-0000-0000-{index:012d}"),
        )

    for index in range(6):
        records = session_records(f"bbbbbbbb-0000-0000-0000-{index:012d}")
        for record in records:
            record["isSidechain"] = True
            record["cwd"] = "/Users/someone/projects/other-thing"
        write_session(root / "project-b" / f"agent-{index}.jsonl", records)

    # One session on a rare client version, to exercise k-anonymity.
    rare = session_records("cccccccc-0000-0000-0000-000000000001")
    for record in rare:
        record["version"] = "9.9.9"
    write_session(root / "project-a" / "rare.jsonl", rare)

    return root
