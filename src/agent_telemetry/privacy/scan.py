"""Independent check that nothing identifying survived.

The allowlist already decides what may be read, so in principle this scan should
never find anything. That is exactly why it exists: it is a second, differently
built check, and a pipeline whose safety rests on one mechanism working correctly
has no way to notice when it stops working.

A finding stops the pipeline. It is deliberately not a filter, because silently
removing a leaked value would hide the fact that the allowlist has a hole.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final

__all__ = ["Finding", "PrivacyError", "scan_frame", "scan_rows", "scan_value"]


class PrivacyError(RuntimeError):
    """Something that looks identifying reached a stage where it should not exist."""


@dataclass(frozen=True, slots=True)
class Finding:
    rule: str
    column: str
    row: int
    excerpt: str
    """A short, redacted excerpt. Enough to locate the problem, not to leak it."""

    def __str__(self) -> str:
        return f"row {self.row}, column {self.column}: {self.rule} ({self.excerpt})"


# Ordered by how much a hit matters. Patterns are intentionally broad: a false
# positive costs a look at one row, a false negative costs a published secret.
RULES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("anthropic api key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}")),
    ("openai api key", re.compile(r"sk-[A-Za-z0-9]{32,}")),
    ("github token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("aws access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("slack token", re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("google api key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer token", re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}")),
    ("email address", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("absolute posix path", re.compile(r"(?:/Users/|/home/|/root/|/var/folders/)\S+")),
    ("windows path", re.compile(r"[A-Za-z]:\\\\?(?:Users|Documents)\\\\?\S*")),
    ("url with host", re.compile(r"https?://[^\s/]+")),
    ("ipv4 address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("long free text", re.compile(r"\S(?:\s+\S+){14,}")),
)

# Columns whose values are expected to look like a path or an identifier and are
# already pseudonymized. Checked against the label format instead.
_LABEL_COLUMNS: Final[frozenset[str]] = frozenset({"session", "project", "branch"})
_LABEL_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?:session|project|branch)-[0-9a-f]{12}$")


def _redact(value: str, match: re.Match[str]) -> str:
    """Show that something matched without reproducing it."""
    text = match.group(0)
    head = text[:4]
    return f"{head}... {len(text)} chars"


def scan_value(column: str, value: Any, row: int) -> Iterator[Finding]:
    """Yield findings for a single cell."""
    if value is None or isinstance(value, bool | int | float):
        return

    text = str(value)
    if not text:
        return

    if column in _LABEL_COLUMNS:
        if not _LABEL_PATTERN.match(text):
            yield Finding(
                "identifier is not a pseudonymized label", column, row, f"{len(text)} chars"
            )
        return

    for rule, pattern in RULES:
        match = pattern.search(text)
        if match:
            yield Finding(rule, column, row, _redact(text, match))


def scan_rows(rows: Iterable[Mapping[str, Any]]) -> list[Finding]:
    """Scan an iterable of mappings and return every finding."""
    findings: list[Finding] = []
    for index, row in enumerate(rows):
        for column, value in row.items():
            findings.extend(scan_value(column, value, index))
    return findings


def scan_frame(frame: Any, sample: int | None = None) -> list[Finding]:
    """Scan a pandas DataFrame.

    ``sample`` limits the scan to the first n rows, which is useful while
    iterating. The pipeline itself always scans everything: a check that only
    looks at part of the data answers a different question than the one being
    asked.
    """
    subset = frame.head(sample) if sample else frame
    return scan_rows(subset.to_dict("records"))


def enforce(frame: Any, sample: int | None = None) -> None:
    """Scan and raise if anything was found."""
    findings = scan_frame(frame, sample)
    if not findings:
        return

    summary = "\n".join(f"  {finding}" for finding in findings[:20])
    more = f"\n  and {len(findings) - 20} more" if len(findings) > 20 else ""
    raise PrivacyError(
        f"{len(findings)} findings in data that was about to be written:\n{summary}{more}\n"
        "This means the field allowlist has a hole. Fix the allowlist rather than "
        "the data, then run the extraction again."
    )
