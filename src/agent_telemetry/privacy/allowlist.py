"""What the extractor is allowed to read.

An allowlist rather than a blocklist, and the distinction is the whole point. A
blocklist protects against the fields you thought of; when the log format grows a
new one it flows straight through into the published data. An allowlist fails the
other way: an unknown field is ignored until someone deliberately adds it here.

Nothing in this module returns text. Fields that carry text are read for their
length or their presence only, and the reading happens in ``extract.py`` against
the names declared here.
"""

from __future__ import annotations

from typing import Any, Final

__all__ = [
    "ALLOWED_CONTENT_BLOCK_FIELDS",
    "ALLOWED_MESSAGE_FIELDS",
    "ALLOWED_TOP_LEVEL_FIELDS",
    "ALLOWED_USAGE_FIELDS",
    "PSEUDONYMIZED_FIELDS",
    "TEXT_BEARING_FIELDS",
    "filter_mapping",
    "unknown_fields",
]

# Read as-is: categories, flags and counters.
ALLOWED_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "type",
        "timestamp",
        "isSidechain",
        "userType",
        "entrypoint",
        "version",
        "permissionMode",
        "effort",
        "subtype",
    }
)

# Read, but hashed before anything keeps them.
PSEUDONYMIZED_FIELDS: Final[frozenset[str]] = frozenset({"sessionId", "cwd", "gitBranch"})

ALLOWED_MESSAGE_FIELDS: Final[frozenset[str]] = frozenset(
    {"role", "model", "stop_reason", "usage"}
)

ALLOWED_USAGE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "service_tier",
    }
)

# Inside a content block, only the shape is read: which kind of block it is, the
# name of the tool it calls, whether a result was an error.
ALLOWED_CONTENT_BLOCK_FIELDS: Final[frozenset[str]] = frozenset({"type", "name", "is_error"})

# Fields that hold text or paths and are never read for their value. Listed
# explicitly so the intent is visible: these are known, and known to be excluded.
TEXT_BEARING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "content",
        "text",
        "thinking",
        "input",
        "slug",
        "lastPrompt",
        "summary",
        "title",
        "toolUseResult",
        "attachment",
        "snapshot",
        "message_content",
        "signature",
    }
)


def filter_mapping(raw: dict[str, Any], allowed: frozenset[str]) -> dict[str, Any]:
    """Return only the allowed keys of ``raw``.

    Values are returned untouched; deciding what may be read is this function's
    job, converting it is the extractor's.
    """
    return {key: value for key, value in raw.items() if key in allowed}


def unknown_fields(raw: dict[str, Any]) -> set[str]:
    """Top level keys that are neither allowed, pseudonymized, nor known text.

    Used by ``agent-telemetry extract --report-unknown`` to notice when the log
    format has grown a field, which is the moment to decide what it is rather
    than to find out later that it was published.
    """
    known = ALLOWED_TOP_LEVEL_FIELDS | PSEUDONYMIZED_FIELDS | TEXT_BEARING_FIELDS
    structural = {"message", "uuid", "parentUuid", "requestId", "promptId", "sessionId"}
    return set(raw) - known - structural
