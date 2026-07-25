"""Turn identifiers into stable, irreversible labels.

Session ids, working directories and branch names are all identifying, and all
three are needed in the output: without them there is no way to group events into
sessions or to see that one project behaves differently from another.

The compromise is a keyed hash. The key lives in a local file that is gitignored,
so the mapping is stable across runs on one machine and worthless to anyone who
only has the published dataset.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from pathlib import Path

__all__ = ["Pseudonymizer", "default_salt_path", "load_or_create_salt"]

SALT_ENV = "AGENT_TELEMETRY_SALT"
SALT_BYTES = 32
DIGEST_LENGTH = 12


def default_salt_path() -> Path:
    return Path.home() / ".agent-telemetry" / "salt"


def load_or_create_salt(path: Path | None = None) -> bytes:
    """Read the salt, creating one on first use.

    ``AGENT_TELEMETRY_SALT`` overrides the file, which is what CI uses so that a
    build never writes into the home directory of whatever runner it landed on.
    """
    from_env = os.environ.get(SALT_ENV)
    if from_env:
        return from_env.encode("utf-8")

    path = path or default_salt_path()
    if path.exists():
        return path.read_bytes()

    salt = secrets.token_bytes(SALT_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(salt)
    path.chmod(0o600)
    return salt


class Pseudonymizer:
    """Keyed hashing with a stable, readable output format.

    The same input always produces the same label for a given salt, and a
    different label for a different salt. Labels are prefixed by what they stand
    for, because ``project-1f4c8a2b`` in a chart is readable and ``1f4c8a2b`` is
    not.
    """

    def __init__(self, salt: bytes | None = None) -> None:
        self._salt = salt if salt is not None else load_or_create_salt()

    def _digest(self, value: str) -> str:
        mac = hmac.new(self._salt, value.encode("utf-8"), hashlib.sha256)
        return mac.hexdigest()[:DIGEST_LENGTH]

    def label(self, prefix: str, value: str | None) -> str:
        """Pseudonymize ``value``. An empty or missing value stays empty."""
        if not value:
            return ""
        return f"{prefix}-{self._digest(value)}"

    def session(self, value: str | None) -> str:
        return self.label("session", value)

    def project(self, value: str | None) -> str:
        return self.label("project", value)

    def branch(self, value: str | None) -> str:
        return self.label("branch", value)
