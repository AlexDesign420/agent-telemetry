"""k-anonymity for the categorical columns.

Aggregate numbers are not the risk in a dataset like this; rare categories are. A
client version that appears in one session, a model nobody else runs, a tool used
exactly once: each of those narrows a row down to a single identifiable session,
and several of them together identify it outright.

Every category with fewer than k occurrences is folded into ``other``. The default
k of 5 is the usual floor for published microdata.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Suppression", "apply_k_anonymity", "rare_values", "verify_k_anonymity"]

OTHER = "other"
DEFAULT_K = 5


@dataclass(frozen=True, slots=True)
class Suppression:
    column: str
    values: tuple[str, ...]
    rows_affected: int

    def __str__(self) -> str:
        return (
            f"{self.column}: folded {len(self.values)} rare values "
            f"into '{OTHER}' across {self.rows_affected} rows"
        )


def rare_values(values: Iterable[str], k: int = DEFAULT_K) -> set[str]:
    """Values occurring fewer than ``k`` times. Empty strings are left alone."""
    counts = Counter(value for value in values if value not in ("", OTHER))
    return {value for value, count in counts.items() if count < k}


def apply_k_anonymity(
    frame: Any,
    columns: Sequence[str],
    k: int = DEFAULT_K,
) -> tuple[Any, list[Suppression]]:
    """Fold rare categories into ``other``.

    Returns the modified frame and a report of what was folded, because a
    suppression nobody is told about is a silent change to the data.
    """
    result = frame.copy()
    suppressions: list[Suppression] = []

    for column in columns:
        if column not in result.columns:
            continue
        rare = rare_values(result[column].astype(str), k)
        if not rare:
            continue
        mask = result[column].astype(str).isin(rare)
        result.loc[mask, column] = OTHER
        suppressions.append(
            Suppression(column=column, values=tuple(sorted(rare)), rows_affected=int(mask.sum()))
        )

    return result, suppressions


def verify_k_anonymity(frame: Any, columns: Sequence[str], k: int = DEFAULT_K) -> list[str]:
    """Return the columns that still hold a category below ``k``."""
    failures: list[str] = []
    for column in columns:
        if column not in frame.columns:
            continue
        remaining = rare_values(frame[column].astype(str), k)
        if remaining:
            failures.append(f"{column}: {sorted(remaining)[:5]}")
    return failures
