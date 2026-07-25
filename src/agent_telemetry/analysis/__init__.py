"""Analyses over the extracted tables.

Each module takes a DataFrame and returns a DataFrame. Nothing here reads from
disk or plots anything, so the notebooks stay thin and the results stay testable.
"""

from . import economics, models, sessions, tools

__all__ = ["economics", "models", "sessions", "tools"]
