"""Centralized env var readers with whitespace stripping.

Guards against docker-compose .env trailing newlines across all modules.
"""

from __future__ import annotations

import os


def env(key: str, default: str = "") -> str:
    """Read an env var, stripping whitespace."""
    return os.environ.get(key, default).strip()


def opt_env(key: str) -> str | None:
    """Read an optional env var. Returns None if absent, stripped value if present."""
    val = os.environ.get(key)
    return val.strip() if val is not None else None


def bool_env(key: str, default: str = "false") -> bool:
    """Read a boolean env var (true/1/yes)."""
    return env(key, default).lower() in ("true", "1", "yes")
