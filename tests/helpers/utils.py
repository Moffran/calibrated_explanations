"""General utility helpers for the test suite."""

import os


def get_env_flag(name: str) -> bool:
    """Return a boolean for environment flags treating common truthy strings as True.

    This treats '1', 'true', 'yes', 'y' (case-insensitive) as True. Everything
    else (including '0', 'false', empty string or unset) is False. Using this
    avoids Python's truthiness where a non-empty string like '0' evaluates to True.
    """
    val = os.getenv(name, "").strip().lower()
    return val in ("1", "true", "yes", "y")
