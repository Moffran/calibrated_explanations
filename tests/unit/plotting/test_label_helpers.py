import logging
import warnings
from pathlib import Path
from typing import Any, Tuple

import pytest

# Suppress deprecation warning for importing plotting
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from calibrated_explanations import plotting

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.mark.parametrize(
    "threshold, expected",
    [
        # Tuple/List (Interval)
        ((0.2, 0.8), ("0.20 <= Y < 0.80", "Outside interval")),
        ([0.2, 0.8], ("0.20 <= Y < 0.80", "Outside interval")),
        (("0.2", "0.8"), ("0.20 <= Y < 0.80", "Outside interval")),
        # Scalar
        (0.5, ("Y < 0.50", "Y >= 0.50")),
        (3, ("Y < 3.00", "Y >= 3.00")),
        ("0.5", ("Y < 0.50", "Y >= 0.50")),
        # Invalid
        (("a", "b"), ("Target within threshold", "Outside threshold")),
        ("invalid", ("Target within threshold", "Outside threshold")),
        (None, ("Target within threshold", "Outside threshold")),
        (["invalid"], ("Target within threshold", "Outside threshold")),
        # Asymmetric bounds
        ((0.1, 0.9), ("0.10 <= Y < 0.90", "Outside interval")),
        # Non-numeric strings
        (("not", "numbers"), ("Target within threshold", "Outside threshold")),
    ],
)
def test_derive_threshold_labels(threshold: Any, expected: Tuple[str, str], caplog):
    """Should derive correct labels for various threshold inputs."""
    # Capture logs to verify debug logging for invalid intervals
    with caplog.at_level(logging.DEBUG):
        labels = plotting._derive_threshold_labels(threshold)

    assert labels == expected

    # Check for logging on invalid interval attempts that fall back
    if threshold == ("a", "b") or threshold == ("not", "numbers"):
        assert "Failed to parse threshold as interval" in caplog.text


@pytest.mark.parametrize(
    "base_path, filename, expected_rule",
    [
        (Path("foo"), "bar.txt", "path_join"),
        ("folder/", "baz", "concat"),
        ("folder", "baz", "path_join"),
        ("", "file.png", "filename_only"),
        ("/some/path/", "file.png", "concat"),
        (None, "file.png", "path_join_str"),
        ("folder\\", "baz", "concat_backslash"),
        ("mixed/separators\\", "file.txt", "path_join"),  # Has / so falls back to Path join
    ],
)
def test_format_save_path(base_path: Any, filename: str, expected_rule: str):
    """Should format save path correctly for various inputs."""
    result = plotting._format_save_path(base_path, filename)

    if expected_rule == "path_join":
        assert result == str(Path(base_path) / filename)
    elif expected_rule == "concat":
        assert result == f"{base_path}{filename}"
    elif expected_rule == "filename_only":
        assert result == filename
    elif expected_rule == "path_join_str":
        assert result == str(Path(str(base_path)) / filename)
    elif expected_rule == "concat_backslash":
        # Implementation detail: only concats if no forward slash in prefix
        if "/" not in base_path[:-1]:
            assert result == f"{base_path}{filename}"
        else:
            assert result == str(Path(base_path) / filename)
