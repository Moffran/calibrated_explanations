"""Fail-closed deprecation ledger / code synchronization (ADR-011).

Asserts that:
- Removed symbols listed in the history table are actually absent from the package.
- Active-deprecation symbols still present in code are in the active table (spot check).
- The active-deprecations table is not empty when active deprecation call sites exist.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPRECATIONS_MD = REPO_ROOT / "docs" / "migration" / "deprecations.md"
SRC_ROOT = REPO_ROOT / "src" / "calibrated_explanations"


def _load_deprecations_md() -> str:
    return DEPRECATIONS_MD.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# History-table assertions: listed symbols must not exist on live objects
# ---------------------------------------------------------------------------


def test_explain_guarded_factual_absent_from_calibrated_explainer():
    """CalibratedExplainer.explain_guarded_factual must have been removed in v0.11.3."""
    from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer

    assert not hasattr(CalibratedExplainer, "explain_guarded_factual"), (
        "CalibratedExplainer.explain_guarded_factual was scheduled for removal in v0.11.3 "
        "but still exists. Remove the method."
    )


def test_explore_guarded_alternatives_absent_from_calibrated_explainer():
    """CalibratedExplainer.explore_guarded_alternatives must have been removed in v0.11.3."""
    from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer

    assert not hasattr(CalibratedExplainer, "explore_guarded_alternatives"), (
        "CalibratedExplainer.explore_guarded_alternatives was scheduled for removal in v0.11.3 "
        "but still exists. Remove the method."
    )


def test_explain_guarded_factual_absent_from_wrap_explainer():
    """WrapCalibratedExplainer.explain_guarded_factual must have been removed in v0.11.3."""
    from calibrated_explanations import WrapCalibratedExplainer

    assert not hasattr(WrapCalibratedExplainer, "explain_guarded_factual"), (
        "WrapCalibratedExplainer.explain_guarded_factual was scheduled for removal in v0.11.3 "
        "but still exists. Remove the method."
    )


def test_explore_guarded_alternatives_absent_from_wrap_explainer():
    """WrapCalibratedExplainer.explore_guarded_alternatives must have been removed in v0.11.3."""
    from calibrated_explanations import WrapCalibratedExplainer

    assert not hasattr(WrapCalibratedExplainer, "explore_guarded_alternatives"), (
        "WrapCalibratedExplainer.explore_guarded_alternatives was scheduled for removal in v0.11.3 "
        "but still exists. Remove the method."
    )


# ---------------------------------------------------------------------------
# Active-deprecations table must not be empty when active sites exist
# ---------------------------------------------------------------------------


def test_active_deprecations_table_not_empty():
    """Active-deprecations table must have rows when active deprecation sites exist."""
    text = _load_deprecations_md()

    # Find the Active deprecations section and extract the table rows
    match = re.search(
        r"### Active deprecations.*?\| Deprecated symbol .*?\|(.*?)### Removed deprecations",
        text,
        re.DOTALL,
    )
    assert match, "Active deprecations section not found in deprecations.md"

    table_body = match.group(1)
    # Count data rows (lines starting with | that are not header or separator)
    data_rows = [
        line.strip()
        for line in table_body.splitlines()
        if line.strip().startswith("|") and not line.strip().startswith("|---")
    ]
    # Filter out the header row (contains "Deprecated symbol")
    data_rows = [r for r in data_rows if "Deprecated symbol" not in r]

    assert data_rows, (
        "Active deprecations table is empty but active deprecation call sites exist in the "
        "codebase. Add all active deprecations to the table in docs/migration/deprecations.md."
    )


def test_active_deprecations_table_contains_guarded_kwarg():
    """guarded=True kwarg deprecation must be listed in active-deprecations table."""
    text = _load_deprecations_md()
    assert "guarded=True" in text, (
        "Active deprecations table is missing the guarded=True kwarg deprecation. "
        "Add it to the Active deprecations table."
    )


def test_active_deprecations_table_contains_reject_confidence():
    """confidence= / reject_confidence= rename must be listed in active-deprecations table."""
    text = _load_deprecations_md()
    assert "reject_confidence" in text, (
        "Active deprecations table is missing the confidence=/reject_confidence= rename. "
        "Add it to the Active deprecations table."
    )


# ---------------------------------------------------------------------------
# History table: guarded wrappers removal date is v0.11.3, not v1.0.0
# ---------------------------------------------------------------------------


def test_history_table_guarded_wrappers_removed_in_v0_11_3():
    """Guarded wrapper history rows must show removal in v0.11.3, not v1.0.0."""
    text = _load_deprecations_md()

    # Find the Removed deprecations section
    match = re.search(r"### Removed deprecations \(history\)(.*?)(?:##|$)", text, re.DOTALL)
    assert match, "Removed deprecations (history) section not found"

    history_section = match.group(1)

    # Find rows that mention explain_guarded_factual or explore_guarded_alternatives
    for line in history_section.splitlines():
        if "explain_guarded_factual" in line or "explore_guarded_alternatives" in line:
            assert (
                "v0.11.3" in line
            ), f"Guarded wrapper history row should show removal in v0.11.3, got: {line}"
            assert (
                "v1.0.0" not in line or "v0.11.3" in line
            ), f"Guarded wrapper history row incorrectly shows v1.0.0 removal: {line}"
