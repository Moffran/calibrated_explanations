"""Tests for _safe_pick endpoint-duplication detection in models.from_legacy_dict (ADR-008 gap 5)."""

from __future__ import annotations

import logging


from calibrated_explanations.explanations.models import from_legacy_dict


def ragged_payload(n_rules: int, n_weights: int) -> dict:
    """Build a payload where feature_weights has fewer entries than rules."""
    return {
        "task": "classification",
        "prediction": {"predict": 0.7},
        "rules": {
            "rule": [f"x{i} <= 1.0" for i in range(n_rules)],
            "feature": list(range(n_rules)),
        },
        "feature_weights": {"predict": [0.1 * j for j in range(n_weights)]},
        "feature_predict": {"predict": [0.5 + 0.01 * j for j in range(n_weights)]},
    }


def test_should_emit_debug_log_when_safe_pick_detects_endpoint_duplication(caplog):
    """from_legacy_dict must log at DEBUG when ragged arrays force endpoint duplication."""
    payload = ragged_payload(n_rules=3, n_weights=1)

    with caplog.at_level(logging.DEBUG, logger="calibrated_explanations.explanations.models"):
        result = from_legacy_dict(0, payload)

    # Should still produce a result (fail-open for ragged payloads)
    assert result is not None
    assert len(result.rules) == 3

    # Endpoint duplication should have been logged at DEBUG
    debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    duplication_logged = any("endpoint duplication" in m for m in debug_msgs)
    assert duplication_logged, f"Expected 'endpoint duplication' in DEBUG log; got: {debug_msgs}"


def test_should_not_log_when_arrays_are_aligned(caplog):
    """from_legacy_dict must not emit endpoint-duplication log when arrays are correctly sized."""
    payload = ragged_payload(n_rules=2, n_weights=2)

    with caplog.at_level(logging.DEBUG, logger="calibrated_explanations.explanations.models"):
        result = from_legacy_dict(0, payload)

    assert len(result.rules) == 2
    debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    duplication_logged = any("endpoint duplication" in m for m in debug_msgs)
    assert (
        not duplication_logged
    ), f"Unexpected endpoint-duplication log when arrays are aligned: {debug_msgs}"


def test_should_return_none_weight_and_log_debug_when_array_is_empty(caplog):
    """_safe_pick must return None and emit a DEBUG log when the weight array is empty."""
    import logging

    payload = {
        "task": "classification",
        "prediction": {"predict": 0.7},
        "rules": {"rule": ["x0 <= 1.0"], "feature": [0]},
        "feature_weights": {"predict": []},
        "feature_predict": {"predict": []},
    }
    with caplog.at_level(logging.DEBUG, logger="calibrated_explanations.explanations.models"):
        result = from_legacy_dict(0, payload)

    assert result is not None
    assert len(result.rules) == 1
    # With an empty array, _safe_pick returns None
    assert result.rules[0].rule_weight == {"predict": None}
    # And should emit a debug log about the empty array
    debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any(
        "empty" in m for m in debug_msgs
    ), f"Expected DEBUG log about empty array; got: {debug_msgs}"
