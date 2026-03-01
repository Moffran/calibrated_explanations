"""Unit tests for MultiClassCalibratedExplanations.from_json round-trip behaviour."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from calibrated_explanations.explanations.explanations import MultiClassCalibratedExplanations
from calibrated_explanations.utils.exceptions import ValidationError


def make_item(index: int, class_index: int, class_label: str) -> Dict[str, Any]:
    # Minimal schema v1 shaped item accepted by serialization.from_json
    return {
        "schema_version": "1.0.0",
        "task": "classification",
        "index": int(index),
        "explanation_type": "factual",
        "prediction": {"predict": 0.0},
        "rules": [],
        "provenance": None,
        "metadata": {"class_index": int(class_index), "class_label": class_label},
    }


def test_multiclass_from_json_preserves_class_annotations():
    payload = {
        "schema_version": "1.0.0",
        "collection": {"size": 2, "class_labels": {0: "a", 1: "b"}},
        "explanations": [
            make_item(0, 0, "a"),
            make_item(0, 1, "b"),
            make_item(1, 0, "a"),
            make_item(1, 1, "b"),
        ],
    }

    exported = MultiClassCalibratedExplanations.from_json(payload)

    # Flat view remains available
    assert len(exported.explanations) == 4

    # Grouped view restores instance -> class mapping structure
    assert len(exported.explanations_by_instance) == 2
    assert set(exported.explanations_by_instance[0].keys()) == {0, 1}
    assert set(exported.explanations_by_instance[1].keys()) == {0, 1}

    # Check that class_index and class_label were propagated into metadata
    for i, item in enumerate(exported.explanations):
        meta = item.metadata or {}
        assert "class_index" in meta
        assert "class_label" in meta
        # simple integer checks
        assert int(meta["class_index"]) in (0, 1)
        assert meta["class_label"] in ("a", "b")


def test_multiclass_from_json_rejects_missing_top_level_schema_version():
    payload = {
        "collection": {"size": 1},
        "explanations": [make_item(0, 0, "a")],
    }

    with pytest.raises(ValidationError, match="schema version"):
        MultiClassCalibratedExplanations.from_json(payload)


def test_multiclass_from_json_rejects_item_schema_mismatch():
    payload = {
        "schema_version": "1.0.0",
        "collection": {"size": 1},
        "explanations": [{**make_item(0, 0, "a"), "schema_version": "9.9.9"}],
    }

    with pytest.raises(ValidationError, match="schema version"):
        MultiClassCalibratedExplanations.from_json(payload)
