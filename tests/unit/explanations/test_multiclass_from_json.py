"""Unit tests for MultiClassCalibratedExplanations.from_json round-trip behaviour."""

from __future__ import annotations

from typing import Any, Dict

from calibrated_explanations.explanations.explanations import MultiClassCalibratedExplanations


def make_item(index: int, class_index: int, class_label: str) -> Dict[str, Any]:
    # Minimal schema v1 shaped item accepted by serialization.from_json
    return {
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
        "collection": {"size": 2, "class_labels": {0: "a", 1: "b"}},
        "explanations": [
            make_item(0, 0, "a"),
            make_item(0, 1, "b"),
            make_item(1, 0, "a"),
            make_item(1, 1, "b"),
        ],
    }

    exported = MultiClassCalibratedExplanations.from_json(payload)

    # Expect four domain explanations
    assert len(exported.explanations) == 4

    # Check that class_index and class_label were propagated into metadata
    for i, item in enumerate(exported.explanations):
        meta = item.metadata or {}
        assert "class_index" in meta
        assert "class_label" in meta
        # simple integer checks
        assert int(meta["class_index"]) in (0, 1)
        assert meta["class_label"] in ("a", "b")
