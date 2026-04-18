"""Test per-class reject_policy mapping into multi-class collections."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import warnings

from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator
from calibrated_explanations.core.explain.orchestrator import coerce_legacy_reject_result
from calibrated_explanations.core.explain.orchestrator import resolve_effective_reject_policy
from calibrated_explanations.core.explain.orchestrator import resolve_reject_policy_spec
from calibrated_explanations.core.explain.orchestrator import resolve_source_indices_for_payload
from calibrated_explanations.explanations.explanations import MultiClassCalibratedExplanations
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import (
    RejectDecisionArtifact,
    RejectPayloadArtifact,
    RejectResult,
    RejectResultV2,
)


class DummyExplainer:
    def __init__(self):
        self.y_cal = np.array([0, 1, 2])
        self.class_labels = {0: "a", 1: "b", 2: "c"}
        self.num_features = 1
        self.interval_summary = None
        self.features_to_ignore = []

        # Provide a simple reject_orchestrator with an apply_policy hook
        class RO:
            def apply_policy(self, policy, x, explain_fn=None, bins=None, **kwargs):
                # Reject even-indexed instances for demonstration
                rejected = [i % 2 == 0 for i in range(len(x))]
                if policy is RejectPolicy.ONLY_ACCEPTED:
                    idxs = [i for i, v in enumerate(rejected) if not v]
                elif policy is RejectPolicy.ONLY_REJECTED:
                    idxs = [i for i, v in enumerate(rejected) if v]
                else:
                    # Keep backward-compatible odd behavior used by earlier tests:
                    # FLAG with rejected-only payload, now supported via fallback.
                    idxs = [i for i, v in enumerate(rejected) if v]
                payload = [SimpleExp(i, 0) for i in idxs] if idxs else None
                return RejectResult(
                    explanation=payload,
                    rejected=np.array(rejected, dtype=bool),
                    policy=policy,
                    metadata={"source_indices": idxs, "original_count": len(x)},
                )

        self.reject_orchestrator = RO()


class SimpleExp:
    def __init__(self, index: int, klass: int):
        self.index = index
        self.klass = klass

    def get_class_labels(self):
        return {0: "a", 1: "b", 2: "c"}


def test_per_class_reject_policy_mapping():
    expl = DummyExplainer()
    orch = ExplanationOrchestrator(expl)

    # Call invoke_factual with multi_labels_enabled and a reject policy
    x = np.array([[0.0], [1.0]])
    coll = orch.invoke_factual(
        x,
        threshold=None,
        low_high_percentiles=None,
        bins=None,
        features_to_ignore=None,
        discretizer=None,
        _use_plugin=True,
        reject_policy=RejectPolicy.FLAG,
        multi_labels_enabled=True,
        interval_summary=None,
    )

    assert isinstance(coll, MultiClassCalibratedExplanations)
    # Our fake reject_orchestrator rejects index 0 and accepts index 1
    # For each class, explanations should be present for index 0 only
    for i, inst in enumerate(coll.explanations):
        # index 0 should have class entries for each class
        if i == 0:
            assert set(inst.keys()) == {0, 1, 2}
        else:
            # index 1: no explanations (apply_policy returned only rejected subset)
            # implementation may leave empty dict for accepted-only
            assert isinstance(inst, dict)


def test_per_class_only_accepted_mapping_uses_source_indices():
    expl = DummyExplainer()
    orch = ExplanationOrchestrator(expl)
    x = np.array([[0.0], [1.0], [2.0], [3.0]])

    coll = orch.invoke_factual(
        x,
        threshold=None,
        low_high_percentiles=None,
        bins=None,
        features_to_ignore=None,
        discretizer=None,
        _use_plugin=True,
        reject_policy=RejectPolicy.ONLY_ACCEPTED,
        multi_labels_enabled=True,
        interval_summary=None,
    )

    assert isinstance(coll, MultiClassCalibratedExplanations)
    # accepted indexes are 1 and 3 for each class
    assert set(coll.explanations[1].keys()) == {0, 1, 2}
    assert set(coll.explanations[3].keys()) == {0, 1, 2}
    assert coll.explanations[0] == {}
    assert coll.explanations[2] == {}


def test_resolve_source_indices_prefers_metadata():
    idxs = resolve_source_indices_for_payload(
        policy=RejectPolicy.ONLY_ACCEPTED,
        metadata={"source_indices": [1, 3], "original_count": 4},
        rejected_mask=np.array([True, False, True, False]),
        payload_count=2,
    )
    assert idxs == [1, 3]


def test_resolve_source_indices_fallback_from_mask_when_missing_metadata():
    with pytest.warns(UserWarning, match="missing source_indices"):
        idxs = resolve_source_indices_for_payload(
            policy=RejectPolicy.ONLY_REJECTED,
            metadata={},
            rejected_mask=np.array([True, False, True, False]),
            payload_count=2,
        )
    assert idxs == [0, 2]


def test_resolve_source_indices_uses_cardinality_inference_for_subset_payload():
    with pytest.warns(UserWarning, match="inferred from accepted subset"):
        idxs = resolve_source_indices_for_payload(
            policy=RejectPolicy.FLAG,
            metadata={},
            rejected_mask=np.array([True, False, False, False]),
            payload_count=3,
        )
    assert idxs == [1, 2, 3]


def test_resolve_source_indices_invalid_metadata_falls_back_to_mask():
    with pytest.warns(UserWarning, match="metadata is invalid"):
        idxs = resolve_source_indices_for_payload(
            policy=RejectPolicy.ONLY_ACCEPTED,
            metadata={"source_indices": [1, -1], "original_count": 4},
            rejected_mask=np.array([True, False, True, False]),
            payload_count=2,
        )
    assert idxs == [1, 3]


def test_resolve_source_indices_rejects_duplicate_metadata_indices():
    with pytest.warns(UserWarning, match="metadata is invalid"):
        idxs = resolve_source_indices_for_payload(
            policy=RejectPolicy.ONLY_ACCEPTED,
            metadata={"source_indices": [1, 1], "original_count": 4},
            rejected_mask=np.array([True, False, True, False]),
            payload_count=2,
        )
    assert idxs == [1, 3]


def test_resolve_source_indices_rejects_unsorted_metadata_indices():
    with pytest.warns(UserWarning, match="metadata is invalid"):
        idxs = resolve_source_indices_for_payload(
            policy=RejectPolicy.ONLY_ACCEPTED,
            metadata={"source_indices": [3, 1], "original_count": 4},
            rejected_mask=np.array([True, False, True, False]),
            payload_count=2,
        )
    assert idxs == [1, 3]


def test_resolve_source_indices_rejects_out_of_range_metadata_indices():
    with pytest.warns(UserWarning, match="metadata is invalid"):
        idxs = resolve_source_indices_for_payload(
            policy=RejectPolicy.ONLY_ACCEPTED,
            metadata={"source_indices": [1, 5], "original_count": 4},
            rejected_mask=np.array([True, False, True, False]),
            payload_count=2,
        )
    assert idxs == [1, 3]


def test_resolve_source_indices_returns_none_when_cardinality_cannot_be_mapped():
    with pytest.warns(UserWarning, match="Unable to map reject payload"):
        idxs = resolve_source_indices_for_payload(
            policy=RejectPolicy.FLAG,
            metadata={},
            rejected_mask=np.array([True, False, True, False]),
            payload_count=1,
        )
    assert idxs is None


def test_resolve_source_indices_none_mask_requires_empty_payload():
    idxs = resolve_source_indices_for_payload(
        policy=RejectPolicy.FLAG,
        metadata={},
        rejected_mask=None,
        payload_count=0,
    )
    assert idxs == []


def test_resolve_source_indices_none_mask_non_empty_payload_returns_none():
    idxs = resolve_source_indices_for_payload(
        policy=RejectPolicy.FLAG,
        metadata={},
        rejected_mask=None,
        payload_count=2,
    )
    assert idxs is None


def test_resolve_source_indices_invalid_policy_value_returns_none_with_warning():
    with pytest.raises(ValueError, match="not-a-policy"):
        resolve_source_indices_for_payload(
            policy="not-a-policy",
            metadata={},
            rejected_mask=np.array([True, False, True]),
            payload_count=2,
        )


def test_resolve_reject_policy_spec_initializes_orchestrators_when_missing():
    plugin_manager = MagicMock()
    explainer = type(
        "Explainer", (), {"reject_orchestrator": None, "plugin_manager": plugin_manager}
    )

    with patch(
        "calibrated_explanations.core.reject.orchestrator.resolve_policy_spec",
        return_value="resolved-policy",
    ) as resolver:
        resolved = resolve_reject_policy_spec("candidate", explainer)

    assert resolved == "resolved-policy"
    plugin_manager.initialize_orchestrators.assert_called_once()
    resolver.assert_called_once_with("candidate", explainer)


def test_resolve_effective_reject_policy_passes_default_policy_and_logger():
    plugin_manager = MagicMock()
    explainer = type(
        "Explainer", (), {"reject_orchestrator": None, "plugin_manager": plugin_manager}
    )

    with patch(
        "calibrated_explanations.core.reject.orchestrator.resolve_effective_reject_policy",
        return_value="effective-policy",
    ) as resolver:
        resolved = resolve_effective_reject_policy(
            "candidate",
            explainer,
            default_policy=RejectPolicy.FLAG,
        )

    assert resolved == "effective-policy"
    plugin_manager.initialize_orchestrators.assert_called_once()
    _, kwargs = resolver.call_args
    assert kwargs["default_policy"] is RejectPolicy.FLAG
    assert kwargs["logger"].name == "calibrated_explanations.core.explain.orchestrator"


def test_coerce_legacy_reject_result_converts_v2_without_deprecation_warning():
    result_v2 = RejectResultV2(
        schema_version="v2",
        policy=RejectPolicy.FLAG,
        decision=RejectDecisionArtifact(
            rejected=np.array([True, False]),
            ambiguity_mask=np.array([False, False]),
            novelty_mask=np.array([False, False]),
            prediction_set_size=np.array([1, 1]),
            prediction_set=None,
            epsilon=0.1,
            confidence=0.9,
            reject_rate=0.5,
            ambiguity_rate=0.0,
            novelty_rate=0.0,
            error_rate=0.0,
            error_rate_defined=True,
            fallback_used=False,
            degraded_mode=(),
        ),
        payload=RejectPayloadArtifact(
            policy=RejectPolicy.FLAG,
            source_indices=(0, 1),
            original_count=2,
            matched_count=2,
            prediction=["p0", "p1"],
            explanation=["e0", "e1"],
        ),
        metadata={},
    )

    with warnings.catch_warnings(record=True) as warning_record:
        warnings.simplefilter("always")
        legacy = coerce_legacy_reject_result(result_v2)
    assert all(not issubclass(w.category, DeprecationWarning) for w in warning_record)
    assert isinstance(legacy, RejectResult)
