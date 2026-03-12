"""Test per-class reject_policy mapping into multi-class collections."""

from __future__ import annotations

import numpy as np

from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator
from calibrated_explanations.explanations.explanations import MultiClassCalibratedExplanations
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import RejectResult


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
