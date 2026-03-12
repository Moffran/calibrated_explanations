"""Pickling round-trip tests for reject-aware collections."""

from __future__ import annotations

import pickle

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import (
    RejectAlternativeExplanations,
    RejectCalibratedExplanations,
)


def train_wrapper(seed: int = 11):
    x, y = make_classification(n_samples=120, n_features=5, random_state=seed)
    x_proper, x_cal, y_proper, y_cal = train_test_split(x, y, test_size=0.4, random_state=seed)
    model = RandomForestClassifier(n_estimators=10, random_state=seed)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x_proper, y_proper)
    wrapper.calibrate(x_cal, y_cal, seed=seed)
    wrapper.explainer.reject_orchestrator.initialize_reject_learner()
    return wrapper, x_cal[:20]


def test_reject_collection_pickles_unpickles_with_metadata_intact():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)

    payload = pickle.dumps(res)
    restored = pickle.loads(payload)

    assert isinstance(restored, RejectCalibratedExplanations)
    assert restored.policy == res.policy
    assert restored.metadata_full().get("raw_reject_counts") == res.metadata_full().get(
        "raw_reject_counts"
    )
    assert list(restored.rejected) == list(res.rejected)


def test_pickled_readonly_flag_is_set():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)

    payload = pickle.dumps(res)
    restored = pickle.loads(payload)

    assert restored.is_readonly_pickled() is True


def test_unpickle_readonly_blocks_runtime_methods():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)

    restored = pickle.loads(pickle.dumps(res))

    # metadata inspection remains safe
    _ = restored.metadata_full()
    with pytest.raises(RuntimeError, match="read-only pickled RejectCalibratedExplanations"):
        _ = restored[:1]


def test_reconstruct_runtime_unblocks_runtime_methods():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)

    restored = pickle.loads(pickle.dumps(res))
    restored.reconstruct_runtime(
        plugin_manager=wrapper.explainer.plugin_manager,
        prediction_orchestrator=wrapper.explainer.prediction_orchestrator,
        reject_orchestrator=wrapper.explainer.reject_orchestrator,
    )
    sliced = restored[:1]
    assert sliced is not None


def test_reject_alternative_collection_pickles_with_readonly_contract():
    wrapper, x_query = train_wrapper()
    alt = wrapper.explore_alternatives(x_query[:6], reject_policy=RejectPolicy.FLAG)
    assert isinstance(alt, RejectAlternativeExplanations)

    state = alt.__getstate__()
    assert state.get("_ce_version") == "reject_v0.11.1"
    for key in ("plugin_manager", "prediction_orchestrator", "_predict_bridge", "rng"):
        assert key not in state

    restored = pickle.loads(pickle.dumps(alt))
    assert isinstance(restored, RejectAlternativeExplanations)
    assert restored.is_readonly_pickled() is True
    with pytest.raises(RuntimeError, match="read-only pickled"):
        _ = restored[:1]

    restored.reconstruct_runtime(
        plugin_manager=wrapper.explainer.plugin_manager,
        prediction_orchestrator=wrapper.explainer.prediction_orchestrator,
        reject_orchestrator=wrapper.explainer.reject_orchestrator,
    )
    sliced = restored[:2]
    assert 1 <= len(sliced.explanations) <= 2
