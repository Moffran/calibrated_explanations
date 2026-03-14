"""Unit-style tests for reject explanation wrappers and metadata handling."""

from __future__ import annotations

import json
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.orchestrator import RejectOrchestrator
from calibrated_explanations.core.reject import policy as reject_policy_module
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.explanations import CalibratedExplanations
from calibrated_explanations.explanations.reject import (
    RejectCalibratedExplanations,
    RejectContext,
    RejectPolicySpec,
    RejectResult,
    _align_reject_field_to_payload,
    _canonicalize_degraded_mode,
    _normalize_contract_metadata,
    _resolve_source_indices_for_wrapper,
)
from calibrated_explanations.utils.exceptions import DataShapeError, ValidationError


def train_wrapper(seed: int = 7):
    x, y = make_classification(n_samples=120, n_features=5, random_state=seed)
    x_proper, x_cal, y_proper, y_cal = train_test_split(x, y, test_size=0.4, random_state=seed)
    model = RandomForestClassifier(n_estimators=10, random_state=seed)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x_proper, y_proper)
    wrapper.calibrate(x_cal, y_cal, seed=seed)
    wrapper.explainer.reject_orchestrator.initialize_reject_learner()
    return wrapper, x_cal[:20]


def test_from_collection_no_aliasing():
    wrapper, x_query = train_wrapper()
    base = wrapper.explain_factual(x_query)
    rejected = np.array([True, False] * 10)
    metadata = {
        "ambiguity_mask": rejected.copy(),
        "novelty_mask": np.zeros_like(rejected),
        "prediction_set_size": np.ones_like(rejected),
        "epsilon": 0.05,
        "raw_total_examples": 20,
        "raw_reject_counts": {"rejected": 10, "ambiguity_mask": 10, "novelty_mask": 0},
    }

    wrapped = RejectCalibratedExplanations.from_collection(
        base, metadata, RejectPolicy.FLAG, rejected
    )
    rejected[0] = False

    assert isinstance(wrapped, RejectCalibratedExplanations)
    assert bool(wrapped.rejected[0]) is True
    assert wrapped.calibrated_explainer is base.calibrated_explainer
    for name in ("x_cal", "_X_cal", "scaled_x_cal", "fast_x_cal", "scaled_y_cal"):
        assert not hasattr(wrapped, name)


def test_from_collection_merges_legacy_raw_count_alias():
    wrapper, x_query = train_wrapper()
    base = wrapper.explain_factual(x_query[:6])
    rejected = np.array([True, False, True, False, False, True])
    metadata = {
        "raw_reject_counts": {"rejected": 3},
        "_raw_reject_counts": {"novelty_mask": 1},
        "prediction_set_size": np.array([1, 0, 1, 1, 1, 0]),
    }

    wrapped = RejectCalibratedExplanations.from_collection(
        base, metadata, RejectPolicy.FLAG, rejected
    )
    counts = wrapped.metadata["raw_reject_counts"]
    assert counts["rejected"] == 3
    assert counts["novelty_mask"] == 1


def test_slice_shapes_valid():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)

    sliced = res[:10]
    assert sliced.metadata["reject_rate"] == res.metadata["reject_rate"]
    assert pytest.approx(sliced.metadata["payload_reject_rate"]) == float(np.mean(sliced.rejected))
    assert sliced.metadata["raw_reject_counts"]["rejected"] == int(np.sum(sliced.rejected))
    assert "_raw_reject_counts" not in sliced.metadata


def test_slice_raw_reject_counts_are_sums():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)

    half = res[: len(res) // 2]
    assert "raw_reject_counts" in half.metadata
    assert half.metadata["raw_reject_counts"]["rejected"] == int(np.sum(half.rejected))
    assert half.metadata["raw_reject_counts"]["ambiguity_mask"] == int(np.sum(half.ambiguity_mask))
    assert half.metadata["raw_reject_counts"]["prediction_set_size"] == int(
        np.sum(half.prediction_set_size)
    )


def test_slice_invalid_boolean_mask_raises():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)

    short_mask = np.array([True, False, True])
    with pytest.raises(DataShapeError, match="boolean mask length mismatch"):
        _ = res[short_mask]


def test_integer_index_out_of_bounds_raises():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)
    with pytest.raises(DataShapeError, match="integer index out of bounds"):
        _ = res[100]


def test_validate_key_indexing_boolean_list_coercion():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)
    sliced = res[:3]
    selected = sliced[[True, False, True]]
    assert len(selected) == 2


def test_validate_key_indexing_integer_list_mixed_types():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)
    sliced = res[:3]
    selected = sliced[[0, -1, 2]]
    assert len(selected) == 3


def test_validate_key_indexing_object_dtype_raises():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)
    sliced = res[:2]
    with pytest.raises(DataShapeError):
        _ = sliced[[1, "x"]]


def test_metadata_lightweight_and_full():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)

    lightweight = res.metadata
    full = res.metadata_full()

    assert "ambiguity_mask" not in lightweight
    assert isinstance(full.get("ambiguity_mask"), list)
    assert "prediction_set_size_summary" in lightweight


def test_metadata_full_prediction_set_is_json_safe():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)

    full = res.metadata_full()
    assert isinstance(full.get("prediction_set"), list)
    assert "_raw_reject_counts" not in full
    json.dumps(full)


def test_prediction_set_slicing_and_sliced_total_examples():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)

    full = res.metadata_full()
    assert "prediction_set" in full
    assert isinstance(full["prediction_set"], list)

    sliced = res[:5]
    full_sliced = sliced.metadata_full()
    assert sliced.metadata["sliced_total_examples"] == len(sliced)
    assert isinstance(full_sliced.get("prediction_set"), list)
    assert sliced.metadata["raw_reject_counts"]["rejected"] == int(np.sum(sliced.rejected))


def test_reject_getitem_wraps_plain_collection_result(monkeypatch):
    wrapper, x_query = train_wrapper()
    reject_res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)
    plain = wrapper.explain_factual(x_query[:2])

    monkeypatch.setattr(CalibratedExplanations, "__getitem__", lambda _self, _key: plain)
    wrapped = RejectCalibratedExplanations.__getitem__(reject_res, slice(0, 2))

    assert isinstance(wrapped, RejectCalibratedExplanations)
    assert wrapped.policy is reject_res.policy
    assert "raw_reject_counts" in wrapped.metadata


def test_resolve_source_indices_for_wrapper_accepts_valid_metadata():
    idxs = _resolve_source_indices_for_wrapper(
        policy=RejectPolicy.ONLY_ACCEPTED,
        metadata={"source_indices": [1, 3], "original_count": 4},
        rejected=np.array([True, False, True, False]),
        payload_count=2,
    )
    np.testing.assert_array_equal(idxs, np.array([1, 3]))


def test_resolve_source_indices_for_wrapper_fallback_from_rejected_mask_warns():
    with pytest.warns(UserWarning, match="missing source_indices"):
        idxs = _resolve_source_indices_for_wrapper(
            policy=RejectPolicy.ONLY_REJECTED,
            metadata={},
            rejected=np.array([True, False, True, False]),
            payload_count=2,
        )
    np.testing.assert_array_equal(idxs, np.array([0, 2]))


def test_align_reject_field_to_payload_slices_by_source_indices():
    aligned = _align_reject_field_to_payload(
        name="ambiguity_mask",
        value=np.array([True, False, True, False]),
        payload_count=2,
        source_indices=np.array([1, 3]),
        ensure_dtype=bool,
    )
    np.testing.assert_array_equal(aligned, np.array([False, False]))


def test_resolve_source_indices_for_wrapper_raises_without_mapping_or_mask():
    with pytest.raises(DataShapeError, match="Cannot align filtered reject payload"):
        _resolve_source_indices_for_wrapper(
            policy=RejectPolicy.ONLY_ACCEPTED,
            metadata={},
            rejected=None,
            payload_count=1,
        )


def test_resolve_source_indices_for_wrapper_empty_payload_without_mapping():
    idxs = _resolve_source_indices_for_wrapper(
        policy=RejectPolicy.ONLY_ACCEPTED,
        metadata={},
        rejected=None,
        payload_count=0,
    )
    np.testing.assert_array_equal(idxs, np.array([], dtype=int))


def test_reject_policy_spec_constructors_and_bounds():
    assert RejectPolicySpec.only_rejected().policy is RejectPolicy.ONLY_REJECTED
    assert RejectPolicySpec.only_accepted().policy is RejectPolicy.ONLY_ACCEPTED
    with pytest.raises(ValueError, match="w must be in"):
        RejectPolicySpec(RejectPolicy.FLAG, ncf="default", w=1.1)


def test_reject_policy_invalid_non_string_raises_value_error():
    with pytest.raises(ValueError):
        RejectPolicy(123)


def test_contract_metadata_normalization_canonicalizes_degraded_mode():
    assert _canonicalize_degraded_mode(None) == ()
    assert _canonicalize_degraded_mode("bulk_to_per_instance_fallback") == (
        "bulk_to_per_instance_fallback",
    )
    assert _canonicalize_degraded_mode(["prediction_payload_failed", None, "custom"]) == (
        "prediction_payload_failed",
        "custom",
    )

    normalized = _normalize_contract_metadata(
        metadata={
            "raw_reject_counts": {"rejected": 2},
            "degraded_mode": "prediction_payload_failed",
        },
        policy=RejectPolicy.FLAG,
        rejected=np.array([True, False, True]),
        source_indices=None,
        original_count=None,
    )
    assert normalized["original_count"] == 3
    assert normalized["source_indices"] == [0, 1, 2]
    assert normalized["rejected_count"] == 2
    assert normalized["accepted_count"] == 1
    assert normalized["fallback_used"] is True


def test_clear_reject_arrays_can_drop_summary_metadata():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)
    res.clear_reject_arrays(keep_summary=False)
    meta = res.metadata
    assert meta["raw_reject_counts"] == {}
    required = {
        "policy",
        "reject_rate",
        "accepted_count",
        "rejected_count",
        "effective_confidence",
        "effective_threshold",
        "source_indices",
        "original_count",
        "init_ok",
        "fallback_used",
        "init_error",
        "degraded_mode",
    }
    assert required.issubset(meta.keys())


def test_slice_recomputes_error_rate_from_prediction_set_size():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)

    sliced = res[:10]
    sizes = np.asarray(sliced.prediction_set_size)
    epsilon = float(sliced.epsilon)
    singleton = int(np.sum(sizes == 1))
    empty = int(np.sum(sizes == 0))
    if singleton == 0:
        assert sliced.metadata["error_rate_defined"] is False
        assert sliced.metadata["error_rate"] == 0.0
    else:
        expected = max(0.0, min(1.0, (len(sliced) * epsilon - empty) / singleton))
        assert sliced.metadata["error_rate_defined"] is True
        assert sliced.metadata["error_rate"] == pytest.approx(expected)


def test_policy_spec_equality_and_serialization():
    spec_a = RejectPolicySpec(RejectPolicy.FLAG, ncf="default", w=0.5)
    spec_b = RejectPolicySpec.flag("default", 0.5)
    assert spec_a == spec_b

    payload = spec_a.to_dict()
    restored = RejectPolicySpec.from_dict(payload)
    assert restored == spec_a
    assert hash(restored) == hash(spec_a)


def test_policy_spec_eq_policy_and_notimplemented_paths():
    spec = RejectPolicySpec.flag("default", 0.5)
    assert spec == RejectPolicy.FLAG
    assert spec.__eq__(object()) is NotImplemented
    assert spec.value.startswith("flag[ncf=default,w=")


def test_policy_spec_w_normalization_non_ensured_and_ensured_sensitivity():
    default_a = RejectPolicySpec.flag(ncf="default", w=0.1)
    default_b = RejectPolicySpec.flag(ncf="default", w=0.9)
    entropy_a = RejectPolicySpec.flag(ncf="entropy", w=0.2)  # legacy compat -> default
    entropy_b = RejectPolicySpec.flag(ncf="entropy", w=0.8)  # legacy compat -> default
    assert default_a == default_b
    assert entropy_a == entropy_b
    assert hash(default_a) == hash(default_b)
    assert hash(entropy_a) == hash(entropy_b)

    ensured_a = RejectPolicySpec.flag(ncf="ensured", w=0.2)
    ensured_b = RejectPolicySpec.flag(ncf="ensured", w=0.8)
    assert ensured_a != ensured_b
    assert hash(ensured_a) != hash(ensured_b)


def test_policy_spec_callable_serialization_rejected():
    with pytest.raises(ValueError, match="ncf must be one of"):
        _ = RejectPolicySpec(RejectPolicy.FLAG, ncf=5, w=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="string ncf"):
        _ = RejectPolicySpec.from_dict({"policy": RejectPolicy.FLAG.value, "ncf": 5, "w": 0.5})


def test_policy_spec_from_dict_missing_key_raises_validation_error():
    with pytest.raises(ValidationError, match="Missing key"):
        _ = RejectPolicySpec.from_dict({"policy": RejectPolicy.FLAG.value, "ncf": "default"})


def test_error_rate_recompute_denominator_preserved():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)
    sliced = res[:8]

    assert sliced.metadata["raw_total_examples"] == res.metadata["raw_total_examples"]
    assert sliced.metadata["error_rate_defined"] is True


def test_deprecation_warning():
    with pytest.warns(DeprecationWarning):
        deprecated = reject_policy_module.__getattr__("PREDICT_AND_FLAG")
    assert deprecated is RejectPolicy.FLAG


def test_metadata_summary_alias_and_memory_profile_after_clear():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)
    res.clear_reject_arrays(keep_summary=True)

    assert res.metadata_summary() == res.metadata
    profile = res.memory_profile()
    assert profile["rejected"] == 0
    assert profile["ambiguity_mask"] == 0
    assert profile["novelty_mask"] == 0
    assert profile["prediction_set_size"] == 0
    assert profile["prediction_set"] == 0


def test_getitem_with_unsupported_key_type_raises_data_shape_error():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query[:5], reject_policy=RejectPolicy.FLAG)
    with pytest.raises(DataShapeError, match="Unsupported key type for reject slicing"):
        _ = res[{"bad": "key"}]  # type: ignore[index]


def test_resolve_source_indices_validation_errors_from_metadata():
    with pytest.raises(DataShapeError, match="one-dimensional integer sequence"):
        _resolve_source_indices_for_wrapper(
            policy=RejectPolicy.FLAG,
            metadata={"source_indices": [0.1, 2.5], "original_count": 3},
            rejected=np.array([False, False, False]),
            payload_count=2,
        )

    with pytest.raises(DataShapeError, match="must be unique"):
        _resolve_source_indices_for_wrapper(
            policy=RejectPolicy.FLAG,
            metadata={"source_indices": [1, 1], "original_count": 3},
            rejected=np.array([False, False, False]),
            payload_count=2,
        )

    with pytest.raises(DataShapeError, match="must preserve source ordering"):
        _resolve_source_indices_for_wrapper(
            policy=RejectPolicy.FLAG,
            metadata={"source_indices": [2, 1], "original_count": 3},
            rejected=np.array([False, False, False]),
            payload_count=2,
        )

    with pytest.raises(DataShapeError, match="must be < original_count"):
        _resolve_source_indices_for_wrapper(
            policy=RejectPolicy.FLAG,
            metadata={"source_indices": [0, 4], "original_count": 4},
            rejected=np.array([False, False, False, False]),
            payload_count=2,
        )


def test_align_reject_field_to_payload_error_paths():
    with pytest.raises(DataShapeError, match="cannot be aligned"):
        _align_reject_field_to_payload(
            name="prediction_set_size",
            value=np.array(1),
            payload_count=1,
            source_indices=np.array([0]),
            ensure_dtype=int,
        )

    with pytest.raises(DataShapeError, match="shorter than source_indices require"):
        _align_reject_field_to_payload(
            name="prediction_set",
            value=np.array([True, False]),
            payload_count=3,
            source_indices=np.array([0, 1, 2]),
            ensure_dtype=bool,
        )


def test_deprecated_explainer_reject_wrappers_delegate_to_orchestrator():
    wrapper, _ = train_wrapper()

    with pytest.warns(DeprecationWarning, match="initialize_reject_learner"):
        wrapper.explainer.initialize_reject_learner(ncf="default", w=0.5)
    assert wrapper.explainer.reject_ncf == "default"

    with pytest.warns(DeprecationWarning, match="predict_reject"):
        rejected, _, _ = wrapper.explainer.predict_reject(wrapper.explainer.x_cal[:5])
    assert len(rejected) == 5


def test_deprecated_wrapper_reject_wrappers_delegate_to_orchestrator():
    wrapper, _ = train_wrapper()

    with pytest.warns(DeprecationWarning, match="initialize_reject_learner"):
        wrapper.initialize_reject_learner(ncf="default", w=0.5)
    assert wrapper.explainer.reject_ncf == "default"

    with pytest.warns(DeprecationWarning, match="predict_reject"):
        rejected, _, _ = wrapper.predict_reject(wrapper.explainer.x_cal[:5])
    assert len(rejected) == 5


def test_matched_count_defined_on_explain_error():
    class DummyPredictionOrchestrator:
        def predict(self, *_args, **_kwargs):
            return None

    class DummyExplainer:
        reject_learner = object()
        prediction_orchestrator = DummyPredictionOrchestrator()
        reject_ncf = None
        reject_ncf_w = None
        reject_ncf_auto_selected = None

    orchestrator = RejectOrchestrator(DummyExplainer())
    orchestrator.predict_reject_breakdown = lambda *args, **kwargs: {
        "rejected": np.array([True, False]),
        "error_rate": 0.0,
        "reject_rate": 0.5,
        "ambiguity_rate": 0.5,
        "novelty_rate": 0.0,
        "ambiguity": np.array([True, False]),
        "novelty": np.array([False, False]),
        "prediction_set_size": np.array([2, 1]),
        "prediction_set": np.array([[1, 1], [1, 0]]),
        "epsilon": 0.05,
        "raw_total_examples": 2,
        "raw_reject_counts": {"rejected": 1},
        "error_rate_defined": True,
    }

    def _raise(*_args, **_kwargs):
        raise TypeError("boom")

    result = orchestrator.apply_policy(RejectPolicy.FLAG, np.array([[0], [1]]), explain_fn=_raise)
    assert isinstance(result, RejectResult)
    assert result.metadata is not None
    assert "matched_count" in result.metadata
    assert result.metadata["matched_count"] is None


def test_metadata_is_lightweight():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)
    meta = res.metadata

    heavy_keys = {
        "ambiguity_mask",
        "novelty_mask",
        "prediction_set_size",
        "rejected",
        "prediction_set",
    }
    assert heavy_keys.isdisjoint(meta.keys())


def test_wrapper_metadata_contains_required_contract_keys():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query[:10], reject_policy=RejectPolicy.ONLY_ACCEPTED)
    required = {
        "policy",
        "reject_rate",
        "accepted_count",
        "rejected_count",
        "effective_confidence",
        "effective_threshold",
        "source_indices",
        "original_count",
        "init_ok",
        "fallback_used",
        "init_error",
        "degraded_mode",
    }
    assert required.issubset(res.metadata.keys())
    assert res.metadata["original_count"] == 10
    assert len(res.metadata["source_indices"]) == len(res.explanations)


def test_wrapper_slice_preserves_original_batch_counts():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query[:12], reject_policy=RejectPolicy.FLAG)
    sliced = res[:3]
    assert sliced.metadata["original_count"] == res.metadata["original_count"]
    assert sliced.metadata["rejected_count"] == res.metadata["rejected_count"]
    assert sliced.metadata["accepted_count"] == res.metadata["accepted_count"]
    assert sliced.metadata["reject_rate"] == res.metadata["reject_rate"]


def test_getstate_prunes_runtime_objects():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)
    state = res.__getstate__()

    for key in ("plugin_manager", "prediction_orchestrator", "_predict_bridge", "rng"):
        assert key not in state
    assert state.get("_ce_version") == "reject_v0.11.1"


def test_apply_policy_unexpected_explain_error_suppressed_by_default():
    class DummyPredictionOrchestrator:
        def predict(self, *_args, **_kwargs):
            return None

    class DummyExplainer:
        reject_learner = object()
        prediction_orchestrator = DummyPredictionOrchestrator()
        reject_ncf = None
        reject_ncf_w = None
        reject_ncf_auto_selected = None

    orchestrator = RejectOrchestrator(DummyExplainer())
    orchestrator.predict_reject_breakdown = lambda *args, **kwargs: {
        "rejected": np.array([True, False]),
        "error_rate": 0.0,
        "reject_rate": 0.5,
        "ambiguity_rate": 0.5,
        "novelty_rate": 0.0,
        "ambiguity": np.array([True, False]),
        "novelty": np.array([False, False]),
        "prediction_set_size": np.array([2, 1]),
        "prediction_set": np.array([[1, 1], [1, 0]]),
        "epsilon": 0.05,
        "raw_total_examples": 2,
        "raw_reject_counts": {"rejected": 1},
        "error_rate_defined": True,
    }

    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    result = orchestrator.apply_policy(RejectPolicy.FLAG, np.array([[0], [1]]), explain_fn=_raise)
    assert isinstance(result, RejectResult)
    assert result.explanation is None


def test_apply_policy_unexpected_explain_error_raises_when_opt_in():
    class DummyPredictionOrchestrator:
        def predict(self, *_args, **_kwargs):
            return None

    class DummyExplainer:
        reject_learner = object()
        prediction_orchestrator = DummyPredictionOrchestrator()
        reject_ncf = None
        reject_ncf_w = None
        reject_ncf_auto_selected = None

    orchestrator = RejectOrchestrator(DummyExplainer())
    orchestrator.predict_reject_breakdown = lambda *args, **kwargs: {
        "rejected": np.array([True, False]),
        "error_rate": 0.0,
        "reject_rate": 0.5,
        "ambiguity_rate": 0.5,
        "novelty_rate": 0.0,
        "ambiguity": np.array([True, False]),
        "novelty": np.array([False, False]),
        "prediction_set_size": np.array([2, 1]),
        "prediction_set": np.array([[1, 1], [1, 0]]),
        "epsilon": 0.05,
        "raw_total_examples": 2,
        "raw_reject_counts": {"rejected": 1},
        "error_rate_defined": True,
    }

    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        orchestrator.apply_policy(
            RejectPolicy.FLAG,
            np.array([[0], [1]]),
            explain_fn=_raise,
            _reject_raise=True,
        )


def test_clear_reject_arrays_keeps_summary():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)
    assert res.rejected is not None
    summary_before = dict(res.metadata)
    res.clear_reject_arrays(keep_summary=True)
    assert res.rejected is None
    assert res.ambiguity_mask is None
    assert res.metadata["raw_reject_counts"] == summary_before["raw_reject_counts"]


def test_packed_bits_round_trip():
    arr = np.array([True, False, True, True, False, False, True], dtype=np.bool_)
    from calibrated_explanations.explanations.reject import as_packed_bits, as_unpacked_bits

    packed = as_packed_bits(arr)
    unpacked = as_unpacked_bits(packed, len(arr))
    assert np.array_equal(arr, unpacked)


def test_memory_profile_reports_total_bytes():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)
    profile = res.memory_profile()
    assert "total_bytes" in profile
    assert profile["total_bytes"] >= 0


def test_to_packed_masks_packs_boolean_arrays():
    wrapper, x_query = train_wrapper()
    res = wrapper.explain_factual(x_query, reject_policy=RejectPolicy.FLAG)
    res.to_packed_masks()
    assert np.asarray(res.rejected).dtype == np.uint8
    assert np.asarray(res.ambiguity_mask).dtype == np.uint8
    assert np.asarray(res.novelty_mask).dtype == np.uint8


def test_reject_context_materialize_prediction_set_variants():
    class DummyExplainer:
        class_labels = {0: "neg", 1: "pos"}

    ctx = RejectContext(rejected=True, prediction_set_ref={"indices": [0, 1]})
    assert ctx.materialize_prediction_set(DummyExplainer()) == {"neg", "pos"}

    ctx_summary = RejectContext(rejected=False, prediction_set_ref={"summary": "fallback"})
    assert ctx_summary.materialize_prediction_set(DummyExplainer()) == "fallback"

    ctx_none = RejectContext(rejected=False)
    assert ctx_none.materialize_prediction_set(DummyExplainer()) is None
