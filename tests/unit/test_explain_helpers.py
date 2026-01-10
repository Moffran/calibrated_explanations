import numpy as np

from calibrated_explanations.core.explain import _helpers as helpers


def test_compute_weight_delta_scalar_and_array():
    # scalar baseline and array perturbed
    base = 1.0
    pert = np.array([0.2, 0.3])
    delta = helpers.compute_weight_delta(base, pert)
    assert np.allclose(delta, np.array([0.8, 0.7]))

    # array shapes equal
    base_arr = np.array([0.5, 0.4])
    pert_arr = np.array([0.2, 0.1])
    delta2 = helpers.compute_weight_delta(base_arr, pert_arr)
    assert np.allclose(delta2, np.array([0.3, 0.3]))


def test_merge_ignore_features_and_slice_helpers():
    class DummyExpl:
        features_to_ignore = [0, 2]

    expl = DummyExpl()
    merged = helpers.merge_ignore_features(expl, None)
    assert np.array_equal(np.sort(merged), np.array([0, 2]))

    merged2 = helpers.merge_ignore_features(expl, [3, 2])
    assert np.array_equal(np.sort(merged2), np.array([0, 2, 3]))

    # slice_threshold with scalar and list
    assert helpers.slice_threshold(0.5, 0, 1, 1) == 0.5
    arr = [1, 2, 3]
    sliced = helpers.slice_threshold(arr, 1, 3, 3)
    assert sliced == [2, 3]


def test_merge_feature_result_updates_buffers():
    n_instances = 2
    n_features = 3
    feature_index = 1

    weights_predict = np.zeros((n_instances, n_features), dtype=float)
    weights_low = np.zeros_like(weights_predict)
    weights_high = np.zeros_like(weights_predict)
    predict_matrix = np.zeros_like(weights_predict)
    low_matrix = np.zeros_like(weights_predict)
    high_matrix = np.zeros_like(weights_predict)

    # create per-feature arrays for feature_index
    feature_weights_predict = np.array([0.1, 0.2])
    feature_weights_low = np.array([0.01, 0.02])
    feature_weights_high = np.array([0.11, 0.22])
    feature_predict_values = np.array([0.5, 0.6])
    feature_low_values = np.array([0.4, 0.45])
    feature_high_values = np.array([0.55, 0.65])

    rule_values_entries = [None, "rv"]
    # entry: predict_row, low_row, high_row, current_bin, counts_row, fractions_row
    binned_entries = [
        None,
        (
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            0,
            np.array([], dtype=float),
            np.array([], dtype=float),
        ),
    ]

    rule_values = [dict() for _ in range(n_instances)]
    # use lists for predict/low/high so per-feature assignments accept arrays
    instance_binned = [
        {"predict": [None] * n_features, "low": [None] * n_features, "high": [None] * n_features, "current_bin": np.full(n_features, -1), "counts": [None] * n_features, "fractions": [None] * n_features},
        {"predict": [None] * n_features, "low": [None] * n_features, "high": [None] * n_features, "current_bin": np.full(n_features, -1), "counts": [None] * n_features, "fractions": [None] * n_features},
    ]

    rule_boundaries = np.zeros((n_instances, n_features, 2), dtype=float)

    result = (
        feature_index,
        feature_weights_predict,
        feature_weights_low,
        feature_weights_high,
        feature_predict_values,
        feature_low_values,
        feature_high_values,
        rule_values_entries,
        binned_entries,
        np.array([0.1, 0.2]),
        np.array([0.2, 0.3]),
    )

    helpers.merge_feature_result(
        result,
        weights_predict,
        weights_low,
        weights_high,
        predict_matrix,
        low_matrix,
        high_matrix,
        rule_values,
        instance_binned,
        rule_boundaries,
    )

    assert np.allclose(weights_predict[:, feature_index], feature_weights_predict)
    assert np.allclose(predict_matrix[:, feature_index], feature_predict_values)
    assert rule_values[1].get(feature_index) == "rv"
    # instance_binned updated for instance 1
    assert np.allclose(instance_binned[1]["predict"][feature_index], binned_entries[1][0])
    # boundaries updated
    assert np.allclose(rule_boundaries[:, feature_index, 0], np.array([0.1, 0.2]))


def test_feature_effect_for_index_uses_explainer_prediction():
    class DummyExpl:
        def predict_calibrated(self, x, **kwargs):
            # return predict, low, high, _
            return np.array([0.5]), np.array([0.4]), np.array([0.6]), None

    expl = DummyExpl()
    baseline_prediction = {"predict": np.array([0.2])}
    res = helpers.feature_effect_for_index(
        expl,
        0,
        x=np.array([[1.0]]),
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        baseline_prediction=baseline_prediction,
    )

    assert res[0] == 0
    # delta arrays computed
    assert res[1].shape[0] == 1
