"""Instance-wise contract for thresholded (probabilistic) regression.

A calibrated probability ``P(y <= t | x)`` and its Venn-Abers interval must be a
function of the test object and the frozen calibration state only. It must not
depend on which other objects share the batch, their order, or the batch size.

Regression guard for the defect where crepes' smoothed conformal p-values drew
the smoothing variable by row index (``np.random.rand(len(y_hat))`` after a
per-call reseed), so the same object received a different calibrated
probability depending on its position in the batch.
"""

from __future__ import annotations

import numpy as np
import pytest
from crepes.extras import MondrianCategorizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from calibrated_explanations import WrapCalibratedExplainer

pytestmark = pytest.mark.integration

N_TEST = 16


def _make_data(seed: int, n: int, *, integer_targets: bool):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = 2.0 * x[:, 0] - x[:, 1] + 0.7 * x[:, 2] + rng.normal(scale=0.5, size=n)
    if integer_targets:
        y = np.round(y)
    return x, y


def _estimator(kind: str, seed: int):
    if kind == "forest":
        return RandomForestRegressor(n_estimators=30, random_state=seed)
    if kind == "linear":
        return LinearRegression()
    # Shallow tree: only a handful of distinct predictions, so test predictions tie.
    return DecisionTreeRegressor(max_depth=2, random_state=seed)


def _fit(kind: str, seed: int, *, integer_targets: bool = False, mondrian: bool = False):
    """Fit a model that ignores x2 entirely; return wrapper, test panel and targets."""
    x, y = _make_data(seed, 600 + 2 * N_TEST, integer_targets=integer_targets)
    model = Pipeline(
        [
            ("keep_x0_x1", ColumnTransformer([("keep", "passthrough", [0, 1])], remainder="drop")),
            ("est", _estimator(kind, seed)),
        ]
    )
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x[:300], y[:300])
    if mondrian:
        mc = MondrianCategorizer()
        mc.fit(x[300:600], f=lambda v: v[:, 0], no_bins=3)
        wrapper.calibrate(x[300:600], y[300:600], mc=mc)
    else:
        wrapper.calibrate(x[300:600], y[300:600])
    base = x[600 : 600 + N_TEST // 2].copy()
    twin = base.copy()
    twin[:, 2] += 1.5  # change only the unused feature
    panel = np.vstack([base, twin])  # row i and row i + N_TEST // 2 form a pair
    return wrapper, panel, y


def _proba(wrapper, x, threshold):
    proba, (low, high) = wrapper.predict_proba(x, uq_interval=True, threshold=threshold)
    return np.column_stack([proba[:, 1], low, high])


def _thresholds(y):
    p05, p50, p95 = np.quantile(y, [0.05, 0.5, 0.95])
    return {
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "within": (float(p05), float(p95)),
    }


CASES = [
    pytest.param(("forest", 0, False, False), id="forest-s0"),
    pytest.param(("forest", 1, False, False), id="forest-s1"),
    pytest.param(("linear", 2, False, False), id="linear-s2"),
    pytest.param(("tree", 3, False, False), id="tied-tree-s3"),
    pytest.param(("tree", 4, True, False), id="tied-tree-integer-targets-s4"),
    pytest.param(("forest", 5, False, True), id="forest-mondrian-s5"),
]


@pytest.fixture(scope="module", params=CASES)
def fitted(request):
    kind, seed, integer_targets, mondrian = request.param
    return _fit(kind, seed, integer_targets=integer_targets, mondrian=mondrian)


def test_unused_feature_leaves_raw_prediction_unchanged(fitted):
    wrapper, panel, _ = fitted
    raw = wrapper.learner.predict(panel)
    half = N_TEST // 2
    np.testing.assert_array_equal(raw[:half], raw[half:])


@pytest.mark.parametrize("query", ["p05", "p50", "p95", "within"])
def test_threshold_probability_is_batch_invariant(fitted, query):
    wrapper, panel, y = fitted
    threshold = _thresholds(y)[query]
    singleton = np.vstack([_proba(wrapper, panel[i : i + 1], threshold) for i in range(N_TEST)])

    rng = np.random.default_rng(0)
    perm = rng.permutation(N_TEST)
    contexts = {
        "full": _proba(wrapper, panel, threshold),
        "reversed": _proba(wrapper, panel[::-1], threshold)[::-1],
        "permuted": _proba(wrapper, panel[perm], threshold)[np.argsort(perm)],
        "halves": np.vstack(
            [
                _proba(wrapper, panel[: N_TEST // 2], threshold),
                _proba(wrapper, panel[N_TEST // 2 :], threshold),
            ]
        ),
        "padded": _proba(wrapper, np.vstack([panel, panel, panel]), threshold)[:N_TEST],
    }
    for name, values in contexts.items():
        np.testing.assert_array_equal(values, singleton, err_msg=f"context={name}")


@pytest.mark.parametrize("query", ["p05", "p95", "within"])
def test_unused_feature_twins_receive_identical_calibrated_output(fitted, query):
    wrapper, panel, y = fitted
    out = _proba(wrapper, panel, _thresholds(y)[query])
    half = N_TEST // 2
    # point estimate, lower and upper Venn-Abers bounds must all coincide
    np.testing.assert_array_equal(out[:half], out[half:])


def test_repeated_identical_calls_are_deterministic(fitted):
    wrapper, panel, y = fitted
    threshold = _thresholds(y)["p05"]
    np.testing.assert_array_equal(
        _proba(wrapper, panel, threshold), _proba(wrapper, panel, threshold)
    )


def test_scalar_threshold_matches_per_instance_threshold_path(fitted):
    """Scalar thresholds (vectorised path) must equal per-instance thresholds (row loop path)."""
    wrapper, panel, y = fitted
    threshold = _thresholds(y)["p95"]
    scalar = _proba(wrapper, panel, threshold)
    per_instance = _proba(wrapper, panel, np.full(N_TEST, threshold))
    np.testing.assert_array_equal(scalar, per_instance)


def test_public_predict_interval_matches_predict_proba_and_is_batch_invariant(fitted):
    """``predict`` returns event labels; its probability interval must match ``predict_proba``."""
    wrapper, panel, y = fitted
    threshold = _thresholds(y)["p05"]
    _, (low, high) = wrapper.predict(panel, uq_interval=True, threshold=threshold)
    _, (low_rev, high_rev) = wrapper.predict(panel[::-1], uq_interval=True, threshold=threshold)
    expected = _proba(wrapper, panel, threshold)[:, 1:]
    np.testing.assert_array_equal(np.column_stack([low, high]), expected)
    np.testing.assert_array_equal(np.column_stack([low_rev, high_rev])[::-1], expected)


def test_relevant_feature_changes_still_change_probabilities():
    """Determinism must not collapse genuinely different predictions."""
    wrapper, _, y = _fit("linear", 7)
    grid = np.zeros((25, 3))
    grid[:, 0] = np.linspace(-2.5, 2.5, 25)  # x0 is used by the model
    raw = wrapper.learner.predict(grid)
    out = _proba(wrapper, grid, float(np.median(y)))
    assert np.all(np.diff(raw) > 0)
    assert len(np.unique(out[:, 0])) > 5
    # P(y <= t) cannot increase as the model's prediction increases
    assert np.all(np.diff(out[:, 0]) <= 0)
    assert out[0, 0] > 0.9 > 0.1 > out[-1, 0]
    assert np.all((out[:, 1] <= out[:, 0]) & (out[:, 0] <= out[:, 2]))


def test_factual_explanation_does_not_depend_on_batch():
    wrapper, panel, y = _fit("forest", 8)
    threshold = _thresholds(y)["p05"]
    alone = wrapper.explain_factual(panel[:1], threshold=threshold)[0]
    batched = wrapper.explain_factual(panel, threshold=threshold)[0]
    for key in ("predict", "low", "high"):
        np.testing.assert_array_equal(alone.prediction[key], batched.prediction[key])
        np.testing.assert_array_equal(alone.feature_weights[key], batched.feature_weights[key])


def test_restored_state_reproduces_instance_wise_outputs(tmp_path):
    """ADR-031: a save/load round trip keeps thresholded outputs exact and batch-invariant."""
    wrapper, panel, y = _fit("forest", 9)
    threshold = _thresholds(y)["p95"]
    expected = _proba(wrapper, panel, threshold)
    path = wrapper.save_state(tmp_path / "state")

    restored = WrapCalibratedExplainer.load_state(path, learner=wrapper.learner)

    np.testing.assert_array_equal(_proba(restored, panel, threshold), expected)
    np.testing.assert_array_equal(_proba(restored, panel[::-1], threshold)[::-1], expected)
