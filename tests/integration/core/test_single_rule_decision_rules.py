"""End-to-end tests for single-condition decision rules.

These tests build synthetic datasets with multiple unrelated features and a
one-level decision tree learner. They verify that ``WrapCalibratedExplainer``
identifies the important feature and surfaces the learned decision rule for
classification, regression, and probabilistic regression tasks.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from calibrated_explanations.core import WrapCalibratedExplainer


def _single_split_dataset(
    n_samples: int = 500, n_features: int = 6, noise: float = 0.05
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create a dataset where half the features share a latent signal.

    The first half of the features are correlated copies of a latent variable
    while the remaining features are independent noise. The target values in
    the tests derive from the latent variable so that at least half of the
    features are informative but a one-level tree can still capture the rule.
    """

    rng = np.random.default_rng(7)
    latent = rng.normal(size=n_samples)
    informative = latent[:, None] + rng.normal(scale=noise, size=(n_samples, n_features // 2))
    distractors = rng.normal(size=(n_samples, n_features - informative.shape[1]))
    x = np.concatenate([informative, distractors], axis=1)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return x, latent, feature_names


def _split_data(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    """Split data into train, calibration, and test folds."""

    split_train = int(0.6 * len(x))
    split_cal = int(0.8 * len(x))
    x_train, x_cal, x_test = x[:split_train], x[split_train:split_cal], x[split_cal:]
    y_train, y_cal, y_test = y[:split_train], y[split_train:split_cal], y[split_cal:]
    return x_train, y_train, x_cal, y_cal, x_test, y_test


def _top_rule_feature(explanation) -> int:
    """Return the feature index associated with the strongest rule weight."""

    rules = explanation._get_rules()  # noqa: SLF001 - test needs rule internals
    top_idx = int(np.argmax(np.asarray(rules["weight"])))
    return int(rules["feature"][top_idx])


def test_single_split_classification_recovers_rule():
    """Classification: CE should surface the single-feature split from the tree."""

    x, latent, feature_names = _single_split_dataset()
    y = (latent > 0.0).astype(int)
    x_train, y_train, x_cal, y_cal, x_test, y_test = _split_data(x, y)

    explainer = WrapCalibratedExplainer(DecisionTreeClassifier(max_depth=1, random_state=11))
    explainer.fit(x_train, y_train)
    explainer.calibrate(x_cal, y_cal, feature_names=feature_names)

    y_pred = explainer.predict_proba(x_test, calibrated=True)
    assert np.all((y_pred >= 0.0) & (y_pred <= 1.0))    # probabilities map to numeric labels

    y_pred = explainer.predict(x_test, calibrated=True).astype(int)
    assert (y_pred == y_test).mean() > 0.8

    factual = explainer.explain_factual(x_test[:3])
    top_feature = _top_rule_feature(factual[0])
    assert top_feature < len(feature_names) // 2
    assert feature_names[top_feature] in factual[0]._get_rules()["rule"][0]

    alternative = explainer.explore_alternatives(x_test[:3])
    top_feature = _top_rule_feature(alternative[0])
    assert top_feature < len(feature_names) // 2
    assert feature_names[top_feature] in alternative[0]._get_rules()["rule"][0]

def test_single_split_regression_recovers_rule():
    """Regression: CE should highlight the informative feature behind the split."""

    x, latent, feature_names = _single_split_dataset(noise=0.02)
    y = np.where(latent > 0.0, 2.5, -2.5) + np.random.default_rng(3).normal(
        scale=0.05, size=len(latent)
    )
    x_train, y_train, x_cal, y_cal, x_test, y_test = _split_data(x, y)

    explainer = WrapCalibratedExplainer(DecisionTreeRegressor(max_depth=1, random_state=19))
    explainer.fit(x_train, y_train)
    explainer.calibrate(x_cal, y_cal, feature_names=feature_names)

    preds, (low, high) = explainer.predict(x_test, uq_interval=True)
    assert np.mean(np.abs(preds - y_test)) < 0.2
    assert np.all(low <= preds) and np.all(preds <= high)

    factual = explainer.explain_factual(x_test[:3])
    top_feature = _top_rule_feature(factual[0])
    assert top_feature < len(feature_names) // 2
    assert feature_names[top_feature] in factual[0]._get_rules()["rule"][0]

    alternative = explainer.explore_alternatives(x_test[:3])
    top_feature = _top_rule_feature(alternative[0])
    assert top_feature < len(feature_names) // 2
    assert feature_names[top_feature] in alternative[0]._get_rules()["rule"][0]

def test_single_split_probabilistic_regression_rule_and_intervals():
    """Probabilistic regression: intervals should reflect the simple split."""

    x, latent, feature_names = _single_split_dataset(noise=0.1)
    noise = np.random.default_rng(13).normal(scale=0.25, size=len(latent))
    y = np.where(latent > 0.0, 1.8, -1.8) + noise
    x_train, y_train, x_cal, y_cal, x_test, y_test = _split_data(x, y)

    explainer = WrapCalibratedExplainer(DecisionTreeRegressor(max_depth=1, random_state=23))
    explainer.fit(x_train, y_train)
    explainer.calibrate(x_cal, y_cal, feature_names=feature_names)

    preds, (low, high) = explainer.predict_proba(x_test, uq_interval=True, calibrated=True, threshold=y_test)
    assert np.all((preds > 0.0) & (preds < 1.0))  # probabilities map to numeric labels

    factual = explainer.explain_factual(x_test[:3], threshold=y_test[:3])
    top_feature = _top_rule_feature(factual[0])
    assert top_feature < len(feature_names) // 2
    rule_texts = factual[0]._get_rules()["rule"]
    assert any(feature_names[top_feature] in rule for rule in rule_texts)

    alternative = explainer.explore_alternatives(x_test[:3], threshold=y_test[:3])
    top_feature = _top_rule_feature(alternative[0])
    assert top_feature < len(feature_names) // 2
    rule_texts = alternative[0]._get_rules()["rule"]
    assert any(feature_names[top_feature] in rule for rule in rule_texts)

def test_single_split_classification_recovers_rule__with_condition_source_prediction():
    """Classification: CE should surface the single-feature split from the tree."""

    x, latent, feature_names = _single_split_dataset()
    y = (latent > 0.0).astype(int)
    x_train, y_train, x_cal, y_cal, x_test, y_test = _split_data(x, y)

    explainer = WrapCalibratedExplainer(DecisionTreeClassifier(max_depth=1, random_state=11))
    explainer.fit(x_train, y_train)
    explainer.calibrate(x_cal, y_cal, feature_names=feature_names)

    factual = explainer.explain_factual(x_test[:3], condition_source="prediction")
    top_feature = _top_rule_feature(factual[0])
    assert top_feature < len(feature_names) // 2
    assert feature_names[top_feature] in factual[0]._get_rules()["rule"][0]

    alternative = explainer.explore_alternatives(x_test[:3], condition_source="prediction")
    top_feature = _top_rule_feature(alternative[0])
    assert top_feature < len(feature_names) // 2
    assert feature_names[top_feature] in alternative[0]._get_rules()["rule"][0]


def test_single_split_regression_recovers_rule__with_condition_source_prediction():
    """Regression: CE should highlight the informative feature behind the split."""

    x, latent, feature_names = _single_split_dataset(noise=0.02)
    y = np.where(latent > 0.0, 2.5, -2.5) + np.random.default_rng(3).normal(
        scale=0.05, size=len(latent)
    )
    x_train, y_train, x_cal, y_cal, x_test, y_test = _split_data(x, y)

    explainer = WrapCalibratedExplainer(DecisionTreeRegressor(max_depth=1, random_state=19))
    explainer.fit(x_train, y_train)
    explainer.calibrate(x_cal, y_cal, feature_names=feature_names)

    factual = explainer.explain_factual(x_test[:3], condition_source="prediction")
    top_feature = _top_rule_feature(factual[0])
    assert top_feature < len(feature_names) // 2
    assert feature_names[top_feature] in factual[0]._get_rules()["rule"][0]

    alternative = explainer.explore_alternatives(x_test[:3], condition_source="prediction")
    top_feature = _top_rule_feature(alternative[0])
    assert top_feature < len(feature_names) // 2
    assert feature_names[top_feature] in alternative[0]._get_rules()["rule"][0]


def test_single_split_probabilistic_regression_rule_and_intervals__with_condition_source_prediction():
    """Probabilistic regression: intervals should reflect the simple split."""

    x, latent, feature_names = _single_split_dataset(noise=0.1)
    noise = np.random.default_rng(13).normal(scale=0.25, size=len(latent))
    y = np.where(latent > 0.0, 1.8, -1.8) + noise
    x_train, y_train, x_cal, y_cal, x_test, y_test = _split_data(x, y)

    explainer = WrapCalibratedExplainer(DecisionTreeRegressor(max_depth=1, random_state=23))
    explainer.fit(x_train, y_train)
    explainer.calibrate(x_cal, y_cal, feature_names=feature_names)

    factual = explainer.explain_factual(x_test[:3], threshold=y_test[:3], condition_source="prediction")
    top_feature = _top_rule_feature(factual[0])
    assert top_feature < len(feature_names) // 2
    rule_texts = factual[0]._get_rules()["rule"]
    assert any(feature_names[top_feature] in rule for rule in rule_texts)

    alternative = explainer.explore_alternatives(x_test[:3], threshold=y_test[:3], condition_source="prediction")
    top_feature = _top_rule_feature(alternative[0])
    assert top_feature < len(feature_names) // 2
    rule_texts = alternative[0]._get_rules()["rule"]
    assert any(feature_names[top_feature] in rule for rule in rule_texts)
