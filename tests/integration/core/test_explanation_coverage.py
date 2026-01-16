"""
Integration tests for src/calibrated_explanations/explanations/explanation.py.
Focusing on plotting, rule filtering, and property flags which have low coverage.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from calibrated_explanations import CalibratedExplainer


def setup_classification():
    """Helper to setup a simple classification scenario."""
    # Ensure n_features >= n_informative + n_redundant + n_repeated (defaults 2, 2, 0)
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    cal_X, cal_y = X[:20], y[:20]
    test_X = X[20:25]

    explainer = CalibratedExplainer(model, cal_X, cal_y, mode="classification")
    return explainer, test_X


def test_explanation_plots_smoke():
    """Smoke test for plot() methods in Explanation classes."""
    # Setup classification
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    cal_X, cal_y = X[:20], y[:20]
    test_X = X[20:25]

    explainer = CalibratedExplainer(model, cal_X, cal_y, mode="classification")

    # 1. Factual Explanation Plot
    explanations = explainer.explain_factual(test_X)
    expl = explanations[0]
    # Call plot without showing to avoid blocking
    expl.plot(show=False)
    plt.close()

    # 2. Alternative Explanation Plot
    # Use explore_alternatives instead of explain_alternative
    alt_explanations = explainer.explore_alternatives(test_X)
    alt_expl = alt_explanations[0]
    alt_expl.plot(show=False)
    plt.close()


def test_explanation_filtering_and_ranking():
    """Test filtering and ranking logic in explanations."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    cal_X, cal_y = X[:50], y[:50]
    test_X = X[50:55]

    explainer = CalibratedExplainer(model, cal_X, cal_y, mode="classification")

    # Alternative explanation has more complex rules to filter
    explanations = explainer.explore_alternatives(test_X)
    expl = explanations[0]

    # Test filtering methods (smoke test mostly, verifying no crash)
    # These properties are used in code but often not tested
    assert isinstance(expl.is_one_sided(), bool)
    assert isinstance(expl.has_conjunctive_rules, bool)

    # expl.metric is not a public property, removed access

    # Rank rules logic is implicitly called.


def test_regression_threshold_explanation():
    """Test standard regression explanation logic including thresholds."""
    X, y = make_regression(n_samples=50, n_features=3, noise=0.1, random_state=42)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    cal_X, cal_y = X[:20], y[:20]
    test_X = X[20:25]

    # With threshold
    explainer = CalibratedExplainer(model, cal_X, cal_y, mode="regression")
    # Pass threshold to explain_factual
    explanations = explainer.explain_factual(test_X, threshold=0.5)
    expl = explanations[0]
    # Check if threshold is correctly propagated
    assert expl.is_thresholded()
    assert expl.y_threshold == 0.5

    # Without threshold
    explanations_no_thresh = explainer.explain_factual(test_X)
    expl_nt = explanations_no_thresh[0]
    assert not expl_nt.is_thresholded()


def test_semifactual_counterfactual_flags():
    """Test boolean flags for counter/semi-factual."""
    # Use binary classification to easily trigger these
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    cal_X, cal_y = X[:20], y[:20]
    test_X = X[20:25]

    explainer = CalibratedExplainer(model, cal_X, cal_y, mode="classification")

    # Alternatives
    explanations = explainer.explore_alternatives(test_X)
    expl = explanations[0]

    # Just verifying properties exist and return bools
    assert isinstance(expl.is_counter_explanation(), bool)
    assert isinstance(expl.is_semi_explanation(), bool)


def test_ignored_features_for_instance():
    """Test ignored_features_for_instance method."""
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    cal_X, cal_y = X[:20], y[:20]
    test_X = X[20:25]

    explainer = CalibratedExplainer(model, cal_X, cal_y, mode="classification")

    explanations = explainer.explain_factual(test_X)
    expl = explanations[0]

    # Test with default (no ignores)
    ignored = expl.ignored_features_for_instance()
    assert isinstance(ignored, set)

    # Test with global ignore
    expl.calibrated_explanations.features_to_ignore = (0, 2)
    ignored = expl.ignored_features_for_instance()
    assert 0 in ignored
    assert 2 in ignored

    # Test with per instance ignore
    expl.calibrated_explanations.feature_filter_per_instance_ignore = [[1], None, None, None, None]
    ignored = expl.ignored_features_for_instance()
    assert 0 in ignored
    assert 2 in ignored
    assert 1 in ignored
