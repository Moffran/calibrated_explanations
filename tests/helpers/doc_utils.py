"""Support utilities for documentation quickstarts used in tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def resolve_doc(slug: str) -> Path:
    """Return the documentation file path for `slug`, preferring Markdown then reStructuredText."""
    slug = slug.lstrip("./")
    candidate = DOCS / slug
    if candidate.suffix:
        return candidate
    md_candidate = candidate.with_suffix(".md")
    if md_candidate.exists():
        return md_candidate
    rst_candidate = candidate.with_suffix(".rst")
    return rst_candidate


def extract_threshold_value(threshold: Any) -> float | None:
    """Extract a numeric value from threshold structures used in docs/tests."""
    if threshold is None:
        return None
    if isinstance(threshold, dict):
        for key in ("value", "threshold", "amount"):
            value = threshold.get(key)
            if isinstance(value, (int, float)):
                return value
        return None
    if isinstance(threshold, (list, tuple)):
        for item in threshold:
            value = extract_threshold_value(item)
            if value is not None:
                return value
        return None
    if isinstance(threshold, (int, float)):
        return threshold
    return None


def run_quickstart_classification() -> SimpleNamespace:
    """Build the classification quickstart explainer fixture returned to tests."""
    # Binary classification dataset (malignant vs benign tumours)
    dataset = load_breast_cancer()
    x = dataset.data
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=0
    )
    x_proper, x_cal, y_proper, y_cal = train_test_split(
        x_train, y_train, test_size=0.25, stratify=y_train, random_state=0
    )

    explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
    explainer.fit(x_proper, y_proper)
    explainer.calibrate(x_cal, y_cal, feature_names=dataset.feature_names)

    factual = explainer.explain_factual(x_test[:5])
    # print(factual[0])  # Removed print

    alternatives = explainer.explore_alternatives(x_test[:2])

    return SimpleNamespace(
        dataset=dataset,
        X=x,
        y=y,
        X_train=x_train,
        X_test=x_test,
        y_train=y_train,
        y_test=y_test,
        X_proper=x_proper,
        X_cal=x_cal,
        y_proper=y_proper,
        y_cal=y_cal,
        explainer=explainer,
        factual=factual,
        alternatives=alternatives,
    )


def run_quickstart_regression() -> SimpleNamespace:
    """Build the regression quickstart explainer fixture returned to tests."""
    dataset = load_diabetes()
    x = dataset.data
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_proper, x_cal, y_proper, y_cal = train_test_split(
        x_train, y_train, test_size=0.25, random_state=0
    )

    explainer = WrapCalibratedExplainer(RandomForestRegressor(random_state=0))
    explainer.fit(x_proper, y_proper)
    explainer.calibrate(
        x_cal,
        y_cal,
        feature_names=dataset.feature_names,
    )

    factual = explainer.explain_factual(x_test[:3])
    # print(f"Prediction interval: {factual.prediction_interval[0]}")

    probabilities, probability_interval = explainer.predict_proba(
        x_test[:1], threshold=150, uq_interval=True
    )
    # print("Calibrated probability:", probabilities[0, 1])

    alternatives = explainer.explore_alternatives(x_test[:2], threshold=150)

    return SimpleNamespace(
        dataset=dataset,
        X=x,
        y=y,
        X_train=x_train,
        X_test=x_test,
        y_train=y_train,
        y_test=y_test,
        X_proper=x_proper,
        X_cal=x_cal,
        y_proper=y_proper,
        y_cal=y_cal,
        explainer=explainer,
        factual=factual,
        probabilistic=probabilities,
        probabilistic_interval=probability_interval,
        alternatives=alternatives,
    )
