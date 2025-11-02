from __future__ import annotations

from types import SimpleNamespace

import pytest


def _run_quickstart_classification() -> SimpleNamespace:
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from calibrated_explanations import WrapCalibratedExplainer

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
    print(factual[0])  # first explanation with rule details

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


def test_quickstart_classification_snippet_output(capsys):
    context = _run_quickstart_classification()
    captured = capsys.readouterr().out
    assert "Prediction" in captured
    assert len(context.factual) == 5
    assert len(context.alternatives) == 2


def test_quickstart_classification_metadata():
    context = _run_quickstart_classification()
    batch = context.factual
    assert len(batch) == 5
    telemetry = getattr(batch, "telemetry", {})
    assert "interval_source" in telemetry
    assert telemetry.get("mode") == "factual"
    first_instance = batch[0]
    prediction = first_instance.prediction
    uncertainty = telemetry.get("uncertainty", {})
    assert uncertainty.get("representation")
    assert uncertainty.get("calibrated_value") == pytest.approx(
        prediction.get("predict", prediction.get("calibrated_value"))
    )
    alternatives = context.alternatives
    assert len(alternatives) == 2
