from __future__ import annotations

from types import SimpleNamespace


def _run_quickstart_regression() -> SimpleNamespace:
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from calibrated_explanations import WrapCalibratedExplainer

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
    print(f"Prediction interval: {factual.prediction_interval[0]}")

    probabilities, probability_interval = explainer.predict_proba(
        x_test[:1], threshold=150, uq_interval=True
    )
    print("Calibrated probability:", probabilities[0, 1])

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


def test_quickstart_regression_snippet_output(capsys):
    context = _run_quickstart_regression()
    captured = capsys.readouterr().out
    assert "Prediction interval" in captured
    assert "Calibrated probability" in captured
    assert len(context.factual) == 3
    assert len(context.alternatives) == 2


def test_quickstart_regression_metadata():
    context = _run_quickstart_regression()
    batch = context.explainer.explore_alternatives(context.X_test[:3], threshold=150)
    telemetry = getattr(batch, "telemetry", {})
    assert telemetry.get("task") == "regression"
    uncertainty = telemetry.get("uncertainty", {})
    assert uncertainty.get("representation")
    assert 0.0 <= context.probabilistic[0, 1] <= 1.0
    assert len(context.probabilistic_interval) == 2
