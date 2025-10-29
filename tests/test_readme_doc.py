from __future__ import annotations

from __future__ import annotations

from types import SimpleNamespace


def _run_readme_snippet() -> SimpleNamespace:
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from calibrated_explanations import WrapCalibratedExplainer

    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        stratify=dataset.target,
        random_state=0,
    )
    X_proper, X_cal, y_proper, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        stratify=y_train,
        random_state=0,
    )

    explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
    explainer.fit(X_proper, y_proper)
    explainer.calibrate(X_cal, y_cal, feature_names=dataset.feature_names)

    factual = explainer.explain_factual(X_test[:1])
    alternatives = explainer.explore_alternatives(X_test[:1])
    probability, (low, high) = explainer.predict(
        X_test[:1], uq_interval=True
    )

    print(f"Calibrated probability: {probability[0]:.3f}")
    print(factual[0])

    return SimpleNamespace(
        explainer=explainer,
        factual=factual,
        alternatives=alternatives,
        probability=probability,
        low=low,
        high=high,
    )


def test_readme_quickstart_snippet(capsys):
    context = _run_readme_snippet()
    output = capsys.readouterr().out
    assert "Calibrated probability:" in output
    assert 0.0 <= context.probability[0] <= 1.0
    assert context.alternatives
