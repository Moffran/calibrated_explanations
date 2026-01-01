from __future__ import annotations

from __future__ import annotations

from types import SimpleNamespace


def run_readme_snippet() -> SimpleNamespace:
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from calibrated_explanations import WrapCalibratedExplainer

    dataset = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        stratify=dataset.target,
        random_state=0,
    )
    x_proper, x_cal, y_proper, y_cal = train_test_split(
        x_train,
        y_train,
        test_size=0.25,
        stratify=y_train,
        random_state=0,
    )

    explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
    explainer.fit(x_proper, y_proper)
    explainer.calibrate(x_cal, y_cal, feature_names=dataset.feature_names)

    factual = explainer.explain_factual(x_test[:1])
    alternatives = explainer.explore_alternatives(x_test[:1])
    probabilities, probability_interval = explainer.predict_proba(x_test[:1], uq_interval=True)
    low, high = probability_interval

    print(f"Calibrated probability: {probabilities[0, 1]:.3f}")
    print(factual[0])

    return SimpleNamespace(
        explainer=explainer,
        factual=factual,
        alternatives=alternatives,
        probabilities=probabilities,
        low=low,
        high=high,
    )


def test_readme_quickstart_snippet(enable_fallbacks, capsys):
    context = run_readme_snippet()
    output = capsys.readouterr().out
    assert "Calibrated probability:" in output
    assert 0.0 <= context.probabilities[0, 1] <= 1.0
    assert context.alternatives
