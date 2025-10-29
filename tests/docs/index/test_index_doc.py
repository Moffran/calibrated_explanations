from __future__ import annotations

from types import SimpleNamespace


def _run_index_snippet() -> SimpleNamespace:
    from sklearn.datasets import load_breast_cancer, load_diabetes
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from calibrated_explanations import WrapCalibratedExplainer

    breast_cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        breast_cancer.data,
        breast_cancer.target,
        test_size=0.2,
        stratify=breast_cancer.target,
        random_state=0,
    )
    explainer_cls = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
    explainer_cls.fit(X_train, y_train)
    explainer_cls.calibrate(
        X_train,
        y_train,
        feature_names=breast_cancer.feature_names,
    )
    factual_cls = explainer_cls.explain_factual(X_test[:1])
    probabilities_cls, probability_interval_cls = explainer_cls.predict_proba(
        X_test[:1], uq_interval=True
    )
    low_cls, high_cls = probability_interval_cls

    boston = load_diabetes()
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        boston.data,
        boston.target,
        test_size=0.2,
        random_state=0,
    )
    explainer_reg = WrapCalibratedExplainer(RandomForestRegressor(random_state=0))
    explainer_reg.fit(X_train_reg, y_train_reg)
    explainer_reg.calibrate(
        X_train_reg,
        y_train_reg,
        feature_names=boston.feature_names,
    )
    factual_reg = explainer_reg.explain_factual(X_test_reg[:1])
    prob_reg, (low_reg, high_reg) = explainer_reg.predict(
        X_test_reg[:1], uq_interval=True
    )

    print(
        f"Classification probability: {probabilities_cls[0, 1]:.3f} "
        f"[{low_cls[0]:.3f}, {high_cls[0]:.3f}]"
    )
    print(factual_cls[0])
    print(
        f"Regression prediction: {prob_reg[0]:.3f} "
        f"[{low_reg[0]:.3f}, {high_reg[0]:.3f}]"
    )
    print(factual_reg[0])

    return SimpleNamespace(
        prob_cls=probabilities_cls,
        low_cls=low_cls,
        high_cls=high_cls,
        factual_cls=factual_cls,
        prob_reg=prob_reg,
        low_reg=low_reg,
        high_reg=high_reg,
        factual_reg=factual_reg,
    )


def test_index_snippet_execution(capsys):
    context = _run_index_snippet()
    out = capsys.readouterr().out
    assert "Classification probability" in out
    assert "Regression prediction" in out
    assert context.factual_cls[0].prediction
    assert context.factual_reg[0].prediction
