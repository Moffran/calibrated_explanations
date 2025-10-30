from __future__ import annotations

from types import SimpleNamespace


def _build_pipeline_context() -> SimpleNamespace:
    from sklearn.compose import ColumnTransformer
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from calibrated_explanations.api.config import ExplainerConfig
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

    numeric = [0, 1, 2]
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        ],
        remainder="drop",
    )

    config = ExplainerConfig(
        model=RandomForestClassifier(random_state=0), preprocessor=preprocessor
    )
    explainer = WrapCalibratedExplainer._from_config(config)

    explainer.fit(X_proper, y_proper)
    explainer.calibrate(X_cal, y_cal)
    factual = explainer.explain_factual(X_test)

    telemetry = explainer.runtime_telemetry
    pre = telemetry.get("preprocessor", {})
    print(pre.get("identifier"))
    print(pre.get("pipeline"))

    return SimpleNamespace(
        explainer=explainer,
        factual=factual,
        telemetry=telemetry,
    )


def test_integrate_with_pipelines_snippet():
    context = _build_pipeline_context()
    assert context.factual
    assert "preprocessor" in context.telemetry
