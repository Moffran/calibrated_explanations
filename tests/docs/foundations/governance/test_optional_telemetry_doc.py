from __future__ import annotations

from types import SimpleNamespace

import json
import logging
import pytest


def _build_classification_context() -> SimpleNamespace:
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
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
    explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
    explainer.fit(X_train, y_train)
    explainer.calibrate(
        X_train,
        y_train,
        feature_names=breast_cancer.feature_names,
    )
    return SimpleNamespace(explainer=explainer, X_test=X_test)


def test_optional_telemetry_snippets(tmp_path):
    context = _build_classification_context()
    explainer = context.explainer
    X_test = context.X_test

    payload = explainer.runtime_telemetry
    pre = payload.get("preprocessor", {})
    print(pre.get("identifier"))  # e.g. sklearn.compose:ColumnTransformer
    print(pre.get("auto_encode"))

    # Only test explain_fast if the explainer is already fast or if fast plugins are available
    if hasattr(explainer, "explain_fast") and hasattr(explainer.explainer, "is_fast"):
        try:
            if explainer.explainer.is_fast():
                explainer.explain_fast(X_test[:5], _use_plugin=False)
                fast_meta = explainer.runtime_telemetry
                print(fast_meta.get("interval_source"))
        except Exception:
            # Fast explanations may not be available without external plugins
            pass

    output = tmp_path / "batch.telemetry.json"
    with output.open("w", encoding="utf-8") as fh:
        json.dump(explainer.runtime_telemetry, fh, indent=2)

    prometheus_client = pytest.importorskip("prometheus_client")
    Gauge = prometheus_client.Gauge

    logger = logging.getLogger("calibrated_explanations.telemetry")
    interval_source = Gauge(
        "ce_interval_source", "Active interval calibrator", ["identifier"]
    )

    batch = explainer.explain_factual(X_test[:10])
    payload = getattr(batch, "telemetry", {})

    logger.info("explain_factual", extra={"telemetry": payload})
    interval_source.labels(payload.get("interval_source", "unknown")).set(1)
