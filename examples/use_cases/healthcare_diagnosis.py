"""Healthcare diagnosis (binary classification) with calibrated explanations.

Scenario: binary-classification.
Outputs: JSON with factual table, alternatives, and probability interval.
"""

from __future__ import annotations

import json

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer


def main() -> None:
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
    probabilities, (low, high) = explainer.predict_proba(x_test[:1], uq_interval=True)
    prob_value = float(probabilities[0, 1] if probabilities.ndim > 1 else probabilities[0])
    low_value = float(low[0, 1] if low.ndim > 1 else low[0])
    high_value = float(high[0, 1] if high.ndim > 1 else high[0])

    summary = {
        "scenario": "healthcare_diagnosis",
        "factual_table": str(factual[0]),
        "alternative_table": str(alternatives[0]),
        "probability": prob_value,
        "probability_interval": {"low": low_value, "high": high_value},
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
