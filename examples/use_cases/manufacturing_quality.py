"""Manufacturing quality (regression) with calibrated explanations.

Scenario: regression.
Outputs: JSON with factual table and prediction interval.
"""

from __future__ import annotations

import json

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer


def main() -> None:
    features, target = make_regression(
        n_samples=600,
        n_features=10,
        n_informative=6,
        noise=0.3,
        random_state=0,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=0,
    )
    x_proper, x_cal, y_proper, y_cal = train_test_split(
        x_train,
        y_train,
        test_size=0.25,
        random_state=0,
    )

    explainer = WrapCalibratedExplainer(RandomForestRegressor(random_state=0))
    explainer.fit(x_proper, y_proper)
    explainer.calibrate(x_cal, y_cal, feature_names=[f"f{i}" for i in range(features.shape[1])])

    factual = explainer.explain_factual(x_test[:1])
    prediction, (low, high) = explainer.predict(x_test[:1], uq_interval=True)

    summary = {
        "scenario": "manufacturing_quality",
        "factual_table": str(factual[0]),
        "prediction": float(prediction[0]),
        "prediction_interval": {"low": float(low[0]), "high": float(high[0])},
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
