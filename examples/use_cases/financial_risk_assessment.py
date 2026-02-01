"""Financial risk assessment (probabilistic regression) with calibrated explanations.

Scenario: probabilistic-regression.
Outputs: JSON with factual table and thresholded probabilities (with uncertainty bounds).

Probabilistic regression in CE is a thresholded probability query for a real-valued target:
- threshold=t queries exceedance probability (depending on formulation)
- threshold=(low, high) queries P(true value ∈ [low, high])
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer


def main() -> None:
    rng = np.random.default_rng(0)
    n_samples = 600
    annual_income = rng.normal(loc=75000, scale=20000, size=n_samples)
    credit_utilization = rng.uniform(0.05, 0.95, size=n_samples)
    debt_to_income = rng.uniform(0.1, 0.6, size=n_samples)
    years_employed = rng.uniform(0, 20, size=n_samples)
    open_accounts = rng.integers(1, 12, size=n_samples)
    delinquencies = rng.poisson(lam=0.3, size=n_samples)

    x = np.column_stack(
        [
            annual_income,
            credit_utilization,
            debt_to_income,
            years_employed,
            open_accounts,
            delinquencies,
        ]
    )
    noise = rng.normal(scale=5000, size=n_samples)
    y = (
        0.35 * annual_income
        - 20000 * credit_utilization
        - 15000 * debt_to_income
        + 800 * years_employed
        - 1200 * delinquencies
        - 500 * open_accounts
        + noise
    )
    feature_names = [
        "annual_income",
        "credit_utilization",
        "debt_to_income",
        "years_employed",
        "open_accounts",
        "delinquencies",
    ]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
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
    explainer.calibrate(x_cal, y_cal, feature_names=feature_names)

    factual = explainer.explain_factual(x_test[:1])

    # Two probabilistic regression queries:
    # 1) Exceedance probability for a scalar threshold
    exceed_threshold = float(np.quantile(y_cal, 0.75))
    p_exceed, (p_exceed_low, p_exceed_high) = explainer.predict_proba(
        x_test[:1],
        uq_interval=True,
        threshold=exceed_threshold,
    )

    # 2) Probability that true value lies inside a user-chosen interval
    interval_low = float(np.quantile(y_cal, 0.40))
    interval_high = float(np.quantile(y_cal, 0.60))
    p_in, (p_in_low, p_in_high) = explainer.predict_proba(
        x_test[:1],
        uq_interval=True,
        threshold=(interval_low, interval_high),
    )

    # Ensure JSON-serializable scalars
    p_exceed_v = float(np.ravel(p_exceed)[0])
    p_exceed_lo = float(np.ravel(p_exceed_low)[0])
    p_exceed_hi = float(np.ravel(p_exceed_high)[0])
    p_in_v = float(np.ravel(p_in)[0])
    p_in_lo = float(np.ravel(p_in_low)[0])
    p_in_hi = float(np.ravel(p_in_high)[0])

    summary = {
        "scenario": "financial_risk_assessment",
        "factual_table": str(factual[0]),
        "threshold_exceed": exceed_threshold,
        "probability_exceed": p_exceed_v,
        "probability_exceed_interval": {"low": p_exceed_lo, "high": p_exceed_hi},
        "threshold_interval": {"low": interval_low, "high": interval_high},
        "probability_in_interval": p_in_v,
        "probability_in_interval_interval": {"low": p_in_lo, "high": p_in_hi},
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
