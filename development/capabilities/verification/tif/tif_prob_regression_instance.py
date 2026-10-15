"""TIF verification interface for instance-wise probabilistic regression outputs.

TIF ID: CE-TIF-PRED-PROB-002

Requirements served:
  CE-REQ-PRED-PROB-INSTANCE-001 — thresholded regression outputs are batch-invariant

Tests call run_prob_regression_instance_tif_scenario() and assert on the returned
ProbRegressionInstanceObservation against acceptance criteria from the requirement file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

_RNG_SEED = 42
_N_SAMPLES = 200
_N_FEATURES = 4
_N_TEST = 12


@dataclass
class ProbRegressionInstanceObservation:
    """Structured observation returned by the instance-wise TIF scenario.

    Fields
    ------
    max_abs_batch_deviation : float
        Largest absolute difference between batched and one-at-a-time outputs
        over the point estimate and both interval bounds.
    max_abs_permutation_deviation : float
        Largest absolute difference between reversed-batch (mapped back) and
        one-at-a-time outputs over the same values.
    threshold : float
        The threshold used.
    n_instances : int
        Number of test instances.
    """

    max_abs_batch_deviation: float
    max_abs_permutation_deviation: float
    threshold: float
    n_instances: int


def _build_regression_explainer() -> tuple:
    """Build a deterministic fitted+calibrated WrapCalibratedExplainer for regression."""
    X_all, y_all = make_regression(
        n_samples=_N_SAMPLES,
        n_features=_N_FEATURES,
        n_informative=3,
        random_state=_RNG_SEED,
        noise=10.0,
    )
    X_train_cal, X_test, y_train_cal, _ = train_test_split(
        X_all, y_all, test_size=_N_TEST, random_state=_RNG_SEED
    )
    X_proper, X_cal, y_proper, y_cal = train_test_split(
        X_train_cal, y_train_cal, test_size=0.4, random_state=_RNG_SEED
    )
    explainer = WrapCalibratedExplainer(
        RandomForestRegressor(n_estimators=10, random_state=_RNG_SEED)
    )
    explainer.fit(X_proper, y_proper)
    explainer.calibrate(X_cal, y_cal)

    assert explainer.fitted, "TIF sanity: explainer must be fitted"
    assert explainer.calibrated, "TIF sanity: explainer must be calibrated"

    return explainer, X_test


def run_prob_regression_instance_tif_scenario(
    *,
    threshold: float = 0.0,
) -> ProbRegressionInstanceObservation:
    """Stimulate CE-REQ-PRED-PROB-INSTANCE-001.

    TIF ID: CE-TIF-PRED-PROB-002

    Requirements served:
      CE-REQ-PRED-PROB-INSTANCE-001 (observation: max_abs_batch_deviation,
      max_abs_permutation_deviation)

    Parameters
    ----------
    threshold : float
        Scalar threshold for the event ``y <= threshold``. Default 0.0.

    Returns
    -------
    ProbRegressionInstanceObservation
        Structured observations. Tests assert on these fields.
    """
    explainer, X_test = _build_regression_explainer()

    def _outputs(x):
        proba, (low, high) = explainer.predict_proba(x, uq_interval=True, threshold=threshold)
        return np.column_stack([np.asarray(proba)[:, 1], low, high])

    single = np.vstack([_outputs(X_test[i : i + 1]) for i in range(len(X_test))])
    batched = _outputs(X_test)
    reversed_back = _outputs(X_test[::-1])[::-1]
    return ProbRegressionInstanceObservation(
        max_abs_batch_deviation=float(np.max(np.abs(batched - single))),
        max_abs_permutation_deviation=float(np.max(np.abs(reversed_back - single))),
        threshold=threshold,
        n_instances=len(X_test),
    )


_DATASET_ID = (
    "sklearn make_regression n_samples=200 n_features=4 n_informative=3 noise=10 random_seed=42"
)


def build_evidence_payload(
    *,
    commit_sha: str,
    timestamp: str,
    date_suffix: str,
    package_version: str,
    python_version: str,
    platform_str: str,
    spec_claim_ids: list,
    spec_requirement_ids: list,
    spec_adr_refs: list,
    spec_tif_id: str,
    spec_verification_type: str,
    spec_evidence_key: str,
) -> dict:
    """Build a complete evidence payload for CE-TIF-PRED-PROB-002.

    Called by scripts/generate_tif_evidence.py during dynamic discovery.
    Envelope metadata is injected from the TIF spec by the caller.
    """
    from tif_evidence_helpers import (
        acceptance_entry,
        build_payload,
        obs_to_dict,
        scenario_entry,
    )

    obs = run_prob_regression_instance_tif_scenario(threshold=0.0)
    scenarios = [
        scenario_entry(
            "prob_regression_batch_invariance_threshold_0",
            obs_to_dict(obs),
            [
                acceptance_entry(
                    "CE-REQ-PRED-PROB-INSTANCE-001",
                    "max_abs_batch_deviation == 0.0",
                    True,
                    obs.max_abs_batch_deviation == 0.0,
                ),
                acceptance_entry(
                    "CE-REQ-PRED-PROB-INSTANCE-001",
                    "max_abs_permutation_deviation == 0.0",
                    True,
                    obs.max_abs_permutation_deviation == 0.0,
                ),
            ],
        ),
    ]
    return build_payload(
        spec_evidence_key,
        claim_ids=spec_claim_ids,
        requirement_ids=spec_requirement_ids,
        adr_refs=spec_adr_refs,
        tif_ids=[spec_tif_id],
        verification_type=spec_verification_type,
        dataset_id=_DATASET_ID,
        scenarios=scenarios,
        commit_sha=commit_sha,
        timestamp=timestamp,
        date_suffix=date_suffix,
        package_version=package_version,
        python_version=python_version,
        platform_str=platform_str,
        configuration={"threshold": 0.0},
    )
