"""Legacy tests for the private _exp_to_domain method on CalibratedExplanations.

These tests access _exp_to_domain, a private method, which is not permitted in the
main test tree under the ADR-030 public-contract policy. They are retained here for
regression coverage until the equivalent behavior is fully covered by the public
to_json / from_json API tests in tests/unit/explanations/test_explanation_domain_model.py.

Allowlist entry required in .github/private_member_allowlist.json to allow these
tests to pass the private-member policy check.
"""

from __future__ import annotations

from calibrated_explanations.explanations.models import (
    CalibrationDescriptor,
    Explanation,
    ModelDescriptor,
)


def _make_minimal_factual():
    """Build a minimal CalibratedExplanations with two FactualExplanations via WrapCalibratedExplainer."""
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    from calibrated_explanations import WrapCalibratedExplainer

    rng = np.random.default_rng(42)
    x = rng.random((20, 3))
    y = (x[:, 0] > 0.5).astype(int)
    model = DecisionTreeClassifier(random_state=42).fit(x[:10], y[:10])
    ce = WrapCalibratedExplainer(model)
    ce.calibrate(x[10:], y[10:])
    result = ce.explain_factual(x[:2])
    return result


def test_exp_to_domain_returns_explanation_instance():
    """`_exp_to_domain` must return an Explanation dataclass instance."""
    ce = _make_minimal_factual()
    domain = ce._exp_to_domain(ce.explanations[0])
    assert isinstance(domain, Explanation)


def test_exp_to_domain_populates_calibration_metadata():
    """`_exp_to_domain` must set calibration_metadata as a CalibrationDescriptor with a non-None method."""
    ce = _make_minimal_factual()
    domain = ce._exp_to_domain(ce.explanations[0])
    assert isinstance(domain.calibration_metadata, CalibrationDescriptor)
    assert domain.calibration_metadata.method is not None
    assert domain.calibration_metadata.method == ce.calibrated_explainer.mode


def test_exp_to_domain_populates_model_metadata():
    """`_exp_to_domain` must set model_metadata as a ModelDescriptor with the learner class name."""
    ce = _make_minimal_factual()
    domain = ce._exp_to_domain(ce.explanations[0])
    assert isinstance(domain.model_metadata, ModelDescriptor)
    assert domain.model_metadata.type == "DecisionTreeClassifier"
