from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import RejectResult


def build_wrapper_classifier(seed: int = 0):
    X, y = load_breast_cancer(return_X_y=True)
    x_proper, x_cal, y_proper, y_cal = train_test_split(X, y, test_size=0.25, random_state=seed)
    model = RandomForestClassifier(n_estimators=8, random_state=seed)
    w = WrapCalibratedExplainer(model)
    w.fit(x_proper, y_proper)
    w.calibrate(x_cal, y_cal, seed=seed)
    # deterministic reject behaviour
    w.explainer.seed = seed
    return w, X[:5]


def test_predict_proba_envelope_and_metadata():
    w, Xq = build_wrapper_classifier(seed=7)
    res = w.predict_proba(Xq, uq_interval=True, reject_policy=RejectPolicy.FLAG)
    assert isinstance(res, RejectResult)
    # legacy-shaped prediction must be preserved inside envelope for uq_interval=True
    pred = res.prediction
    assert isinstance(pred, tuple) and len(pred) == 2
    proba, (low, high) = pred
    assert proba.shape[0] == len(Xq)
    # metadata keys
    meta = res.metadata
    assert isinstance(meta, dict)
    for key in ("ambiguity_mask", "novelty_mask", "prediction_set_size", "epsilon"):
        assert key in meta


def test_predict_proba_legacy_without_reject():
    w, Xq = build_wrapper_classifier(seed=9)
    legacy = w.predict_proba(Xq, uq_interval=True)
    # legacy returns (proba, (low, high)) tuple when uq_interval=True
    assert isinstance(legacy, tuple) and len(legacy) == 2
    proba, (low, high) = legacy
    assert proba.shape[0] == len(Xq)
