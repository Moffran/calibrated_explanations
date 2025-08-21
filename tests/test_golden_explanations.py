import json
import warnings
from pathlib import Path

from calibrated_explanations import CalibratedExplainer
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

GOLDEN_DIR = Path(__file__).parent / "data" / "golden"
CLASS_FILE = GOLDEN_DIR / "classification.json"
REG_FILE = GOLDEN_DIR / "regression.json"


def _serialize_classification(exp):
    # Pull first 3 explanation objects for compact rule snapshot (string repr truncated)
    rule_summaries = []
    for e in exp.explanations[:3]:
        s = str(e).splitlines()[:6]
        rule_summaries.append(" ".join(s))
    probs = None
    if hasattr(exp, "probabilities") and getattr(exp, "probabilities") is not None:
        probs = [
            [round(float(v), 6) for v in row]
            for row in exp.probabilities[:5]  # type: ignore[index]
        ]
    return {
        "mode": "classification",
        "n_instances": len(exp.explanations),
        "predictions": (exp.predictions[:5].tolist() if hasattr(exp, "predictions") else None),
        "probabilities_head": probs,
        "class_labels": getattr(exp, "class_labels", None),
        "feature_names": getattr(exp, "feature_names", None),
        "rule_summaries": rule_summaries,
    }


def _serialize_regression(exp):
    rule_summaries = []
    for e in exp.explanations[:3]:
        s = str(e).splitlines()[:6]
        rule_summaries.append(" ".join(s))
    lows = highs = None
    if (
        hasattr(exp, "lower")
        and hasattr(exp, "upper")
        and getattr(exp, "lower") is not None
        and getattr(exp, "upper") is not None
    ):
        lows = [round(float(v), 6) for v in exp.lower[:5]]  # type: ignore[index]
        highs = [round(float(v), 6) for v in exp.upper[:5]]  # type: ignore[index]
    return {
        "mode": "regression",
        "n_instances": len(exp.explanations),
        "predictions": (exp.predictions[:5].tolist() if hasattr(exp, "predictions") else None),
        "lower_head": lows,
        "upper_head": highs,
        "feature_names": getattr(exp, "feature_names", None),
        "rule_summaries": rule_summaries,
    }


def _write_if_missing(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _approx_equal(a, b):
    if a is None or b is None:
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if abs(x - y) > 1e-12:
                    return False
            else:
                if x != y:
                    return False
        return True
    return a == b


def test_golden_classification():
    rng = 42
    data = load_iris()
    X_train, X_cal, y_train, y_cal = train_test_split(
        data.data, data.target, test_size=0.2, random_state=rng, stratify=data.target
    )
    clf = RandomForestClassifier(n_estimators=25, random_state=rng, max_depth=3)
    clf.fit(X_train, y_train)
    explainer = CalibratedExplainer(clf, X_cal, y_cal, mode="classification", seed=rng)
    factual = explainer.explain_factual(X_cal[:5])
    payload = _serialize_classification(factual)
    if not CLASS_FILE.exists():
        _write_if_missing(CLASS_FILE, payload)
        warnings.warn("Golden classification fixture created; re-run tests.")
        return
    golden = json.loads(CLASS_FILE.read_text())
    # Compare keys & values
    assert golden.keys() == payload.keys()
    for k in golden:
        assert _approx_equal(golden[k], payload[k]), f"Mismatch in field {k}"


def test_golden_regression():
    rng = 42
    data = load_diabetes()
    X_train, X_cal, y_train, y_cal = train_test_split(
        data.data, data.target, test_size=0.2, random_state=rng
    )
    reg = RandomForestRegressor(n_estimators=30, random_state=rng, max_depth=4)
    reg.fit(X_train, y_train)
    explainer = CalibratedExplainer(reg, X_cal, y_cal, mode="regression", seed=rng)
    factual = explainer.explain_factual(X_cal[:5])
    payload = _serialize_regression(factual)
    if not REG_FILE.exists():
        _write_if_missing(REG_FILE, payload)
        warnings.warn("Golden regression fixture created; re-run tests.")
        return
    golden = json.loads(REG_FILE.read_text())
    assert golden.keys() == payload.keys()
    for k in golden:
        assert _approx_equal(golden[k], payload[k]), f"Mismatch in field {k}"
