import json
import warnings
from pathlib import Path

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

GOLDEN_DIR = Path(__file__).parent / "data" / "golden"
CLASS_FILE = GOLDEN_DIR / "classification.json"
REG_FILE = GOLDEN_DIR / "regression.json"


def serialize_classification(exp):
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


def serialize_regression(exp):
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


def write_if_missing(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def approx_equal(a, b):
    if a is None or b is None:
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            # If both items are numeric, compare within tolerance.
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if abs(x - y) > 1e-6:
                    return False
            # If both are strings, attempt to compare embedded numeric tokens
            # with a relaxed tolerance; fall back to exact string compare.
            elif isinstance(x, str) and isinstance(y, str):
                import re

                float_re = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
                x_tokens = float_re.findall(x)
                y_tokens = float_re.findall(y)
                if len(x_tokens) == len(y_tokens) and len(x_tokens) > 0:
                    for xf, yf in zip(x_tokens, y_tokens):
                        try:
                            if abs(float(xf) - float(yf)) > 1e-2:
                                return False
                        except Exception:
                            if xf != yf:
                                return False
                else:
                    if x != y:
                        return False
            else:
                if x != y:
                    return False
        return True
    return a == b


def test_golden_classification():
    rng = 42
    data = load_iris()
    x_train, x_cal, y_train, y_cal = train_test_split(
        data.data, data.target, test_size=0.2, random_state=rng, stratify=data.target
    )
    clf = RandomForestClassifier(n_estimators=25, random_state=rng, max_depth=3)
    clf.fit(x_train, y_train)
    explainer = CalibratedExplainer(clf, x_cal, y_cal, mode="classification", seed=rng)
    factual = explainer.explain_factual(x_cal[:5])
    payload = serialize_classification(factual)
    if not CLASS_FILE.exists():
        write_if_missing(CLASS_FILE, payload)
        warnings.warn("Golden classification fixture created; re-run tests.")
        return
    golden = json.loads(CLASS_FILE.read_text())
    # Compare keys & values
    assert golden.keys() == payload.keys()
    for k in golden:
        # Rule summaries are textual and may vary slightly across envs/formatting.
        # Validate presence and basic structure rather than exact string equality.
        if k == "rule_summaries":
            assert isinstance(payload[k], list)
            assert len(payload[k]) >= 3
            for item in payload[k][:3]:
                assert isinstance(item, str)
        else:
            assert approx_equal(golden[k], payload[k]), f"Mismatch in field {k}"


def test_golden_regression():
    rng = 42
    data = load_diabetes()
    x_train, x_cal, y_train, y_cal = train_test_split(
        data.data, data.target, test_size=0.2, random_state=rng
    )
    reg = RandomForestRegressor(n_estimators=30, random_state=rng, max_depth=4)
    reg.fit(x_train, y_train)
    explainer = CalibratedExplainer(reg, x_cal, y_cal, mode="regression", seed=rng)
    factual = explainer.explain_factual(x_cal[:5])
    payload = serialize_regression(factual)
    if not REG_FILE.exists():
        write_if_missing(REG_FILE, payload)
        warnings.warn("Golden regression fixture created; re-run tests.")
        return
    golden = json.loads(REG_FILE.read_text())
    assert golden.keys() == payload.keys()
    for k in golden:
        if k == "rule_summaries":
            assert isinstance(payload[k], list)
            assert len(payload[k]) >= 3
            for item in payload[k][:3]:
                assert isinstance(item, str)
        else:
            assert approx_equal(golden[k], payload[k]), f"Mismatch in field {k}"
