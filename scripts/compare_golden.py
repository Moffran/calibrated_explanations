import json, re
from pathlib import Path
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from sklearn.datasets import load_iris, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

GOLDEN_DIR = Path(__file__).parent.parent / "tests" / "integration" / "core" / "data" / "golden"
CLASS_FILE = GOLDEN_DIR / "classification.json"
REG_FILE = GOLDEN_DIR / "regression.json"


def float_tokens(s):
    return re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?").findall(s)


def approx_equal(a, b):
    if a is None or b is None:
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if abs(x - y) > 1e-6:
                    return False
            elif isinstance(x, str) and isinstance(y, str):
                x_tokens = float_tokens(x)
                y_tokens = float_tokens(y)
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

# classification
rng = 42
data = load_iris()
x_train, x_cal, y_train, y_cal = train_test_split(data.data, data.target, test_size=0.2, random_state=rng, stratify=data.target)
clf = RandomForestClassifier(n_estimators=25, random_state=rng, max_depth=3)
clf.fit(x_train, y_train)
explainer = CalibratedExplainer(clf, x_cal, y_cal, mode="classification", seed=rng)
factual = explainer.explain_factual(x_cal[:5])

rule_summaries_payload = []
for e in factual.explanations[:3]:
    s = str(e).splitlines()[:6]
    rule_summaries_payload.append(" ".join(s))

golden = json.loads(CLASS_FILE.read_text())
rule_summaries_golden = golden['rule_summaries']

print('Payload lines:')
for l in rule_summaries_payload:
    print(l)
print('\nGolden lines:')
for l in rule_summaries_golden:
    print(l)

print('\nComparison results:')
for i,(g,p) in enumerate(zip(rule_summaries_golden, rule_summaries_payload)):
    print(i, approx_equal(g,p))
    print('gold tokens:', float_tokens(g))
    print('payload tokens:', float_tokens(p))

# regression
rng = 42
data = load_diabetes()
x_train, x_cal, y_train, y_cal = train_test_split(data.data, data.target, test_size=0.2, random_state=rng)
reg = RandomForestRegressor(n_estimators=30, random_state=rng, max_depth=4)
reg.fit(x_train, y_train)
explainer = CalibratedExplainer(reg, x_cal, y_cal, mode="regression", seed=rng)
factual = explainer.explain_factual(x_cal[:5])

rule_summaries_payload = []
for e in factual.explanations[:3]:
    s = str(e).splitlines()[:6]
    rule_summaries_payload.append(" ".join(s))

golden = json.loads(REG_FILE.read_text())
rule_summaries_golden = golden['rule_summaries']

print('\nREG Payload lines:')
for l in rule_summaries_payload:
    print(l)
print('\nREG Golden lines:')
for l in rule_summaries_golden:
    print(l)

print('\nREG Comparison results:')
for i,(g,p) in enumerate(zip(rule_summaries_golden, rule_summaries_payload)):
    print(i, approx_equal(g,p))
    print('gold tokens:', float_tokens(g))
    print('payload tokens:', float_tokens(p))
