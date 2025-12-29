import logging
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.api.config import ExplainerBuilder

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Reproduce notebook setup
X, y = make_classification(n_samples=2000, n_features=200, n_informative=5, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=42)
X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(X_train, y_train, test_size=200, random_state=42)

learner = RandomForestClassifier(n_estimators=50, random_state=42)
learner.fit(X_train_proper, y_train_proper)

builder = ExplainerBuilder(learner)
config = (
    builder
    .task("classification")
    .perf_parallel(True, backend="threads", workers=4, granularity="feature")
    .perf_feature_filter(True, per_instance_top_k=5)
    .build_config()
)

wrapper = WrapCalibratedExplainer.from_config(config)
wrapper.calibrate(X_cal, y_cal)
explainer = wrapper.explainer

print("Calling explain_factual with debug logging...")
explanations = wrapper.explain_factual(X_test)

print("explanations.features_to_ignore:", getattr(explanations, "features_to_ignore", None))
print("explanations.features_to_ignore_per_instance present:", getattr(explanations, "features_to_ignore_per_instance", None) is not None)

# Inspect first explanation
e = explanations[0]
import numpy as _np
w = _np.asarray(e.feature_weights["predict"]).reshape(-1)
nonzero_idx = _np.where(_np.abs(w) > 1e-12)[0]
print("first explanation nonzero weight count:", nonzero_idx.size)
print("first explanation nonzero sample:", nonzero_idx[:30].tolist())
print("first explanation len(rule):", len(e._get_rules()["rule"]))
