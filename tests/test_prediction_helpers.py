import numpy as np
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core import prediction_helpers as ph
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def test_prediction_helpers_round_trip():
    data = load_iris()
    X_train, X_cal, y_train, y_cal = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    clf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    clf.fit(X_train, y_train)
    explainer = CalibratedExplainer(clf, X_cal, y_cal, mode="classification", seed=42)

    X_test = X_cal[:2]
    # Delegated validation should produce identical array
    X_valid = ph.validate_and_prepare_input(explainer, X_test)
    assert np.array_equal(X_valid, X_test)

    explanation = ph.initialize_explanation(
        explainer, X_valid, (5, 95), None, None, features_to_ignore=None
    )
    assert explanation is not None

    # Need a discretizer assigned prior to predict step
    explainer.set_discretizer("binaryEntropy")
    # Predict step returns tuple; sanity check shape length invariants
    predict, low, high, prediction, *_rest = ph.explain_predict_step(
        explainer, X_valid, None, (5, 95), None, features_to_ignore=[]
    )
    # The internal predict step includes perturbed instances, so length should be >= original
    assert len(predict) >= len(X_valid)
    assert len(low) == len(predict)
    assert len(high) == len(predict)
    # The base prediction vector stored in prediction dict should align with original test size
    base_pred_len = (
        len(prediction["predict"])
        if hasattr(prediction["predict"], "__len__")
        else prediction["predict"].shape[0]
    )  # type: ignore[index]
    assert base_pred_len == len(X_valid)
