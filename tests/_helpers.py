def assert_predictions_match(y_pred1, y_pred2, msg="Predictions don't match"):
    """Verify predictions match exactly."""
    assert len(y_pred1) == len(y_pred2), f"{msg}: Different lengths"
    assert all(y1 == y2 for y1, y2 in zip(y_pred1, y_pred2)), msg


def assert_valid_confidence_bounds(predictions, bounds, msg="Invalid confidence bounds"):
    """Ensure confidence bounds contain predictions."""
    low, high = bounds
    for i, pred in enumerate(predictions):
        assert low[i] <= pred <= high[i], f"{msg} at index {i}"


def generic_test(cal_exp, x_prop_train, y_prop_train, x, y):
    """Run a generic calibrated explainer test routine.

    This function encapsulates repeated assertions used across several
    test modules. Tests should import and call this helper rather than
    importing from another test module.
    """
    cal_exp.fit(x_prop_train, y_prop_train)
    assert cal_exp.fitted
    assert cal_exp.calibrated

    learner = cal_exp.learner
    explainer = cal_exp.explainer

    from calibrated_explanations.core import WrapCalibratedExplainer

    new_exp = WrapCalibratedExplainer(learner)
    assert new_exp.fitted
    assert not new_exp.calibrated
    assert new_exp.learner == learner

    new_exp = WrapCalibratedExplainer(explainer)
    assert new_exp.fitted
    assert new_exp.calibrated
    assert new_exp.explainer == explainer
    assert new_exp.learner == learner

    cal_exp.plot(x, show=False)
    cal_exp.plot(x, y, show=False)
    return cal_exp


def get_classification_model(model_name, x_prop_train, y_prop_train):
    """Return a fitted classification model (RF or DT)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    import os

    fast = bool(os.getenv("FAST_TESTS"))
    t1 = DecisionTreeClassifier()
    r1 = RandomForestClassifier(n_estimators=3 if fast else 10)
    model_dict = {"RF": (r1, "RF"), "DT": (t1, "DT")}

    model, model_name = model_dict[model_name]
    model.fit(x_prop_train, y_prop_train)
    return model, model_name


def get_regression_model(model_name, x_prop_train, y_prop_train):
    """Return a fitted regression model (RF or DT)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    import os

    fast = bool(os.getenv("FAST_TESTS"))
    t1 = DecisionTreeRegressor()
    r1 = RandomForestRegressor(n_estimators=3 if fast else 10)
    model_dict = {"RF": (r1, "RF"), "DT": (t1, "DT")}

    model, model_name = model_dict[model_name]
    model.fit(x_prop_train, y_prop_train)
    return model, model_name


def initiate_explainer(
    model,
    x_cal,
    y_cal,
    feature_names,
    categorical_features,
    mode,
    class_labels=None,
    difficulty_estimator=None,
    bins=None,
    fast=False,
    verbose=False,
    **kwargs,
):
    """Initialize a CalibratedExplainer instance."""
    from calibrated_explanations.core import CalibratedExplainer

    return CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        feature_names=feature_names,
        categorical_features=categorical_features,
        mode=mode,
        class_labels=class_labels,
        bins=bins,
        fast=fast,
        difficulty_estimator=difficulty_estimator,
        verbose=verbose,
        **kwargs,
    )
