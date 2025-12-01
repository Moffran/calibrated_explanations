import numpy as np
from calibrated_explanations.viz import (
    build_probabilistic_bars_spec,
    build_regression_bars_spec,
    build_alternative_probabilistic_spec,
    build_alternative_regression_spec,
    build_triangular_plotspec_dict,
    build_global_plotspec_dict,
)


def factual_probabilistic_no_uncertainty():
    predict = {"predict": 0.83, "low": 0.8, "high": 0.86}
    feature_weights = [0.35, -0.12, 0.0]
    spec = build_probabilistic_bars_spec(
        title="factual_no_unc",
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=[0, 1, 2],
        column_names=["f0", "f1", "f2"],
        instance=[45.0, 0.2, "x"],
        y_minmax=[0.0, 1.0],
        interval=False,
    )
    return spec


def factual_probabilistic_zero_crossing():
    predict = {"predict": 0.5, "low": 0.4, "high": 0.6}
    # one feature crosses the previous probability pivot; pivot behavior removed
    feature_weights = {
        "predict": [0.6],
        "low": [0.45],
        "high": [0.55],
    }
    spec = build_probabilistic_bars_spec(
        title="factual_cross",
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=[0],
        column_names=["f0"],
        instance=[1],
        y_minmax=[0.0, 1.0],
        interval=True,
    )
    return spec


def factual_regression_interval():
    predict = {"predict": 3.6, "low": 3.2, "high": 4.1}
    feature_weights = [0.25, -0.1]
    spec = build_regression_bars_spec(
        title="reg_interval",
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=[0, 1],
        column_names=["r0", "r1"],
        instance=[2.3, 0.5],
        y_minmax=[-5.0, 100.0],
        interval=True,
    )
    return spec


def alternative_probabilistic_cross_05():
    predict = {"predict": 0.6, "low": 0.45, "high": 0.65}
    feature_predict = {
        "predict": [0.3, 0.7],
        "low": [0.2, 0.6],
        "high": [0.4, 0.8],
    }
    spec = build_alternative_probabilistic_spec(
        title="alt_cross",
        predict=predict,
        feature_weights=feature_predict,
        features_to_plot=[0, 1],
        column_names=["a0", "a1"],
        instance=[0.1, 0.2],
        y_minmax=[0.0, 1.0],
        interval=True,
    )
    return spec


def alternative_regression_interval():
    predict = {"predict": 1.2, "low": 0.5, "high": 2.0}
    feature_predict = {
        "predict": [0.9, -0.2],
        "low": [0.8, -0.4],
        "high": [1.0, 0.1],
    }
    spec = build_alternative_regression_spec(
        title="alt_reg",
        predict=predict,
        feature_weights=feature_predict,
        features_to_plot=[0, 1],
        column_names=["r0", "r1"],
        instance=[0.5, -1.2],
        y_minmax=[-1.0, 2.5],
        interval=True,
    )
    return spec


def alternative_regression_point():
    predict = {"predict": 0.5, "low": 0.2, "high": 0.8}
    feature_predict = [1.1, -0.4]
    spec = build_alternative_regression_spec(
        title="alt_reg_point",
        predict=predict,
        feature_weights=feature_predict,
        features_to_plot=[0, 1],
        column_names=["r0", "r1"],
        instance=[0.05, -0.12],
        y_minmax=[-1.0, 2.0],
        interval=False,
    )
    return spec


def alternative_regression_probability_scale():
    predict = {"predict": 0.22, "low": 0.18, "high": 0.24}
    feature_predict = {
        "predict": [0.01, 0.3],
        "low": [0.0, 0.28],
        "high": [0.02, 0.32],
    }
    xticks = [float(x) for x in np.linspace(0.0, 1.0, 11)]
    spec = build_alternative_probabilistic_spec(
        title="alt_reg_prob",
        predict=predict,
        feature_weights=feature_predict,
        features_to_plot=[0, 1],
        column_names=["r0", "r1"],
        instance=[0.05, -0.12],
        y_minmax=None,
        interval=True,
        neg_label="Y ≥ 180000.00",
        pos_label="Y < 180000.00",
        xlabel="Probability of target being below 180000.00",
        xlim=(0.0, 1.0),
        xticks=xticks,
    )
    return spec


def triangular_probabilistic():
    proba = np.array([0.7])
    uncertainty = np.array([0.05])
    rule_proba = np.array([0.6])
    rule_uncertainty = np.array([0.02])
    spec = build_triangular_plotspec_dict(
        title="triangle",
        proba=proba,
        uncertainty=uncertainty,
        rule_proba=rule_proba,
        rule_uncertainty=rule_uncertainty,
        num_to_show=1,
    )
    return spec


def global_probabilistic_multiclass():
    # small synthetic multiclass proba array
    proba = np.array([[0.1, 0.9], [0.8, 0.2]])
    low = proba - 0.01
    high = proba + 0.01
    y_test = np.array([1, 0])
    # uncertainty per-sample (use small values)
    uncertainty = np.full(shape=(proba.shape[0], proba.shape[1]), fill_value=0.01)
    spec = build_global_plotspec_dict(
        title="global_multi",
        proba=proba,
        predict=None,
        low=low,
        high=high,
        uncertainty=uncertainty,
        y_test=y_test,
        is_regularized=True,
    )
    # attach save behavior metadata post-build (builder does not accept save_behavior kw)
    spec.setdefault("plot_spec", {}).setdefault("save_behavior", {})
    spec["plot_spec"]["save_behavior"].update(
        {
            "path": "plots/global",
            "title": "global_multi",
            "default_exts": ["svg", "png"],
            "save_global": True,
        }
    )
    return spec
