from __future__ import annotations

from tests.docs.get_started.test_quickstart_classification_doc import (
    _run_quickstart_classification,
)


def test_interpret_explanations_snippets():
    context = _run_quickstart_classification()
    explainer = context.explainer
    X_test = context.X_test

    factual_batch = explainer.explain_factual(X_test[:1])
    factual = factual_batch[0]
    alternative_batch = explainer.explore_alternatives(X_test[:1])
    alternative = alternative_batch[0]

    print(f"Mode: {factual.get_mode()}  Calibrated prediction: {factual.predict:.3f}")
    print(f"Mode: {alternative.get_mode()}  Calibrated prediction: {alternative.predict:.3f}")

    factual_payload = factual.build_rules_payload()
    factual_core = factual_payload["core"]
    for rule in factual_core["feature_rules"]:
        interval = rule["weight"]["uncertainty_interval"]
        print(
            f"{rule['condition']['feature']:>20} "
            f"| condition={rule['condition']['operator']} {rule['condition']['value']} "
            f"| weight={rule['weight']['value']:+.3f} "
            f"| weight interval=({interval['lower']:.3f}, {interval['upper']:.3f})"
        )

    factual_metadata = factual_payload["metadata"]["feature_rules"]
    for extra in factual_metadata:
        print(
            f"  repr={extra['weight_uncertainty']['representation']} "
            f"prediction interval=({extra['prediction_uncertainty']['lower_bound']:.3f}, "
            f"{extra['prediction_uncertainty']['upper_bound']:.3f})"
        )

    alternative_payload = alternative.build_rules_payload()
    alternative_core = alternative_payload["core"]
    for rule in alternative_core["feature_rules"]:
        interval = rule["prediction"]["uncertainty_interval"]
        print(
            f"{rule['condition']['feature']:>20} "
            f"| new condition={rule['condition']['operator']} {rule['condition']['value']} "
            f"| predicted value={rule['prediction']['value']:+.3f} "
            f"| interval=({interval['lower']:.3f}, {interval['upper']:.3f})"
        )

    alternative_metadata = alternative_payload["metadata"]["feature_rules"]
    for extra in alternative_metadata:
        print(
            f"  Δ prediction={extra['weight_value']:+.3f} "
            f"(repr={extra['weight_uncertainty']['representation']})"
        )

    fig = factual.plot(uncertainty=True, filter_top=6)
    try:
        from matplotlib import pyplot as plt

        plt.close(fig)
    except Exception:
        pass

    telemetry = getattr(factual_batch, "telemetry", {})
    print(telemetry["interval_source"])
    print(telemetry["plot_source"])
    print(telemetry["uncertainty"])

    assert factual_core["feature_rules"]
    assert alternative_core["feature_rules"]
