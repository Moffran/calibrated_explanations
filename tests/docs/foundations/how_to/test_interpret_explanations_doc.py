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

    factual_rules = factual.build_rules_payload()
    for rule in factual_rules:
        print(
            f"{rule['feature']:>20} "
            f"| condition={rule['condition']['operator']} {rule['condition']['value']} "
            f"| weight={rule['weight']:+.3f} "
            f"| weight interval=({rule['uncertainty']['lower_bound']:.3f}, "
            f"{rule['uncertainty']['upper_bound']:.3f})"
        )

    alternative_rules = alternative.build_rules_payload()
    for item in alternative_rules:
        if item['kind'] == 'alternative':
            print('Suggested changes:', item['conditions'])
            print('Resulting calibrated prediction:', item['calibrated_prediction'])
            for feature_rule in item['feature_rules']:
                print(
                    f"  {feature_rule['feature']}: weight={feature_rule['weight']:+.3f} "
                    f"condition={feature_rule['condition']['operator']} {feature_rule['condition']['value']}"
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

    assert factual_rules
    assert alternative_rules
