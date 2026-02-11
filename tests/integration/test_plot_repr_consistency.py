import re
from calibrated_explanations import CalibratedExplainer


def parse_repr_rules(repr_str):
    """Extract rule feature names and weights from __repr__ string."""
    rules = {}
    lines = repr_str.strip().split("\n")

    # Identify data start
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Value"):
            start_idx = i + 1
            break

    for line in lines[start_idx:]:
        if not line.strip():
            continue

        # Split by first colon to separate Value
        parts = line.split(":", 1)
        if len(parts) < 2:
            continue

        rest = parts[1].strip()  # "Feature ... Weight ... [ ... ]"

        # Regex to find the weight before the bracket
        m = re.search(r"^(.*?)\s+([-\d\.]+)\s+\[", rest)
        if m:
            rule_text = m.group(1).strip()
            weight = float(m.group(2))
            rules[rule_text] = weight

    return rules


def test_invariant_plot_repr_consistency():
    """Verify that Plot and Repr agree on rule weights/signs using public APIs.

    Using synthetic data to ensure non-zero feature importance.
    """
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    # Generate synthetic regression data
    x, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    columns = [f"Feature_{i}" for i in range(x.shape[1])]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20, random_state=42)
    x_prop_train, x_cal, y_prop_train, y_cal = train_test_split(
        x_train, y_train, test_size=40, random_state=42
    )

    model = LinearRegression()
    model.fit(x_prop_train, y_prop_train)

    calibrated_explainer = CalibratedExplainer(
        model, x_cal, y_cal, mode="regression", feature_names=columns
    )

    X_test = x_test
    explanation = calibrated_explainer.explain_factual(X_test[0])

    # 1. Get PlotSpec (Public API)
    plot_spec = explanation.plot(return_plot_spec=True, show=False)
    # Access internal data structure of the Spec
    # New object-based access
    assert plot_spec.body is not None
    plot_data = plot_spec.body.bars

    # 2. Get Repr Rules (Public API)
    repr_str = str(explanation)
    repr_rules = parse_repr_rules(repr_str)

    # Assert we parsed something
    assert len(repr_rules) > 0, "Failed to parse rules from __repr__"

    # 3. Assert Intersection
    matched_count = 0
    for bar in plot_data:
        rule_text = bar.label
        color_role = bar.color_role

        # Only check items that are rules
        if rule_text and rule_text in repr_rules:
            weight = repr_rules[rule_text]

            # Determine expected color role from repr weight
            expected_role = "positive" if weight > 0 else "negative"
            # Allow neutral if close to zero
            if abs(weight) < 1e-9:
                expected_role = "neutral"

            assert (
                color_role == expected_role
            ), f"Mismatch for rule '{rule_text}': Repr weight {weight} implies {expected_role}, Plot has {color_role}"
            matched_count += 1

    assert matched_count > 0, "No rules matched between Plot and Repr"
