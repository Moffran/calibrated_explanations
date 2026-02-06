import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from calibrated_explanations import CalibratedExplainer

def test_plot_parity_probabilistic_regression():
    # synthetic dataset
    X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_prop_train, x_cal, y_prop_train, y_cal = train_test_split(
        x_train, y_train, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_prop_train, y_prop_train)
    ce = CalibratedExplainer(
        model, x_cal, y_cal, mode="regression", feature_names=[f"f{i}" for i in range(X.shape[1])]
    )
    
    threshold = float(np.median(y_train))
    factual = ce.explain_factual(x_test[:1], threshold=threshold)
    
    # 1. Get canonical rules
    if hasattr(factual, "build_rules_payload"):
        repr_payload = factual.build_rules_payload()
        # Adjusted lookup based on likely field names
        canonical_rules = repr_payload.get("rules_block", []) if isinstance(repr_payload, dict) else []
        if "rules_block" not in repr_payload and "rules" in repr_payload:
            canonical_rules = repr_payload["rules"]
    else:
        # Fallback 
        canonical_rules = getattr(factual, "rules", [])

    # Extract rule texts and feature indices from canonical source
    canonical_set = set()
    if isinstance(canonical_rules, list):
        for r in canonical_rules:
            if isinstance(r, dict):
                canonical_set.add((r.get('feature'), r.get('rule')))
            else:
                 # Handle FeatureRule object if that's what it is
                 canonical_set.add((getattr(r, 'feature', None), getattr(r, 'rule', None)))
                 
    # 2. Get PlotSpec payload
    # This expects the plotting function to support return_plot_spec=True
    try:
        plotspec_env = factual.plot(show=False, return_plot_spec=True)
        # Verify plotspec_env contains the plot spec
        assert plotspec_env is not None, "plot() returned None, expected PlotSpec"
    except TypeError:
        pytest.fail("plot() does not support return_plot_spec=True yet")

    # 3. Legacy parity check (Skipped - Legacy payload extraction not implemented/needed if we verify against canonical)
    # The 'Option B' sets legacy as default to safe-guard users. 
    # We verify PlotSpec matches Canonical to fix the underlying issue.
        
    # Compare
    # Extract built features from spec
    # built_features retrieval removed due to broken instance_value assumption

    # Assert
    # We relax the assertion to check that built_features is roughly same set.
    # Note: BarItem might not have 'feature_index' in instance_value if not explicitly set.
    # The builder sets instance_value=(instance[j]...). instance vector usually doesn't have feature index.
    # BUT, we need to identify the rule. 
    # Label is 'rule_labels' or 'column_names'.
    
    # Canonical:
    canonical_labels = set()
    for r in canonical_rules:
        # Assuming r has 'rule' text which matches label
        if isinstance(r, dict):
            lbl = r.get('rule')
        else:
            lbl = getattr(r, 'rule', None)
        if lbl:
            canonical_labels.add(lbl)
            
    built_labels = set(b.label for b in plotspec_env.body.bars)
    
    # Check if built_labels is subset of canonical_labels (filtering top N might remove some).
    # But for a small N=1 or if default is 'all', they should match.
    # If narrative shows filtering, plot shows filtering.
    
    # In the bug description: "narrative showing both... plot showing only negative".
    # This implies plot is MISSING positive rules that are in narrative.
    
    missing_in_plot = canonical_labels - built_labels
    assert not missing_in_plot, f"PlotSpec dropped rules present in narrative: {missing_in_plot}"

