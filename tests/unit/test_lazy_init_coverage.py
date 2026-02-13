import importlib

import pytest




def test_plotting_deprecation_warning(monkeypatch) -> None:
    """Test that accessing 'plotting' attribute triggers deprecation warning."""
    import calibrated_explanations

    # Clear cached value to force fresh __getattr__ call
    monkeypatch.delitem(calibrated_explanations.__dict__, "plotting", raising=False)

    with pytest.warns(DeprecationWarning, match="plotting"):
        # Access the deprecated plotting attribute
        _ = calibrated_explanations.plotting


