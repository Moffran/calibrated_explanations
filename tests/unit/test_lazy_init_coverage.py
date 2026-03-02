import pytest


def test_plotting_deprecation_warning(monkeypatch) -> None:
    """Test that package attribute access no longer exposes deprecated plotting alias."""
    import calibrated_explanations

    monkeypatch.delitem(calibrated_explanations.__dict__, "plotting", raising=False)
    with pytest.raises(AttributeError):
        _ = calibrated_explanations.plotting

    import calibrated_explanations.plotting as plotting_module

    assert plotting_module is not None


def test_mappingproxy_reducer_registration_failure_is_non_fatal(monkeypatch) -> None:
    import copyreg
    import importlib

    import calibrated_explanations

    def fail_registration(*_args, **_kwargs):
        raise TypeError("simulated reducer registration failure")

    monkeypatch.setattr(copyreg, "pickle", fail_registration)
    reloaded = importlib.reload(calibrated_explanations)
    assert reloaded is not None
