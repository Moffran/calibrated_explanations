from __future__ import annotations

import pytest

import calibrated_explanations.plotting as plotting
from calibrated_explanations.core import config_helpers
from calibrated_explanations.core.reject import policy as reject_policy


def test_coerce_string_tuple_non_iterable_returns_empty_tuple() -> None:
    assert config_helpers.coerce_string_tuple(1234) == ()


def test_reject_policy_deprecated_attr_and_invalid_policy_path() -> None:
    with pytest.warns(DeprecationWarning):
        deprecated_value = reject_policy.__getattr__("PREDICT_AND_FLAG")
    assert deprecated_value.value == "flag"
    assert reject_policy.is_policy_enabled("not-a-policy") is False
    assert reject_policy.is_policy_enabled("flag") is True


def test_read_plot_pyproject_returns_empty_on_missing_or_invalid_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    assert plotting.read_plot_pyproject() == {}
    (tmp_path / "pyproject.toml").write_text("not: [toml", encoding="utf-8")
    assert plotting.read_plot_pyproject() == {}


def test_plot_alternative_resolves_explainer_from_container_paths() -> None:
    class ContainerWithPrivate:
        def _get_explainer(self):
            return object()

    class ExplanationWithPrivateContainer:
        calibrated_explanations = ContainerWithPrivate()

    plotting.plot_alternative(
        explanation=ExplanationWithPrivateContainer(),
        instance=[0.1],
        predict={"predict": 0.5, "low": 0.4, "high": 0.6},
        feature_predict={"predict": [0.5], "low": [0.4], "high": [0.6]},
        features_to_plot=[0],
        num_to_show=1,
        column_names=["f0"],
        title="noop",
        path=None,
        show=False,
        save_ext=None,
    )

    class ContainerWithAttr:
        calibrated_explainer = object()

    class ExplanationWithAttrContainer:
        calibrated_explanations = ContainerWithAttr()

    plotting.plot_alternative(
        explanation=ExplanationWithAttrContainer(),
        instance=[0.1],
        predict={"predict": 0.5, "low": 0.4, "high": 0.6},
        feature_predict={"predict": [0.5], "low": [0.4], "high": [0.6]},
        features_to_plot=[0],
        num_to_show=1,
        column_names=["f0"],
        title="noop",
        path=None,
        show=False,
        save_ext=None,
    )
