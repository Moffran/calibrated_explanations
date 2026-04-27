from __future__ import annotations

from types import SimpleNamespace

import pytest

from calibrated_explanations import plotting


def explanation_without_explainer() -> SimpleNamespace:
    return SimpleNamespace(
        prediction={"classes": 0},
        y_minmax=(0.0, 1.0),
        get_class_labels=lambda: ("neg", "pos"),
        is_thresholded=lambda: False,
        y_threshold=None,
    )


@pytest.mark.parametrize(
    ("plot_func_name", "legacy_attr", "call_kwargs"),
    (
        (
            "plot_probabilistic",
            "_plot_probabilistic",
            {
                "instance": [0.5],
                "predict": {"predict": 0.5, "low": 0.4, "high": 0.6},
                "feature_weights": {"predict": [0.5], "low": [0.4], "high": [0.6]},
                "features_to_plot": [0],
                "num_to_show": 1,
                "column_names": ["f1"],
                "title": "legacy default",
                "path": None,
                "show": False,
            },
        ),
        (
            "plot_regression",
            "plot_regression",
            {
                "instance": [0.5],
                "predict": {"predict": 0.5, "low": 0.4, "high": 0.6},
                "feature_weights": {"predict": [0.5], "low": [0.4], "high": [0.6]},
                "features_to_plot": [0],
                "num_to_show": 1,
                "column_names": ["f1"],
                "title": "legacy default",
                "path": None,
                "show": False,
            },
        ),
        (
            "plot_alternative",
            "plot_alternative",
            {
                "instance": [0.5],
                "predict": {"predict": 0.5, "low": 0.4, "high": 0.6},
                "feature_predict": {"predict": [0.5], "low": [0.4], "high": [0.6]},
                "features_to_plot": [0],
                "num_to_show": 1,
                "column_names": ["f1"],
                "title": "legacy default",
                "path": None,
                "show": False,
            },
        ),
    ),
)
def test_should_keep_legacy_default_for_chain_based_plotters_when_chain_prefers_legacy(
    monkeypatch: pytest.MonkeyPatch,
    plot_func_name: str,
    legacy_attr: str,
    call_kwargs: dict[str, object],
) -> None:
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(plotting, "_resolve_explainer_from_explanation", lambda explanation: object())
    monkeypatch.setattr(
        plotting,
        "_resolve_plot_style_chain",
        lambda explainer, style: ("legacy", "plot_spec.default"),
    )
    monkeypatch.setattr(
        plotting.legacy,
        legacy_attr,
        lambda *args, **kwargs: legacy_calls.append((args, kwargs)),
    )

    plot_func = getattr(plotting, plot_func_name)
    plot_func(explanation_without_explainer(), use_legacy=None, **call_kwargs)

    assert legacy_calls


@pytest.mark.parametrize("plot_func_name", ("plot_probabilistic", "plot_regression", "plot_alternative"))
def test_should_select_plotspec_for_chain_based_plotters_when_style_override_requests_it(
    monkeypatch: pytest.MonkeyPatch,
    plot_func_name: str,
) -> None:
    captured_styles: list[str | None] = []

    monkeypatch.setattr(
        plotting,
        "_render_instance_plot_plugin",
        lambda explanation, explicit_style=None, **kwargs: captured_styles.append(explicit_style)
        or {"style": explicit_style},
    )

    plot_func = getattr(plotting, plot_func_name)
    common = {
        "instance": [0.5],
        "predict": {"predict": 0.5, "low": 0.4, "high": 0.6},
        "features_to_plot": [0],
        "num_to_show": 1,
        "column_names": ["f1"],
        "title": "plotspec opt-in",
        "path": None,
        "show": False,
        "style_override": "plot_spec.default",
        "use_legacy": None,
    }
    if plot_func_name == "plot_alternative":
        common["feature_predict"] = {"predict": [0.5], "low": [0.4], "high": [0.6]}
    else:
        common["feature_weights"] = {"predict": [0.5], "low": [0.4], "high": [0.6]}

    plot_func(explanation_without_explainer(), **common)

    assert captured_styles == ["plot_spec.default"]


@pytest.mark.parametrize("plot_func_name", ("plot_probabilistic", "plot_regression", "plot_alternative"))
def test_should_use_plotspec_first_when_no_explainer_is_available(
    monkeypatch: pytest.MonkeyPatch,
    plot_func_name: str,
) -> None:
    captured_styles: list[str | None] = []
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(
        plotting,
        "_render_instance_plot_plugin",
        lambda explanation, explicit_style=None, **kwargs: captured_styles.append(explicit_style)
        or {"style": explicit_style},
    )
    monkeypatch.setattr(plotting.legacy, "_plot_probabilistic", lambda *args, **kwargs: legacy_calls.append((args, kwargs)))
    monkeypatch.setattr(plotting.legacy, "plot_regression", lambda *args, **kwargs: legacy_calls.append((args, kwargs)))
    monkeypatch.setattr(plotting.legacy, "plot_alternative", lambda *args, **kwargs: legacy_calls.append((args, kwargs)))

    plot_func = getattr(plotting, plot_func_name)
    common = {
        "instance": [0.5],
        "predict": {"predict": 0.5, "low": 0.4, "high": 0.6},
        "features_to_plot": [0],
        "num_to_show": 1,
        "column_names": ["f1"],
        "title": "no explainer",
        "path": None,
        "show": False,
        "use_legacy": None,
    }
    if plot_func_name == "plot_alternative":
        common["feature_predict"] = {"predict": [0.5], "low": [0.4], "high": [0.6]}
    else:
        common["feature_weights"] = {"predict": [0.5], "low": [0.4], "high": [0.6]}

    plot_func(explanation_without_explainer(), **common)

    assert captured_styles == ["plot_spec.default"]
    assert not legacy_calls


def test_should_keep_legacy_default_for_triangular_when_use_legacy_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(plotting.legacy, "plot_triangular", lambda *args, **kwargs: legacy_calls.append((args, kwargs)))

    plotting.plot_triangular(
        explanation=None,
        proba=[0.2],
        uncertainty=[0.1],
        rule_proba=[0.3],
        rule_uncertainty=[0.05],
        num_to_show=1,
        title="triangular",
        path=None,
        show=False,
    )

    assert legacy_calls


def test_should_keep_legacy_default_for_global_when_use_legacy_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(plotting.legacy, "plot_global", lambda *args, **kwargs: legacy_calls.append((args, kwargs)))

    explainer = SimpleNamespace()
    plotting.plot_global(explainer, x=[1, 2], show=False)

    assert legacy_calls
