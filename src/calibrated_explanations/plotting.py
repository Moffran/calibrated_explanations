"""Plotting utilities used by explanation objects and the wrap-first workflow.

The public README surface still calls ``CalibratedExplanation.plot``; this
module houses the implementation while ``calibrated_explanations._plots`` remains
as a deprecated shim for backwards compatibility.
"""

from __future__ import annotations

import configparser
import contextlib
import os
import warnings
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

# Legacy import to ensure legacy plotting is still working
# while development of plotspec, adapters, and builders are unfinished.
from .legacy import plotting as legacy

try:
    import tomllib as _plot_tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    try:  # pragma: no cover - optional dependency path
        import tomli as _plot_tomllib  # type: ignore[assignment]
    except ModuleNotFoundError:  # pragma: no cover - tomllib unavailable
        _plot_tomllib = None  # type: ignore[assignment]

try:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
except Exception as _e:  # pragma: no cover - optional dependency guard
    mcolors = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_IMPORT_ERROR = _e
else:
    _MATPLOTLIB_IMPORT_ERROR = None

def _read_plot_pyproject() -> Dict[str, Any]:
    """Return ``pyproject.toml`` plot configuration when available."""

    if _plot_tomllib is None:
        return {}

    candidate = Path.cwd() / "pyproject.toml"
    if not candidate.exists():
        return {}
    try:
        with candidate.open("rb") as fh:  # type: ignore[arg-type]
            data = _plot_tomllib.load(fh)
    except Exception:  # pragma: no cover - permissive fallback
        return {}

    cursor: Any = data
    for key in ("tool", "calibrated_explanations", "plots"):
        if isinstance(cursor, dict) and key in cursor:
            cursor = cursor[key]
        else:
            return {}
    if isinstance(cursor, dict):
        return dict(cursor)
    return {}


def _split_csv(value: Any) -> Sequence[str]:
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence):
        return tuple(str(item).strip() for item in value if isinstance(item, str) and item.strip())
    return ()


def _resolve_plot_style_chain(explainer, explicit_style: str | None) -> Sequence[str]:
    """Determine the ordered style fallback chain for plot builders/renderers."""

    chain: List[str] = []
    if isinstance(explicit_style, str) and explicit_style:
        chain.append(explicit_style)

    env_style = os.environ.get("CE_PLOT_STYLE")
    if env_style:
        chain.append(env_style.strip())
    chain.extend(_split_csv(os.environ.get("CE_PLOT_STYLE_FALLBACKS")))

    py_settings = _read_plot_pyproject()
    py_style = py_settings.get("style")
    if isinstance(py_style, str) and py_style:
        chain.append(py_style)
    chain.extend(_split_csv(py_settings.get("fallbacks")))

    mode = getattr(explainer, "_last_explanation_mode", None)
    plot_fallbacks = getattr(explainer, "_plot_plugin_fallbacks", {})
    if mode and isinstance(plot_fallbacks, dict):
        chain.extend(plot_fallbacks.get(mode, ()))

    chain.append("legacy")

    ordered: List[str] = []
    seen: set[str] = set()
    for identifier in chain:
        if identifier and identifier not in seen:
            ordered.append(identifier)
            seen.add(identifier)
    if "plot_spec.default" not in seen:
        if "legacy" in ordered:
            legacy_index = ordered.index("legacy")
            ordered.insert(legacy_index, "plot_spec.default")
        else:
            ordered.append("plot_spec.default")
    if "legacy" not in ordered:
        ordered.append("legacy")
    return tuple(ordered)


# pylint: disable=unknown-option-value
# pylint: disable=too-many-arguments, too-many-statements, too-many-branches, too-many-locals, too-many-positional-arguments, fixme


def load_plot_config():
    """Load plot configuration from INI file."""
    config = configparser.ConfigParser()
    config_path = os.path.join(
        os.path.dirname(__file__), "../calibrated-explanations/utils/configurations/plot_config.ini"
    )

    # Set default values
    config["style"] = {"base": "seaborn-v0_8-whitegrid"}
    config["fonts"] = {
        "family": "sans-serif",
        "sans_serif": "Arial",
        "axes_label_size": "12",
        "tick_label_size": "10",
        "legend_size": "10",
        "title_size": "14",
    }
    config["lines"] = {"width": "2"}
    config["grid"] = {"style": "--", "alpha": "0.5"}
    config["figure"] = {
        "dpi": "300",
        "save_dpi": "300",
        "facecolor": "white",
        "axes_facecolor": "white",
        "width": "10",  # Add default width value
    }
    config["colors"] = {
        "background": "#f8f9fa",
        "text": "#666666",
        "grid": "#cccccc",
        "regression": "red",
        "positive": "red",
        "negative": "blue",
        "uncertainty": "lightgrey",
        "alpha": "0.2",
    }

    # Read config file if it exists
    config.read(config_path)
    return config


def update_plot_config(new_config):
    """Update plot configuration file with new values."""
    config = load_plot_config()

    # Update configuration with new values
    for section, values in new_config.items():
        if section not in config:
            config[section] = {}
        for key, value in values.items():
            config[section][key] = str(value)

    # Write updated config to file
    config_path = os.path.join(
        os.path.dirname(__file__), "../calibrated-explanations/utils/configurations/plot_config.ini"
    )
    os.makedirs(os.path.dirname(config_path), exist_ok=True)  # Ensure directory exists
    with open(config_path, "w", encoding="utf-8") as f:
        config.write(f)


def __require_matplotlib() -> None:
    """Ensure matplotlib is available before using plotting functions."""
    if plt is None or mcolors is None:
        msg = (
            "Plotting requires matplotlib. Install the 'viz' extra: "
            "pip install calibrated_explanations[viz]"
        )
        if _MATPLOTLIB_IMPORT_ERROR is not None:
            msg += f"\nOriginal import error: {_MATPLOTLIB_IMPORT_ERROR}"
        raise RuntimeError(msg)


def __setup_plot_style(style_override=None):
    """Set up plot style using configuration with optional runtime overrides."""
    __require_matplotlib()
    config = load_plot_config()

    # Apply style overrides if provided
    if style_override:
        for section, values in style_override.items():
            if section not in config:
                config[section] = {}
                warnings.warn(
                    f'Unknown style section "{section}" in style override.', Warning, stacklevel=2
                )
            for key, value in values.items():
                config[section][key] = str(value)

    plt.style.use(config["style"]["base"])

    # Font settings
    plt.rcParams["font.family"] = config["fonts"]["family"]
    plt.rcParams["font.sans-serif"] = [config["fonts"]["sans_serif"]]
    plt.rcParams["axes.labelsize"] = float(config["fonts"]["axes_label_size"])
    plt.rcParams["xtick.labelsize"] = float(config["fonts"]["tick_label_size"])
    plt.rcParams["ytick.labelsize"] = float(config["fonts"]["tick_label_size"])
    plt.rcParams["legend.fontsize"] = float(config["fonts"]["legend_size"])
    plt.rcParams["axes.titlesize"] = float(config["fonts"]["title_size"])

    # Line settings
    plt.rcParams["lines.linewidth"] = float(config["lines"]["width"])

    # Grid settings
    plt.rcParams["grid.linestyle"] = config["grid"]["style"]
    plt.rcParams["grid.alpha"] = float(config["grid"]["alpha"])

    # Figure settings
    plt.rcParams["figure.dpi"] = float(config["figure"]["dpi"])
    plt.rcParams["savefig.dpi"] = float(config["figure"]["save_dpi"])
    plt.rcParams["figure.facecolor"] = config["figure"]["facecolor"]
    plt.rcParams["axes.facecolor"] = config["figure"]["axes_facecolor"]

    return config


def _plot_probabilistic(
    explanation,
    instance,
    predict,
    feature_weights,
    features_to_plot,
    num_to_show,
    column_names,
    title,
    path,
    show,
    interval=False,
    idx=None,
    save_ext=None,
    style_override=None,
    use_legacy=None,
):
    """
    Plot regular and uncertainty explanations.

    Parameters
    ----------
    explanation : object
        The explanation object containing the details of the explanation.
    instance : array-like
        The instance for which the explanation is generated.
    predict : dict
        The prediction details including 'predict', 'low', and 'high'.
    feature_weights : dict or array-like
        The weights of the features.
    features_to_plot : list
        The list of features to plot.
    num_to_show : int
        The number of features to show in the plot.
    column_names : list
        The names of the columns.
    title : str
        The title of the plot.
    path : str
        The path to save the plot.
    show : bool
        Whether to show the plot.
    interval : bool, optional
        Whether to plot intervals.
    idx : int, optional
        The index for interval plotting.
    save_ext : list, optional
        The list of file extensions to save the plot.
    """
    explainer = None
    if use_legacy is None:
        try:
            explainer = explanation._get_explainer()
        except Exception:  # pragma: no cover - defensive
            explainer = getattr(explanation, "calibrated_explanations", None)
            if explainer is not None:
                explainer = getattr(explainer, "calibrated_explainer", None)
        if explainer is not None:
            chain = _resolve_plot_style_chain(explainer, style_override)
        else:
            chain = ("plot_spec.default", "legacy")
        selected_style = chain[0] if chain else "legacy"
        use_legacy = selected_style == "legacy"
    else:
        selected_style = None

    if use_legacy:
        legacy._plot_probabilistic(
            explanation,
            instance,
            predict,
            feature_weights,
            features_to_plot,
            num_to_show,
            column_names,
            title,
            path,
            show,
            interval,
            idx,
            save_ext,
        )
        return

    # If matplotlib is unavailable and we're not showing, perform a no-op to avoid failing
    if not show and plt is None:  # lightweight path for tests/CI without viz extra
        return
    # If we're not showing and not saving, perform a no-op to avoid requiring matplotlib
    if not show and (save_ext is None or len(save_ext) == 0):
        return

    __require_matplotlib()
    # config = __setup_plot_style(style_override) # Local variable `config` is assigned to but never used

    if save_ext is None:
        save_ext = ["svg", "pdf", "png"]
    if interval is True:
        assert idx is not None
    # Build a PlotSpec and render via matplotlib adapter to centralize logic
    # Build a PlotSpec and render via matplotlib adapter to centralize logic
    from .viz.builders import build_probabilistic_bars_spec
    from .viz.matplotlib_adapter import render as render_plotspec

    # Attempt to extract class labels for header annotation (neg/pos)
    class_labels = None
    try:
        class_labels = explanation.get_class_labels()
    except Exception:
        class_labels = None

    neg_label = None
    pos_label = None
    if class_labels is not None and len(class_labels) >= 2:
        # class_labels is indexed by class index; positive class corresponds to prediction["classes"]
        pos_idx = (
            int(explanation.prediction.get("classes", 1))
            if explanation.prediction is not None
            else 1
        )
        # fallback ordering: [neg, pos]
        # try to derive neg/pos by index
        try:
            pos_label = class_labels[pos_idx]
            # pick the other label as neg
            neg_idx = 0 if pos_idx != 0 else 1
            neg_label = class_labels[neg_idx]
        except Exception:
            neg_label, pos_label = None, None

    spec = build_probabilistic_bars_spec(
        title=title,
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=features_to_plot,
        column_names=column_names,
        rule_labels=column_names,
        instance=instance,
        y_minmax=getattr(explanation, "y_minmax", None),
        interval=interval,
        sort_by=None,
        ascending=False,
        neg_label=neg_label,
        pos_label=pos_label,
    )

    # Use the adapter's render function. For save behavior, construct a temporary
    # save_path if saving multiple extensions was requested by caller.
    save_path = None
    if save_ext is not None and len(save_ext) > 0 and path is not None and title is not None:
        # The matplotlib adapter accepts a single save_path; callers previously saved
        # multiple files by iterating. We'll mimic that behavior here below.
        save_path = None

    try:
        # Render to show (no-op if neither show nor save requested)
        render_plotspec(spec, show=show, save_path=save_path)

        # If requested, also save multiple extensions like the legacy function did
        if save_ext is not None and len(save_ext) > 0 and path is not None and title is not None:
            from .viz.matplotlib_adapter import render as _render

            for ext in save_ext:
                _render(spec, show=False, save_path=path + title + ext)
    except Exception as exc:  # pragma: no cover - fallback path
        warnings.warn(
            f"PlotSpec rendering failed with '{exc}'. Falling back to legacy plot.",
            stacklevel=2,
        )
        legacy._plot_probabilistic(
            explanation,
            instance,
            predict,
            feature_weights,
            features_to_plot,
            num_to_show,
            column_names,
            title,
            path,
            show,
            interval,
            idx,
            save_ext,
        )
    return


# pylint: disable=too-many-branches, too-many-statements, too-many-locals
def _plot_regression(
    explanation,
    instance,
    predict,
    feature_weights,
    features_to_plot,
    num_to_show,
    column_names,
    title,
    path,
    show,
    interval=False,
    idx=None,
    save_ext=None,
    style_override=None,
    use_legacy=None,
):
    """
    Plot regular and uncertainty explanations.

    Parameters
    ----------
    explanation : object
        The explanation object containing the details of the explanation.
    instance : array-like
        The instance for which the explanation is generated.
    predict : dict
        The prediction details including 'predict', 'low', and 'high'.
    feature_weights : dict or array-like
        The weights of the features.
    features_to_plot : list
        The list of features to plot.
    num_to_show : int
        The number of features to show in the plot.
    column_names : list
        The names of the columns.
    title : str
        The title of the plot.
    path : str
        The path to save the plot.
    show : bool
        Whether to show the plot.
    interval : bool, optional
        Whether to plot intervals.
    idx : int, optional
        The index for interval plotting.
    save_ext : list, optional
        The list of file extensions to save the plot.
    """
    explainer = None
    if use_legacy is None:
        try:
            explainer = explanation._get_explainer()
        except Exception:  # pragma: no cover - defensive
            explainer = getattr(explanation, "calibrated_explanations", None)
            if explainer is not None:
                explainer = getattr(explainer, "calibrated_explainer", None)
        if explainer is not None:
            chain = _resolve_plot_style_chain(explainer, style_override)
        else:
            chain = ("plot_spec.default", "legacy")
        selected_style = chain[0] if chain else "legacy"
        use_legacy = selected_style == "legacy"
    else:
        selected_style = None

    if use_legacy:
        legacy._plot_regression(
            explanation,
            instance,
            predict,
            feature_weights,
            features_to_plot,
            num_to_show,
            column_names,
            title,
            path,
            show,
            interval,
            idx,
            save_ext,
        )
        return

    # If matplotlib is unavailable and we're not showing, perform a no-op to avoid failing
    if not show and plt is None:  # lightweight path for tests/CI without viz extra
        return
    # If we're not showing and not saving, perform a no-op to avoid requiring matplotlib
    if not show and (save_ext is None or len(save_ext) == 0):
        return

    # Build PlotSpec via builder and render via adapter
    from .viz.builders import build_regression_bars_spec
    from .viz.matplotlib_adapter import render as render_plotspec

    spec = build_regression_bars_spec(
        title=title,
        predict=predict,
        feature_weights=feature_weights,
        features_to_plot=features_to_plot,
        column_names=column_names,
        rule_labels=column_names,
        instance=instance,
        y_minmax=getattr(explanation, "y_minmax", None),
        interval=interval,
        sort_by=None,
        ascending=False,
    )

    try:
        # Render once and then save multiple extensions if requested
        render_plotspec(spec, show=show, save_path=None)
        if save_ext is not None and len(save_ext) > 0 and path is not None and title is not None:
            for ext in save_ext:
                render_plotspec(spec, show=False, save_path=path + title + ext)
    except Exception as exc:  # pragma: no cover - fallback path
        warnings.warn(
            f"PlotSpec rendering failed with '{exc}'. Falling back to legacy plot.",
            stacklevel=2,
        )
        legacy._plot_regression(
            explanation,
            instance,
            predict,
            feature_weights,
            features_to_plot,
            num_to_show,
            column_names,
            title,
            path,
            show,
            interval,
            idx,
            save_ext,
        )
    return


# pylint: disable=duplicate-code
def _plot_triangular(
    explanation,
    proba,
    uncertainty,
    rule_proba,
    rule_uncertainty,
    num_to_show,
    title,
    path,
    show,
    save_ext=None,
    style_override=None,
    use_legacy=True,
):
    """
    Plot triangular explanations.

    Parameters
    ----------
    explanation : object
        The explanation object containing the details of the explanation.
    proba : array-like
        The probabilities of the predictions.
    uncertainty : array-like
        The uncertainties of the predictions.
    rule_proba : array-like
        The probabilities of the rules.
    rule_uncertainty : array-like
        The uncertainties of the rules.
    num_to_show : int
        The number of rules to show in the plot.
    title : str
        The title of the plot.
    path : str
        The path to save the plot.
    show : bool
        Whether to show the plot.
    save_ext : list, optional
        The list of file extensions to save the plot.
    """
    if use_legacy:
        legacy._plot_triangular(
            explanation,
            proba,
            uncertainty,
            rule_proba,
            rule_uncertainty,
            num_to_show,
            title,
            path,
            show,
            save_ext,
        )
        return

    # If matplotlib is unavailable and we're not showing, perform a no-op to avoid failing
    if not show and plt is None:  # lightweight path for tests/CI without viz extra
        return
    # If we're not showing and not saving, perform a no-op to avoid requiring matplotlib
    if not show and (save_ext is None or len(save_ext) == 0):
        return

    # Build triangular PlotSpec dict via the builder and delegate rendering to the adapter.
    # This ensures the adapter emits canonical primitives (triangle_background,
    # quiver/scatter, save_fig) matching ADR-016 and keeps a single rendering path.
    from .viz.builders import build_triangular_plotspec_dict
    from .viz.matplotlib_adapter import render as render_plotspec

    if save_ext is None:
        save_ext = ["svg", "pdf", "png"]

    spec = build_triangular_plotspec_dict(
        title=title,
        proba=proba,
        uncertainty=uncertainty,
        rule_proba=rule_proba,
        rule_uncertainty=rule_uncertainty,
        num_to_show=num_to_show,
        is_probabilistic=True,
    )

    # Let adapter perform the rendering and handle saving behavior.
    render_plotspec(spec, show=show, save_path=None)
    # If caller requested saving in multiple extensions, call adapter for each
    # extension so it can emit the expected save primitives (and actually save
    # if matplotlib is available).
    if save_ext is not None and len(save_ext) > 0 and path is not None and title is not None:
        for ext in save_ext:
            render_plotspec(
                {
                    **spec,
                    "plot_spec": {
                        **spec.get("plot_spec", {}),
                        "save_behavior": {"default_exts": [ext]},
                    },
                },
                show=False,
                save_path=path + title + ext,
            )
    return


# `__plot_triangular`
def __plot_proba_triangle():
    x = np.arange(0, 1, 0.01)
    plt.plot((x / (1 + x)), x, color="black")
    plt.plot(x, ((1 - x) / x), color="black")
    x = np.arange(0.5, 1, 0.005)
    plt.plot((0.5 + x - 0.5) / (1 + x - 0.5), x - 0.5, color="black")
    x = np.arange(0, 0.5, 0.005)
    plt.plot((x + 0.5 - x) / (1 + x), x, color="black")


# pylint: disable=too-many-arguments, too-many-locals, invalid-name, too-many-branches, too-many-statements
def _plot_alternative(
    explanation,
    instance,
    predict,
    feature_predict,
    features_to_plot,
    num_to_show,
    column_names,
    title,
    path,
    show,
    save_ext=None,
    style_override=None,
    use_legacy=None,
):
    """
    Plot alternative explanations.

    Parameters
    ----------
    explanation : object
        The explanation object containing the details of the explanation.
    instance : array-like
        The instance for which the explanation is generated.
    predict : dict
        The prediction details including 'predict', 'low', and 'high'.
    feature_predict : dict
        The predictions for the features.
    features_to_plot : list
        The list of features to plot.
    num_to_show : int
        The number of features to show in the plot.
    column_names : list
        The names of the columns.
    title : str
        The title of the plot.
    path : str
        The path to save the plot.
    show : bool
        Whether to show the plot.
    save_ext : list, optional
        The list of file extensions to save the plot.
    """
    explainer = None
    if use_legacy is None:
        try:
            explainer = explanation._get_explainer()
        except Exception:  # pragma: no cover - defensive
            explainer = getattr(explanation, "calibrated_explanations", None)
            if explainer is not None:
                explainer = getattr(explainer, "calibrated_explainer", None)
        if explainer is not None:
            chain = _resolve_plot_style_chain(explainer, style_override)
        else:
            chain = ("plot_spec.default", "legacy")
        selected_style = chain[0] if chain else "legacy"
        use_legacy = selected_style == "legacy"
    else:
        selected_style = None

    if use_legacy:
        legacy._plot_alternative(
            explanation,
            instance,
            predict,
            feature_predict,
            features_to_plot,
            num_to_show,
            column_names,
            title,
            path,
            show,
            save_ext,
        )
        return

    # If matplotlib is unavailable and we're not showing, perform a no-op to avoid failing
    if not show and plt is None:  # lightweight path for tests/CI without viz extra
        return
    # If we're not showing and not saving, perform a no-op to avoid requiring matplotlib
    if not show and (save_ext is None or len(save_ext) == 0):
        return

    __require_matplotlib()

    if save_ext is None:
        save_ext = ["svg", "pdf", "png"]

    features_to_plot = list(features_to_plot or [])

    # Ensure y_minmax is finite when available
    y_minmax = getattr(explanation, "y_minmax", None)
    normalised_y_minmax: tuple[float, float] | None = None
    if isinstance(y_minmax, Sequence) and len(y_minmax) >= 2:
        try:
            y0, y1 = float(y_minmax[0]), float(y_minmax[1])
        except Exception:
            normalised_y_minmax = None
        else:
            if np.isfinite(y0) and np.isfinite(y1):
                normalised_y_minmax = (y0, y1)

    def _safe_float(value: Any) -> float | None:
        try:
            val = float(value)
        except Exception:
            return None
        if not np.isfinite(val):
            return None
        return val

    predict_payload: Dict[str, Any] = {}
    if isinstance(predict, Mapping):
        predict_payload.update(predict)

    base_pred = _safe_float(predict_payload.get("predict"))
    if base_pred is None:
        base_pred = 0.0

    low_default = base_pred
    high_default = base_pred
    if normalised_y_minmax is not None:
        low_default = normalised_y_minmax[0]
        high_default = normalised_y_minmax[1]

    pred_low = _safe_float(predict_payload.get("low"))
    pred_high = _safe_float(predict_payload.get("high"))

    predict_payload["predict"] = base_pred
    predict_payload["low"] = pred_low if pred_low is not None else low_default
    predict_payload["high"] = pred_high if pred_high is not None else high_default

    feature_payload: Any = feature_predict
    if isinstance(feature_predict, Mapping):
        fallback_map = {
            "predict": base_pred,
            "low": predict_payload["low"],
            "high": predict_payload["high"],
        }
        sanitised: Dict[str, List[float]] = {}
        for key, values in feature_predict.items():
            if isinstance(values, np.ndarray):
                seq = values.tolist()
            elif isinstance(values, Sequence):
                seq = list(values)
            else:
                seq = [values]
            fallback = fallback_map.get(key, base_pred)
            sanitised[key] = [
                val if (val := _safe_float(item)) is not None else float(fallback)
                for item in seq
            ]
        feature_payload = sanitised
    else:
        values_seq: Sequence[Any]
        if isinstance(feature_predict, np.ndarray):
            values_seq = feature_predict.tolist()
        elif isinstance(feature_predict, Sequence):
            values_seq = list(feature_predict)
        else:
            values_seq = [feature_predict]
        feature_payload = [
            val if (val := _safe_float(item)) is not None else float(base_pred)
            for item in values_seq
        ]

    feature_count = 0
    if isinstance(feature_payload, Mapping):
        for values in feature_payload.values():
            if isinstance(values, Sequence):
                try:
                    feature_count = max(feature_count, len(values))
                except TypeError:  # pragma: no cover - defensive
                    continue
                if feature_count:
                    break
    elif isinstance(feature_payload, Sequence):
        try:
            feature_count = len(feature_payload)
        except TypeError:  # pragma: no cover - defensive
            feature_count = 0

    normalised_indices: list[int] = []
    for idx in features_to_plot:
        try:
            value = int(idx)
        except Exception:
            continue
        if value < 0:
            continue
        normalised_indices.append(value)
    features_to_plot = normalised_indices

    if not features_to_plot and feature_count:
        features_to_plot = list(range(feature_count))

    if len(features_to_plot) == 0 or num_to_show <= 0:
        return

    if column_names is None and feature_count:
        column_names = [str(idx) for idx in range(feature_count)]

    interval = (
        isinstance(feature_payload, Mapping)
        and "low" in feature_payload
        and "high" in feature_payload
    )

    try:
        mode = ""
        try:
            mode = str(explanation.get_mode() or "")
        except Exception:
            mode = ""
        is_regression = "regression" in mode

        neg_label = None
        pos_label = None
        if not is_regression:
            class_labels = None
            try:
                class_labels = explanation.get_class_labels()
            except Exception:
                class_labels = None
            if class_labels is not None and len(class_labels) >= 2:
                try:
                    prediction = getattr(explanation, "prediction", {})
                    cls_idx = 1
                    if isinstance(prediction, Mapping):
                        cls_idx = int(prediction.get("classes", 1))
                    pos_label = class_labels[cls_idx]
                    neg_idx = 0 if cls_idx != 0 else 1
                    neg_label = class_labels[neg_idx]
                except Exception:
                    neg_label = None
                    pos_label = None

        from .viz.builders import (
            build_alternative_probabilistic_spec,
            build_alternative_regression_spec,
        )
        from .viz.matplotlib_adapter import render as render_plotspec

        builder_kwargs = {
            "title": title,
            "predict": predict_payload,
            "feature_weights": feature_payload,
            "features_to_plot": features_to_plot,
            "column_names": column_names,
            "rule_labels": column_names,
            "instance": instance,
            "y_minmax": normalised_y_minmax,
            "interval": interval,
            "sort_by": None,
            "ascending": False,
            "legacy_solid_behavior": True,
        }
        if is_regression:
            spec = build_alternative_regression_spec(**builder_kwargs)
        else:
            builder_kwargs.update({
                "neg_label": neg_label,
                "pos_label": pos_label,
            })
            spec = build_alternative_probabilistic_spec(**builder_kwargs)

        try:
            render_plotspec(spec, show=show, save_path=None)
            if save_ext and path is not None and title is not None:
                for ext in save_ext:
                    render_plotspec(spec, show=False, save_path=path + title + ext)
        except Exception as exc:  # pragma: no cover - fallback path
            warnings.warn(
                f"PlotSpec rendering failed with '{exc}'. Falling back to legacy plot.",
                stacklevel=2,
            )
            legacy._plot_alternative(
                explanation,
                instance,
                predict,
                feature_predict,
                features_to_plot,
                num_to_show,
                column_names,
                title,
                path,
                show,
                save_ext,
            )
    except Exception as exc:  # pragma: no cover - fallback path
        warnings.warn(
            f"PlotSpec rendering failed with '{exc}'. Falling back to legacy plot.",
            stacklevel=2,
        )
        legacy._plot_alternative(
            explanation,
            instance,
            predict,
            feature_predict,
            features_to_plot,
            num_to_show,
            column_names,
            title,
            path,
            show,
            save_ext,
        )
        return


# pylint: disable=duplicate-code, too-many-branches, too-many-statements, too-many-locals
def _plot_global(explainer, x, y=None, threshold=None, **kwargs):
    """
    Generate a global explanation plot for the given test data.

    This plot is based on the
    probability distribution and the uncertainty quantification intervals.
    The plot is only available for calibrated probabilistic learners (both classification and
    thresholded regression).

    Parameters
    ----------
    explainer : object
        The explainer object used to generate the explanations.
    x : array-like
        The input data for which predictions are to be made.
    y : array-like, optional
        The true labels of the test data.
    threshold : float, int, optional
        The threshold value used with regression to get probability of being below the threshold.
    **kwargs : dict
        Additional keyword arguments.
    """
    show = kwargs.get("show", True)
    bins = kwargs.get("bins")
    # style_override = kwargs.get("style_override")
    use_legacy = kwargs.get("use_legacy", True)
    if use_legacy:
        legacy._plot_global(explainer, x, y, threshold, **kwargs)
        return

    style = kwargs.get("style")
    path = kwargs.get("path")
    save_ext_value = kwargs.get("save_ext")
    if isinstance(save_ext_value, (list, tuple)):
        save_ext_value = tuple(save_ext_value)

    # Gather model outputs in the same way legacy code did
    is_regularized = True
    if "predict_proba" not in dir(explainer.learner) and threshold is None:
        predict, (low, high) = explainer.predict(x, uq_interval=True, bins=bins)
        proba = None
        is_regularized = False
    else:
        proba, (low, high) = explainer.predict_proba(
            x, uq_interval=True, threshold=threshold, bins=bins
        )
        predict = None

    uncertainty = (
        (np.array(high) - np.array(low)) if (low is not None and high is not None) else None
    )

    payload = {
        "proba": proba,
        "predict": predict,
        "low": low,
        "high": high,
        "uncertainty": uncertainty,
        "y": (list(y) if y is not None else None),
        "is_regularized": is_regularized,
        "threshold": threshold,
    }

    from .plugins import PlotRenderContext
    from .plugins.registry import (
        ensure_builtin_plugins,
        find_plot_plugin,
        find_plot_plugin_trusted,
    )

    ensure_builtin_plugins()

    chain = _resolve_plot_style_chain(explainer, style)
    errors: List[str] = []

    for identifier in chain:
        plugin = find_plot_plugin_trusted(identifier)
        if plugin is None:
            plugin = find_plot_plugin(identifier)
        if plugin is None:
            errors.append(f"{identifier}: not registered")
            continue
        if not hasattr(plugin, "build") or not hasattr(plugin, "render"):
            errors.append(f"{identifier}: missing build/render implementation")
            continue

        if identifier == "legacy" or getattr(plugin, "plugin_meta", {}).get("style") == "legacy":
            __require_matplotlib()
        elif show and plt is None:  # pragma: no cover - optional dep path
            errors.append(f"{identifier}: matplotlib backend unavailable")
            continue

        context = PlotRenderContext(
            explanation=getattr(explainer, "latest_explanation", None),
            instance_metadata=MappingProxyType({"type": "global"}),
            style=identifier,
            intent=MappingProxyType(
                {
                    "type": "global",
                    "explainer_mode": getattr(explainer, "_last_explanation_mode", None),
                }
            ),
            show=show,
            path=path,
            save_ext=save_ext_value,
            options=MappingProxyType({"payload": payload}),
        )
        try:
            artifact = plugin.build(context)
            result = plugin.render(artifact, context=context)
        except Exception as exc:  # pragma: no cover - plugin failures
            errors.append(f"{identifier}: {exc}")
            continue
        return result

    from .core.exceptions import ConfigurationError as _PlotConfigurationError

    raise _PlotConfigurationError(
        "Unable to resolve plot plugin for global explanations; "
        + "tried: "
        + ", ".join(chain)
        + ("; errors: " + "; ".join(errors) if errors else "")
    )


def _plot_proba_triangle():
    __require_matplotlib()
    fig = plt.figure()
    x = np.arange(0, 1, 0.01)
    plt.plot((x / (1 + x)), x, color="black")
    plt.plot(x, ((1 - x) / x), color="black")
    x = np.arange(0.5, 1, 0.005)
    plt.plot((0.5 + x - 0.5) / (1 + x - 0.5), x - 0.5, color="black")
    x = np.arange(0, 0.5, 0.005)
    plt.plot((x + 0.5 - x) / (1 + x), x, color="black")
    return fig


# pylint: disable=invalid-name
