"""Plotting utilities used by explanation objects and the wrap-first workflow.

The public README surface still calls ``CalibratedExplanation.plot``; this
module houses the implementation while ``calibrated_explanations._plots`` remains
as a deprecated shim for backwards compatibility.
"""

from __future__ import annotations

import configparser
import contextlib
import logging
import os
import sys
import warnings
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


def __getattr__(name: str) -> Any:
    """Lazily load legacy plotting module."""
    if name == "legacy":
        from .legacy import plotting as legacy

        globals()["legacy"] = legacy
        return legacy
    raise AttributeError(f"module {__name__} has no attribute {name}")


def derive_threshold_labels(threshold: Any | None) -> tuple[str, str]:
    """Return positive/negative labels summarising a regression threshold."""
    try:
        if (
            isinstance(threshold, Sequence)
            and not isinstance(threshold, (str, bytes))
            and len(threshold) >= 2
        ):
            lo = float(threshold[0])
            hi = float(threshold[1])
            return (f"{lo:.2f} <= Y < {hi:.2f}", "Outside interval")
    except Exception:  # adr002_allow
        logging.getLogger(__name__).debug(
            "Failed to parse threshold as interval: %s", sys.exc_info()[1]
        )
    try:
        value = float(threshold)
    except:  # noqa: E722
        if not isinstance(sys.exc_info()[1], Exception):
            raise
        return ("Target within threshold", "Outside threshold")
    return (f"Y < {value:.2f}", f"Y >= {value:.2f}")


try:
    import tomllib as _plot_tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    try:  # pragma: no cover - optional dependency path
        import tomli as _plot_tomllib  # type: ignore[assignment]
    except ModuleNotFoundError:  # pragma: no cover - tomllib unavailable
        _plot_tomllib = None  # type: ignore[assignment]

_MATPLOTLIB_IMPORT_ERROR = None
mcolors = None
plt = None

try:
    import matplotlib  # noqa: F401
except (ImportError, RuntimeError) as e:
    _MATPLOTLIB_IMPORT_ERROR = e


def __require_matplotlib() -> None:
    """Ensure matplotlib is available before using plotting functions."""
    global mcolors, plt, _MATPLOTLIB_IMPORT_ERROR
    from .utils.exceptions import ConfigurationError

    if plt is None or mcolors is None:
        if _MATPLOTLIB_IMPORT_ERROR is None:
            try:
                import matplotlib.artist  # noqa: F401
                import matplotlib.axes  # noqa: F401
                import matplotlib.colors as mcolors_local

                # Preload lazy-loaded submodules to avoid AttributeError when coverage runs
                import matplotlib.image  # noqa: F401
                import matplotlib.pyplot as plt_local

                mcolors = mcolors_local
                plt = plt_local
            except Exception:  # adr002_allow
                _MATPLOTLIB_IMPORT_ERROR = sys.exc_info()[1]

        if plt is None or mcolors is None:
            msg = (
                "Plotting requires matplotlib. Install the 'viz' extra: "
                "pip install calibrated_explanations[viz]"
            )
            if _MATPLOTLIB_IMPORT_ERROR is not None:
                msg += f"\nOriginal import error: {_MATPLOTLIB_IMPORT_ERROR}"
            raise ConfigurationError(
                msg,
                details={
                    "requirement": "matplotlib",
                    "extra": "viz",
                    "reason": "import_failed" if _MATPLOTLIB_IMPORT_ERROR else "not_installed",
                    "error": str(_MATPLOTLIB_IMPORT_ERROR) if _MATPLOTLIB_IMPORT_ERROR else None,
                },
            )


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
    except Exception:  # adr002_allow
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


def split_csv(value: Any) -> Sequence[str]:
    """Normalize comma-separated labels into a tuple."""
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence):
        return tuple(str(item).strip() for item in value if isinstance(item, str) and item.strip())
    return ()


def _format_save_path(base_path: Any, filename: str) -> str:
    """Return a string path while preserving caller formatting when possible."""
    if isinstance(base_path, (Path, PurePath, os.PathLike)):
        return str(Path(base_path) / filename)
    if isinstance(base_path, str):
        if base_path.strip() == "":
            return filename
        if base_path.endswith("/") and "\\" not in base_path[:-1]:
            return f"{base_path}{filename}"
        if base_path.endswith("\\") and "/" not in base_path[:-1]:
            return f"{base_path}{filename}"
        return str(Path(base_path) / filename)
    return str(Path(str(base_path)) / filename)


def _resolve_explainer_from_explanation(explanation: Any) -> Any:
    """Best-effort resolver for explanation -> explainer references."""
    getter = getattr(explanation, "get_explainer", None)
    if callable(getter):
        return getter()
    getter = getattr(explanation, "_get_explainer", None)
    if callable(getter):
        return getter()
    container = getattr(explanation, "calibrated_explanations", None)
    if container is not None:
        getter = getattr(container, "get_explainer", None)
        if callable(getter):
            return getter()
        getter = getattr(container, "_get_explainer", None)
        if callable(getter):
            return getter()
        return getattr(container, "calibrated_explainer", None)
    return None


def _resolve_plot_style_chain(explainer, explicit_style: str | None) -> Sequence[str]:
    """Determine the ordered style fallback chain for plot builders/renderers."""
    chain: List[str] = []
    if isinstance(explicit_style, str) and explicit_style:
        chain.append(explicit_style)

    env_style = os.environ.get("CE_PLOT_STYLE")
    if env_style:
        chain.append(env_style.strip())
    chain.extend(split_csv(os.environ.get("CE_PLOT_STYLE_FALLBACKS")))

    py_settings = _read_plot_pyproject()
    py_style = py_settings.get("style")
    if isinstance(py_style, str) and py_style:
        chain.append(py_style)
    chain.extend(split_csv(py_settings.get("fallbacks")))

    # If no explicit style provided, prepend the default style from registry
    if not chain:
        from .plugins.registry import list_plot_style_descriptors

        for descriptor in list_plot_style_descriptors():
            if descriptor.metadata.get("is_default", False):
                chain.append(descriptor.identifier)
                break

    mode = getattr(explainer, "last_explanation_mode", None)
    plot_fallbacks = getattr(explainer, "plot_plugin_fallbacks", {})
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


def resolve_plot_style_chain(explainer, explicit_style: str | None = None) -> Sequence[str]:
    """Determine the ordered style fallback chain for plot builders/renderers.

    Parameters
    ----------
    explainer : CalibratedExplainer
        The explainer instance.
    explicit_style : str, optional
        An explicit style identifier to use as the primary style.

    Returns
    -------
    Sequence[str]
        The ordered list of style identifiers to attempt.
    """
    return _resolve_plot_style_chain(explainer, explicit_style)


# pylint: disable=unknown-option-value
# pylint: disable=too-many-arguments, too-many-statements, too-many-branches, too-many-locals, too-many-positional-arguments, fixme


def _plot_config_path() -> Path:
    """Return the resolved path to the plot configuration file."""
    return Path(__file__).resolve().parent / "utils" / "configurations" / "plot_config.ini"


def load_plot_config():
    """Load plot configuration from INI file."""
    config = configparser.ConfigParser()
    config_path = _plot_config_path()

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
    config.read(str(config_path), encoding="utf-8")
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

    # Write updated config to file, normalising trailing newlines so
    # repeated writes don't accumulate blank lines.
    from io import StringIO

    config_path = _plot_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    buf = StringIO()
    config.write(buf)
    text = buf.getvalue()
    # Normalise line endings and ensure exactly one trailing newline
    text = text.replace("\r\n", "\n").rstrip("\n") + "\n"
    with config_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


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


def plot_probabilistic(
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
    style_override : str, optional
        The style to use for plotting.
    use_legacy : bool, optional
        Whether to use the legacy plotting system.
    """
    explainer = _resolve_explainer_from_explanation(explanation)
    if use_legacy is None:
        if explainer is not None:
            chain = _resolve_plot_style_chain(explainer, style_override)
        else:
            chain = ("plot_spec.default", "legacy")
        selected_style = chain[0] if chain else "legacy"
        use_legacy = selected_style == "legacy"
    else:
        selected_style = None

    predict_payload = dict(predict or {})

    if use_legacy:
        from .legacy import plotting as legacy

        legacy._plot_probabilistic(
            explanation,
            instance,
            predict_payload,
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
    __require_matplotlib()
    if save_ext is None:
        save_ext = ["svg", "pdf", "png"]
    if not show and plt is None:  # lightweight path for tests/CI without viz extra
        return
    # If we're not showing and not saving, perform a no-op to avoid requiring matplotlib
    if not show and (save_ext is None or len(save_ext) == 0):
        return
    if interval is True:
        assert idx is not None

    def _finite_or(value: Any, fallback: float) -> float:
        try:
            val = float(value)
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            return fallback
        if not np.isfinite(val):
            return fallback
        return val

    y_minmax = getattr(explanation, "y_minmax", None)
    base_pred = _finite_or(predict_payload.get("predict"), 0.0)
    predict_payload["predict"] = base_pred

    low_fallback = base_pred
    high_fallback = base_pred
    if isinstance(y_minmax, Sequence) and len(y_minmax) >= 2:
        try:
            low_fallback = float(y_minmax[0])
            high_fallback = float(y_minmax[1])
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            low_fallback = base_pred
            high_fallback = base_pred

    predict_payload["low"] = _finite_or(predict_payload.get("low"), low_fallback)
    predict_payload["high"] = _finite_or(predict_payload.get("high"), high_fallback)
    # Build a PlotSpec and render via matplotlib adapter to centralize logic
    # Build a PlotSpec and render via matplotlib adapter to centralize logic
    from .viz.builders import build_probabilistic_bars_spec
    from .viz.matplotlib_adapter import render as render_plotspec

    # Attempt to extract class labels for header annotation and captioning
    class_labels = None
    with contextlib.suppress(Exception):
        class_labels = explanation.get_class_labels()

    neg_caption: str | None = None
    pos_caption: str | None = None
    neg_label = None
    pos_label = None

    prediction_classes = None
    if getattr(explanation, "prediction", None) is not None:
        prediction_classes = explanation.prediction.get("classes")

    def _format_class(value: Any) -> str:
        try:
            return str(value)
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            return ""

    is_thresholded = False
    with contextlib.suppress(Exception):
        is_thresholded = explanation.is_thresholded()

    if is_thresholded:
        threshold = getattr(explanation, "y_threshold", None)
        try:
            if np.isscalar(threshold):  # type: ignore[arg-type]
                thr_val = float(threshold)  # type: ignore[arg-type]
                neg_caption = f"P(y>{thr_val:.2f})"
                pos_caption = f"P(y<={thr_val:.2f})"
            elif isinstance(threshold, Sequence) and len(threshold) >= 2:
                lo = float(threshold[0])
                hi = float(threshold[1])
                neg_caption = f"y_hat <= {lo:.3f} || y_hat > {hi:.3f}"
                pos_caption = f"{lo:.3f} < y_hat <= {hi:.3f}"
            else:
                neg_caption = "P(y>threshold)"
                pos_caption = "P(y<=threshold)"
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            neg_caption = "P(y>threshold)"
            pos_caption = "P(y<=threshold)"
    else:
        is_multiclass = False
        if explainer is not None:
            with contextlib.suppress(Exception):
                is_multiclass = bool(explainer.is_multiclass())
        if not is_multiclass:
            is_multiclass = bool(getattr(explanation, "is_multiclass", False))

        if class_labels is None:
            if is_multiclass:
                label_val = _format_class(prediction_classes)
                neg_caption = f"P(y!={label_val})"
                pos_caption = f"P(y={label_val})"
            else:
                neg_caption = "P(y=0)"
                pos_caption = "P(y=1)"
        elif bool(getattr(explanation, "is_multiclass", False)) or is_multiclass:
            pred_idx = 0
            if prediction_classes is not None:
                try:
                    pred_idx = int(prediction_classes)
                except:  # noqa: E722
                    if not isinstance(sys.exc_info()[1], Exception):
                        raise
                    pred_idx = 0
            try:
                label_val = class_labels[pred_idx]
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                label_val = prediction_classes
            label_str = _format_class(label_val)
            neg_caption = f"P(y!={label_str})"
            pos_caption = f"P(y={label_str})"
        else:
            try:
                neg_label = class_labels[0]
                pos_label = class_labels[1]
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                neg_label = class_labels[0] if class_labels else None
                pos_label = class_labels[1] if class_labels and len(class_labels) > 1 else None
            if neg_label is not None:
                neg_caption = f"P(y={_format_class(neg_label)})"
            if pos_label is not None:
                pos_caption = f"P(y={_format_class(pos_label)})"

    spec = build_probabilistic_bars_spec(
        title=title,
        predict=predict_payload,
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
        neg_caption=neg_caption,
        pos_caption=pos_caption,
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
                _render(spec, show=False, save_path=_format_save_path(path, title + ext))
    except:  # noqa: E722
        if not isinstance(sys.exc_info()[1], Exception):
            raise
        warnings.warn(
            f"PlotSpec rendering failed with '{sys.exc_info()[1]}'. Falling back to legacy plot.",
            stacklevel=2,
        )
        from .legacy import plotting as legacy_module

        legacy_module._plot_probabilistic(
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
def plot_regression(
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
    if interval and hasattr(explanation, "is_one_sided"):
        try:
            if explanation.is_one_sided():
                raise Warning("Interval plot is not supported for one-sided explanations.")
        except Warning:
            raise
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            # If the guard fails unexpectedly, defer to legacy parity by proceeding.
            logging.getLogger(__name__).debug("Guard check failed: %s", sys.exc_info()[1])

    explainer = _resolve_explainer_from_explanation(explanation)
    if use_legacy is None:
        if explainer is not None:
            chain = _resolve_plot_style_chain(explainer, style_override)
        else:
            chain = ("plot_spec.default", "legacy")
        selected_style = chain[0] if chain else "legacy"
        use_legacy = selected_style == "legacy"
    else:
        selected_style = None

    if use_legacy:
        from .legacy import plotting as legacy

        legacy.plot_regression(
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
    if save_ext is None:
        save_ext = ["svg", "pdf", "png"]
    from .viz.builders import build_regression_bars_spec
    from .viz.matplotlib_adapter import render as render_plotspec

    confidence = None
    try:
        calibrated = getattr(explanation, "calibrated_explanations", None)
        if calibrated is not None:
            confidence = calibrated.get_confidence()
    except:  # noqa: E722
        if not isinstance(sys.exc_info()[1], Exception):
            raise
        confidence = None

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
        confidence=confidence,
        sort_by=None,
        ascending=False,
    )

    try:
        # Render once and then save multiple extensions if requested
        render_plotspec(spec, show=show, save_path=None)
        if save_ext is not None and len(save_ext) > 0 and path is not None and title is not None:
            for ext in save_ext:
                render_plotspec(spec, show=False, save_path=_format_save_path(path, title + ext))
    except:  # noqa: E722
        if not isinstance(sys.exc_info()[1], Exception):
            raise
        warnings.warn(
            f"PlotSpec rendering failed with '{sys.exc_info()[1]}'. Falling back to legacy plot.",
            stacklevel=2,
        )
        from .legacy import plotting as legacy_module

        legacy_module.plot_regression(
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
def plot_triangular(
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
        from .legacy import plotting as legacy

        legacy.plot_triangular(
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
                save_path=_format_save_path(path, title + ext),
            )
    return


# `__plot_triangular`
def __plot_proba_triangle():
    """Render the static probability triangle overlay."""
    x = np.arange(0, 1, 0.01)
    plt.plot((x / (1 + x)), x, color="black")
    plt.plot(x, ((1 - x) / x), color="black")
    x = np.arange(0.5, 1, 0.005)
    plt.plot((0.5 + x - 0.5) / (1 + x - 0.5), x - 0.5, color="black")
    x = np.arange(0, 0.5, 0.005)
    plt.plot((x + 0.5 - x) / (1 + x), x, color="black")


# pylint: disable=too-many-arguments, too-many-locals, invalid-name, too-many-branches, too-many-statements
def plot_alternative(
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
    explainer = _resolve_explainer_from_explanation(explanation)
    if use_legacy is None:
        if explainer is not None:
            chain = _resolve_plot_style_chain(explainer, style_override)
        else:
            chain = ("plot_spec.default", "legacy")
        selected_style = chain[0] if chain else "legacy"
        use_legacy = selected_style == "legacy"
    else:
        selected_style = None

    if use_legacy:
        from .legacy import plotting as legacy

        legacy.plot_alternative(
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
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            normalised_y_minmax = None
        else:
            if np.isfinite(y0) and np.isfinite(y1):
                normalised_y_minmax = (y0, y1)

    def _safe_float(value: Any) -> float | None:
        try:
            val = float(value)
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
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
                val if (val := _safe_float(item)) is not None else float(fallback) for item in seq
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
                except:  # noqa: E722
                    if not isinstance(sys.exc_info()[1], TypeError):
                        raise
                    continue
                if feature_count:
                    break
    elif isinstance(feature_payload, Sequence):
        try:
            feature_count = len(feature_payload)
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], TypeError):
                raise
            feature_count = 0

    normalised_indices: list[int] = []
    for idx in features_to_plot:
        try:
            value = int(idx)
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            logging.getLogger(__name__).debug(
                "Failed to convert feature index to int: %s", sys.exc_info()[1]
            )
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
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            mode = ""
        is_regression = "regression" in mode

        neg_label = None
        pos_label = None
        x_axis_label = None
        xlim_override: tuple[float, float] | None = None
        xticks_override: Sequence[float] | None = None
        is_thresholded = False
        if not is_regression:
            class_labels = None
            with contextlib.suppress(Exception):
                class_labels = explanation.get_class_labels()
            if class_labels is not None and len(class_labels) >= 2:
                try:
                    prediction = getattr(explanation, "prediction", {})
                    cls_idx = 1
                    if isinstance(prediction, Mapping):
                        cls_idx = int(prediction.get("classes", 1))
                    pos_label = class_labels[cls_idx]
                    neg_idx = 0 if cls_idx != 0 else 1
                    neg_label = class_labels[neg_idx]
                except:  # noqa: E722
                    if not isinstance(sys.exc_info()[1], Exception):
                        raise
                    neg_label = None
                    pos_label = None
            # Legacy plots fixed the probability axis to [0,1] with evenly spaced ticks
            xlim_override = (0.0, 1.0)
            xticks_override = [float(x) for x in np.linspace(0.0, 1.0, 11)]
            with contextlib.suppress(Exception):
                is_thresholded = bool(explanation.is_thresholded())
            if is_thresholded:
                threshold_value = getattr(explanation, "y_threshold", None)
                if np.isscalar(threshold_value):
                    try:
                        x_axis_label = (
                            f"Probability of target being below {float(threshold_value):.2f}"
                        )
                    except:  # noqa: E722
                        if not isinstance(sys.exc_info()[1], Exception):
                            raise
                        x_axis_label = "Probability"
                elif isinstance(threshold_value, tuple) and len(threshold_value) >= 2:
                    try:
                        x_axis_label = (
                            "Probability of target being between "
                            f"{float(threshold_value[0]):.3f} and {float(threshold_value[1]):.3f}"
                        )
                    except:  # noqa: E722
                        if not isinstance(sys.exc_info()[1], Exception):
                            raise
                        x_axis_label = "Probability"
                else:
                    try:
                        x_axis_label = (
                            f"Probability of target being below {float(threshold_value):.2f}"
                        )
                    except:  # noqa: E722
                        if not isinstance(sys.exc_info()[1], Exception):
                            raise
                        x_axis_label = "Probability"
            else:
                predicted_idx = None
                try:
                    prediction = getattr(explanation, "prediction", None)
                    if isinstance(prediction, Mapping):
                        predicted_idx = prediction.get("classes")
                except:  # noqa: E722
                    if not isinstance(sys.exc_info()[1], Exception):
                        raise
                    predicted_idx = None
                idx_value = None
                if predicted_idx is not None:
                    with contextlib.suppress(Exception):
                        idx_value = int(predicted_idx)
                is_multi = False
                if explainer is not None:
                    with contextlib.suppress(Exception):
                        is_multi = bool(explainer.is_multiclass())
                if class_labels is None:
                    if is_multi:
                        x_axis_label = f"Probability for class '{predicted_idx}'"
                    else:
                        x_axis_label = "Probability for the positive class"
                else:
                    if is_multi and idx_value is not None and 0 <= idx_value < len(class_labels):
                        x_axis_label = f"Probability for class '{class_labels[idx_value]}'"
                    elif not is_multi and len(class_labels) > 1:
                        x_axis_label = f"Probability for class '{class_labels[1]}'"
                    else:
                        x_axis_label = "Probability"
                if x_axis_label is None:
                    x_axis_label = "Probability"
        else:
            with contextlib.suppress(Exception):
                is_thresholded = bool(explanation.is_thresholded())
            if normalised_y_minmax is not None:
                try:
                    xlim_override = (
                        float(normalised_y_minmax[0]),
                        float(normalised_y_minmax[1]),
                    )
                except:  # noqa: E722
                    if not isinstance(sys.exc_info()[1], Exception):
                        raise
                    xlim_override = None
            if xlim_override is None:
                try:
                    xlim_override = (
                        float(predict_payload.get("low", 0.0)),
                        float(predict_payload.get("high", 0.0)),
                    )
                except:  # noqa: E722
                    if not isinstance(sys.exc_info()[1], Exception):
                        raise
                    xlim_override = None
            confidence = None
            try:
                calibrated = getattr(explanation, "calibrated_explanations", None)
                if calibrated is not None:
                    confidence = calibrated.get_confidence()
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                confidence = None
            if is_thresholded:
                xlim_override = (0.0, 1.0)
                xticks_override = [float(x) for x in np.linspace(0.0, 1.0, 11)]
                threshold_value = getattr(explanation, "y_threshold", None)
                if np.isscalar(threshold_value):
                    try:
                        x_axis_label = (
                            f"Probability of target being below {float(threshold_value):.2f}"
                        )
                    except:  # noqa: E722
                        if not isinstance(sys.exc_info()[1], Exception):
                            raise
                        x_axis_label = "Probability"
                elif isinstance(threshold_value, tuple) and len(threshold_value) >= 2:
                    try:
                        x_axis_label = (
                            "Probability of target being between "
                            f"{float(threshold_value[0]):.3f} and {float(threshold_value[1]):.3f}"
                        )
                    except:  # noqa: E722
                        if not isinstance(sys.exc_info()[1], Exception):
                            raise
                        x_axis_label = "Probability"
                else:
                    try:
                        x_axis_label = (
                            f"Probability of target being below {float(threshold_value):.2f}"
                        )
                    except:  # noqa: E722
                        if not isinstance(sys.exc_info()[1], Exception):
                            raise
                        x_axis_label = "Probability"
            else:
                if confidence is not None:
                    x_axis_label = f"Prediction interval with {confidence}% confidence"
                else:
                    x_axis_label = "Prediction interval"

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
            if is_thresholded:
                threshold_value = getattr(explanation, "y_threshold", None)
                pos_label, neg_label = derive_threshold_labels(threshold_value)
                classification_kwargs = builder_kwargs.copy()
                classification_kwargs["neg_label"] = neg_label
                classification_kwargs["pos_label"] = pos_label
                classification_kwargs["xlabel"] = x_axis_label or "Probability"
                classification_kwargs["xlim"] = (0.0, 1.0)
                classification_kwargs["xticks"] = [float(x) for x in np.linspace(0.0, 1.0, 11)]
                classification_kwargs["y_minmax"] = None
                spec = build_alternative_probabilistic_spec(**classification_kwargs)
            else:
                builder_kwargs.update(
                    {
                        "xlabel": x_axis_label,
                        "xlim": xlim_override,
                        "xticks": xticks_override,
                    }
                )
                spec = build_alternative_regression_spec(**builder_kwargs)
        else:
            builder_kwargs.update(
                {
                    "neg_label": neg_label,
                    "pos_label": pos_label,
                    "xlabel": x_axis_label,
                    "xlim": xlim_override,
                    "xticks": xticks_override,
                }
            )
            spec = build_alternative_probabilistic_spec(**builder_kwargs)

        try:
            render_plotspec(spec, show=show, save_path=None)
            if save_ext and path is not None and title is not None:
                for ext in save_ext:
                    render_plotspec(
                        spec, show=False, save_path=_format_save_path(path, title + ext)
                    )
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            warnings.warn(
                f"PlotSpec rendering failed with '{sys.exc_info()[1]}'. Falling back to legacy plot.",
                stacklevel=2,
            )
            from .legacy import plotting as legacy

            legacy.plot_alternative(
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
    except:  # noqa: E722
        if not isinstance(sys.exc_info()[1], Exception):
            raise
        warnings.warn(
            f"PlotSpec rendering failed with '{sys.exc_info()[1]}'. Falling back to legacy plot.",
            stacklevel=2,
        )
        from .legacy import plotting as legacy

        legacy.plot_alternative(
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
def plot_global(explainer, x, y=None, threshold=None, **kwargs):
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
        from .legacy import plotting as legacy

        legacy.plot_global(explainer, x, y, threshold, **kwargs)
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

    from .plugins import (
        PlotRenderContext,
        ensure_builtin_plugins,
        find_plot_plugin,
        find_plot_plugin_trusted,
    )
    from .plugins.registry import find_plot_renderer

    ensure_builtin_plugins()

    chain = _resolve_plot_style_chain(explainer, style)
    # Resolve renderer override from kwargs/env/pyproject
    renderer_override = kwargs.get("renderer")
    if not renderer_override:
        renderer_override = os.environ.get("CE_PLOT_RENDERER")
    if not renderer_override:
        py_settings = _read_plot_pyproject()
        renderer_override = py_settings.get("renderer")
    errors: List[str] = []

    for identifier in chain:
        plugin = find_plot_plugin_trusted(identifier)
        if plugin is None:
            plugin = find_plot_plugin(identifier)
        if plugin is None:
            errors.append(f"{identifier}: not registered")
            continue
        # If no renderer override, use default_renderer from builder metadata
        effective_renderer_override = renderer_override
        if not effective_renderer_override:
            builder_meta = getattr(plugin.builder, "plugin_meta", {})
            effective_renderer_override = builder_meta.get("default_renderer")
        # If a renderer override is specified, try to substitute the renderer
        if effective_renderer_override:
            try:
                override_renderer = find_plot_renderer(effective_renderer_override)
            except Exception:  # adr002_allow
                import logging
                import warnings

                logging.getLogger(__name__).info(
                    "Failed to find plot renderer '%s'; falling back to default",
                    effective_renderer_override,
                )
                warnings.warn(
                    f"Failed to find plot renderer '{effective_renderer_override}'; falling back to default",
                    UserWarning,
                    stacklevel=2,
                )
                override_renderer = None
            if override_renderer is not None:
                # Combined plugin returned by registry exposes .builder and .renderer
                # best-effort: ignore substitution failures
                with contextlib.suppress(Exception):
                    plugin.renderer = override_renderer
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
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            errors.append(f"{identifier}: {sys.exc_info()[1]}")
            continue
        return result

    from .utils.exceptions import ConfigurationError as _PlotConfigurationError

    raise _PlotConfigurationError(
        "Unable to resolve plot plugin for global explanations; "
        + "tried: "
        + ", ".join(chain)
        + ("; errors: " + "; ".join(errors) if errors else "")
    )


def _plot_proba_triangle():
    """Build a Matplotlib figure showcasing the probability triangle layout."""
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
