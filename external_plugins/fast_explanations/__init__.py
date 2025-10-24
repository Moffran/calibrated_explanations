"""External fast explanation plugins for calibrated_explanations.

These plugins mirror the legacy FAST implementations but ship outside the core
package so projects must opt in explicitly. Import :mod:`external_plugins.fast_explanations`
and call :func:`register` (or install the aggregated ``external-plugins`` extra)
to make the FAST interval and explanation identifiers available at runtime.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any

from calibrated_explanations import __version__ as package_version
from calibrated_explanations.explanations.explanation import FastExplanation
from calibrated_explanations.plugins.builtins import _LegacyExplanationBase
from calibrated_explanations.plugins.intervals import (
    IntervalCalibratorContext,
    IntervalCalibratorPlugin,
)
from calibrated_explanations.plugins.registry import (
    register_explanation_plugin,
    register_interval_plugin,
)
from calibrated_explanations.utils.perturbation import perturb_dataset


class FastIntervalCalibratorPlugin(IntervalCalibratorPlugin):
    """FAST adapter returning the precomputed list of interval learners."""

    plugin_meta = {
        "name": "core.interval.fast",
        "schema_version": 1,
        "version": package_version,
        "provider": "calibrated_explanations",
        "capabilities": ["interval:classification", "interval:regression"],
        "modes": ("classification", "regression"),
        "dependencies": (),
        "trusted": True,
        "trust": {"trusted": True},
        "fast_compatible": True,
        "requires_bins": False,
        "confidence_source": "fast",
        "legacy_compatible": True,
    }

    def create(self, context: IntervalCalibratorContext, *, fast: bool = True) -> Any:
        """Return the FAST calibrator list already prepared by the explainer."""

        metadata = context.metadata
        task = str(metadata.get("task") or metadata.get("mode") or "")
        explainer = metadata.get("explainer")
        if explainer is None:
            raise RuntimeError("FAST interval context missing 'explainer' handle")

        x_cal, y_cal = context.calibration_splits[0]
        bins = context.bins.get("calibration")
        learner = context.learner
        difficulty = metadata.get("difficulty_estimator")
        categorical_features = tuple(metadata.get("categorical_features", ()))
        noise_cfg = metadata.get("noise_config", {})
        (
            explainer.fast_x_cal,
            explainer.scaled_x_cal,
            explainer.scaled_y_cal,
            scale_factor,
        ) = perturb_dataset(
            x_cal,
            y_cal,
            categorical_features,
            noise_type=noise_cfg.get("noise_type"),
            scale_factor=noise_cfg.get("scale_factor"),
            severity=noise_cfg.get("severity"),
            seed=noise_cfg.get("seed"),
            rng=noise_cfg.get("rng"),
        )
        expanded_bins = np.tile(bins.copy(), scale_factor) if bins is not None else None
        original_bins = explainer.bins
        original_x_cal = explainer.x_cal
        original_y_cal = explainer.y_cal
        explainer.bins = expanded_bins

        calibrators: list[Any] = []
        num_features = int(metadata.get("num_features", 0) or 0)
        if "classification" in task:
            from calibrated_explanations.core.venn_abers import (
                VennAbers,
            )  # local import to avoid circular dependency

            for f in range(num_features):
                fast_x_cal = explainer.scaled_x_cal.copy()
                fast_x_cal[:, f] = explainer.fast_x_cal[:, f]
                calibrators.append(
                    VennAbers(
                        fast_x_cal,
                        explainer.scaled_y_cal,
                        learner,
                        explainer.bins,
                        difficulty_estimator=difficulty,
                    )
                )
        else:
            from calibrated_explanations.core.interval_regressor import (
                IntervalRegressor,
            )  # local import to avoid circular dependency

            for f in range(num_features):
                fast_x_cal = explainer.scaled_x_cal.copy()
                fast_x_cal[:, f] = explainer.fast_x_cal[:, f]
                explainer.x_cal = fast_x_cal
                explainer.y_cal = explainer.scaled_y_cal
                calibrators.append(IntervalRegressor(explainer))

        explainer.x_cal = original_x_cal
        explainer.y_cal = original_y_cal
        explainer.bins = original_bins

        if "classification" in task:
            from calibrated_explanations.core.venn_abers import (
                VennAbers,
            )  # local import to avoid circular dependency

            calibrators.append(
                VennAbers(
                    x_cal,
                    y_cal,
                    learner,
                    bins,
                    difficulty_estimator=difficulty,
                    predict_function=(
                        metadata.get("predict_function")
                        if metadata.get("predict_function") is not None
                        else getattr(explainer, "predict_function", None)
                    ),
                )
            )
        else:
            from calibrated_explanations.core.interval_regressor import (
                IntervalRegressor,
            )  # local import to avoid circular dependency

            calibrators.append(IntervalRegressor(explainer))

        if isinstance(metadata, dict):
            metadata.setdefault("fast_calibrators", tuple(calibrators))
        return calibrators


@dataclass
class FastExplanationPlugin(_LegacyExplanationBase):
    """Plugin wrapping ``CalibratedExplainer.explain_fast``."""

    plugin_meta = {
        "name": "core.explanation.fast",
        "schema_version": 1,
        "version": package_version,
        "provider": "calibrated_explanations",
        "capabilities": ["explain", "explanation:fast", "task:classification", "task:regression"],
        "modes": ("fast",),
        "tasks": ("classification", "regression"),
        "dependencies": ("core.interval.fast", "legacy"),
        "interval_dependency": "core.interval.fast",
        "plot_dependency": "legacy",
        "trusted": True,
        "trust": {"trusted": True},
    }

    def __init__(self) -> None:
        super().__init__(
            _mode="fast",
            _explanation_attr="explain_fast",
            _expected_cls=FastExplanation,
            plugin_meta=self.plugin_meta,
        )


def register() -> None:
    """Register the FAST interval and explanation plugins with the core registry."""

    register_interval_plugin("core.interval.fast", FastIntervalCalibratorPlugin())
    register_explanation_plugin("core.explanation.fast", FastExplanationPlugin())


__all__ = [
    "FastIntervalCalibratorPlugin",
    "FastExplanationPlugin",
    "register",
]
