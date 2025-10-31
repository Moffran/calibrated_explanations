"""LIME bridge helpers used by :class:`~calibrated_explanations.core.calibrated_explainer.CalibratedExplainer`.

The optional :mod:`lime` dependency is imported lazily through :func:`safe_import`.  When LIME is
missing the helper simply returns ``(None, None)`` so that callers can decide how to degrade.
When the dependency is available the helper instantiates and caches the
:class:`lime.lime_tabular.LimeTabularExplainer` alongside a reference explanation.  Subsequent
requests reuse those cached objects unless the caller explicitly clears the cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Tuple

from ..utils.helper import safe_import

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from ..core.calibrated_explainer import CalibratedExplainer


@dataclass
class LimeHelper:
    """Manage the lifecycle of the optional LIME integration."""

    explainer: "CalibratedExplainer"
    _enabled: bool = field(default=False, init=False)
    _explainer_instance: Any = field(default=None, init=False)
    _reference_explanation: Any = field(default=None, init=False)

    def is_enabled(self) -> bool:
        """Return whether the helper has produced a cached LIME explainer."""

        return self._enabled

    def set_enabled(self, value: bool) -> None:
        """Force the helper's enabled flag, primarily used in tests."""

        if not value:
            self.reset()
        else:
            self._enabled = True

    @property
    def explainer_instance(self) -> Any:
        """Expose the cached LIME explainer, triggering preload if necessary."""

        instance, _ = self.preload()
        return instance

    @property
    def reference_explanation(self) -> Any:
        """Return the cached reference explanation, triggering preload if required."""

        _, explanation = self.preload()
        return explanation

    def preload(self, x_cal: Optional[Any] = None) -> Tuple[Any, Any]:
        """Materialize and cache the LIME explainer if the dependency is present."""

        lime_cls = safe_import("lime.lime_tabular", "LimeTabularExplainer")
        if not lime_cls:
            return None, None

        if not self._enabled or self._explainer_instance is None:
            features = self.explainer.feature_names
            x_cal_source = self.explainer.x_cal[:1, :] if x_cal is None else x_cal
            if self.explainer.mode == "classification":
                self._explainer_instance = lime_cls(
                    x_cal_source,
                    feature_names=features,
                    class_names=["0", "1"],
                    mode=self.explainer.mode,
                )
                self._reference_explanation = self._explainer_instance.explain_instance(
                    self.explainer.x_cal[0, :],
                    self.explainer.learner.predict_proba,
                    num_features=self.explainer.num_features,
                )
            elif "regression" in self.explainer.mode:
                self._explainer_instance = lime_cls(
                    x_cal_source,
                    feature_names=features,
                    mode="regression",
                )
                self._reference_explanation = self._explainer_instance.explain_instance(
                    self.explainer.x_cal[0, :],
                    self.explainer.learner.predict,
                    num_features=self.explainer.num_features,
                )
            self._enabled = self._explainer_instance is not None

        return self._explainer_instance, self._reference_explanation

    def reset(self) -> None:
        """Drop cached objects so that future calls rebuild the integration."""

        self._enabled = False
        self._explainer_instance = None
        self._reference_explanation = None
