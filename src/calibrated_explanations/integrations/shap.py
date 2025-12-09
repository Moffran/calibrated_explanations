"""SHAP bridge helpers used by :class:`~calibrated_explanations.core.calibrated_explainer.CalibratedExplainer`.

The helper isolates all optional :mod:`shap` imports.  When the dependency is installed the
helper lazily constructs and caches a :class:`shap.Explainer` together with an initial
explanation so that subsequent requests are inexpensive.  If :mod:`shap` is not available the
helper simply returns ``(None, None)`` and keeps the integration disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Tuple

from ..utils import safe_import

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from ..core.calibrated_explainer import CalibratedExplainer


@dataclass
class ShapHelper:
    """Manage construction and caching of the optional SHAP explainer."""

    explainer: "CalibratedExplainer"
    _enabled: bool = field(default=False, init=False)
    _explainer_instance: Any = field(default=None, init=False)
    _reference_explanation: Any = field(default=None, init=False)

    def is_enabled(self) -> bool:
        """Return whether the helper has produced a cached SHAP explainer."""
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        """Force the helper enabled flag, primarily for compatibility tests."""
        if not value:
            self.reset()
        else:
            self._enabled = True

    @property
    def explainer_instance(self) -> Any:
        """Expose the cached SHAP explainer, preloading it if necessary."""
        instance, _ = self.preload()
        return instance

    @property
    def reference_explanation(self) -> Any:
        """Return the cached explanation, preloading if required."""
        _, explanation = self.preload()
        return explanation

    def preload(self, num_test: Optional[int] = None) -> Tuple[Any, Any]:
        """Materialize the SHAP explainer when :mod:`shap` is available."""
        if self._enabled and self._explainer_instance is not None:
            if num_test is None:
                return self._explainer_instance, self._reference_explanation
            shape = getattr(self._reference_explanation, "shape", None)
            if shape is not None and shape[0] == num_test:
                return self._explainer_instance, self._reference_explanation

        shap_module = safe_import("shap")
        if not shap_module:
            return None, None

        def _predict(x):
            return self.explainer._predict(x)[0]  # pylint: disable=protected-access

        self._explainer_instance = shap_module.Explainer(
            _predict,
            self.explainer.x_cal,
            feature_names=self.explainer.feature_names,
        )
        self._reference_explanation = (
            self._explainer_instance(self.explainer.x_cal[0, :].reshape(1, -1))
            if num_test is None
            else self._explainer_instance(self.explainer.x_cal[:num_test, :])
        )
        self._enabled = self._explainer_instance is not None

        return self._explainer_instance, self._reference_explanation

    def reset(self) -> None:
        """Drop cached objects so a future preload rebuilds them."""
        self._enabled = False
        self._explainer_instance = None
        self._reference_explanation = None
