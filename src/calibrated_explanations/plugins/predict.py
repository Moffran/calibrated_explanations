"""Predict bridge protocol for explanation plugins (ADR-015).

The predict bridge encapsulates calibrated inference helpers that explanation
plugins rely on. It provides a narrow surface so plugins can trigger
calibrated predictions without depending on the concrete calibrator
implementations. The precise behaviours are intentionally conservative – the
bridge is a frozen adapter owned by :class:`CalibratedExplainer` and treated as
read-only by plugins.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class PredictBridge(Protocol):
    """Protocol describing calibrated prediction helpers.

    The bridge exposes high-level hooks rather than the raw calibrators. This
    keeps plugins decoupled from the in-tree implementations while ensuring all
    predictions continue to flow through the calibrated pathways established by
    :class:`CalibratedExplainer`.
    """

    def predict(
        self,
        x: Any,
        *,
        mode: str,
        task: str,
        bins: Any | None = None,
    ) -> Mapping[str, Any]:
        """Return calibrated predictions for *X*.

        Parameters
        ----------
        X:
            Feature matrix or payload understood by the host explainer.
        mode:
            Explanation mode requesting the prediction (``"factual"``,
            ``"alternative"``, ``"fast"``, or vendor extensions).
        task:
            Calibrated task identifier (``"classification"`` or
            ``"regression"``).
        """

    def predict_interval(
        self,
        x: Any,
        *,
        task: str,
        bins: Any | None = None,
    ) -> Sequence[Any]:
        """Return calibrated interval predictions for *X*.

        The return payload mirrors the interval calibrator outputs and is left
        intentionally loose so that downstream plugins can translate it into
        their preferred artefacts.
        """

    def predict_proba(self, x: Any, bins: Any | None = None) -> Sequence[Any]:
        """Return calibrated probability estimates for *X* when available."""


__all__ = ["PredictBridge"]
