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
        """Return calibrated predictions for *x*.

        Parameters
        ----------
        x:
            Feature matrix or payload understood by the host explainer.
        mode:
            Explanation mode requesting the prediction (``"factual"``,
            ``"alternative"``, ``"fast"``, or vendor extensions).
        task:
            Calibrated task identifier (``"classification"`` or
            ``"regression"``).
        bins : optional
            Mondrian bins for conditional calibration, forwarded verbatim to
            the underlying calibrator.

        Returns
        -------
        Mapping[str, Any]
            A payload containing at minimum ``predict``, ``mode``, and
            ``task`` keys.  When the underlying calibrator produces uncertainty
            bounds the payload also contains ``low`` and ``high`` arrays.
            For classification tasks a ``classes`` key with calibrated class
            labels is included.

        Notes
        -----
        **The bridge is a lifecycle-check shim, not the interval-shaping path.**

        The bridge protocol accepts only ``mode``, ``task``, and ``bins``.
        It always uses the explainer's default percentiles (5, 95) to produce
        ``low`` / ``high``; it has no parameter for customising them.

        Plugin implementers who need to honour a non-default
        ``low_high_percentiles`` value from an ``ExplanationRequest`` must
        call the explainer handle directly — **not** forward the kwarg here:

        .. code-block:: python

            # CORRECT — custom percentiles go to the explainer handle
            explainer = context.helper_handles["explainer"]
            preds, (low, high) = explainer.predict(
                x,
                uq_interval=True,
                low_high_percentiles=request.low_high_percentiles,
                bins=request.bins,
            )

            # WRONG — bridge.predict() does not accept low_high_percentiles;
            #         passing it raises TypeError, which propagates as ENGINE_FAILURE
            bridge.predict(x, mode=mode, task=task, low_high_percentiles=...)  # noqa

        In CE's own ``explain_batch`` implementations, the bridge call is
        made solely to honour the lifecycle contract; its return value is either
        used as-is (for default-percentile payloads) or discarded, and interval
        shaping is performed separately through the explanation callable.
        """

    def predict_interval(
        self,
        x: Any,
        *,
        task: str,
        bins: Any | None = None,
    ) -> Sequence[Any]:
        """Return calibrated interval predictions for *x*.

        The return payload mirrors the interval calibrator outputs and is left
        intentionally loose so that downstream plugins can translate it into
        their preferred artefacts.
        """

    def predict_proba(self, x: Any, bins: Any | None = None) -> Sequence[Any]:
        """Return calibrated probability estimates for *x* when available."""


__all__ = ["PredictBridge"]
