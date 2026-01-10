"""Orchestration layer for reject learner initialization and inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from crepes import ConformalClassifier
from crepes.extras import hinge

from ...utils.exceptions import ValidationError
from .policy import RejectPolicy
from ...explanations.reject import RejectResult


class RejectOrchestrator:
    """Coordinate reject learner lifecycle and predictions."""

    def __init__(self, explainer: Any) -> None:
        self.explainer = explainer
        # Lightweight registry for reject strategies. Keys are string identifiers
        # (e.g., 'builtin.default') and values are callables with the same
        # signature as `apply_policy` that return a `RejectResult`.
        self._strategies: dict[str, Any] = {}
        # Register the builtin default strategy preserving existing semantics
        # under the well-known identifier `builtin.default`.
        self.register_strategy("builtin.default", self._builtin_strategy)

    def initialize_reject_learner(self, calibration_set=None, threshold=None):
        """Initialize the reject learner with a threshold value."""
        if calibration_set is None:
            x_cal, y_cal = self.explainer.x_cal, self.explainer.y_cal
        elif calibration_set is tuple:
            x_cal, y_cal = calibration_set
        else:
            x_cal, y_cal = calibration_set[0], calibration_set[1]
        self.explainer.reject_threshold = None
        if self.explainer.mode in "regression":
            proba_1, _, _, _ = self.explainer.interval_learner.predict_probability(
                x_cal, y_threshold=threshold, bins=self.explainer.bins
            )
            proba = np.array([[1 - proba_1[i], proba_1[i]] for i in range(len(proba_1))])
            classes = (y_cal < threshold).astype(int)
            self.explainer.reject_threshold = threshold
        elif self.explainer.is_multiclass():  # pylint: disable=protected-access
            proba, classes = self.explainer.interval_learner.predict_proba(
                x_cal, bins=self.explainer.bins
            )
            proba = np.array([[1 - proba[i, c], proba[i, c]] for i, c in enumerate(classes)])
            classes = (classes == y_cal).astype(int)
        else:
            proba = self.explainer.interval_learner.predict_proba(x_cal, bins=self.explainer.bins)
            classes = y_cal
        alphas_cal = hinge(proba, np.unique(classes), classes)
        self.explainer.reject_learner = ConformalClassifier().fit(alphas=alphas_cal, bins=classes)
        return self.explainer.reject_learner

    def predict_reject(self, x, bins=None, confidence=0.95):
        """Predict whether to reject the explanations for the test data."""
        if bins is not None:
            bins = np.asarray(bins)
        if self.explainer.mode in "regression":
            if self.explainer.reject_threshold is None:
                raise ValidationError(
                    "The reject learner is only available for regression with a threshold."
                )
            proba_1, _, _, _ = self.explainer.interval_learner.predict_probability(
                x, y_threshold=self.explainer.reject_threshold, bins=bins
            )
            proba = np.array([[1 - proba_1[i], proba_1[i]] for i in range(len(proba_1))])
            classes = [0, 1]
        elif self.explainer.is_multiclass():  # pylint: disable=protected-access
            proba, classes = self.explainer.interval_learner.predict_proba(x, bins=bins)
            proba = np.array([[1 - proba[i, c], proba[i, c]] for i, c in enumerate(classes)])
            classes = [0, 1]
        else:
            proba = self.explainer.interval_learner.predict_proba(x, bins=bins)
            classes = np.unique(self.explainer.y_cal)
        alphas_test = hinge(proba)

        prediction_set = np.array(
            [
                self.explainer.reject_learner.predict_set(
                    alphas_test, np.full(len(alphas_test), classes[c]), confidence=confidence
                )[:, c]
                for c in range(len(classes))
            ]
        ).T
        singleton = np.sum(np.sum(prediction_set, axis=1) == 1)
        empty = np.sum(np.sum(prediction_set, axis=1) == 0)
        n = len(x)

        epsilon = 1 - confidence
        error_rate = (n * epsilon - empty) / singleton
        reject_rate = 1 - singleton / n

        rejected = np.sum(prediction_set, axis=1) != 1
        return rejected, error_rate, reject_rate

    def apply_policy(self, policy: RejectPolicy, x, explain_fn=None, bins=None, confidence=0.95, **kwargs):
        """Apply a `RejectPolicy` to inputs and optionally produce predictions/explanations.

        Parameters
        - policy: RejectPolicy selected by the caller
        - x: input instances
        - explain_fn: optional callable `explain_fn(x_subset, **kwargs)` returning explanations
        - bins, confidence: passed to reject prediction

        Returns
        - RejectResult envelope with `prediction`, `explanation`, `rejected`, `policy`, and `metadata`.
        """
        # Allow callers to select a strategy identifier via the `strategy` kwarg.
        # By default, resolve to `builtin.default` which preserves legacy semantics.
        strategy_name = kwargs.pop("strategy", None)
        strategy = self.resolve_strategy(strategy_name)
        return strategy(policy, x, explain_fn=explain_fn, bins=bins, confidence=confidence, **kwargs)

    # --- Registry helpers -------------------------------------------------
    def register_strategy(self, name: str, fn: Any) -> None:
        """Register a reject strategy callable under *name*.

        The callable must accept the same parameters as `apply_policy` and
        return a `RejectResult`.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("strategy name must be a non-empty string")
        if not callable(fn):
            raise ValueError("strategy must be callable")
        self._strategies[name] = fn

    def resolve_strategy(self, identifier: str | None):
        """Resolve a registered strategy by identifier.

        If *identifier* is None, return the builtin default strategy.
        Raises KeyError when an unknown identifier is requested.
        """
        if identifier is None:
            # default fallback
            try:
                return self._strategies["builtin.default"]
            except KeyError:
                raise KeyError("builtin.default strategy not registered")
        if identifier in self._strategies:
            return self._strategies[identifier]
        raise KeyError(f"Reject strategy '{identifier}' is not registered")

    # --- Builtin strategy (preserve previous apply_policy impl) -----------
    def _builtin_strategy(self, policy: RejectPolicy, x, explain_fn=None, bins=None, confidence=0.95, **kwargs):
        """Builtin strategy that preserves the previous `apply_policy` semantics."""
        try:
            policy = RejectPolicy(policy)
        except Exception:
            policy = RejectPolicy.NONE

        # If NONE, return a simple envelope indicating no action
        if policy is RejectPolicy.NONE:
            return RejectResult(prediction=None, explanation=None, rejected=None, policy=policy, metadata=None)

        # Ensure reject learner is initialized (implicit enable)
        if getattr(self.explainer, "reject_learner", None) is None:
            # Best-effort initialization using explainer calibration set
            try:
                self.initialize_reject_learner()
            except Exception:
                # If initialization fails, surface minimal metadata but continue
                return RejectResult(prediction=None, explanation=None, rejected=None, policy=policy, metadata={"init_error": True})

        rejected, error_rate, reject_rate = self.predict_reject(x, bins=bins, confidence=confidence)

        prediction = None
        explanation = None

        # Obtain predictions when requested by policy
        if policy in (
            RejectPolicy.PREDICT_AND_FLAG,
            RejectPolicy.EXPLAIN_ALL,
            RejectPolicy.EXPLAIN_REJECTS,
            RejectPolicy.EXPLAIN_NON_REJECTS,
            RejectPolicy.SKIP_ON_REJECT,
        ):
            # Delegate to explainer's public predict method. To avoid
            # re-entering the reject orchestration (which would cause
            # recursion), signal the explainer to skip reject handling for
            # this inner prediction by setting an internal-only flag.
            try:
                inner_kwargs = dict(kwargs) if kwargs is not None else {}
                inner_kwargs["_ce_skip_reject"] = True
                prediction = self.explainer.predict(x, **inner_kwargs)
            except Exception:
                prediction = None

        # Obtain explanations via provided callable according to policy
        if explain_fn is not None:
            try:
                if policy is RejectPolicy.EXPLAIN_ALL:
                    explanation = explain_fn(x, **kwargs)
                elif policy is RejectPolicy.EXPLAIN_REJECTS:
                    # explain only rejected instances
                    idx = [i for i, r in enumerate(rejected) if r]
                    explanation = explain_fn([x[i] for i in idx], **kwargs) if idx else None
                elif policy is RejectPolicy.EXPLAIN_NON_REJECTS:
                    idx = [i for i, r in enumerate(rejected) if not r]
                    explanation = explain_fn([x[i] for i in idx], **kwargs) if idx else None
                elif policy is RejectPolicy.SKIP_ON_REJECT:
                    idx = [i for i, r in enumerate(rejected) if not r]
                    explanation = explain_fn([x[i] for i in idx], **kwargs) if idx else None
                elif policy is RejectPolicy.PREDICT_AND_FLAG:
                    # do not generate explanations, only flag
                    explanation = None
            except Exception:
                explanation = None

        metadata = {"error_rate": error_rate, "reject_rate": reject_rate}
        return RejectResult(prediction=prediction, explanation=explanation, rejected=rejected, policy=policy, metadata=metadata)
