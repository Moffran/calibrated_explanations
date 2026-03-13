"""Orchestration layer for reject learner initialization and inference."""

from __future__ import annotations

import inspect
import logging
import threading
import warnings
from dataclasses import dataclass
from math import isclose
from typing import Any

import numpy as np
from crepes import ConformalClassifier
from crepes.extras import hinge

from ...explanations.reject import (
    RejectResult,
    canonical_reject_ncf_w,
    normalize_reject_ncf_choice,
)
from ...utils.exceptions import ValidationError
from .policy import RejectPolicy

_VALID_NCF = frozenset({"default", "ensured"})


@dataclass(frozen=True)
class RejectPolicyResolution:
    """Canonical reject-policy resolution result."""

    policy: RejectPolicy
    used_default: bool
    fallback_used: bool
    reason: str | None = None


def validate_reject_confidence(confidence: Any) -> float:
    """Validate reject confidence and return canonical float in (0, 1)."""
    try:
        value = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "confidence must be a float in the open interval (0, 1).",
            details={"confidence": confidence},
        ) from exc
    if not 0.0 < value < 1.0:
        raise ValidationError(
            "confidence must be a float in the open interval (0, 1).",
            details={"confidence": value},
        )
    return value


def validate_reject_w(w: Any) -> float:
    """Validate reject blending weight and return canonical float in [0, 1]."""
    try:
        value = float(w)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "w must be a float in the closed interval [0, 1].",
            details={"w": w},
        ) from exc
    if not 0.0 <= value <= 1.0:
        raise ValidationError(
            "w must be a float in the closed interval [0, 1].",
            details={"w": value},
        )
    return value


def _thresholds_equal(lhs: Any, rhs: Any) -> bool:
    """Return True when two threshold payloads are numerically equivalent."""
    try:
        left = np.asarray(lhs)
        right = np.asarray(rhs)
    except Exception:  # adr002_allow
        return lhs == rhs
    if left.shape != right.shape:
        return False
    with np.errstate(invalid="ignore"):
        return bool(np.allclose(left, right, equal_nan=True))


def _interval_width_score(proba: np.ndarray) -> np.ndarray:
    """Compute instance-level interval-width score from a 2-column VA output."""
    proba = np.asarray(proba, dtype=float)
    if proba.shape[1] < 2:
        return np.zeros(len(proba))
    return proba[:, 1] - proba[:, 0]


def _margin_score(proba: np.ndarray) -> np.ndarray:
    """Compute instance-level margin score (higher means less conforming)."""
    proba = np.asarray(proba, dtype=float)
    if proba.shape[1] < 2:
        return np.zeros(len(proba))
    sorted_proba = np.sort(proba, axis=1)[:, ::-1]
    return 1.0 - (sorted_proba[:, 0] - sorted_proba[:, 1])


def _default_ncf_kind(is_multiclass: bool) -> str:
    """Return internal default score kind for the current task."""
    return "margin" if is_multiclass else "hinge"


def _default_score_cal(
    proba: np.ndarray,
    classes: np.ndarray,
    labels: np.ndarray,
    default_kind: str,
) -> np.ndarray:
    """Compute 1-D calibration scores for the internal default reject score."""
    if default_kind == "hinge":
        return hinge(proba, classes, labels)
    if default_kind == "margin":
        return _margin_score(proba)
    raise ValidationError(
        f"Unsupported internal default score kind {default_kind!r}.",
        details={"default_kind": default_kind},
    )


def _default_score_test(proba: np.ndarray, default_kind: str) -> np.ndarray:
    """Compute 2-D test scores for the internal default reject score."""
    if default_kind == "hinge":
        return hinge(proba)
    if default_kind == "margin":
        base = _margin_score(proba)
        k = np.asarray(proba).shape[1]
        return np.repeat(base[:, np.newaxis], k, axis=1)
    raise ValidationError(
        f"Unsupported internal default score kind {default_kind!r}.",
        details={"default_kind": default_kind},
    )


def _normalize_stored_ncf(value: Any) -> str | None:
    """Normalize persisted/legacy NCF names for re-init equality checks."""
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in ("default", "hinge", "margin", "entropy"):
        return "default"
    if lowered == "ensured":
        return "ensured"
    return lowered


def _legacy_base_ncf(proba: np.ndarray, ncf: str) -> np.ndarray:
    """Compute instance-level legacy NCF scores.

    Parameters
    ----------
    proba : ndarray of shape (n, k)
        Calibrated probability matrix.  In multiclass mode this is a binarized
        ``(n, 2)`` matrix ``[1 - p_argmax, p_argmax]``, not the full K-class
        distribution.  Consequently ``entropy`` and ``margin`` operate on this
        two-column representation rather than the full K-class probabilities:

        * ``entropy`` → binary entropy of ``p_argmax``.
        * ``margin``  → ``2 * (1 - p_argmax)`` (confidence-based, not true
          K-class margin).

        For binary classification and regression the full probability matrix
        is passed and the semantics match the documented definitions.

        For Venn-Abers binary output, column 0 = predict_low and column 1 =
        predict_high, so the width ``proba[:,1] - proba[:,0]`` is the
        calibrated uncertainty interval (used by ``ensured``).
    ncf : {'ensured', 'entropy', 'margin'}
        Non-conformity function type.

    Returns
    -------
    ndarray of shape (n,) with scores in [0, 1].
    """
    proba = np.asarray(proba, dtype=float)
    if ncf == "ensured":
        return _interval_width_score(proba)
    if ncf == "entropy":
        proba_clipped = np.clip(proba, 1e-12, 1.0)
        k = max(proba.shape[1], 2)
        return -np.sum(proba_clipped * np.log2(proba_clipped), axis=1) / np.log2(k)
    if ncf == "margin":
        return _margin_score(proba)
    raise ValidationError(
        f"Unsupported ncf type {ncf!r}; expected one of {sorted(_VALID_NCF)!r}",
        details={"ncf": ncf},
    )


def _ncf_scores_cal(  # pylint: disable=invalid-name
    proba: np.ndarray,
    classes: np.ndarray,
    labels: np.ndarray,
    ncf: str,
    w: float,
    default_kind: str,
) -> np.ndarray:
    """Compute 1-D calibration non-conformity scores.

    Returns
    -------
    ndarray of shape (n,)
    """
    default_score = _default_score_cal(proba, classes, labels, default_kind)
    if ncf == "default":
        return default_score
    if ncf == "ensured":
        interval = _interval_width_score(proba)
        return (1.0 - w) * interval + w * default_score
    return _legacy_base_ncf(proba, ncf)


def _ncf_scores_test(  # pylint: disable=invalid-name
    proba: np.ndarray,
    ncf: str,
    w: float,
    default_kind: str,
) -> np.ndarray:
    """Compute 2-D test non-conformity scores (one column per class).

    Returns
    -------
    ndarray of shape (n, k)
    """
    default_score = _default_score_test(proba, default_kind)
    if ncf == "default":
        return default_score
    if ncf == "ensured":
        interval = _interval_width_score(proba)
        k = np.asarray(proba).shape[1]
        return (1.0 - w) * np.repeat(interval[:, np.newaxis], k, axis=1) + w * default_score
    base = _legacy_base_ncf(proba, ncf)  # (n,)
    k = np.asarray(proba).shape[1]
    return np.repeat(base[:, np.newaxis], k, axis=1)


def resolve_policy_spec(reject_policy_kw: Any, explainer: Any) -> Any:
    """Resolve reject policy inputs to a canonical policy value.

    This function accepts:
      - RejectPolicy enum members
      - RejectPolicySpec instances
      - dict payloads produced by RejectPolicySpec.to_dict()
            - plain policy strings ("flag", "only_rejected", ...)
      - None (returned unchanged)

    Returns
    -------
    RejectPolicy | None
        The canonical RejectPolicy value, or None when input is None.

    Raises
    ------
    ValidationError
        When the input cannot be parsed into a known policy/spec.
    """
    from ...explanations.reject import RejectPolicySpec  # pylint: disable=import-outside-toplevel

    if reject_policy_kw is None:
        return None

    if isinstance(reject_policy_kw, RejectPolicy):
        return reject_policy_kw

    spec: RejectPolicySpec | None = None
    if isinstance(reject_policy_kw, RejectPolicySpec):
        spec = reject_policy_kw
    elif isinstance(reject_policy_kw, dict):
        try:
            spec = RejectPolicySpec.from_dict(reject_policy_kw)
        except ValueError as exc:
            raise ValidationError(str(exc), details={"payload": reject_policy_kw}) from exc
        except ValidationError as exc:
            raise ValidationError(
                "Invalid RejectPolicySpec dict; expected keys 'policy','ncf','w'.",
                details={"payload": reject_policy_kw},
            ) from exc
    elif isinstance(reject_policy_kw, str):
        stripped = reject_policy_kw.strip()
        if stripped.startswith("{"):
            raise ValidationError(
                "JSON string reject policy payloads are unsupported; pass a dict or RejectPolicySpec.",
                details={"payload": reject_policy_kw},
            )
        try:
            return RejectPolicy(stripped.lower())
        except ValueError as exc:
            raise ValidationError(
                "Unknown reject policy string.",
                details={"policy": reject_policy_kw},
            ) from exc
    else:
        raise ValidationError(
            "Unsupported reject_policy input type.",
            details={"type": type(reject_policy_kw).__name__, "value": repr(reject_policy_kw)},
        )

    if spec is not None:
        stored_ncf = _normalize_stored_ncf(getattr(explainer, "reject_ncf", None))
        stored_w = getattr(explainer, "reject_ncf_w", None)
        effective_spec_w = canonical_reject_ncf_w(spec.ncf, float(spec.w))
        effective_stored_w = (
            None
            if stored_w is None or stored_ncf is None
            else canonical_reject_ncf_w(str(stored_ncf), float(stored_w))
        )
        if (
            stored_ncf != spec.ncf
            or effective_stored_w is None
            or not isclose(
                float(effective_stored_w), float(effective_spec_w), rel_tol=1e-9, abs_tol=0.0
            )
        ):
            reject_orchestrator = getattr(explainer, "reject_orchestrator", None)
            if reject_orchestrator is None:
                plugin_manager = getattr(explainer, "plugin_manager", None)
                if plugin_manager is None:
                    raise ValidationError(
                        "Reject orchestrator is unavailable for policy initialization.",
                        details={"reason": "missing_plugin_manager"},
                    )
                plugin_manager.initialize_orchestrators()
                reject_orchestrator = getattr(explainer, "reject_orchestrator", None)
            if reject_orchestrator is None:
                raise ValidationError(
                    "Reject orchestrator is unavailable for policy initialization.",
                    details={"reason": "missing_reject_orchestrator"},
                )
            reject_orchestrator.initialize_reject_learner(ncf=spec.ncf, w=effective_spec_w)
        return spec.policy

    raise ValidationError(
        "Failed to resolve reject_policy to a canonical form.",
        details={"input": repr(reject_policy_kw)},
    )


def resolve_effective_reject_policy(
    reject_policy_kw: Any,
    explainer: Any,
    *,
    default_policy: Any = RejectPolicy.NONE,
    logger: logging.Logger | None = None,
) -> RejectPolicyResolution:
    """Resolve explicit/default reject policy to a canonical effective policy.

    Behavior contract
    -----------------
    - Explicit invalid per-call inputs fail fast with ``ValidationError``.
    - Invalid explainer defaults fall back to ``RejectPolicy.NONE`` and emit
      both a ``UserWarning`` and an INFO log event.
    """
    used_default = reject_policy_kw is None
    candidate_policy = default_policy if used_default else reject_policy_kw
    active_logger = logger or logging.getLogger(__name__)

    try:
        resolved = resolve_policy_spec(candidate_policy, explainer)
    except ValidationError as exc:
        if not used_default:
            raise
        message = "Invalid default_reject_policy; falling back to RejectPolicy.NONE."
        active_logger.info("%s %s", message, str(exc))
        warnings.warn(f"{message} {exc!s}", UserWarning, stacklevel=3)
        return RejectPolicyResolution(
            policy=RejectPolicy.NONE,
            used_default=True,
            fallback_used=True,
            reason="invalid_default_reject_policy",
        )

    resolved_policy = RejectPolicy.NONE if resolved is None else RejectPolicy(resolved)

    return RejectPolicyResolution(
        policy=resolved_policy,
        used_default=used_default,
        fallback_used=False,
        reason=None,
    )


class RejectOrchestrator:
    """Coordinate reject learner lifecycle and predictions."""

    def __init__(self, explainer: Any) -> None:
        self.explainer = explainer
        self._logger = logging.getLogger(__name__)
        # Lightweight registry for reject strategies. Keys are string identifiers
        # (e.g., 'builtin.default') and values are callables with the same
        # signature as `apply_policy` that return a `RejectResult`.
        self._strategies: dict[str, Any] = {}
        self._strategies_lock = threading.RLock()
        # Register the builtin default strategy preserving existing semantics
        # under the well-known identifier `builtin.default`.
        self.register_strategy("builtin.default", self._builtin_strategy)

    def __getstate__(self) -> dict:
        """Support pickling by excluding unpicklable attributes (RLock/loggers).

        When the orchestrator is used with multiprocessing or joblib, instances
        may be pickled. Exclude the thread lock and logger and recreate them
        during unpickling to avoid "cannot pickle '_thread.RLock' object".
        """
        state = self.__dict__.copy()
        state.pop("_strategies_lock", None)
        state.pop("_logger", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state and recreate unpicklable attributes."""
        self.__dict__.update(state)
        self._strategies_lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _filter_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in signature.parameters}

    def _resolve_regression_effective_threshold(
        self, threshold: Any
    ) -> tuple[Any | None, str | None]:
        """Resolve effective threshold for reject-enabled calls.

        Regression contract:
        - Call threshold is mandatory.
        - Stored reject_threshold is never used as fallback.
        """
        if self.explainer.mode != "regression":
            return None, None
        if threshold is None:
            raise ValidationError("reject learner unavailable for regression without threshold")

        effective_threshold = threshold
        threshold_source = "call"
        current_threshold = getattr(self.explainer, "reject_threshold", None)
        learner_missing = getattr(self.explainer, "reject_learner", None) is None
        mismatch = current_threshold is None or not _thresholds_equal(
            current_threshold, effective_threshold
        )
        if learner_missing or mismatch:
            if not learner_missing and mismatch:
                self._logger.info(
                    "Reject threshold mismatch detected; reinitializing reject learner with call threshold."
                )
                warnings.warn(
                    "Reject threshold mismatch detected; reinitializing reject learner "
                    "with call threshold.",
                    UserWarning,
                    stacklevel=3,
                )
            self.initialize_reject_learner(
                threshold=effective_threshold,
                ncf=getattr(self.explainer, "reject_ncf", None),
                w=(
                    getattr(self.explainer, "reject_ncf_w", 0.5)
                    if getattr(self.explainer, "reject_ncf_w", None) is not None
                    else 0.5
                ),
            )
            threshold_source = "call_reinitialized" if mismatch else "call"
        return effective_threshold, threshold_source

    def initialize_reject_learner(  # pylint: disable=invalid-name
        self, calibration_set=None, threshold=None, ncf=None, w=0.5
    ):
        """Initialize the reject learner with calibration data and NCF settings.

        Parameters
        ----------
        calibration_set : tuple (x_cal, y_cal) or None
            Calibration data. Uses the explainer's calibration set when None.
        threshold : float or None
            Decision threshold for **regression only**. **Required** when the
            explainer is in regression mode — omitting it raises
            ``ValidationError``.

            The threshold defines a binary event: *"will the target be below
            this value?"*  The framework converts regression into threshold-
            binarized conformal classification (``P(y ≤ threshold)``). This is
            **not** conformal prediction interval regression; it is conformal
            prediction for a user-defined threshold crossing. For classification
            this parameter is unused and should remain ``None``.
        ncf : str or None, default None
            Non-conformity function type: 'default' or 'ensured'. The
            internal default score is task-dependent: margin for multiclass
            and hinge for binary/regression. Legacy 'entropy' input is
            accepted and silently mapped to 'default'. Explicit 'hinge' and
            'margin' inputs are rejected.
        w : float, default 0.5
            Blending weight in [0, 1] used only when ``ncf='ensured'``.
            ``score = (1-w) * interval_width + w * default_score``.
            Ignored for ``ncf='default'``. ``w=0.0`` raises
            ``ValidationError``; ``w < 0.1`` emits a ``UserWarning``.

        Raises
        ------
        ValidationError
            If ``threshold`` is ``None`` and the explainer is in regression
            mode, if ``ncf`` is not a recognised value, if explicit
            ``ncf='hinge'`` or ``ncf='margin'`` is supplied, or if ``w=0.0``
            with ``ncf='ensured'``.
        """
        validated_w = validate_reject_w(w)
        bins_cal = self.explainer.bins if calibration_set is None else None
        if calibration_set is None:
            x_cal, y_cal = self.explainer.x_cal, self.explainer.y_cal
        elif isinstance(calibration_set, (tuple, list)) and len(calibration_set) == 2:
            x_cal, y_cal = calibration_set
        else:
            raise ValidationError("calibration_set must be a (x_cal, y_cal) pair or None")

        # Resolve user NCF; internal default score remains task-dependent.
        ncf_explicit = ncf is not None
        if ncf is None:
            ncf = "default"
        try:
            ncf = normalize_reject_ncf_choice(ncf)
        except ValueError as exc:
            raise ValidationError(
                str(exc),
                details={"ncf": ncf},
            ) from exc
        if ncf not in _VALID_NCF:
            raise ValidationError(
                f"ncf must be one of {sorted(_VALID_NCF)!r}; got {ncf!r}",
                details={"ncf": ncf},
            )
        if ncf == "ensured":
            if validated_w == 0.0:
                raise ValidationError(
                    "w=0.0 with ncf='ensured' is not allowed. Use w > 0.0 "
                    "(recommended w >= 0.1).",
                    details={"w": validated_w, "ncf": ncf},
                )
            if validated_w < 0.1:
                warnings.warn(
                    f"ncf='ensured' with w={validated_w} (near 0) may produce unstable reject "
                    "behavior. Consider w >= 0.1.",
                    UserWarning,
                    stacklevel=2,
                )

        self.explainer.reject_threshold = None
        self.explainer.reject_ncf = ncf
        self.explainer.reject_ncf_w = canonical_reject_ncf_w(ncf, validated_w)
        self.explainer.reject_ncf_auto_selected = not ncf_explicit
        default_kind = _default_ncf_kind(
            bool(self.explainer.is_multiclass())  # pylint: disable=protected-access
        )

        if self.explainer.mode == "regression":
            if threshold is None:
                raise ValidationError("reject learner unavailable for regression without threshold")
            proba_1, _, _, _ = self.explainer.interval_learner.predict_probability(
                x_cal, y_threshold=threshold, bins=bins_cal
            )
            proba = np.array([[1 - proba_1[i], proba_1[i]] for i in range(len(proba_1))])
            calibration_bins = (y_cal < threshold).astype(int)
            self.explainer.reject_threshold = threshold
        elif self.explainer.is_multiclass():  # pylint: disable=protected-access
            proba, predicted_labels = self.explainer.interval_learner.predict_proba(
                x_cal, bins=bins_cal
            )
            proba = np.array(
                [[1 - proba[i, c], proba[i, c]] for i, c in enumerate(predicted_labels)]
            )
            calibration_bins = (predicted_labels == y_cal).astype(int)
        else:
            proba = self.explainer.interval_learner.predict_proba(x_cal, bins=bins_cal)
            calibration_bins = y_cal

        effective_w = canonical_reject_ncf_w(ncf, validated_w)
        alphas_cal = _ncf_scores_cal(
            proba, np.unique(calibration_bins), calibration_bins, ncf, effective_w, default_kind
        )
        self.explainer.reject_learner = ConformalClassifier().fit(alphas=alphas_cal, bins=bins_cal)
        _ = ncf_explicit  # used above; suppress unused-variable warning
        return self.explainer.reject_learner

    def _compute_prediction_set(
        self, x, bins=None, confidence: float = 0.95, threshold=None
    ) -> tuple[np.ndarray, float]:
        confidence = validate_reject_confidence(confidence)
        if bins is not None:
            bins = np.asarray(bins)

        if self.explainer.mode == "regression":
            if threshold is None:
                raise ValidationError("reject learner unavailable for regression without threshold")
            proba_1, _, _, _ = self.explainer.interval_learner.predict_probability(
                x, y_threshold=threshold, bins=bins
            )
            proba = np.array([[1 - proba_1[i], proba_1[i]] for i in range(len(proba_1))])
        elif self.explainer.is_multiclass():  # pylint: disable=protected-access
            proba, predicted_labels = self.explainer.interval_learner.predict_proba(x, bins=bins)
            proba = np.array(
                [[1 - proba[i, c], proba[i, c]] for i, c in enumerate(predicted_labels)]
            )
        else:
            proba = self.explainer.interval_learner.predict_proba(x, bins=bins)

        ncf = getattr(self.explainer, "reject_ncf", "default")
        ncf_w = getattr(self.explainer, "reject_ncf_w", 1.0)
        default_kind = _default_ncf_kind(
            bool(self.explainer.is_multiclass())  # pylint: disable=protected-access
        )
        alphas_test = np.asarray(_ncf_scores_test(proba, ncf, ncf_w, default_kind))

        seed = getattr(self.explainer, "seed", None)
        epsilon = 1 - confidence

        prediction_set = None
        # Preferred: compute p-values once (deterministic) and threshold at epsilon.
        # This guarantees prediction sets are nested as confidence increases.
        if hasattr(self.explainer.reject_learner, "predict_p"):
            predict_p_kwargs = {
                "bins": bins,
                "all_classes": True,
                "smoothing": False,
                "seed": seed,
            }
            predict_p_kwargs = self._filter_kwargs(
                self.explainer.reject_learner.predict_p, predict_p_kwargs
            )
            try:
                p_values = self.explainer.reject_learner.predict_p(alphas_test, **predict_p_kwargs)
                p_values = np.asarray(p_values, dtype=float)
                prediction_set = p_values > epsilon
            except Exception as exc:  # adr002_allow
                self._logger.info(
                    "Reject predict_p failed; falling back to predict_set.",
                    exc_info=True,
                )
                warnings.warn(
                    (
                        "Reject prediction fallback engaged: predict_p failed "
                        f"({exc!s}); using predict_set."
                    ),
                    UserWarning,
                    stacklevel=2,
                )

        # Fallback: use predict_set directly but force smoothing=False for determinism.
        if prediction_set is None:
            # Provide a classes_array hint for conformal implementations
            # that expect an explicit classes argument (tests use a Dummy
            # Conformal that requires `classes_array`). Derive from the
            # probability matrix where possible.
            classes_array = None
            try:
                if hasattr(proba, "ndim") and proba.ndim == 2:
                    classes_array = np.arange(proba.shape[1])
            except Exception:  # adr002_allow - defensive fallback for classes derivation
                classes_array = None

            predict_set_kwargs = {
                "bins": bins,
                "confidence": confidence,
                "smoothing": False,
                "seed": seed,
                "classes_array": classes_array,
            }
            predict_set_kwargs = self._filter_kwargs(
                self.explainer.reject_learner.predict_set, predict_set_kwargs
            )
            try:
                prediction_set = self.explainer.reject_learner.predict_set(
                    alphas_test, **predict_set_kwargs
                )
                prediction_set = np.asarray(prediction_set, dtype=bool)
            except Exception as exc:  # adr002_allow
                self._logger.info(
                    "Reject predict_set bulk call failed; using per-instance fallback.",
                    exc_info=True,
                )
                warnings.warn(
                    (
                        "Reject prediction fallback engaged: bulk predict_set failed "
                        f"({exc!s}); using per-instance calls."
                    ),
                    UserWarning,
                    stacklevel=2,
                )
                prediction_set = None

        expected_rows = len(alphas_test)
        if (
            prediction_set is None
            or prediction_set.ndim != 2
            or prediction_set.shape[0] != expected_rows
        ):
            if prediction_set is not None:
                self._logger.info(
                    "Reject predict_set returned unexpected shape %s; using per-instance fallback.",
                    getattr(prediction_set, "shape", None),
                )
                warnings.warn(
                    (
                        "Reject prediction fallback engaged: predict_set returned unexpected "
                        "shape; using per-instance calls."
                    ),
                    UserWarning,
                    stacklevel=2,
                )
            collected: list[np.ndarray] = []
            for i in range(expected_rows):
                per_bins = None
                if bins is not None and np.ndim(bins) == 1 and len(bins) == expected_rows:
                    per_bins = bins[i]
                if hasattr(self.explainer.reject_learner, "predict_p"):
                    kwargs_i = {
                        "bins": per_bins,
                        "all_classes": True,
                        "smoothing": False,
                        "seed": seed,
                    }
                    kwargs_i = self._filter_kwargs(
                        self.explainer.reject_learner.predict_p, kwargs_i
                    )
                    per_p = self.explainer.reject_learner.predict_p(
                        alphas_test[i : i + 1],
                        **kwargs_i,
                    )
                    per_p = np.asarray(per_p, dtype=float).reshape(-1)
                    collected.append((per_p > epsilon).astype(bool))
                else:
                    # Ensure per-instance calls also receive a classes_array
                    # hint when the conformal implementation expects it.
                    kwargs_i = {
                        "bins": per_bins,
                        "confidence": confidence,
                        "smoothing": False,
                        "seed": seed,
                        "classes_array": classes_array,
                    }
                    kwargs_i = self._filter_kwargs(
                        self.explainer.reject_learner.predict_set, kwargs_i
                    )
                    per_set = self.explainer.reject_learner.predict_set(
                        alphas_test[i : i + 1],
                        **kwargs_i,
                    )
                    collected.append(np.asarray(per_set, dtype=bool).reshape(-1))
            prediction_set = np.vstack(collected).astype(bool)

        return prediction_set.astype(bool), float(epsilon)

    def predict_reject_breakdown(
        self, x, bins=None, confidence: float = 0.95, threshold=None
    ) -> dict[str, Any]:
        """Return reject decision plus ambiguity/novelty breakdown.

        Notes
        -----
        For nested conformal prediction sets, as confidence increases, the
        *ambiguity* rate (multi-label sets) is non-decreasing while the
        *novelty* rate (empty sets) is non-increasing.
        """
        confidence = validate_reject_confidence(confidence)
        # Backwards compatibility: if a subclass has overridden the legacy
        # `predict_reject` method (tests and some mocks do this), prefer its
        # lightweight result shape. This keeps the mocking pattern in unit
        # tests working without requiring a full ConformalClassifier.
        legacy_predict = getattr(type(self), "predict_reject", None)
        if legacy_predict is not None and legacy_predict is not RejectOrchestrator.predict_reject:
            try:
                try:
                    legacy_res = self.predict_reject(
                        x, bins=bins, confidence=confidence, threshold=threshold
                    )
                except TypeError:
                    # Backward compatibility for legacy overrides that have not
                    # adopted the new threshold parameter.
                    legacy_res = self.predict_reject(x, bins=bins, confidence=confidence)
                # Expected legacy shape: (rejected_array, error_rate, reject_rate)
                if isinstance(legacy_res, tuple) and len(legacy_res) >= 1:
                    rejected = np.asarray(legacy_res[0], dtype=bool)
                    error_rate = legacy_res[1] if len(legacy_res) > 1 else None
                    reject_rate = (
                        legacy_res[2]
                        if len(legacy_res) > 2
                        else float(np.mean(rejected))
                        if len(rejected) > 0
                        else 0.0
                    )
                else:
                    # Fallback to full computation when legacy result is unexpected
                    raise ValidationError(
                        "legacy predict_reject returned unexpected shape",
                        details={"legacy_result_length": len(legacy_res)},
                    )

                # Build minimal breakdown from the boolean rejected mask.
                # We cannot infer true ambiguity/novelty without full prediction
                # sets, so represent rejected instances as "novelty" (empty set)
                # and non-rejected as singletons. This satisfies test expectations
                # around subset sizing and reject rates while remaining conservative.
                set_sizes = np.where(rejected, 0, 1)
                ambiguity = set_sizes >= 2
                novelty = set_sizes == 0
                num_instances = len(x)
                singleton = int(np.sum(set_sizes == 1))
                empty = int(np.sum(novelty))

                # preserve any reject_rate returned by the legacy method; fall
                # back to computed mean only when absent
                reject_rate = (
                    reject_rate
                    if "reject_rate" in locals() and reject_rate is not None
                    else (0.0 if num_instances == 0 else float(np.mean(rejected)))
                )
                ambiguity_rate = 0.0 if num_instances == 0 else float(np.mean(ambiguity))
                novelty_rate = 0.0 if num_instances == 0 else float(np.mean(novelty))

                # Clamp error_rate from legacy override to [0, 1]; it may be
                # None (undefined) when the legacy method could not compute it.
                legacy_error_rate_defined = error_rate is not None
                if legacy_error_rate_defined and isinstance(error_rate, (int, float)):
                    error_rate = max(0.0, min(1.0, float(error_rate)))
                else:
                    error_rate = 0.0
                    legacy_error_rate_defined = False

                return {
                    "rejected": rejected,
                    "ambiguity": ambiguity,
                    "novelty": novelty,
                    "prediction_set_size": set_sizes,
                    "reject_rate": reject_rate,
                    "ambiguity_rate": ambiguity_rate,
                    "novelty_rate": novelty_rate,
                    "error_rate": error_rate,
                    "error_rate_defined": legacy_error_rate_defined,
                    "epsilon": 1.0 - float(confidence),
                    "raw_total_examples": int(len(rejected)),
                    "raw_reject_counts": {
                        "rejected": int(np.sum(rejected)),
                        "ambiguity_mask": int(np.sum(ambiguity)),
                        "novelty_mask": int(np.sum(novelty)),
                        "prediction_set_size": int(np.sum(set_sizes)),
                    },
                }
            except Exception as exc:  # adr002_allow - graceful fallback for legacy override
                # If the legacy override misbehaves, fall through to full computation
                self._logger.debug(
                    "Legacy predict_reject override misbehaved: %s", exc, exc_info=True
                )

        prediction_set, epsilon = self._compute_prediction_set(
            x, bins=bins, confidence=confidence, threshold=threshold
        )
        set_sizes = np.sum(prediction_set, axis=1)
        rejected = set_sizes != 1
        ambiguity = set_sizes >= 2
        novelty = set_sizes == 0

        num_instances = len(x)
        singleton = int(np.sum(set_sizes == 1))
        empty = int(np.sum(novelty))
        # When there are no singleton prediction sets (all empty or ambiguous),
        # fall back to a numeric sentinel (0.0) rather than None so callers
        # expecting a numeric error_rate do not error on np.isnan checks.
        # error_rate_defined=False signals the value is a sentinel, not a
        # meaningful estimate (e.g. all instances were rejected as novel/ambiguous).
        if num_instances == 0 or singleton == 0:
            error_rate = 0.0
            error_rate_defined = False
        else:
            # Clamp to [0, 1]: the formula can go negative when empty > n*epsilon
            # (high novelty rate with small epsilon), which is not a valid rate.
            error_rate = max(0.0, min(1.0, (num_instances * epsilon - empty) / singleton))
            error_rate_defined = True

        reject_rate = 0.0 if num_instances == 0 else float(np.mean(rejected))
        ambiguity_rate = 0.0 if num_instances == 0 else float(np.mean(ambiguity))
        novelty_rate = 0.0 if num_instances == 0 else float(np.mean(novelty))

        return {
            "rejected": rejected,
            "ambiguity": ambiguity,
            "novelty": novelty,
            "prediction_set_size": set_sizes,
            "prediction_set": prediction_set,
            "reject_rate": reject_rate,
            "ambiguity_rate": ambiguity_rate,
            "novelty_rate": novelty_rate,
            "error_rate": error_rate,
            "error_rate_defined": error_rate_defined,
            "epsilon": epsilon,
            "raw_total_examples": int(num_instances),
            "raw_reject_counts": {
                "rejected": int(np.sum(rejected)),
                "ambiguity_mask": int(np.sum(ambiguity)),
                "novelty_mask": int(np.sum(novelty)),
                "prediction_set_size": int(np.sum(set_sizes)),
            },
        }

    def predict_reject(self, x, bins=None, confidence=0.95, threshold=None):
        """Predict whether to reject the explanations for the test data."""
        breakdown = self.predict_reject_breakdown(
            x, bins=bins, confidence=confidence, threshold=threshold
        )
        return breakdown["rejected"], breakdown["error_rate"], breakdown["reject_rate"]

    def apply_policy(
        self,
        policy: RejectPolicy,
        x,
        explain_fn=None,
        bins=None,
        confidence=0.95,
        threshold=None,
        **kwargs,
    ):
        """Apply a `RejectPolicy` to inputs and optionally produce predictions/explanations.

        Parameters
        ----------
        policy : RejectPolicy
            Selected by the caller.
        x :
            Input instances.
        explain_fn : callable, optional
            Callable `explain_fn(x_subset, **kwargs)` returning explanations.
        bins :
            Passed to reject prediction.
        confidence : float, default 0.95
            Passed to reject prediction.

        Returns
        -------
        RejectResult
            Envelope with `prediction`, `explanation`, `rejected`, `policy`, and `metadata`.
        """
        confidence = validate_reject_confidence(confidence)
        # Allow callers to select a strategy identifier via the `strategy` kwarg.
        # By default, resolve to `builtin.default` which preserves legacy semantics.
        strategy_name = kwargs.pop("strategy", None)
        strategy = self.resolve_strategy(strategy_name)
        return strategy(
            policy,
            x,
            explain_fn=explain_fn,
            bins=bins,
            confidence=confidence,
            threshold=threshold,
            **kwargs,
        )

    # --- Registry helpers -------------------------------------------------
    def register_strategy(self, name: str, fn: Any) -> None:
        """Register a reject strategy callable under *name*.

        The callable must accept the same parameters as `apply_policy` and
        return a `RejectResult`.
        """
        if not isinstance(name, str) or not name:
            raise ValidationError("strategy name must be a non-empty string")
        if not callable(fn):
            raise ValidationError("strategy must be callable")
        with self._strategies_lock:
            self._strategies[name] = fn

    def resolve_strategy(self, identifier: str | None):
        """Resolve a registered strategy by identifier.

        If *identifier* is None, return the builtin default strategy.
        Raises KeyError when an unknown identifier is requested.
        """
        with self._strategies_lock:
            if identifier is None:
                # default fallback
                try:
                    return self._strategies["builtin.default"]
                except KeyError as exc:
                    raise KeyError("builtin.default strategy not registered") from exc
            if identifier in self._strategies:
                return self._strategies[identifier]
            raise KeyError(f"Reject strategy '{identifier}' is not registered")

    # --- Builtin strategy (preserve previous apply_policy impl) -----------
    def _builtin_strategy(
        self,
        policy: RejectPolicy,
        x,
        explain_fn=None,
        bins=None,
        confidence=0.95,
        threshold=None,
        **kwargs,
    ):
        """Builtin strategy that preserves the previous `apply_policy` semantics."""
        confidence = validate_reject_confidence(confidence)
        try:
            policy = RejectPolicy(policy)
        except Exception:  # adr002_allow
            policy = RejectPolicy.NONE

        # If NONE, return a simple envelope indicating no action
        if policy is RejectPolicy.NONE:
            return RejectResult(
                prediction=None, explanation=None, rejected=None, policy=policy, metadata=None
            )

        effective_threshold = None
        threshold_source = None
        try:
            explainer_mode = getattr(self.explainer, "mode", "classification")
            if explainer_mode == "regression":
                effective_threshold, threshold_source = (
                    self._resolve_regression_effective_threshold(threshold)
                )
            elif getattr(self.explainer, "reject_learner", None) is None:
                self.initialize_reject_learner()
        except ValidationError:
            raise
        except Exception as exc:  # adr002_allow
            self._logger.info(
                "Reject learner init failed; returning empty RejectResult.",
                exc_info=True,
            )
            warnings.warn(
                f"Reject initialization failed; reject policy will not run ({exc!s}).",
                UserWarning,
                stacklevel=2,
            )
            return RejectResult(
                prediction=None,
                explanation=None,
                rejected=None,
                policy=policy,
                metadata={"init_error": True, "error_message": str(exc)},
            )

        breakdown = self.predict_reject_breakdown(
            x,
            bins=bins,
            confidence=confidence,
            threshold=effective_threshold,
        )
        rejected = breakdown["rejected"]
        error_rate = breakdown["error_rate"]
        reject_rate = breakdown["reject_rate"]

        prediction = None
        explanation = None

        # Obtain predictions when requested by policy (all non-NONE policies)
        if policy in (
            RejectPolicy.FLAG,
            RejectPolicy.ONLY_REJECTED,
            RejectPolicy.ONLY_ACCEPTED,
        ):
            try:
                # Use the prediction orchestrator directly. The explainer's
                # public predict method is a thin facade over the same
                # orchestrator and does not own reject orchestration.
                prediction_kwargs = dict(kwargs)
                if getattr(self.explainer, "mode", "classification") == "regression":
                    prediction_kwargs["threshold"] = effective_threshold
                prediction = self.explainer.prediction_orchestrator.predict(x, **prediction_kwargs)
            except Exception as exc:  # adr002_allow
                self._logger.info(
                    "Reject policy prediction failed; returning prediction=None.",
                    exc_info=True,
                )
                warnings.warn(
                    f"Reject policy prediction failed; returning prediction=None ({exc!s}).",
                    UserWarning,
                    stacklevel=2,
                )
                prediction = None

        # matched_count records how many instances matched the policy filter;
        # None for FLAG (all instances processed), 0 when subset was empty.
        matched_count = None
        source_indices: list[int] = list(range(len(rejected)))
        if policy is RejectPolicy.ONLY_REJECTED:
            source_indices = [i for i, r in enumerate(rejected) if r]
        elif policy is RejectPolicy.ONLY_ACCEPTED:
            source_indices = [i for i, r in enumerate(rejected) if not r]

        # Obtain explanations via provided callable according to policy
        if explain_fn is not None:
            try:
                if policy is RejectPolicy.FLAG:
                    # Process all instances, tag rejection status
                    explanation = explain_fn(x, **kwargs)
                elif policy is RejectPolicy.ONLY_REJECTED:
                    # Process only rejected instances
                    idx = source_indices
                    matched_count = len(idx)
                    if idx:
                        subset = (
                            np.asarray(x)[idx] if isinstance(x, np.ndarray) else [x[i] for i in idx]
                        )
                        explanation = explain_fn(subset, **kwargs)
                    else:
                        explanation = None
                elif policy is RejectPolicy.ONLY_ACCEPTED:
                    # Process only non-rejected (accepted) instances
                    idx = source_indices
                    matched_count = len(idx)
                    if idx:
                        subset = (
                            np.asarray(x)[idx] if isinstance(x, np.ndarray) else [x[i] for i in idx]
                        )
                        explanation = explain_fn(subset, **kwargs)
                    else:
                        explanation = None
            except Exception as exc:  # adr002_allow
                self._logger.info(
                    "Reject policy explanation failed; returning explanation=None.",
                    exc_info=True,
                )
                warnings.warn(
                    f"Reject policy explanation failed; returning explanation=None ({exc!s}).",
                    UserWarning,
                    stacklevel=2,
                )
                explanation = None
                if kwargs.get("_reject_raise", False):
                    raise

        metadata = {
            "error_rate": error_rate,
            "error_rate_defined": breakdown.get("error_rate_defined", True),
            "reject_rate": reject_rate,
            "ambiguity_rate": breakdown.get("ambiguity_rate"),
            "novelty_rate": breakdown.get("novelty_rate"),
            # Per-instance breakdown so callers can inspect ambiguity vs novelty
            "ambiguity_mask": breakdown.get("ambiguity"),
            "novelty_mask": breakdown.get("novelty"),
            "prediction_set_size": breakdown.get("prediction_set_size"),
            "prediction_set": breakdown.get("prediction_set"),
            "epsilon": breakdown.get("epsilon"),
            "raw_total_examples": breakdown.get("raw_total_examples"),
            "raw_reject_counts": breakdown.get("raw_reject_counts"),
            # NCF provenance: which function was used and whether it was auto-selected
            "reject_ncf": getattr(self.explainer, "reject_ncf", None),
            "reject_ncf_w": getattr(self.explainer, "reject_ncf_w", None),
            "reject_ncf_auto_selected": getattr(self.explainer, "reject_ncf_auto_selected", None),
            # How many instances matched the policy filter (None for FLAG, 0 when empty)
            "matched_count": matched_count,
            "source_indices": source_indices,
            "original_count": int(len(rejected)),
            "effective_confidence": confidence,
            "effective_threshold": effective_threshold,
            "threshold_source": threshold_source,
            "effective_w": validate_reject_w(
                getattr(self.explainer, "reject_ncf_w", 0.0)
                if getattr(self.explainer, "reject_ncf_w", None) is not None
                else 0.0
            ),
        }
        return RejectResult(
            prediction=prediction,
            explanation=explanation,
            rejected=rejected,
            policy=policy,
            metadata=metadata,
        )
