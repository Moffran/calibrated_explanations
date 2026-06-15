"""Call-time tuning options for the guarded explanation API (ADR-038)."""

from __future__ import annotations

from dataclasses import dataclass

from ..utils.exceptions import ValidationError


@dataclass(frozen=True)
class GuardedOptions:
    """Per-call tuning for the KNN-based in-distribution guard (ADR-038).

    Parameters
    ----------
    confidence : float, default=0.9
        Conformity coverage target. A candidate representative is accepted when its
        guard p-value is ``>= 1 - confidence``. Equivalent to the legacy
        ``significance = 1 - confidence`` convention; the value is inverted so that
        the convention matches ``reject_confidence`` (higher is stricter coverage,
        not stricter rejection).
        Must be in the open interval ``(0, 1)``.
    n_neighbors : int, default=5
        Number of nearest neighbours used by :class:`~...utils.distribution_guard.InDistributionGuard`.
    normalize : bool, default=True
        Apply per-feature normalisation inside the guard.
    merge_adjacent : bool, default=False
        When True, merge adjacent conforming bins into wider interval conditions.
        Merged representatives are re-tested; merges that fail conformity are skipped.
    verbose : bool, default=False
        When True, emit ``UserWarning`` messages for guarded-explanation diagnostics
        (e.g. empty bins rejected unconditionally).

    Notes
    -----
    ``GuardedOptions(confidence=0.9)`` is the exact replacement for the legacy
    ``significance=0.1`` parameter. The value is inverted:
    ``confidence = 1 - significance``.

    Examples
    --------
    >>> from calibrated_explanations import GuardedOptions
    >>> opts = GuardedOptions(confidence=0.9)
    >>> opts.confidence
    0.9
    """

    confidence: float = 0.9
    n_neighbors: int = 5
    normalize: bool = True
    merge_adjacent: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate field ranges after construction."""
        if not (0.0 < self.confidence < 1.0):
            raise ValidationError(
                "GuardedOptions.confidence must be in the open interval (0, 1).",
                details={"confidence": self.confidence},
            )
        if self.n_neighbors < 1:
            raise ValidationError(
                "GuardedOptions.n_neighbors must be >= 1.",
                details={"n_neighbors": self.n_neighbors},
            )
