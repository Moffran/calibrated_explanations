# pylint: disable=unknown-option-value, too-many-arguments
# pylint: disable=too-many-lines, too-many-public-methods, invalid-name, too-many-positional-arguments, line-too-long
"""Containers for storing, exporting, and visualising calibrated explanations."""

from __future__ import annotations

import warnings
from copy import copy, deepcopy
from dataclasses import dataclass
from time import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np

from ..utils.discretizers import EntropyDiscretizer, RegressorDiscretizer
from ..utils.helper import prepare_for_saving
from .adapters import legacy_to_domain
from .explanation import AlternativeExplanation, FactualExplanation, FastExplanation
from .models import Explanation as DomainExplanation


@dataclass(frozen=True)
class ExportedExplanationCollection:
    """Lightweight representation of exported explanations plus collection metadata."""

    metadata: Mapping[str, Any]
    explanations: Sequence[DomainExplanation]


def _jsonify(value: Any) -> Any:
    """Convert numpy objects and arrays into JSON-serialisable primitives."""
    if isinstance(value, np.ndarray):
        return [_jsonify(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, np.generic):  # numpy scalars
        return value.item()
    return value


class CalibratedExplanations:  # pylint: disable=too-many-instance-attributes
    """A class for storing and visualizing calibrated explanations.

    This class is created by :class:`.CalibratedExplainer` and provides methods for managing
    and accessing explanations for test instances.
    """

    def __init__(self, calibrated_explainer, x, y_threshold, bins, features_to_ignore=None) -> None:
        """Initialize the explanation collection for a calibrated explainer.

        Parameters
        ----------
        calibrated_explainer : CalibratedExplainer
            The calibrated explainer object.
        x : array-like
            The test data.
        y_threshold : float or tuple
            The threshold for regression explanations.
        bins : array-like
            The bins for conditional explanations.
        """
        self.calibrated_explainer: FrozenCalibratedExplainer = FrozenCalibratedExplainer(
            calibrated_explainer
        )
        self.x_test: np.ndarray = x
        self.y_threshold: Optional[Union[float, Tuple[float, float], List[Tuple[float, float]]]] = (
            y_threshold
        )
        self.low_high_percentiles: Optional[Tuple[float, float]] = None
        self.explanations: List[
            Union[FactualExplanation, AlternativeExplanation, FastExplanation]
        ] = []
        self.start_index: int = 0
        self.current_index: int = self.start_index
        self.end_index: int = len(x[:, 0])
        self.bins: Optional[Sequence[Any]] = bins
        self.total_explain_time: Optional[float] = None
        self.features_to_ignore: List[int] = (
            features_to_ignore if features_to_ignore is not None else []
        )
        # Derived caches (set during finalize of individual explanations)
        self._feature_names_cache: Optional[Sequence[str]] = None  # populated lazily
        self._predictions_cache: Optional[np.ndarray] = None
        self._probabilities_cache: Optional[np.ndarray] = None  # classification only
        self._lower_cache: Optional[np.ndarray] = None  # regression only
        self._upper_cache: Optional[np.ndarray] = None
        self._class_labels_cache: Optional[Sequence[str]] = None  # classification only

    def __iter__(self):
        """Return an iterator for the explanations."""
        self.current_index = self.start_index
        return self

    def __next__(self):
        """Return the next explanation."""
        if self.current_index >= self.end_index:
            raise StopIteration
        result = self[self.current_index]
        self.current_index += 1
        return result

    def __len__(self):
        """Return the number of explanations."""
        return len(self.x_test[:, 0])

    def build_rules_payload(self) -> List[Dict[str, Any]]:
        """Delegate payload materialisation to each stored explanation."""
        return [exp.build_rules_payload() for exp in self.explanations]

    def __getitem__(self, key: Union[int, slice, List[int], List[bool], np.ndarray]):
        """Return the explanation for the given key.

        In case the index key is an integer (or results in a single result), the function returns the explanation
        corresponding to the index. If the key is a slice or an integer or boolean list (or numpy array)
        resulting in more than one explanation, the function returns a new `CalibratedExplanations`
        object with the indexed explanations.
        """
        if isinstance(key, int):
            # Handle single item access
            return self.explanations[key]
        if isinstance(key, (slice, list, np.ndarray)):
            new_ = copy(self)
            if isinstance(key, slice):
                # Handle slicing
                new_.explanations = list(self.explanations[key])
            if isinstance(key, (list, np.ndarray)):
                if isinstance(key[0], (bool, np.bool_)):
                    # Handle boolean indexing
                    new_.explanations = [
                        exp for exp, include in zip(self.explanations, key) if include
                    ]
                elif isinstance(key[0], int):
                    # Handle integer list indexing
                    new_.explanations = [self.explanations[i] for i in key]
            if len(new_.explanations) == 1:
                return new_.explanations[0]
            new_.start_index = 0
            new_.current_index = new_.start_index
            new_.end_index = len(new_.explanations)
            new_.bins = None if self.bins is None else [self.bins[e.index] for e in new_]
            new_.x_test = np.array([self.x_test[e.index, :] for e in new_])
            if self.y_threshold is None:
                new_.y_threshold = None
            elif isinstance(self.y_threshold, (int, float)):
                new_.y_threshold = float(self.y_threshold)
            elif isinstance(self.y_threshold, tuple):
                new_.y_threshold = self.y_threshold
            else:
                # assume list of tuples aligned with instances
                new_.y_threshold = [self.y_threshold[e.index] for e in new_]
            # Reset cached aggregates to avoid referencing stale state from the source
            new_._feature_names_cache = None
            new_._predictions_cache = None
            new_._probabilities_cache = None
            new_._lower_cache = None
            new_._upper_cache = None
            new_._class_labels_cache = None
            for i, e in enumerate(new_):
                e.index = i
            return new_
        raise TypeError("Invalid argument type.")

    def __repr__(self) -> str:
        """Return the string representation of the CalibratedExplanations object."""
        explanations_str = "\n".join([str(e) for e in self.explanations])
        return f"CalibratedExplanations({len(self)} explanations):\n{explanations_str}"

    def __str__(self) -> str:
        """Return the string representation of the CalibratedExplanations object."""
        return self.__repr__()

    # ------------------------------------------------------------------
    # Plugin bridge helpers
    # ------------------------------------------------------------------

    def to_batch(self):
        """Serialise the collection into an :class:`ExplanationBatch`."""
        from ..plugins.builtins import _collection_to_batch  # lazy import

        return _collection_to_batch(self)

    @classmethod
    def from_batch(cls, batch):
        """Reconstruct a collection from an :class:`ExplanationBatch`."""
        from ..core.exceptions import SerializationError, ValidationError

        container = batch.collection_metadata.get("container")
        if container is None:
            raise SerializationError(
                "ExplanationBatch is missing container metadata",
                details={"artifact": "ExplanationBatch", "field": "container"},
            )
        if not isinstance(container, cls):
            raise ValidationError(
                "ExplanationBatch container metadata has unexpected type",
                details={
                    "param": "container",
                    "expected_type": cls.__name__,
                    "actual_type": type(container).__name__,
                },
            )
        return container

    # ------------------------------------------------------------------
    # JSON export helpers (schema v1 wrappers)
    # ------------------------------------------------------------------

    def to_json(self, *, include_version: bool = True) -> Mapping[str, Any]:
        """Return a JSON-friendly payload describing this collection.

        The payload wraps each explanation using the schema v1 helpers from
        :mod:`calibrated_explanations.serialization` and adds collection-level
        metadata (mode, thresholds, feature names, telemetry snapshot).

        Parameters
        ----------
        include_version:
            When ``True`` (default) the ``schema_version`` field is included on
            the top-level payload as well as on each explanation entry.
        """
        from ..serialization import to_json as _explanation_to_json

        instances = []
        for exp in self.explanations:
            domain = legacy_to_domain(exp.index, self._legacy_payload(exp))
            provenance = getattr(exp, "provenance", None)
            metadata = getattr(exp, "metadata", None)
            if provenance is not None:
                domain.provenance = cast(Optional[Mapping[str, Any]], _jsonify(provenance))
            if metadata is not None:
                domain.metadata = cast(Optional[Mapping[str, Any]], _jsonify(metadata))
            instances.append(_explanation_to_json(domain, include_version=include_version))

        payload: dict[str, Any] = {
            "collection": self._collection_metadata(),
            "explanations": instances,
        }
        if include_version:
            payload.setdefault("schema_version", "1.0.0")
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> ExportedExplanationCollection:
        """Materialise domain explanations from a :meth:`to_json` payload."""
        from ..serialization import from_json as _explanation_from_json

        explanations_blob = payload.get("explanations", []) or []
        domain: list[DomainExplanation] = []
        for item in explanations_blob:
            domain.append(_explanation_from_json(item))
        metadata = payload.get("collection", {}) or {}
        return ExportedExplanationCollection(
            metadata=cast(Mapping[str, Any], _jsonify(metadata)),
            explanations=tuple(domain),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _legacy_payload(self, exp) -> Mapping[str, Any]:
        """Build a legacy-shaped payload from an explanation instance."""
        rules_blob = None
        # prefer conjunctive rules when present and populated
        if getattr(exp, "_has_conjunctive_rules", False):
            rules_blob = getattr(exp, "conjunctive_rules", None)
        if not rules_blob:
            rules_blob = getattr(exp, "rules", None)
        if not rules_blob and hasattr(exp, "_get_rules"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    rules_blob = exp._get_rules()  # type: ignore[attr-defined]
                except Exception:  # pragma: no cover - defensive
                    rules_blob = {}

        payload: dict[str, Any] = {
            "task": getattr(
                exp, "get_mode", lambda: getattr(self.calibrated_explainer, "mode", None)
            )(),
            "rules": _jsonify(rules_blob or {}),
            "feature_weights": _jsonify(getattr(exp, "feature_weights", {})),
            "feature_predict": _jsonify(getattr(exp, "feature_predict", {})),
            "prediction": _jsonify(getattr(exp, "prediction", {})),
        }
        return payload

    def _collection_metadata(self) -> Mapping[str, Any]:
        """Collect calibration metadata required to interpret the payload."""
        base = getattr(self, "calibrated_explainer", None)
        underlying = getattr(base, "_explainer", None)

        feature_names = None
        try:
            names = self.feature_names
            if names is not None:
                feature_names = list(names)
        except Exception:  # pragma: no cover - defensive
            feature_names = None

        class_labels = None
        if hasattr(base, "class_labels"):
            try:
                class_labels = _jsonify(base.class_labels)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                class_labels = None

        sample_percentiles = None
        if hasattr(base, "sample_percentiles"):
            try:
                sample_percentiles = _jsonify(base.sample_percentiles)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                sample_percentiles = None

        runtime_telemetry = None
        if underlying is not None:
            try:
                runtime_telemetry = getattr(underlying, "runtime_telemetry", None)
                if callable(runtime_telemetry):
                    runtime_telemetry = runtime_telemetry()
            except Exception:  # pragma: no cover - defensive
                runtime_telemetry = None

        metadata = {
            "size": len(self),
            "mode": getattr(base, "mode", None),
            "y_threshold": _jsonify(self.y_threshold),
            "low_high_percentiles": _jsonify(self.low_high_percentiles),
            "feature_names": _jsonify(feature_names),
            "class_labels": class_labels,
            "sample_percentiles": sample_percentiles,
            "runtime_telemetry": _jsonify(runtime_telemetry),
        }
        return {k: v for k, v in metadata.items() if v is not None}

    @property
    def prediction_interval(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """Return the prediction intervals for each explanation.

        Returns
        -------
        list of tuples
            A list of tuples containing (low, high) values of the prediction interval.
        """
        return [e.prediction_interval for e in self.explanations]

    @property
    def predict(self) -> List[Any]:
        """Return the scalar prediction for every explanation.

        Returns
        -------
        list
            A list of prediction value.
        """
        return [e.predict for e in self.explanations]

    # ---- Rich baseline exposure (Phase 1A golden snapshot enrichment) ----
    @property
    def feature_names(self):  # consistent naming with underlying explainer
        """Return cached feature names sourced from the underlying explainer."""
        if self._feature_names_cache is None:
            # Underlying FrozenCalibratedExplainer exposes feature_names via original explainer
            try:
                self._feature_names_cache = self.calibrated_explainer._explainer.feature_names  # noqa: SLF001
            except Exception:  # pragma: no cover - defensive
                self._feature_names_cache = None
        return self._feature_names_cache

    @property
    def class_labels(self):
        """Return class labels for classification explanations if available."""
        if self._class_labels_cache is None:
            try:
                labels = getattr(self.calibrated_explainer._explainer, "class_labels", None)  # noqa: SLF001
                if labels is not None and isinstance(labels, dict):
                    # normalize to list ordered by class index if dict provided
                    # assume keys are numeric class indices
                    labels = [labels[k] for k in sorted(labels.keys())]
                self._class_labels_cache = labels
            except Exception:  # pragma: no cover
                self._class_labels_cache = None
        return self._class_labels_cache

    @property
    def predictions(self):  # noqa: D401
        """Vector of scalar predictions for the explained instances (cached)."""
        if self._predictions_cache is None:
            try:
                self._predictions_cache = np.asarray([e.predict for e in self.explanations])
            except Exception:  # pragma: no cover
                self._predictions_cache = None
        return self._predictions_cache

    @property
    def probabilities(self):  # classification only
        """Return cached probability matrices for classification explanations."""
        if self._probabilities_cache is None:
            try:
                # Each explanation may store:
                #  (a) its own probability vector (shape (n_classes,)) OR
                #  (b) the full matrix (n_instances, n_classes) due to earlier enrichment
                raw = [getattr(e, "prediction_probabilities", None) for e in self.explanations]
                if all(r is not None for r in raw):
                    # If first is a tuple (should not now), handle defensively
                    first = raw[0]
                    if isinstance(first, tuple):  # pragma: no cover - defensive
                        first = first[0]
                    first = np.asarray(first)
                    if first.ndim == 2 and first.shape[0] == len(self.explanations):
                        # Case (b): each explanation redundantly holds full matrix
                        self._probabilities_cache = first
                    else:
                        # Case (a): stack per-instance vectors
                        self._probabilities_cache = np.vstack(raw)
            except Exception:  # pragma: no cover
                self._probabilities_cache = None
        return self._probabilities_cache

    @property
    def lower(self):  # regression only
        """Return cached lower bounds for regression prediction intervals."""
        if self._lower_cache is None:
            try:
                lows = [
                    getattr(e, "prediction_interval", (None, None))[0] for e in self.explanations
                ]
                if any(low is not None for low in lows):
                    self._lower_cache = np.asarray(lows)
            except Exception:  # pragma: no cover
                self._lower_cache = None
        return self._lower_cache

    @property
    def upper(self):  # regression only
        """Return cached upper bounds for regression prediction intervals."""
        if self._upper_cache is None:
            try:
                highs = [
                    getattr(e, "prediction_interval", (None, None))[1] for e in self.explanations
                ]
                if any(h is not None for h in highs):
                    self._upper_cache = np.asarray(highs)
            except Exception:  # pragma: no cover
                self._upper_cache = None
        return self._upper_cache

    def _is_probabilistic_regression(self) -> bool:
        """Check if the explanations use probabilistic regression (thresholded).

        Probabilistic regression and thresholded regression are synonymous terms.
        See ADR-021 for terminology guidance.
        """
        return self.y_threshold is not None

    def _is_thresholded(self) -> bool:
        """Use _is_probabilistic_regression() instead (deprecated).

        Backward-compatible alias kept until v0.10.0.
        This method is identical to _is_probabilistic_regression().
        """
        return self._is_probabilistic_regression()

    def _is_one_sided(self) -> bool:
        """Check if the explanations are one-sided."""
        if self.low_high_percentiles is None:
            return False
        return np.isinf(self.get_low_percentile()) or np.isinf(self.get_high_percentile())

    def get_confidence(self) -> float:
        """Return the confidence level of the explanations.

        This method calculates the confidence interval for regression tasks by determining the distance between the lower and upper percentiles. By default, these percentiles are set to 5 and 95.

        Returns
        -------
        float
            The difference between the high and low percentiles, representing the confidence interval.

        Notes
        -----
        - This method is only applicable to regression tasks.
        - If the high percentile is infinite, the confidence is calculated as `100 - low_percentile`.
        - If the low percentile is infinite, the confidence is calculated as `high_percentile`.
        """
        if np.isinf(self.get_high_percentile()):
            return 100 - self.get_low_percentile()
        if np.isinf(self.get_low_percentile()):
            return self.get_high_percentile()
        return self.get_high_percentile() - self.get_low_percentile()

    def get_low_percentile(self) -> float:
        """Return the low percentile of the explanations.

        This method returns the first element of the `low_high_percentiles` attribute,
        which represents the lower bound of the percentile range for the explanation.

        Returns
        -------
        float
            The low percentile value of the explanation.
        """
        # mypy: low_high_percentiles is Optional; ensure it's set by callers before use
        assert self.low_high_percentiles is not None, "low_high_percentiles not set"
        return self.low_high_percentiles[0]  # pylint: disable=unsubscriptable-object

    def get_high_percentile(self) -> float:
        """Return the high percentile of the explanations.

        Returns
        -------
        float
            The high percentile value of the explanation.
        """
        assert self.low_high_percentiles is not None, "low_high_percentiles not set"
        return self.low_high_percentiles[1]  # pylint: disable=unsubscriptable-object

    # pylint: disable=too-many-arguments
    def finalize(
        self,
        binned,
        feature_weights,
        feature_predict,
        prediction,
        instance_time=None,
        total_time=None,
    ) -> "CalibratedExplanations":
        """
        Finalize the explanation by adding the binned data and the feature weights.

        Parameters
        ----------
        binned : array-like
            The binned data for the features.
        feature_weights : array-like
            The weights of the features.
        feature_predict : array-like
            The predicted values for the features.
        prediction : array-like
            The prediction values.
        instance_time : array-like, optional
            The time taken to explain each instance, by default None.
        total_time : float, optional
            The total time taken to explain all instances, by default None.

        Returns
        -------
        self : object
            Returns the instance of the class with explanations finalized.
        """
        for i, instance in enumerate(self.x_test):
            instance_bin = self.bins[i] if self.bins is not None else None
            if self._is_alternative():
                explanation: Union[FactualExplanation, AlternativeExplanation, FastExplanation]
                explanation = AlternativeExplanation(
                    self,
                    i,
                    instance,
                    binned,
                    feature_weights,
                    feature_predict,
                    prediction,
                    self.y_threshold,
                    instance_bin=instance_bin,
                )
            else:
                explanation = FactualExplanation(
                    self,
                    i,
                    instance,
                    binned,
                    feature_weights,
                    feature_predict,
                    prediction,
                    self.y_threshold,
                    instance_bin=instance_bin,
                )
            explanation.explain_time = instance_time[i] if instance_time is not None else None
            self.explanations.append(explanation)
        self.total_explain_time = time() - total_time if total_time is not None else None
        if self._is_alternative():
            return self.__convert_to_alternative_explanations()
        return self

    def __convert_to_alternative_explanations(self) -> "AlternativeExplanations":
        """Return an ``AlternativeExplanations`` view sharing this collection's backing data."""
        alternative_explanations = AlternativeExplanations.__new__(AlternativeExplanations)
        alternative_explanations.__dict__.update(self.__dict__)
        return alternative_explanations

    # pylint: disable=too-many-arguments
    def finalize_fast(
        self, feature_weights, feature_predict, prediction, instance_time=None, total_time=None
    ) -> None:
        """
        Finalize the explanation by adding the binned data and the feature weights.

        Parameters
        ----------
        binned : array-like
            The binned data for the features.
        feature_weights : array-like
            The weights of the features.
        feature_predict : array-like
            The predicted values for the features.
        prediction : array-like
            The prediction values.
        instance_time : array-like, optional
            The time taken to explain each instance, by default None.
        total_time : float, optional
            The total time taken to explain all instances, by default None.

        Notes
        -----
        - This method iterates over the test instances and creates a `FastExplanation` object for each instance.
        - The `FastExplanation` object is initialized with the provided feature weights, predictions, and other relevant data.
        - The explanation time for each instance is recorded if `instance_time` is provided.
        - The total explanation time is calculated if `total_time` is provided.
        """
        for i, instance in enumerate(self.x_test):
            instance_bin = self.bins[i] if self.bins is not None else None
            explanation = FastExplanation(
                self,
                i,
                instance,
                feature_weights,
                feature_predict,
                prediction,
                self.y_threshold,
                instance_bin=instance_bin,
            )
            explanation.explain_time = instance_time[i] if instance_time is not None else None
            self.explanations.append(explanation)
        self.total_explain_time = time() - total_time if total_time is not None else None

    def _get_explainer(self):
        """Return the underlying :class:`~calibrated_explanations.core.calibrated_explainer.CalibratedExplainer` instance."""
        return self.calibrated_explainer

    def _get_rules(self):
        """Return the materialised rule payload for each explanation in the collection."""
        return [
            # pylint: disable=protected-access
            explanation._get_rules()
            for explanation in self.explanations
        ]

    def add_conjunctions(self, n_top_features=5, max_rule_size=2):
        """
        Add conjunctive rules to the explanations.

        The conjunctive rules are added to the `conjunctive_rules` attribute of the `CalibratedExplanations`
        object.

        Parameters
        ----------
        n_top_features : int, optional
            The number of most important factual rules to try to combine into conjunctive rules. Defaults to 5.
        max_rule_size : int, optional
            The maximum size of the conjunctions. Defaults to 2 (meaning `rule_one and rule_two`).

        Returns
        -------
        CalibratedExplanations
            Returns a self reference, to allow for method chaining.
        """
        for explanation in self.explanations:
            explanation.add_conjunctions(n_top_features, max_rule_size)
        return self

    def reset(self):
        """Reset the explanations to their original state."""
        for explanation in self.explanations:
            explanation.reset()
        return self

    def remove_conjunctions(self):
        """Remove any conjunctive rules."""
        for explanation in self.explanations:
            explanation.remove_conjunctions()
        return self

    def get_explanation(self, index):
        """Return the explanation corresponding to the index.

        Parameters
        ----------
        index : int
            The index of the explanation to retrieve.

        Returns
        -------
        CalibratedExplanation
            The explanation at the specified index.

        Warnings
        --------
        Deprecated: This method is deprecated and may be removed in future versions. Use indexing instead.
        """
        from ..core.exceptions import ValidationError
        from ..utils.deprecations import deprecate

        deprecate(
            "This method is deprecated and may be removed in future versions. Use indexing instead.",
            key="CalibratedExplanations.get_explanation",
            stacklevel=3,
        )
        if not isinstance(index, int):
            raise ValidationError(
                "index must be an integer",
                details={"param": "index", "expected_type": "int", "actual_type": type(index).__name__},
            )
        if index < 0:
            raise ValidationError(
                "index must be greater than or equal to 0",
                details={"param": "index", "value": index, "requirement": "non-negative"},
            )
        if index >= len(self.x_test):
            raise ValidationError(
                "index must be less than the number of test instances",
                details={
                    "param": "index",
                    "value": index,
                    "max_index": len(self.x_test) - 1,
                    "n_instances": len(self.x_test),
                },
            )
        return self.explanations[index]

    def _is_alternative(self):
        """Return True when the collection represents an alternative explanation workflow."""
        return isinstance(
            self.calibrated_explainer.discretizer, (RegressorDiscretizer, EntropyDiscretizer)
        )

    # pylint: disable=too-many-arguments, too-many-locals, unused-argument
    def plot(
        self,
        index=None,
        filter_top=10,
        show=True,
        filename="",
        uncertainty=False,
        style="regular",
        rnk_metric=None,
        rnk_weight=0.5,
        style_override=None,
        **kwargs,
    ):
        """Plot explanations for a given instance, with the option to show or save the plots.

        Parameters
        ----------
        index : int or None, default=None
            The index of the instance for which you want to plot the explanation. If None, the
            function will plot all the explanations.
        filter_top : int or None, default=10
            The number of top features to display in the plot. If set to `None`, all the
            features will be shown.
        show : bool, default=True
            Determines whether the plots should be displayed immediately after they are
            generated. Suitable to set to False when saving the plots to a file.
        filename : str, default=''
            The full path and filename of the plot image file that will be saved. If empty, the
            plot will not be saved.
        uncertainty : bool, default=False
            Determines whether to include uncertainty information in the plots.
        style : str, default='regular'
            The style of the plot. Supported styles are 'regular' and 'triangular'
            (experimental).
        rnk_metric : str, default=None
            The metric used to rank the features. Supported metrics are 'ensured',
            'feature_weight', and 'uncertainty'. If None, the default from the explanation
            class is used.
        rnk_weight : float, default=0.5
            The weight of the uncertainty in the ranking. Used with the 'ensured' ranking
            metric.

        Returns
        -------
        None

        See Also
        --------
        FactualExplanation.plot
            Refer to the docstring for plot in FactualExplanation for details on default
            ranking ('feature_weight').
        AlternativeExplanation.plot
            Refer to the docstring for plot in AlternativeExplanation for details on default
            ranking ('ensured').
        FastExplanation.plot
            Refer to the docstring for plot in FastExplanation for details on default ranking
            ('feature_weight').
        """
        if style == "narrative":
            from ..viz.narrative_plugin import NarrativePlotPlugin

            template_path = kwargs.pop("template_path", None)
            expertise_level = kwargs.pop(
                "expertise_level", ("beginner", "intermediate", "advanced")
            )
            output_format = kwargs.pop("output", "dataframe")

            plugin = NarrativePlotPlugin(template_path=template_path)

            if index is not None:
                # Delegate to single explanation helper when a specific index is requested
                return self[index].to_narrative(
                    template_path=template_path,
                    expertise_level=expertise_level,
                    output_format=output_format,
                    **kwargs,
                )

            return plugin.plot(
                self,
                template_path=template_path,
                expertise_level=expertise_level,
                output=output_format,
                **kwargs,
            )

        if len(filename) > 0:
            path, filename, title, ext = prepare_for_saving(filename)

        if index is not None:
            if len(filename) > 0:
                filename = path + title + str(index) + ext
            self[index].plot(
                filter_top=filter_top,
                show=show,
                filename=filename,
                uncertainty=uncertainty,
                style=style,
                rnk_metric=rnk_metric,
                rnk_weight=rnk_weight,
                style_override=style_override,
            )
        else:
            for i, explanation in enumerate(self.explanations):
                if len(filename) > 0:
                    filename = path + title + str(i) + ext
                explanation.plot(
                    filter_top=filter_top,
                    show=show,
                    filename=filename,
                    uncertainty=uncertainty,
                    style=style,
                    rnk_metric=rnk_metric,
                    rnk_weight=rnk_weight,
                    style_override=style_override,
                )

    def to_narrative(
        self,
        template_path="exp.yaml",
        expertise_level=("beginner", "advanced"),
        output_format="dataframe",
        **kwargs,
    ):
        """
        Generate narrative explanations for the collection.

        This method provides a clean API for generating human-readable narratives
        from calibrated explanations using customizable templates.

        Parameters
        ----------
        template_path : str, default="exp.yaml"
            Path to the narrative template file (YAML or JSON).
            If the file doesn't exist, the default template will be used.
        expertise_level : str or tuple of str, default=("beginner", "advanced")
            The expertise level(s) for narrative generation. Can be a single
            level or a tuple of levels. Valid values: "beginner", "intermediate", "advanced".
        output_format : str, default="dataframe"
            Output format. Valid values: "dataframe", "text", "html", "dict".
        **kwargs : dict
            Additional keyword arguments passed to the narrative plugin.

        Returns
        -------
        pd.DataFrame or str or list of dict
            The generated narratives in the requested format:
            - "dataframe": pandas DataFrame with columns for each expertise level
            - "text": formatted text string with all narratives
            - "html": HTML table with all narratives
            - "dict": list of dictionaries, one per instance

        Raises
        ------
        FileNotFoundError
            If the template file is not found and no default is available.
        ValueError
            If an invalid expertise level or output format is specified.
        ImportError
            If pandas is not available and output_format="dataframe" is requested.

        Examples
        --------
        >>> from calibrated_explanations import CalibratedExplainer
        >>> explainer = CalibratedExplainer(model, X_train, y_train)
        >>> explanations = explainer.explain_factual(X_test)
        >>> narratives = explanations.to_narrative(
        ...     template_path="exp.yaml",
        ...     expertise_level=("beginner", "advanced"),
        ...     output_format="dataframe"
        ... )
        >>> print(narratives)

        See Also
        --------
        :meth:`.plot` : Plot explanations with various visual styles.
        """
        from ..viz.narrative_plugin import NarrativePlotPlugin

        # Create plugin instance
        plugin = NarrativePlotPlugin(template_path=template_path)

        # Generate narratives using the plugin
        return plugin.plot(
            self,
            template_path=template_path,
            expertise_level=expertise_level,
            output=output_format,
            **kwargs,
        )

    # pylint: disable=protected-access
    def as_lime(self, num_features_to_show=None):
        """Transform the explanations into LIME explanation objects.

        Returns
        -------
        list of lime.Explanation
            List of LIME explanation objects with the same values as the `CalibratedExplanations`.
        """
        _, lime_exp = self.calibrated_explainer._preload_lime()  # pylint: disable=protected-access
        exp = []
        for explanation in self.explanations:  # range(len(self.x[:,0])):
            tmp = deepcopy(lime_exp)
            tmp.intercept[1] = 0
            tmp.local_pred = explanation.prediction["predict"]
            if "regression" in self.calibrated_explainer.mode:
                tmp.predicted_value = explanation.prediction["predict"]
                tmp.min_value = np.min(self.calibrated_explainer.y_cal)
                tmp.max_value = np.max(self.calibrated_explainer.y_cal)
            else:
                tmp.predict_proba[0], tmp.predict_proba[1] = (
                    1 - explanation.prediction["predict"],
                    explanation.prediction["predict"],
                )

            feature_weights = explanation.feature_weights["predict"]
            num_to_show = (
                num_features_to_show
                if num_features_to_show is not None
                else self.calibrated_explainer.num_features
            )
            features_to_plot = explanation._rank_features(feature_weights, num_to_show=num_to_show)
            rules = explanation._define_conditions()
            for j, f in enumerate(features_to_plot[::-1]):  # pylint: disable=invalid-name
                tmp.local_exp[1][j] = (f, feature_weights[f])
            del tmp.local_exp[1][num_to_show:]
            tmp.domain_mapper.discretized_feature_names = rules
            tmp.domain_mapper.feature_values = explanation.x_test
            exp.append(tmp)
        return exp

    def as_shap(self):
        """Transform the explanations into a SHAP explanation object.

        Returns
        -------
        shap.Explanation
            SHAP explanation object with the same values as the explanation.
        """
        _, shap_exp = self.calibrated_explainer._preload_shap()  # pylint: disable=protected-access
        shap_exp.base_values = np.resize(shap_exp.base_values, len(self))
        shap_exp.values = np.resize(shap_exp.values, (len(self), len(self.x_test[0, :])))
        shap_exp.data = self.x_test
        for i, explanation in enumerate(self.explanations):  # range(len(self.x[:,0])):
            # shap_exp.base_values[i] = explanation.prediction['predict']
            for f in range(len(self.x_test[0, :])):
                shap_exp.values[i][f] = -explanation.feature_weights["predict"][f]
        return shap_exp


class AlternativeExplanations(CalibratedExplanations):
    """A class for storing and visualizing alternative explanations.

    Inherits from :class:`.CalibratedExplanations` and provides methods specific to
    alternative explanations, such as filtering explanations by type.
    """

    def super_explanations(self, only_ensured=False, include_potential=True):
        """
        Return a copy with only super-explanations.

        Super-explanations are individual rules with higher probability that support the predicted class.

        Parameters
        ----------
        only_ensured : bool, default=False
            Determines whether to return only ensured explanations.
        include_potential : bool, default=True
            Determines whether to include potential explanations in the super-explanations.

        Returns
        -------
        AlternativeExplanations
            A new `AlternativeExplanations` object containing only super-factual or super-potential explanations.

        Notes
        -----
        Super-explanations are only available for `AlternativeExplanation` explanations.
        """
        for explanation in self.explanations:
            explanation.super_explanations(
                only_ensured=only_ensured, include_potential=include_potential
            )
        return self

    def semi_explanations(self, only_ensured=False, include_potential=True):
        """
        Return a copy with only semi-explanations.

        Semi-explanations are individual rules with lower probability that support the predicted class.

        Parameters
        ----------
        only_ensured : bool, default=False
            Determines whether to return only ensured explanations.
        include_potential : bool, default=True
            Determines whether to include potential explanations in the semi-explanations.

        Returns
        -------
        AlternativeExplanations
            A new `AlternativeExplanations` object containing only semi-factual or semi-potential explanations.

        Notes
        -----
        Semi-explanations are only available for `AlternativeExplanation` explanations.
        """
        for explanation in self.explanations:
            explanation.semi_explanations(
                only_ensured=only_ensured, include_potential=include_potential
            )
        return self

    def counter_explanations(self, only_ensured=False, include_potential=True):
        """
        Return a copy with only counter-explanations.

        Counter-explanations are individual rules that do not support the predicted class.

        Parameters
        ----------
        only_ensured : bool, default=False
            Determines whether to return only ensured explanations.
        include_potential : bool, default=True
            Determines whether to include potential explanations in the counter-explanations.

        Returns
        -------
        AlternativeExplanations
            A new `AlternativeExplanations` object containing only counter-factual or counter-potential explanations.

        Notes
        -----
        Counter-explanations are only available for `AlternativeExplanation` explanations.
        """
        for explanation in self.explanations:
            explanation.counter_explanations(
                only_ensured=only_ensured, include_potential=include_potential
            )
        return self

    def ensured_explanations(self):
        """
        Return a copy with only ensured explanations.

        Ensured explanations are individual rules that have a smaller confidence interval.

        Returns
        -------
        AlternativeExplanations
            A new `AlternativeExplanations` object containing only ensured explanations.
        """
        for explanation in self.explanations:
            explanation.ensured_explanations()
        return self


class FrozenCalibratedExplainer:
    """A class that wraps an explainer to provide a read-only interface.

    Prevents modification of the underlying explainer, ensuring its state remains unchanged.
    """

    def __init__(self, explainer):
        """Initialize a new instance of the FrozenCalibratedExplainer class.

        Parameters
        ----------
        explainer : CalibratedExplainer
            The explainer to be wrapped.
        """
        self._explainer = deepcopy(explainer)

    @property
    def x_cal(self):
        """
        Retrieves the calibrated feature matrix from the underlying explainer.

        This property provides access to the feature matrix used in the explainer, allowing users to understand the data being analyzed.

        Returns
        -------
            numpy.ndarray: The calibrated feature matrix.
        """
        return self._explainer.x_cal

    @property
    def y_cal(self):
        """
        Retrieves the calibrated target values from the underlying explainer.

        This property provides access to the target values used in the explainer, allowing users to understand the data being analyzed.

        Returns
        -------
            numpy.ndarray: The calibrated target values.
        """
        return self._explainer.y_cal

    @property
    def num_features(self):
        """
        Retrieves the number of features in the dataset.

        This property provides access to the count of features that the underlying explainer is using.
        It is useful for understanding the dimensionality of the data being analyzed.

        Returns
        -------
            int: The number of features in the dataset.
        """
        return self._explainer.num_features

    @property
    def categorical_features(self):
        """
        Retrieves the indices of categorical features from the underlying explainer.

        This property provides access to the indices of categorical features used in the explainer, allowing users to understand the data being analyzed.

        Returns
        -------
            list: The indices of categorical features.
        """
        return self._explainer.categorical_features

    @property
    def categorical_labels(self):
        """
        Retrieves the labels for categorical features from the underlying explainer.

        This property provides access to the labels for categorical features used in the explainer, allowing users to understand the data being analyzed.

        Returns
        -------
            list: The labels for categorical features.
        """
        return self._explainer.categorical_labels

    @property
    def feature_values(self):
        """
        Retrieves the unique values for each feature from the underlying explainer.

        This property provides access to the unique values for each feature used in the explainer, allowing users to understand the data being analyzed.

        Returns
        -------
            list: The unique values for each feature.
        """
        return self._explainer.feature_values

    @property
    def feature_names(self):
        """
        Retrieves the names of the features from the underlying explainer.

        This property provides access to the names of the features used in the explainer, allowing users to understand the data being analyzed.

        Returns
        -------
            list: The names of the features.
        """
        return self._explainer.feature_names

    @property
    def class_labels(self):
        """
        Retrieves the labels for the classes from the underlying explainer.

        This property provides access to the labels for the classes used in the explainer, allowing users to understand the data being analyzed.

        Returns
        -------
            list: The labels for the classes.
        """
        return self._explainer.class_labels

    @property
    def sample_percentiles(self):
        """
        Retrieves the sample percentiles from the underlying explainer.

        This property provides access to the percentiles of the samples used in the explainer,
        allowing users to understand the distribution of the data being analyzed.

        Returns
        -------
            list: The sample percentiles as a list.
        """
        return self._explainer.sample_percentiles

    @property
    def mode(self):
        """
        Retrieves the mode of the explainer from the underlying explainer.

        This property provides access to the mode of the explainer, allowing users to understand the type of problem being analyzed.

        Returns
        -------
            str: The mode of the explainer.
        """
        return self._explainer.mode

    @property
    def is_multiclass(self):
        """
        Retrieves a boolean indicating if the problem is multiclass from the underlying explainer.

        This property provides access to a boolean value indicating if the problem is multiclass, allowing users to understand the type of problem being analyzed.

        Returns
        -------
            bool: True if the problem is multiclass, False otherwise.
        """
        return self._explainer.is_multiclass

    @property
    def discretizer(self):
        """
        Retrieves the discretizer used by the explainer from the underlying explainer.

        This property provides access to the discretizer used by the explainer, allowing users to understand the discretization process.

        Returns
        -------
            Discretizer: The discretizer used by the explainer.
        """
        return self._explainer.discretizer

    @property
    def _discretize(self):
        """
        Retrieves the discretize function from the underlying explainer.

        This property provides access to the discretize function used by the explainer, allowing users to understand the discretization process.

        Returns
        -------
            function: The discretize function used by the explainer.
        """
        return self._explainer._discretize  # pylint: disable=protected-access

    @property
    def rule_boundaries(self):
        """
        Retrieves the boundaries for rules in the explainer from the underlying explainer.

        This property provides access to the boundaries for rules used in the explainer, allowing users to understand the discretization process.

        Returns
        -------
            list: The boundaries for rules in the explainer.
        """
        return self._explainer.rule_boundaries

    @property
    def learner(self):
        """
        Retrieves the learner associated with the explainer from the underlying explainer.

        This property provides access to the learner associated with the explainer, allowing users to understand the learning process.

        Returns
        -------
            object: The learner associated with the explainer.
        """
        return self._explainer.learner

    @property
    def difficulty_estimator(self):
        """
        Retrieves the estimator for difficulty levels from the underlying explainer.

        This property provides access to the estimator for difficulty levels used in the explainer, allowing users to understand the learning process.

        Returns
        -------
            object: The estimator for difficulty levels.
        """
        return self._explainer.difficulty_estimator

    @property
    def _predict(self):
        """
        Retrieves the predict function from the underlying explainer.

        This property provides access to the predict function used by the explainer, allowing users to understand the prediction process.

        Returns
        -------
            function: The predict function used by the explainer.
        """
        return self._explainer._predict  # pylint: disable=protected-access

    @property
    def _preload_lime(self):
        """
        Retrieves the preload_lime function from the underlying explainer.

        This property provides access to the preload_lime function used by the explainer, allowing users to understand the prediction process.

        Returns
        -------
            function: The preload_lime function used by the explainer.
        """
        return self._explainer._preload_lime  # pylint: disable=protected-access

    @property
    def _preload_shap(self):
        """
        Retrieves the preload_shap function from the underlying explainer.

        This property provides access to the preload_shap function used by the explainer, allowing users to understand the prediction process.

        Returns
        -------
            function: The preload_shap function used by the explainer.
        """
        return self._explainer._preload_shap  # pylint: disable=protected-access

    def __setattr__(self, key, value):
        """Prevent modification of attributes except for '_explainer'."""
        if key == "_explainer":
            super().__setattr__(key, value)
        else:
            raise AttributeError("Cannot modify frozen instance")
