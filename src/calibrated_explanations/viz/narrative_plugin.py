"""Narrative generation plot plugin for calibrated explanations.

This module provides a plot plugin that generates human-readable narratives
from calibrated explanations instead of traditional visualizations.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.narrative_generator import NarrativeGenerator

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    _PANDAS_AVAILABLE = False


class NarrativePlotPlugin:
    """Plot plugin that generates narrative explanations.

    This plugin integrates with the calibrated explanations plot system to
    generate human-readable narratives instead of traditional visualizations.

    Parameters
    ----------
    template_path : str, optional
        Path to the narrative template file (YAML or JSON).
        If not provided, uses the default template from the project root.

    Examples
    --------
    >>> from calibrated_explanations import CalibratedExplainer
    >>> explainer = CalibratedExplainer(model, X_train, y_train)
    >>> explanations = explainer.explain_factual(X_test)
    >>> result = explanations.plot(
    ...     style="narrative",
    ...     template_path="explain_template.yaml",
    ...     output="dataframe"
    ... )
    """

    def __init__(self, template_path: Optional[str] = None):
        """Initialize the narrative plot plugin.

        Parameters
        ----------
        template_path : str, optional
            Path to the narrative template file.
        """
        self.default_template = self._get_default_template_path()
        self._template_path = template_path or self.default_template

    @staticmethod
    def _get_default_template_path() -> str:
        """Get the default template path from the package resources.

        Returns
        -------
        str
            Absolute path to the default explain_template.yaml file.
        """
        # Navigate from viz/ to templates/
        current_file = Path(__file__).resolve()
        # src/calibrated_explanations/viz -> src/calibrated_explanations/templates
        package_root = current_file.parent.parent
        default_template = package_root / "templates" / "explain_template.yaml"
        return str(default_template)

    def plot(
        self,
        explanations,
        template_path: Optional[str] = None,
        expertise_level: Union[str, Tuple[str, ...]] = ("beginner", "intermediate", "advanced"),
        output: str = "dataframe",
        **kwargs,
    ) -> Union[Any, str, List[Dict[str, Any]]]:
        """Generate narratives for a collection of explanations.

        Parameters
        ----------
        explanations : CalibratedExplanations or AlternativeExplanations
            The explanations to generate narratives for.
        template_path : str, optional
            Path to the narrative template file. If not provided, uses the
            default template or the one specified during initialization.
        expertise_level : str or tuple of str, default=("beginner", "intermediate", "advanced")
            The expertise level(s) for narrative generation. Can be a single
            level or a tuple of levels. Valid values: "beginner", "intermediate", "advanced".
        output : str, default="dataframe"
            Output format. Valid values: "dataframe", "text", "html", "dict".
        **kwargs : dict
            Additional keyword arguments (currently unused, reserved for future extensions).

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
            If the template file is not found.
        ValueError
            If an invalid expertise level or output format is specified.
        ImportError
            If pandas is not available and output="dataframe" is requested.
        """
        # Determine template path
        template = template_path or self._template_path

        # Check if template exists, if not fall back to default
        if template and not Path(template).is_absolute() and not Path(template).exists():
            # For relative paths, fall back to default template
            template = self.default_template
        elif template and Path(template).is_absolute() and not Path(template).exists():
            # For absolute paths, fall back to default template
            template = self.default_template

        # Validate expertise level
        valid_levels = {"beginner", "intermediate", "advanced"}
        if isinstance(expertise_level, str):
            if expertise_level not in valid_levels:
                from ..utils.exceptions import ValidationError

                raise ValidationError(
                    f"Invalid expertise level: {expertise_level}. "
                    f"Valid levels: {', '.join(sorted(valid_levels))}",
                    details={
                        "param": "expertise_level",
                        "value": expertise_level,
                        "allowed_values": sorted(valid_levels),
                    },
                )
            levels = (expertise_level,)
        else:
            levels = tuple(expertise_level)
            invalid = [lv for lv in levels if lv not in valid_levels]
            if invalid:
                from ..utils.exceptions import ValidationError

                raise ValidationError(
                    f"Invalid expertise level(s): {', '.join(invalid)}. "
                    f"Valid levels: {', '.join(sorted(valid_levels))}",
                    details={
                        "param": "expertise_level",
                        "invalid_values": invalid,
                        "allowed_values": sorted(valid_levels),
                    },
                )

        # Validate output format
        valid_outputs = {"dataframe", "text", "html", "dict"}
        if output not in valid_outputs:
            from ..utils.exceptions import ValidationError

            raise ValidationError(
                f"Invalid output format: {output}. "
                f"Valid formats: {', '.join(sorted(valid_outputs))}",
                details={
                    "param": "output",
                    "value": output,
                    "allowed_values": sorted(valid_outputs),
                },
            )

        # Check pandas availability for dataframe output
        if output == "dataframe" and not _PANDAS_AVAILABLE:
            raise ImportError(
                "Pandas is required for dataframe output. " "Install with: pip install pandas"
            )

        # Detect problem type and explanation type
        problem_type = self._detect_problem_type(explanations)
        explanation_type = "alternative" if self._is_alternative(explanations) else "factual"

        # Get feature names
        feature_names = self._get_feature_names(explanations)

        # Initialize narrative generator
        narrator = NarrativeGenerator(template)

        # Generate narratives for each explanation
        results = []
        for i, explanation in enumerate(explanations.explanations):
            row = {"instance_index": i}

            # Get threshold if applicable
            threshold = getattr(explanation, "y_threshold", None)

            # Generate narrative for each expertise level
            for level in levels:
                try:
                    narrative = narrator.generate_narrative(
                        explanation,
                        problem_type=problem_type,
                        explanation_type=explanation_type,
                        expertise_level=level,
                        threshold=threshold,
                        feature_names=feature_names,
                    )
                    row[f"{explanation_type}_explanation_{level}"] = narrative
                except:
                    e = sys.exc_info()[1]
                    if not isinstance(e, Exception):
                        raise
                    # Include error message in output for debugging
                    row[f"{explanation_type}_explanation_{level}"] = (
                        f"Error generating narrative: {e}"
                    )

            row["expertise_level"] = levels
            row["problem_type"] = problem_type
            results.append(row)

        # Format output
        return self._format_output(results, output)

    def _detect_problem_type(self, explanations) -> str:
        """Detect the problem type from explanations metadata.

        Parameters
        ----------
        explanations : CalibratedExplanations or AlternativeExplanations
            The explanations object.

        Returns
        -------
        str
            One of: "regression", "binary_classification",
            "multiclass_classification", "probabilistic_regression"
        """
        try:
            # Get the explainer
            explainer = explanations.calibrated_explainer

            # Check if thresholded (probabilistic regression)
            if explanations.y_threshold is not None:
                return "probabilistic_regression"

            # Check mode
            mode = getattr(explainer, "mode", "")

            if "classification" in mode:
                # Check if multiclass
                try:
                    is_multiclass = explainer.is_multiclass()
                except:
                    if not isinstance(sys.exc_info()[1], Exception):
                        raise
                    is_multiclass = getattr(explainer, "is_multiclass", False)

                if is_multiclass:
                    return "multiclass_classification"
                return "binary_classification"

            if "regression" in mode:
                return "regression"

            # Default fallback
            return "regression"

        except:
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            # Fallback to regression if detection fails
            return "regression"

    def _get_feature_names(self, explanations) -> Optional[List[str]]:
        """Extract feature names from the explainer.

        Parameters
        ----------
        explanations : CalibratedExplanations or AlternativeExplanations
            The explanations object.

        Returns
        -------
        list of str or None
            Feature names if available, None otherwise.
        """
        try:
            explainer = explanations.calibrated_explainer

            # Try to get feature names from the underlying explainer
            if hasattr(explainer, "_explainer"):
                underlying = explainer._explainer
                if hasattr(underlying, "feature_names"):
                    return underlying.feature_names

            # Try direct access
            if hasattr(explainer, "feature_names"):
                return explainer.feature_names

            return None

        except:
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            return None

    def _is_alternative(self, explanations) -> bool:
        """Check if explanations are alternative explanations.

        Parameters
        ----------
        explanations : CalibratedExplanations or AlternativeExplanations
            The explanations object.

        Returns
        -------
        bool
            True if alternative explanations, False otherwise.
        """
        # Check class name
        class_name = type(explanations).__name__
        if "Alternative" in class_name:
            return True

        # Check if the explanations have the _is_alternative method
        if hasattr(explanations, "_is_alternative"):
            with contextlib.suppress(AttributeError, TypeError):
                return explanations._is_alternative()
            return False

        return False

    def _format_output(
        self, results: List[Dict[str, Any]], output_format: str
    ) -> Union[Any, str, List[Dict[str, Any]]]:
        """Format the results into the requested output format.

        Parameters
        ----------
        results : list of dict
            The narrative results for each instance.
        output_format : str
            The desired output format: "dataframe", "text", "html", or "dict".

        Returns
        -------
        pd.DataFrame or str or list of dict
            The formatted output.
        """
        if output_format == "dict":
            return results

        if output_format == "dataframe":
            return pd.DataFrame(results)

        if output_format == "text":
            return self._format_as_text(results)

        if output_format == "html":
            return self._format_as_html(results)

        # Should not reach here due to validation
        from ..utils.exceptions import ConfigurationError

        raise ConfigurationError(
            f"Unsupported output format: {output_format}",
            details={
                "param": "output_format",
                "value": output_format,
                "allowed_values": ["dataframe", "text", "html", "dict"],
            },
        )

    def _format_as_text(self, results: List[Dict[str, Any]]) -> str:
        """Format results as plain text.

        Parameters
        ----------
        results : list of dict
            The narrative results.

        Returns
        -------
        str
            Formatted text output.
        """
        lines = []
        for result in results:
            idx = result["instance_index"]
            lines.append(f"\n{'=' * 80}")
            lines.append(f"Instance {idx}")
            lines.append("=" * 80)

            # Get all narrative keys (excluding metadata)
            narrative_keys = [
                k for k in result if k not in ("instance_index", "expertise_level", "problem_type")
            ]

            for key in sorted(narrative_keys):
                # Extract level from key (e.g., "factual_explanation_beginner" -> "Beginner")
                parts = key.split("_")
                if len(parts) >= 3:
                    level = parts[-1].capitalize()
                    exp_type = parts[0].capitalize()
                    lines.append(f"\n{exp_type} Explanation ({level}):")
                    lines.append("-" * 80)
                    lines.append(result[key])

        return "\n".join(lines)

    def _format_as_html(self, results: List[Dict[str, Any]]) -> str:
        """Format results as HTML table.

        Parameters
        ----------
        results : list of dict
            The narrative results.

        Returns
        -------
        str
            HTML table output.
        """
        if not results:
            return "<table></table>"

        # Build HTML table
        html = ['<table border="1" style="border-collapse: collapse; width: 100%;">']

        # Header
        html.append("<thead><tr>")
        first_result = results[0]
        for key in first_result:
            html.append(f"<th style='padding: 8px; text-align: left;'>{key}</th>")
        html.append("</tr></thead>")

        # Body
        html.append("<tbody>")
        for result in results:
            html.append("<tr>")
            for key in result:
                value = result[key]
                # Format value for display
                if isinstance(value, (tuple, list)):
                    value = ", ".join(str(v) for v in value)
                else:
                    value = str(value).replace("\n", "<br>")
                html.append(f"<td style='padding: 8px; vertical-align: top;'>{value}</td>")
            html.append("</tr>")
        html.append("</tbody>")

        html.append("</table>")
        return "\n".join(html)


__all__ = ["NarrativePlotPlugin"]
