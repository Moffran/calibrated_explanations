"""Narrative generation utilities for CalibratedExplainer."""

from __future__ import annotations

import contextlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    _PANDAS_AVAILABLE = False


def load_template_file(filepath: str) -> Dict[str, Any]:
    """Load template file - supports both JSON and YAML formats."""
    from ..utils.exceptions import SerializationError

    path = Path(filepath)

    if not path.exists():
        raise SerializationError(
            f"Template file not found: {filepath}",
            details={"filepath": filepath, "reason": "file_not_found"},
        )

    extension = path.suffix.lower()

    if extension == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise SerializationError(
                f"Failed to parse JSON template: {e}",
                details={"filepath": filepath, "format": "json", "error": str(e)},
            ) from e

    elif extension in (".yaml", ".yml"):
        if yaml is None:
            raise SerializationError(
                f"Cannot load YAML file '{filepath}' - PyYAML not installed. "
                "Install with: pip install pyyaml",
                details={
                    "filepath": filepath,
                    "format": "yaml",
                    "reason": "yaml_not_installed",
                },
            )
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise SerializationError(
                f"Failed to parse YAML template: {e}",
                details={"filepath": filepath, "format": "yaml", "error": str(e)},
            ) from e

    else:
        raise SerializationError(
            f"Unsupported template file format: {extension}. "
            "Supported formats: .json, .yaml, .yml",
            details={
                "filepath": filepath,
                "format": extension,
                "supported": [".json", ".yaml", ".yml"],
            },
        )


def to_py(v: Any) -> Any:
    """Convert numpy scalars/arrays to plain Python types."""
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def fmt_float(x: Optional[float], nd=3) -> str:
    """Format float with specified decimal places."""
    return "N/A" if x is None else f"{x:.{nd}f}"


def _first_or_none(x):
    """Return first element if list/array-like, else the scalar."""
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return to_py(x[0]) if len(x) else None
    return to_py(x)


def clean_condition(rule: str, feat_name: Any) -> str:
    """Remove feature name from beginning of rule."""
    if not rule:
        return ""
    if feat_name is None:
        return rule
    # Ensure feat_name is a string
    feat_name_str = str(feat_name)
    if not feat_name_str:
        return rule
    try:
        return re.sub(rf"(?i)^{re.escape(feat_name_str)}\s*", "", rule).strip()
    except:  # noqa: E722
        if not isinstance(sys.exc_info()[1], Exception):
            raise
        # Fallback if regex fails
        return rule


def crosses_zero(feat: Dict) -> bool:
    """Check if feature weight interval crosses zero (direction uncertain)."""
    wl = feat.get("weight_low")
    wh = feat.get("weight_high")
    with contextlib.suppress(ValueError, TypeError):
        if wl is not None and wh is not None:
            return float(wl) <= 0.0 <= float(wh)
    return False


def has_wide_prediction_interval(feat: Dict, threshold: float = 0.20) -> bool:
    """Check if feature's prediction interval is wide (≥ 0.20)."""
    pl = feat.get("predict_low")
    ph = feat.get("predict_high")
    with contextlib.suppress(ValueError, TypeError):
        if pl is not None and ph is not None:
            width = abs(float(ph) - float(pl))
            return width >= threshold - 1e-12
    return False


class NarrativeGenerator:
    """Generate human-readable narratives from calibrated explanations."""

    def __init__(self, template_path: Optional[str] = None):
        """Initialize with optional template file path."""
        self.templates = None
        if template_path:
            self.load_templates(template_path)

    def load_templates(self, template_path: str):
        """Load narrative templates from file."""
        self.templates = load_template_file(template_path)

    def generate_narrative(
        self,
        explanation,
        problem_type: str,
        explanation_type: str = "factual",
        expertise_level: str = "advanced",
        threshold: Optional[float] = None,
        feature_names: Optional[List[str]] = None,
    ) -> str:
        """Generate narrative for a single explanation."""
        from ..utils.exceptions import ValidationError

        if self.templates is None:
            raise ValidationError(
                "Templates not loaded. Call load_templates() first.",
                details={"state": "uninitialized", "required_method": "load_templates"},
            )

        # Get rules from explanation
        if hasattr(explanation, "get_rules") or hasattr(explanation, "get_rules"):
            rules_dict = explanation.get_rules()
        else:
            raise ValidationError(
                "Explanation has no get_rules method",
                details={
                    "param": "explanation",
                    "required_method": "get_rules",
                    "type": type(explanation).__name__,
                },
            )

        # Extract base prediction values
        bp = _first_or_none(rules_dict.get("base_predict"))
        bl = _first_or_none(rules_dict.get("base_predict_low"))
        bh = _first_or_none(rules_dict.get("base_predict_high"))

        # Build context dictionary
        # Try to get the predicted class/label
        predicted_class = rules_dict.get("classes")
        if predicted_class is not None:
            # For classification, get the class label
            if hasattr(explanation, "get_class_labels"):
                class_labels = explanation.get_class_labels()
                if class_labels is not None and isinstance(class_labels, dict):
                    label = str(class_labels.get(predicted_class, predicted_class))
                else:
                    label = str(predicted_class)
            else:
                label = str(predicted_class)
        else:
            # Fallback: try to infer from prediction
            if bp is not None and bp > 0.5:
                label = "1"
            elif bp is not None:
                label = "0"
            else:
                label = ""

        context = {
            "label": label,
            "calibrated_pred": fmt_float(bp),
            "pred_interval_lower": fmt_float(bl),
            "pred_interval_upper": fmt_float(bh),
            "threshold": fmt_float(threshold) if threshold else "",
            "runner_up_class": "",
            "margin_value": "",
            "interval_width": "",
        }

        # Calculate interval width for regression
        if (
            problem_type in ("regression", "probabilistic_regression")
            and bl is not None
            and bh is not None
        ):
            with contextlib.suppress(ValueError, TypeError):
                width = float(bh) - float(bl)
                context["interval_width"] = fmt_float(width)

        # Get template
        try:
            template = self.templates["narrative_templates"][problem_type][explanation_type][
                expertise_level
            ]
        except KeyError:
            return f"Template not found for {problem_type}/{explanation_type}/{expertise_level}"

        # Get feature rules with proper feature names
        rules = self._serialize_rules(rules_dict, feature_names)

        # Split features by weight sign
        pos_features = [r for r in rules if r.get("weight", 0) > 0]
        neg_features = [r for r in rules if r.get("weight", 0) < 0]

        # Sort by absolute weight
        pos_features.sort(key=lambda r: abs(r.get("weight", 0)), reverse=True)
        neg_features.sort(key=lambda r: abs(r.get("weight", 0)), reverse=True)

        # Take top features (min 3)
        min_features = 3
        max_features = max(min_features, len(pos_features))
        pos_features = pos_features[:max_features]
        max_features = max(min_features, len(neg_features))
        neg_features = neg_features[:max_features]

        # For advanced level: split by prediction interval width
        if expertise_level == "advanced":
            pos_certain = [r for r in pos_features if not has_wide_prediction_interval(r)]
            pos_uncertain = [r for r in pos_features if has_wide_prediction_interval(r)]
            neg_certain = [r for r in neg_features if not has_wide_prediction_interval(r)]
            neg_uncertain = [r for r in neg_features if has_wide_prediction_interval(r)]
            uncertain_all = pos_uncertain + neg_uncertain

            narrative = self._expand_template(
                template,
                pos_certain,
                neg_certain,
                uncertain_all,
                context,
                expertise_level,
                problem_type,  # Pass problem_type
            )
        else:
            narrative = self._expand_template(
                template,
                pos_features,
                neg_features,
                [],
                context,
                expertise_level,
                problem_type,  # Pass problem_type
            )

        return narrative

    def _serialize_rules(
        self, rules_dict: Dict, feature_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """Convert rule dictionary to list of feature dictionaries with proper names."""
        rules_list = rules_dict.get("rule", []) or []
        n = len(rules_list)

        def get_item(key, idx):
            items = rules_dict.get(key, [])
            return items[idx] if idx < len(items) else None

        def extract_feature_name_from_rule(rule: str) -> str:
            """Extract feature name from rule string."""
            if not rule:
                return ""
            # The feature name is typically the first part before any operator
            parts = rule.split()
            if parts:
                # Remove any operators that might be stuck to the name
                import re

                name = re.sub(r"[<=>=!<>]+$", "", parts[0])
                return name
            return ""

        result = []
        for i in range(n):
            feature_idx = get_item("feature", i)

            # Try to get actual feature name
            feature_name = None
            if feature_names is not None and feature_idx is not None:
                with contextlib.suppress(ValueError, TypeError):
                    feat_idx = int(feature_idx)
                    if 0 <= feat_idx < len(feature_names):
                        feature_name = feature_names[feat_idx]

            if feature_name is None:
                # Fallback to extracting from rule
                feature_name = extract_feature_name_from_rule(
                    rules_list[i] if i < len(rules_list) else ""
                )

            feature_dict = {
                "rule": rules_list[i] if i < len(rules_list) else "",
                "value": get_item("value", i),
                "weight": get_item("weight", i),
                "weight_low": get_item("weight_low", i),
                "weight_high": get_item("weight_high", i),
                "feature_name": feature_name,
                "feature_index": feature_idx,
                "predict": get_item("predict", i),
                "predict_low": get_item("predict_low", i),
                "predict_high": get_item("predict_high", i),
            }
            result.append(feature_dict)

        return result

    def _expand_template(
        self,
        template: str,
        pos_features: List[Dict],
        neg_features: List[Dict],
        uncertain_features: List[Dict],
        context: Dict[str, str],
        level: str,
        problem_type: str = "regression",
    ) -> str:
        """Expand template with features and context."""
        # Fill in global context placeholders
        for k, v in context.items():
            template = template.replace(f"{{{k}}}", v)

        # Check if caution is needed based on calibrated probability interval width
        # Only for probability-based tasks (classification and probabilistic regression)
        caution_line = ""
        if problem_type in (
            "binary_classification",
            "multiclass_classification",
            "probabilistic_regression",
        ):
            with contextlib.suppress(ValueError, TypeError):
                pred_low = context.get("pred_interval_lower", "")
                pred_high = context.get("pred_interval_upper", "")

                # Try to parse and calculate width
                if pred_low and pred_high and pred_low != "N/A" and pred_high != "N/A":
                    low_val = float(pred_low)
                    high_val = float(pred_high)
                    width = high_val - low_val

                    # Only show caution if width > 0.20
                    if width > 0.20:
                        if level == "beginner":
                            caution_line = "⚠️ Use caution: uncertainty is high."
                        else:
                            caution_line = f"⚠️ Use caution: calibrated probability interval is wide ({width:.3f})."

        # Build feature lines
        def build_lines(line: str, feats: List[Dict]) -> List[str]:
            rendered = []
            for f in feats:
                # Ensure feature_name is a string
                feat_name_raw = f.get("feature_name")
                if feat_name_raw is not None:
                    feat_name = str(feat_name_raw)
                else:
                    # Fallback: extract from rule
                    rule = f.get("rule", "")
                    feat_name = rule.split()[0] if rule else ""

                rule = f.get("rule", "")
                cond = clean_condition(rule, feat_name)

                # Per-feature uncertainty tags
                tags = []
                if has_wide_prediction_interval(f):
                    tags.append("⚠️ highly uncertain")
                if crosses_zero(f):
                    tags.append("⚠️ direction uncertain")
                uncertainty_tag = " [" + ", ".join(tags) + "]" if tags else ""

                txt = (
                    line.replace("{feature_name}", feat_name)
                    .replace("{feature_actual_value}", str(f.get("value", "")))
                    .replace("{condition}", cond)
                )

                # Handle predict placeholders
                p, pl, ph = f.get("predict"), f.get("predict_low"), f.get("predict_high")
                txt = (
                    txt.replace("{predict}", fmt_float(p))
                    .replace("{predict_low}", fmt_float(pl))
                    .replace("{predict_high}", fmt_float(ph))
                )

                # Handle weight placeholders
                w, wl, wh = f.get("weight"), f.get("weight_low"), f.get("weight_high")
                txt = txt.replace("{feature_weight}", fmt_float(w))

                if level == "advanced" and wl is not None and wh is not None:
                    txt = txt.replace("{feature_weight_low}", fmt_float(wl)).replace(
                        "{feature_weight_high}", fmt_float(wh)
                    )
                else:
                    # Remove interval placeholders
                    txt = re.sub(
                        r"\s*\[\{feature_weight_low\},\s*\{feature_weight_high\}\]", "", txt
                    )
                    txt = txt.replace("{feature_weight_low}", "").replace(
                        "{feature_weight_high}", ""
                    )

                txt += uncertainty_tag
                rendered.append(txt)
            return rendered

        lines = template.splitlines()
        out_lines = []
        placeholder = "{feature_name}"
        pos_done = neg_done = uncertain_done = False

        for line in lines:
            if placeholder in line:
                if not pos_done and pos_features:
                    out_lines.extend(build_lines(line, pos_features))
                    pos_done = True
                elif not neg_done and neg_features:
                    out_lines.extend(build_lines(line, neg_features))
                    neg_done = True
                elif not uncertain_done and uncertain_features and level == "advanced":
                    out_lines.extend(build_lines(line, uncertain_features))
                    uncertain_done = True
            else:
                out_lines.append(line)

        body = "\n".join(out_lines).strip()

        if caution_line:
            return f"{caution_line}\n\n{body}".strip()
        return body
