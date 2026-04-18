"""Outcome-oriented summarizer for the reject evaluation suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ART = Path(__file__).resolve().parent / "artifacts"


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    return str(value)


def summarize() -> None:
    """Regenerate the top-level markdown summary from per-scenario metadata."""
    lines = [
        "# Reject Evaluation Summary",
        "",
        "This report aggregates the outcome of the real `WrapCalibratedExplainer` reject evaluation suite.",
        "",
    ]

    # RQ mapping for all scenarios (1–7)
    rq_map = {
        "scenario_1_binary_coverage": ("RQ1", "Binary marginal coverage preservation", "formal_target"),
        "scenario_2_multiclass_correctness": ("RQ2", "Multiclass correctness classifier", "empirical"),
        "scenario_3_regression_threshold_baseline": ("RQ3", "Threshold regression heuristic baseline", "empirical"),
        "scenario_4_ncf_weight_grid": ("RQ4", "NCF selection and precision-coverage tradeoff", "empirical"),
        "scenario_5_explanation_quality": ("RQ5", "Explanation quality on accepted instances", "empirical"),
        "scenario_6_finite_sample_stress": ("RQ6", "Finite-sample stress tests", "empirical"),
        "scenario_7_ncf_coverage_validity": ("C1", "NCF coverage validity sweep (supplementary)", "empirical"),
    }

    # Collect artifacts
    scenarios: list[tuple[str, dict[str, Any], pd.DataFrame | None]] = []
    for json_file in sorted(ART.glob("*.json")):
        meta = json.loads(json_file.read_text(encoding="utf-8"))
        csv_path = ART / f"{json_file.stem}.csv"
        df = pd.read_csv(csv_path) if csv_path.exists() else None
        scenarios.append((json_file.stem, meta, df))

    if not scenarios:
        lines.append("_No scenario artifacts found._")
    else:
        # Core research evaluation section (Scenarios 1–6)
        lines.extend(["## Core Research Scenarios (1–6)", ""])
        for stem, meta, df in scenarios:
            if stem not in rq_map:
                continue
            if meta.get("supplementary"):
                continue
            rq, title_short, formal_status = rq_map[stem]
            title = meta.get("display_name", stem)
            lines.append(f"### {title} — {rq}: {title_short}")
            lines.append("")
            lines.append(f"- **Status**: {formal_status}")
            lines.append("")
            lines.extend([f"- {item}" for item in (meta.get("highlights") or [])])
            if meta.get("outcome"):
                lines.append("Outcome snapshot:")
                for key, value in (meta.get("outcome") or {}).items():
                    if isinstance(value, dict):
                        lines.append(f"- **{key}**: (see json artifact)")
                    else:
                        lines.append(f"- **{key}**: {_format_value(value)}")
                lines.append("")
            if df is not None:
                lines.append(f"Rows: {len(df)}")
                lines.append(f"Columns: {', '.join(df.columns)}")
                lines.append("")

        # Supplementary section (Scenario 7)
        lines.extend(["## Supplementary Scenarios", ""])
        for stem, meta, df in scenarios:
            if stem not in rq_map:
                continue
            if not meta.get("supplementary"):
                continue
            rq, title_short, formal_status = rq_map[stem]
            title = meta.get("display_name", stem)
            lines.append(f"### {title} — {rq}: {title_short}")
            lines.append("")
            lines.append(f"- **Status**: {formal_status}")
            lines.append("")
            lines.extend([f"- {item}" for item in (meta.get("highlights") or [])])
            if meta.get("outcome"):
                lines.append("Outcome snapshot:")
                for key, value in (meta.get("outcome") or {}).items():
                    if isinstance(value, dict):
                        lines.append(f"- **{key}**: (see json artifact)")
                    else:
                        lines.append(f"- **{key}**: {_format_value(value)}")
                lines.append("")
            if df is not None:
                lines.append(f"Rows: {len(df)}")
                lines.append(f"Columns: {', '.join(df.columns)}")
                lines.append("")

    md = ART / "summary.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary to {md}")


if __name__ == "__main__":
    summarize()
