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

    # RQ mapping for research scenarios
    rq_map = {
        "scenario_e_binary_coverage_sweep": ("RQ1", "Binary marginal coverage preservation", "formal"),
        "scenario_f_multiclass_coverage": ("RQ2", "Multiclass correctness evaluation", "empirical"),
        "scenario_g_regression_coverage": ("RQ3", "Threshold regression accepted-subset behaviour", "empirical"),
        "scenario_h_ncf_grid": ("RQ4", "NCF selection and precision-coverage tradeoff", "empirical"),
        "scenario_i_explanation_quality": ("RQ5", "Explanation quality on accepted instances", "empirical"),
        "scenario_j_stress_tests": ("RQ6", "Finite-sample stress tests", "empirical"),
        "scenario_k_mondrian_regression": ("RQ7", "Difficulty-normalised regression reject (K1)", "target_formal"),
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
        # Integration validation section (A-D)
        lines.extend(["## Integration Validation (A–D)", ""])
        for stem, meta, df in scenarios:
            if stem in ("scenario_a_policy_matrix", "scenario_b_ncf_sweep", "scenario_c_confidence_monotonicity", "scenario_d_regression_threshold"):
                title = meta.get("display_name", stem)
                lines.append(f"### {title}")
                lines.append("")
                lines.extend([f"- {item}" for item in (meta.get("highlights") or [])])
                if meta.get("outcome"):
                    lines.append("Outcome snapshot:")
                    for key, value in (meta.get("outcome") or {}).items():
                        lines.append(f"- **{key}**: {_format_value(value)}")
                    lines.append("")
                if df is not None:
                    lines.append(f"Rows: {len(df)}")
                    lines.append(f"Columns: {', '.join(df.columns)}")
                    lines.append("")

        # Research evaluation section (E–K)
        lines.extend(["## Research Evaluation (E–K)", ""])
        for stem, meta, df in scenarios:
            if stem in rq_map:
                rq, title_short, formal_status = rq_map[stem]
                title = meta.get("display_name", stem)
                lines.append(f"### {title} — {rq}: {title_short}")
                lines.append("")
                # Formal/empirical annotation
                lines.append(f"- **Status**: {formal_status}")
                lines.append("")
                lines.extend([f"- {item}" for item in (meta.get("highlights") or [])])
                if meta.get("outcome"):
                    lines.append("Outcome snapshot:")
                    for key, value in (meta.get("outcome") or {}).items():
                        lines.append(f"- **{key}**: {_format_value(value)}")
                    lines.append("")
                if df is not None:
                    lines.append(f"Rows: {len(df)}")
                    lines.append(f"Columns: {', '.join(df.columns)}")
                    lines.append("")
                for plot in meta.get("plots") or []:
                    lines.append(f"![{plot}]({plot})")
                    lines.append("")

    md = ART / "summary.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary to {md}")


if __name__ == "__main__":
    summarize()
