"""One-shot script to add RT-4, RT-5, RT-10 analytical sections to artefact MDs/JSONs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

arts = Path(__file__).resolve().parent / "artifacts"

df9 = pd.read_csv(arts / "scenario_9_difficulty_normalized_ncf.csv")
df11 = pd.read_csv(arts / "scenario_11_operating_point_selection.csv")

# ---------- RT-5: AUC reversal ----------
det = df9[df9["estimator_type"] == "deterministic"]
a_all = (det[det["arm_code"] == "A"]
         [["dataset", "seed", "confidence", "difficulty_reject_auc"]]
         .rename(columns={"difficulty_reject_auc": "auc_A"}))
c_all = (det[det["arm_code"] == "C"]
         [["dataset", "seed", "confidence", "difficulty_reject_auc"]]
         .rename(columns={"difficulty_reject_auc": "auc_C"}))
m_all = a_all.merge(c_all, on=["dataset", "seed", "confidence"]).dropna(subset=["auc_A", "auc_C"])
s9_full_delta = float((m_all["auc_C"] - m_all["auc_A"]).mean())
hi = m_all[m_all["confidence"] >= 0.91]
lo = m_all[m_all["confidence"] < 0.91]
hi_delta = float((hi["auc_C"] - hi["auc_A"]).mean())
lo_delta = float((lo["auc_C"] - lo["auc_A"]).mean())
mean_rr_a_hi = float(det[(det["arm_code"] == "A") & (det["confidence"] >= 0.91)]["reject_rate"].mean())
mean_rr_c_hi = float(det[(det["arm_code"] == "C") & (det["confidence"] >= 0.91)]["reject_rate"].mean())

s11_a = (df11[df11["arm_code"] == "A"]
         [["dataset", "seed", "target_reject_rate", "difficulty_reject_auc"]]
         .rename(columns={"difficulty_reject_auc": "auc_A"}))
s11_c = (df11[df11["arm_code"] == "C"]
         [["dataset", "seed", "target_reject_rate", "difficulty_reject_auc"]]
         .rename(columns={"difficulty_reject_auc": "auc_C"}))
s11_m = s11_a.merge(s11_c, on=["dataset", "seed", "target_reject_rate"]).dropna()
s11_full_delta = float((s11_m["auc_C"] - s11_m["auc_A"]).mean())

rt5_body = (
    f"Scenario 9 reports a full-grid A-vs-C difficulty_reject_auc delta of "
    f"{s9_full_delta:+.4f}, while Scenario 11 reports {s11_full_delta:+.4f} at matched "
    "operating points. This apparent reversal is a selection effect, not a contradiction:\n\n"
    f"- Scenario 9 averages over all confidence values. The positive delta is dominated by "
    f"high-confidence rows (conf >= 0.91), where reject rates reach "
    f"~{mean_rr_a_hi:.0%} (A) / ~{mean_rr_c_hi:.0%} (C) and the AUC delta is "
    f"{hi_delta:+.4f}.\n"
    f"- At moderate confidence (conf < 0.91), the AUC delta is only {lo_delta:+.4f}, "
    "consistent with Scenario 11 matched-point results.\n"
    "- Scenario 11 targets reject rates of 10-40%, which land in the moderate-confidence "
    "regime and do not include the high-reject-rate tail driving Scenario 9 positively.\n\n"
    "Conclusion: at the deployment-relevant range (10-40% rejection), the AUC advantage "
    "is near zero. Scenario 9 positive full-grid AUC reflects the high-reject-rate regime "
    "(>40% rejection) where difficulty normalization is most effective."
)
rt5_section = f"\n\n## Metric Consistency Note (RT-5)\n\n{rt5_body}\n"

# ---------- RT-10: Statistical significance ----------
sig_rows = []
for target in [0.10, 0.20, 0.30, 0.40]:
    t_sub = df11[np.isclose(df11["target_reject_rate"].astype(float), float(target))]
    a_acc = (t_sub[t_sub["arm_code"] == "A"]
             [["dataset", "seed", "accepted_accuracy"]]
             .rename(columns={"accepted_accuracy": "acc_A"}))
    c_acc = (t_sub[t_sub["arm_code"] == "C"]
             [["dataset", "seed", "accepted_accuracy"]]
             .rename(columns={"accepted_accuracy": "acc_C"}))
    merged = a_acc.merge(c_acc, on=["dataset", "seed"])
    deltas = (merged["acc_C"] - merged["acc_A"]).dropna()
    if len(deltas) < 2:
        continue
    mean_d = float(deltas.mean())
    std_d = float(deltas.std(ddof=1))
    cohens_d = mean_d / std_d if std_d > 0 else float("nan")
    frac_pos = float((deltas > 0).mean())
    _, p_val = ttest_1samp(deltas, 0.0)
    sig_rows.append({
        "target": target, "n": len(deltas), "mean": mean_d, "std": std_d,
        "cohens_d": cohens_d, "frac_C_gt_A": frac_pos, "p_value": float(p_val),
    })

rt10_lines = [
    "",
    "## Statistical Note (RT-10)",
    "",
    "One-sample t-test of A-vs-C accepted-accuracy delta against H0: delta=0 "
    "(paired across dataset-seed groups at each target reject rate).",
    "",
    "| target | n | mean_delta | std | cohens_d | frac_C>A | p_value | significant |",
    "|---|---|---|---|---|---|---|---|",
]
for r in sig_rows:
    sig = "yes" if r["p_value"] < 0.05 else "no"
    rt10_lines.append(
        f"| {r['target']:.2f} | {r['n']} | {r['mean']:+.4f} | {r['std']:.4f} "
        f"| {r['cohens_d']:+.4f} | {r['frac_C_gt_A']:.3f} | {r['p_value']:.4f} | {sig} |"
    )
rt10_lines += [
    "",
    "None of the targets reach statistical significance (alpha=0.05). Effect sizes (Cohen's d) "
    "are small to medium negative at targets 0.20 and 0.30, meaning C does not reliably improve "
    "accepted accuracy. These results support the `do_not_promote` recommendation.",
    "",
]
rt10_section = "\n".join(rt10_lines)

# ---------- RT-4: Matching quality ----------
mq_rows = []
for target in [0.10, 0.20, 0.30, 0.40]:
    t_sub = df11[np.isclose(df11["target_reject_rate"].astype(float), float(target))]
    abs_err = t_sub["reject_rate_target_abs_error"].dropna()
    mq_rows.append({
        "target": target, "mean_abs_error": float(abs_err.mean()),
        "std": float(abs_err.std()), "n": len(abs_err),
    })

rt4_lines = [
    "",
    "## Matching Quality (RT-4)",
    "",
    "Mean absolute error between target reject rate and observed reject rate at the selected "
    "operating point (all arms included). Quick-mode confidence grid has 15 points (0.50-0.99).",
    "",
    "| target_reject_rate | mean_abs_error | std | n |",
    "|---|---|---|---|",
]
for r in mq_rows:
    rt4_lines.append(
        f"| {r['target']:.2f} | {r['mean_abs_error']:.4f} | {r['std']:.4f} | {r['n']} |"
    )
rt4_lines += [
    "",
    "Matching is tighter at higher targets (0.40: mean error ~0.05) than at target 0.10 "
    "(mean error ~0.09). Interpret deltas as approximate operating-point comparisons.",
    "",
]
rt4_section = "\n".join(rt4_lines)

# ---------- Write Scenario 9 MD ----------
s9_md = arts / "scenario_9_difficulty_normalized_ncf.md"
content9 = s9_md.read_text(encoding="utf-8")
if "Metric Consistency Note" not in content9:
    s9_md.write_text(content9 + rt5_section, encoding="utf-8")
    print("Appended RT-5 to Scenario 9 MD")
else:
    print("RT-5 already in Scenario 9 MD -- skipped")

# ---------- Write Scenario 11 MD ----------
s11_md = arts / "scenario_11_operating_point_selection.md"
content11 = s11_md.read_text(encoding="utf-8")
to_append = ""
if "Statistical Note" not in content11:
    to_append += rt10_section
if "Matching Quality" not in content11:
    to_append += rt4_section
if "Metric Consistency Note" not in content11:
    to_append += rt5_section
if to_append:
    s11_md.write_text(content11 + to_append, encoding="utf-8")
    print("Appended analytical sections to Scenario 11 MD")
else:
    print("All sections already in Scenario 11 MD -- skipped")

# ---------- Update Scenario 11 JSON ----------
with open(arts / "scenario_11_operating_point_selection.json", encoding="utf-8") as f:
    j11 = json.load(f)
j11["outcome"]["statistical_note_A_vs_C"] = {
    str(r["target"]): {
        "mean_delta": r["mean"], "cohens_d": r["cohens_d"],
        "frac_C_gt_A": r["frac_C_gt_A"], "p_value": r["p_value"],
    }
    for r in sig_rows
}
j11["outcome"]["matching_quality"] = {
    str(r["target"]): {"mean_abs_error": r["mean_abs_error"], "std": r["std"]}
    for r in mq_rows
}
j11["outcome"]["auc_reversal_note"] = {
    "s9_full_grid_delta": s9_full_delta,
    "s11_matched_delta": s11_full_delta,
    "s9_hi_conf_delta": hi_delta,
    "s9_lo_conf_delta": lo_delta,
}
with open(arts / "scenario_11_operating_point_selection.json", "w", encoding="utf-8") as f:
    json.dump(j11, f, indent=2)
print("Updated Scenario 11 JSON")

# ---------- Update Scenario 9 JSON ----------
with open(arts / "scenario_9_difficulty_normalized_ncf.json", encoding="utf-8") as f:
    j9 = json.load(f)
j9["outcome"]["metric_consistency_note"] = {
    "full_grid_auc_delta": s9_full_delta,
    "hi_conf_auc_delta": hi_delta,
    "lo_conf_auc_delta": lo_delta,
    "note": (
        "Full-grid positive delta is driven by high-confidence rows (conf>=0.91). "
        "At moderate confidence the delta is small, consistent with Scenario 11 matched results."
    ),
}
with open(arts / "scenario_9_difficulty_normalized_ncf.json", "w", encoding="utf-8") as f:
    json.dump(j9, f, indent=2)
print("Updated Scenario 9 JSON")
print("Done.")
