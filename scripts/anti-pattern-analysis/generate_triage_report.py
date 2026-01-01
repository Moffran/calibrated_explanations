import csv
import collections
from pathlib import Path

def generate_triage_report():
    analysis_file = Path("reports/anti-pattern-analysis/private_method_analysis.csv")
    usage_file = Path("reports/anti-pattern-analysis/private_usage_scan.csv")
    output_csv = Path("reports/anti-pattern-analysis/test_only_private_refs.csv")
    output_md = Path("reports/anti-pattern-analysis/triage_next_actions.md")

    if not analysis_file.exists() or not usage_file.exists():
        print("Required analysis files not found. Run analyze_private_methods.py and scan_private_usage.py first.")
        return

    # Load analysis data for in_src status
    analysis_data = {}
    with open(analysis_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            analysis_data[row["name"]] = row["scope"] == "library"

    # Group usages
    usages = collections.defaultdict(list)
    with open(usage_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            usages[row["name"]].append(f"{row['file']}:{row['line']}")

    # Prepare report data
    report_data = []
    for name, samples in usages.items():
        report_data.append({
            "name": name,
            "count": len(samples),
            "samples": ";".join(samples[:3]), # Limit samples for CSV
            "in_src": analysis_data.get(name, False)
        })

    # Sort by count descending
    report_data.sort(key=lambda x: x["count"], reverse=True)

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "count", "samples", "in_src"])
        writer.writeheader()
        writer.writerows(report_data)

    print(f"Triage CSV written to {output_csv}")

    # Generate Markdown
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Manual Triage — Prioritized Next Actions\n\n")
        f.write("This file lists prioritized private-member symbols with recommended actions.\n\n")
        f.write("## Top Symbols to Remediate\n\n")
        f.write("| Name | Count | In Src | Samples |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for row in report_data[:50]:
            f.write(f"| {row['name']} | {row['count']} | {row['in_src']} | {row['samples']} |\n")

    print(f"Triage Markdown written to {output_md}")

if __name__ == "__main__":
    generate_triage_report()
