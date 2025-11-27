"""Enforce per-module coverage thresholds (ADR-019).

Usage:
    python scripts/check_coverage_gates.py [coverage_xml_path]
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, List

# Thresholds defined in ADR-019
# Path suffix -> Minimum coverage percentage
GATES: Dict[str, float] = {
    "src/calibrated_explanations/core/calibrated_explainer.py": 95.0,
    "src/calibrated_explanations/serialization.py": 95.0,
    "src/calibrated_explanations/plugins/registry.py": 95.0,
    "src/calibrated_explanations/core/calibration/interval_regressor.py": 95.0,
}


def parse_coverage(xml_path: str) -> Dict[str, float]:
    """Parse coverage.xml and return a map of filename -> coverage percentage."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    coverage_map = {}

    # Iterate over all class elements (files)
    for cls in root.findall(".//class"):
        filename = cls.get("filename")
        if not filename:
            continue

        # Normalize path separators to forward slashes for consistency
        filename = filename.replace("\\", "/")

        line_rate = float(cls.get("line-rate", 0.0))
        coverage_map[filename] = line_rate * 100.0

    return coverage_map


def check_gates(coverage_map: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Check if coverage meets defined gates."""
    failures = []

    for target_suffix, threshold in GATES.items():
        # Find matching file in coverage map
        # We match by suffix because coverage.xml might have relative paths
        # Extract just the relative path part (everything after src/calibrated_explanations/)
        relative_target = target_suffix.rsplit("src/calibrated_explanations/", maxsplit=1)[-1]

        matched_file = None
        for filename in coverage_map:
            if filename.endswith(relative_target):
                matched_file = filename
                break

        if matched_file:
            current = coverage_map[matched_file]
            if current < threshold:
                failures.append(
                    f"FAIL: {target_suffix} coverage {current:.1f}% < {threshold}%"
                )
            else:
                print(f"PASS: {target_suffix} coverage {current:.1f}% >= {threshold}%")
        else:
            # If file not found in coverage report, assume 0% or missing
            # But maybe it was just not run? For now, warn but don't fail if missing?
            # ADR implies strict enforcement, so missing critical file from report is bad.
            failures.append(f"FAIL: {target_suffix} not found in coverage report")

    return len(failures) == 0, failures


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
    else:
        xml_path = "coverage.xml"

    if not Path(xml_path).exists():
        print(f"Error: Coverage report not found at {xml_path}")
        sys.exit(1)

    print(f"Checking coverage gates against {xml_path}...")
    coverage_map = parse_coverage(xml_path)

    success, failures = check_gates(coverage_map)

    if not success:
        print("\nCoverage Gate Failures:")
        for failure in failures:
            print(f"  {failure}")
        sys.exit(1)

    print("\nAll coverage gates passed!")


if __name__ == "__main__":
    main()
