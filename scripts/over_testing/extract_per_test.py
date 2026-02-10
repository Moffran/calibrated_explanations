"""Extract per-test unique line counts from .coverage contexts and produce baseline CSV from coverage.xml.

Outputs:
 - reports/over_testing/per_test_summary.csv (columns: test,unique_lines,runtime)
 - reports/over_testing/line_coverage_counts.csv (columns: file,line,hit)
"""
from __future__ import annotations
import os
import csv
from coverage import CoverageData
import xml.etree.ElementTree as ET

OUT_DIR = os.path.join("reports", "over_testing")
os.makedirs(OUT_DIR, exist_ok=True)

def build_line_csv(xml_path: str, out_csv: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for class_el in root.findall('.//class'):
        filename = class_el.get('filename')
        if not filename:
            continue
        for line in class_el.findall('.//line'):
            num = int(line.get('number'))
            hits = int(line.get('hits', '0'))
            rows.append((filename, num, hits))
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'line', 'hit'])
        for r in rows:
            w.writerow(r)


def build_per_test(cd: CoverageData, out_csv: str):
    contexts = list(cd.measured_contexts())
    per = {}
    files = list(cd.measured_files())
    for ctx in contexts:
        per[ctx] = 0
    for f in files:
        ctx_map = cd.contexts_by_lineno(f)
        for ln, ctxs in ctx_map.items():
            # ctxs may be tuple/list
            if not ctxs:
                continue
            if len(ctxs) == 1:
                ctx = ctxs[0]
                per.setdefault(ctx, 0)
                per[ctx] += 1
    # write CSV
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['test', 'unique_lines', 'runtime'])
        for ctx, unique in sorted(per.items(), key=lambda x: -x[1]):
            # Normalize pytest nodeids: context may be 'test::nodeid' or similar; leave as-is
            w.writerow([ctx, unique, 0])


def main():
    cov_xml = os.path.join(os.getcwd(), 'coverage.xml')
    cov_db = os.path.join(os.getcwd(), '.coverage')
    out_lines = os.path.join(OUT_DIR, 'line_coverage_counts.csv')
    out_per = os.path.join(OUT_DIR, 'per_test_summary.csv')

    if not os.path.exists(cov_xml):
        print('coverage.xml not found; run pytest with coverage first')
        return
    if not os.path.exists(cov_db):
        print('.coverage DB not found; run pytest with coverage first')
        return

    build_line_csv(cov_xml, out_lines)

    cd = CoverageData()
    cd.read()
    build_per_test(cd, out_per)
    print('Wrote', out_lines, out_per)

if __name__ == '__main__':
    main()
