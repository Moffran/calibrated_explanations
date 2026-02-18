#!/usr/bin/env python3
import csv
from collections import defaultdict

csv_path = 'reports/over_testing/per_test_summary.csv'
files = defaultdict(list)
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        nodeid = row[0]
        try:
            uniq = int(row[1])
        except Exception:
            uniq = 0
        if not nodeid:
            continue
        file_part = nodeid.split('::')[0]
        files[file_part].append(uniq)

candidates = []
for path, uniqs in files.items():
    # normalize path separators so the prefix check works across platforms
    normalized_path = path.replace('\\', '/')
    # consider only test files under tests/
    if not normalized_path.startswith('tests/'):
        continue
    # skip files with no entries
    if len(uniqs) == 0:
        continue
    # if all entries have uniq==0
    if all(u == 0 for u in uniqs):
        candidates.append((path, len(uniqs)))

# sort by number of tests in file descending to maximize removals
candidates.sort(key=lambda x: -x[1])
# limit to 100 files
selected = candidates[:100]
with open('reports/over_testing/zero_unique_files_to_remove.txt', 'w', encoding='utf-8') as out:
    for path, count in selected:
        out.write(f"{path},{count}\n")
print(f"Wrote {len(selected)} files to reports/over_testing/zero_unique_files_to_remove.txt")
