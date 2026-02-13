"""Detect redundant tests by comparing their coverage fingerprints.

A "Coverage Fingerprint" is the unique set of (file, line_number) tuples executed by a test.
Tests with identical fingerprints cover exactly the same code paths and are strong candidates
for being redundant (unless they check different return values/side-effects).

This script:
1. Loads the `.coverage` database (requires `pytest --cov-context=test`).
2. Maps each test context to its full set of covered lines.
3. Groups tests by their fingerprint.
4. Reports groups with >1 test as "Redundant Candidates".

Outputs:
- reports/over_testing/redundant_tests.csv
  (fingerprint_hash, test_count, unique_lines, test_names)
"""
import sqlite3
import hashlib
import json
import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple

def get_coverage_data(db_path: Path) -> Dict[str, Set[Tuple[str, int]]]:
    """
    Reads the .coverage SQLite DB manually to get per-context line data.
    Returns: Mapping of context_name -> Set[(filename, lineno)]
    """
    if not db_path.exists():
        print(f"Error: Coverage DB found at {db_path}. Run `pytest --cov-context=test` first.")
        sys.exit(1)

    print(f"Loading coverage data from {db_path}...")
    
    # We use sqlite3 directly because coverage.py's public API for raw context querying 
    # can be slow or complex for this specific aggregation.
    with sqlite3.connect(db_path) as conn:
        # 1. Get dictionary of file IDs to paths
        cursor = conn.cursor()
        cursor.execute("SELECT id, path FROM file")
        file_map = {row[0]: row[1] for row in cursor.fetchall()}
        
        # 2. Get dictionary of context IDs to names
        cursor.execute("SELECT id, context FROM context")
        context_map = {row[0]: row[1] for row in cursor.fetchall()}
        
        # 3. Get all line hits (context_id, file_id, lineno)
        # Note: 'line_bits' table stores generic hits, 'line_map' logic might be needed 
        # for complex cases, but standard coverage usually populates 'line_bits' or 'line' 
        # dependent on version. 
        # Modern coverage.py (v5+) uses a 'line_bits' table with binary blobs, 
        # or a 'line' table. Let's try the standard SQL schema.
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = dir_res = [r[0] for r in cursor.fetchall()]
        
        test_coverage = defaultdict(set)
        
        if 'line' in tables:
            # Simple schema
            cursor.execute("SELECT context_id, file_id, lineno FROM line")
            rows = cursor.fetchall()
            for ctx_id, file_id, lineno in rows:
                if ctx_id not in context_map or file_id not in file_map:
                    continue
                ctx_name = context_map[ctx_id]
                file_path = file_map[file_id]
                test_coverage[ctx_name].add((file_path, lineno))
                
        elif 'line_bits' in tables:
            # Bitfield schema (more common in newer coverage)
            # This is complex to parse via SQL. Better to use coverage API if possible.
            # Fallback to CoverageData API.
            print("Detected optimized schema, switching to CoverageData API (slower but reliable)...")
            return get_coverage_data_via_api()
            
    return test_coverage

def get_coverage_data_via_api() -> Dict[str, Set[Tuple[str, int]]]:
    """Fallback using official coverage API."""
    from coverage import CoverageData
    
    cd = CoverageData()
    cd.read()
    
    test_coverage = defaultdict(set)
    measured_files = cd.measured_files()
    
    print(f"Processing {len(measured_files)} files...")
    
    for file_path in measured_files:
        # contexts_by_lineno: {lineno: [ctx_names...]}
        ctx_by_line = cd.contexts_by_lineno(file_path)
        for lineno, contexts in ctx_by_line.items():
            for ctx in contexts:
                if ctx == "": # Skip empty context
                    continue
                test_coverage[ctx].add((file_path, lineno))
    
    if not test_coverage:
        print("\nERROR: No test contexts found in coverage data!")
        print("Possible reasons:")
        print("1. Tests were run without '--cov-context=test'")
        print("2. Tests failed to run")
        print("\nPlease run: python scripts/over_testing/run_over_testing_pipeline.py")
        print("(or: pytest --cov-context=test --cov=src/calibrated_explanations)")
        sys.exit(1)
                
    return test_coverage

def calculate_fingerprints(coverage_map: Dict[str, Set[Tuple[str, int]]]) -> Tuple[List[dict], List[dict]]:
    """
    Group tests by fingerprint and detect subset redundancies (aggressive).
    Returns (fingerprint_groups, subset_pairs)
    """
    print(f"Fingerprinting {len(coverage_map)} contexts...")
    
    # 1. Exact Fingerprinting
    groups = defaultdict(list)
    test_lines_map = {} # cache for subset check
    
    for test_name, lines in coverage_map.items():
        if test_name == "": continue
        test_lines_map[test_name] = lines
        
        # Create deterministic string representation of executed lines
        sorted_lines = sorted(list(lines))
        fingerprint_str = str(sorted_lines)
        fingerprint_hash = hashlib.md5(fingerprint_str.encode('utf-8')).hexdigest()
        
        groups[fingerprint_hash].append(test_name)
    
    formatted_groups = []
    
    # Pre-calculate unique lines helper
    global_line_counts = defaultdict(int)
    for lines in coverage_map.values():
        for line_key in lines:
            global_line_counts[line_key] += 1
            
    test_unique_counts = {}
    for test, lines in coverage_map.items():
        unique = sum(1 for line in lines if global_line_counts[line] == 1)
        test_unique_counts[test] = unique

    for fp, tests in groups.items():
        example_test = tests[0]
        line_count = len(coverage_map[example_test])
        unique_val = test_unique_counts[example_test]
        
        formatted_groups.append({
            "fingerprint": fp,
            "test_count": len(tests),
            "line_count": line_count,
            "unique_lines_per_test": unique_val,
            "tests": sorted(tests),
            "lines_set": coverage_map[example_test] # Keep for subset check
        })
    
    formatted_groups.sort(key=lambda x: x['test_count'], reverse=True)

    # 2. Aggressive Subset Detection
    # If Group A's lines are a subset of Group B's lines, then all tests in Group A 
    # are potentially redundant to B.
    print(f"Running aggressive subset analysis on {len(formatted_groups)} distinct fingerprints...")
    
    subset_redundancies = []
    
    # Sort by line_count descending to optimize subset check
    # We only need to check if a smaller set is contained in a larger set
    sorted_groups = sorted(formatted_groups, key=lambda x: x['line_count'], reverse=True)
    
    # O(N^2) comparison - acceptable for N < 5000 groups
    for i in range(len(sorted_groups)):
        subset_cand = sorted_groups[i]
        subset_lines = subset_cand['lines_set']
        
        if not subset_lines: continue
        
        # Check against all groups that have MORE lines (appear earlier in list)
        for j in range(i):
            superset_cand = sorted_groups[j]
            # Optimization: distinct line counts must differ
            # (If equal, they would be in same fingerprint group)
            
            if subset_lines.issubset(superset_cand['lines_set']):
                 subset_redundancies.append({
                     "subset_fingerprint": subset_cand['fingerprint'],
                     "superset_fingerprint": superset_cand['fingerprint'],
                     "subset_tests": subset_cand['tests'],
                     "superset_tests": superset_cand['tests'],
                     "subset_line_count": subset_cand['line_count'],
                     "superset_line_count": superset_cand['line_count']
                 })
                 # Once we find ONE superset, this group is redundant.
                 # No need to find all supersets for the report (reduce noise).
                 break
                 
    return formatted_groups, subset_redundancies

def main():
    report_dir = Path("reports/over_testing")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    print("Step 1: Reading coverage data...")
    coverage_map = get_coverage_data_via_api()
    
    print("Step 2: Calculating fingerprints and subsets...")
    groups, subsets = calculate_fingerprints(coverage_map)
    
    out_csv = report_dir / "redundant_tests.csv"
    
    print(f"Step 3: Writing report to {out_csv}...")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "type",
            "test_count", 
            "lines_covered", 
            "unique_lines_per_test", 
            "description",
            "tests"
        ])
        
        redundant_groups = 0
        total_redundant_tests = 0
        
        # 1. Write Exact Duplicates
        for g in groups:
            if g['test_count'] > 1:
                redundant_groups += 1
                count = g['test_count']
                total_redundant_tests += (count - 1)
                
                writer.writerow([
                    "EXACT_MATCH",
                    count,
                    g['line_count'],
                    g['unique_lines_per_test'],
                    f"Share exact same {g['line_count']} lines",
                    json.dumps(g['tests'])
                ])
        
        # 2. Write Subsets
        for s in subsets:
            count = len(s['subset_tests'])
            total_redundant_tests += count
            
            desc = f"Subset of {s['superset_tests'][0]} ({s['subset_line_count']} lines vs {s['superset_line_count']})"
            
            writer.writerow([
                "SUBSET_MATCH",
                count,
                s['subset_line_count'],
                0, # Effective unique lines is 0 if it's a subset
                desc,
                json.dumps(s['subset_tests'])
            ])

    print("\n" + "="*60)
    print(f"ANALYSIS COMPLETE")
    print(f"Total Unique Contexts: {len(coverage_map)}")
    print(f"Exact Duplicate Groups: {redundant_groups}")
    print(f"Subset Redundancy Groups: {len(subsets)}")
    print(f"Total Potential Redundant Tests: {total_redundant_tests}")
    print(f"Report saved to: {out_csv}")
    print("="*60)



if __name__ == "__main__":
    main()
