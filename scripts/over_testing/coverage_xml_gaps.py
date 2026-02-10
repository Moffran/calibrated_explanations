"""Extract contiguous zero-hit line blocks from coverage.xml.

Outputs CSV lines: file,start,end,length
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict


def parse_coverage_xml(path: str):
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {}
    files_map = defaultdict(dict)
    # find all <class> elements under packages/classes
    for class_el in root.findall('.//class'):
        filename = class_el.get('filename')
        if not filename:
            continue
        for line in class_el.findall('.//line'):
            num = int(line.get('number'))
            hits = int(line.get('hits', '0'))
            files_map[filename][num] = hits
    return files_map


def find_blocks(lines_map, threshold=20):
    blocks = []
    sorted_lines = sorted(lines_map.keys())
    start = None
    prev = None
    for ln in sorted_lines:
        hit = lines_map.get(ln, 0)
        if hit == 0:
            if start is None:
                start = ln
            prev = ln
        else:
            if start is not None:
                if prev - start + 1 >= threshold:
                    blocks.append((start, prev))
                start = None
                prev = None
    if start is not None and prev is not None and prev - start + 1 >= threshold:
        blocks.append((start, prev))
    return blocks


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--xml', default='coverage.xml')
    p.add_argument('--threshold', type=int, default=20)
    args = p.parse_args(argv)

    files_map = parse_coverage_xml(args.xml)
    for fname, lm in files_map.items():
        blocks = find_blocks(lm, args.threshold)
        for s, e in blocks:
            print(f"{fname},{s},{e},{e-s+1}")


if __name__ == '__main__':
    main()
