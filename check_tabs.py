with open("src/calibrated_explanations/plotting.py", "rb") as f:
    for i, line in enumerate(f):
        s = line.decode("utf-8", errors="ignore")
        stripped = s.strip()
        if stripped.startswith("try:") or stripped.startswith("except"):
             indent = len(s) - len(s.lstrip())
             print(f"Line {i+1}: Indent {indent} | {stripped}")
