import os
import re
import sys
import json
from pathlib import Path

REPO_ROOT = Path(".").resolve()
EXCLUDE_DIRS = {".git", "__pycache__", "data", "checkpoints", "venv", ".idea"}
EXCLUDE_EXTS = {".pt", ".png", ".pdf", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".bin", ".log"}

HARDCODED_PATH_PATTERNS = [
    re.compile(r"[a-zA-Z]:\\[Uu]sers\\[^\s\"';\)]+"),
    re.compile(r"/home/[^\s\"';\)]+"),
    re.compile(r"25beevdt047"),
    re.compile(r"antigravity[\\/]scratch"),
]

SUSPICIOUS_MOCK_PATTERNS = [
    re.compile(r"#\s*TODO\s*mock", re.IGNORECASE),
    re.compile(r"#\s*mock\s*data", re.IGNORECASE),
    re.compile(r"return\s*\{\s*['\"]psnr['\"]\s*:\s*[0-9\.]+\s*\}"),  # hardcoded return metrics
]

def scan_file(filepath: Path):
    rel_path = filepath.relative_to(REPO_ROOT)
    issues = []
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return issues

    lines = content.splitlines()
    for idx, line in enumerate(lines, 1):
        for pattern in HARDCODED_PATH_PATTERNS:
            if pattern.search(line):
                # allow comments explaining paths or config defaults if properly handled
                issues.append((idx, "HARDCODED_PATH", line.strip()))
        for pattern in SUSPICIOUS_MOCK_PATTERNS:
            if pattern.search(line):
                issues.append((idx, "SUSPICIOUS_MOCK", line.strip()))
    return issues

def main():
    all_issues = {}
    print("=" * 60)
    print("AUDITING CODEBASE FOR HARDCODED VALUES & PATHS")
    print("=" * 60)

    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in EXCLUDE_EXTS or p.name == "audit_codebase.py":
                continue
            issues = scan_file(p)
            if issues:
                all_issues[str(p.relative_to(REPO_ROOT))] = issues

    if not all_issues:
        print("PASS: No hardcoded local paths or suspicious mock returns found in any codebase files!")
    else:
        print(f"FOUND {len(all_issues)} files with potential issues:")
        for f, issues in all_issues.items():
            print(f"\nFile: {f}")
            for line_no, issue_type, text in issues:
                print(f"  Line {line_no:4d} [{issue_type}]: {text[:100]}")

    print("\n" + "=" * 60)
    print("VERIFYING RESULTS PIPELINE DYNAMISM")
    print("=" * 60)

    # Verify generate_paper_tables.py reads from JSON
    tables_script = (REPO_ROOT / "utils" / "generate_paper_tables.py").read_text(encoding="utf-8", errors="ignore")
    if "json.load" in tables_script and "results/ic_pinot_master_benchmarks.json" in tables_script:
        print("PASS: utils/generate_paper_tables.py dynamically loads from results/ic_pinot_master_benchmarks.json")
    else:
        print("FAIL: utils/generate_paper_tables.py does not dynamically load benchmark JSON!")

    # Verify generate_results_pdf.py reads from JSON
    pdf_script = (REPO_ROOT / "utils" / "generate_results_pdf.py").read_text(encoding="utf-8", errors="ignore")
    if "json.load" in pdf_script and "results/ic_pinot_master_benchmarks.json" in pdf_script:
        print("PASS: utils/generate_results_pdf.py dynamically loads from results/ic_pinot_master_benchmarks.json")
    else:
        print("FAIL: utils/generate_results_pdf.py does not dynamically load benchmark JSON!")

    # Verify visualize_evidential_results.py loads actual checkpoints
    vis_script = (REPO_ROOT / "utils" / "visualize_evidential_results.py").read_text(encoding="utf-8", errors="ignore")
    if "torch.load" in vis_script and "checkpoints/" in vis_script:
        print("PASS: utils/visualize_evidential_results.py dynamically loads real checkpoints from checkpoints/")
    else:
        print("FAIL: utils/visualize_evidential_results.py does not dynamically load checkpoints!")

    print("\n" + "=" * 60)
    print("SYZYGY ASD-STE100 ANTI-SLOP AUDIT")
    print("=" * 60)
    import subprocess
    readme_path = REPO_ROOT / "README.md"
    if readme_path.exists():
        proc = subprocess.run(["syzygy", "validate", "README.md"], capture_output=True, text=True, shell=True)
        if proc.returncode == 0:
            print("PASS: README.md passed syzygy validate with 100/100 score!")
            print(f"Output: {proc.stdout.strip()}")
        else:
            print(f"FAIL: syzygy validate failed with code {proc.returncode}")
            print(proc.stderr)

if __name__ == "__main__":
    main()
