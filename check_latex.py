# coding: utf-8
"""
check_latex.py  — static pre-flight check for the paper/ directory.
Checks:
  1. Every \\input{} target exists on disk
  2. Every \\label{} is matched by at least one \\ref{}
  3. Every \\ref{} has a corresponding \\label{}
  4. Every \\cite{} key exists in references.bib
  5. Every \\includegraphics{} target exists on disk
Run: python check_latex.py
"""
import os, re, sys

PAPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper")

def read_all_tex() -> str:
    """Return concatenated content of all .tex files under paper/."""
    chunks = []
    for dirpath, _, files in os.walk(PAPER_DIR):
        for fn in files:
            if fn.endswith(".tex"):
                with open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace") as f:
                    chunks.append(f.read())
    return "\n".join(chunks)

def read_bib() -> set:
    bib_path = os.path.join(PAPER_DIR, "references.bib")
    if not os.path.exists(bib_path):
        return set()
    txt = open(bib_path, encoding="utf-8", errors="replace").read()
    return set(re.findall(r"@\w+\{([^,]+),", txt))

errors = []
warnings = []

all_tex = read_all_tex()

# 1. \input{} targets
for m in re.finditer(r"\\input\{([^}]+)\}", all_tex):
    rel = m.group(1)
    if not rel.endswith(".tex"):
        rel += ".tex"
    full = os.path.join(PAPER_DIR, rel)
    if not os.path.exists(full):
        errors.append(f"Missing \\input target: {rel}")

# 2 & 3. \label / \ref consistency
labels  = set(re.findall(r"\\label\{([^}]+)\}", all_tex))
refs    = set(re.findall(r"\\ref\{([^}]+)\}", all_tex))
for r in sorted(refs - labels):
    errors.append(f"Undefined \\ref{{{r}}} — no matching \\label")
for l in sorted(labels - refs):
    warnings.append(f"Unused \\label{{{l}}} — never \\ref'd")

# 4. \cite keys
bib_keys = read_bib()
cited = set()
for m in re.finditer(r"\\cite[tp]?\*?\{([^}]+)\}", all_tex):
    for key in m.group(1).split(","):
        cited.add(key.strip())
for c in sorted(cited - bib_keys):
    errors.append(f"Undefined \\cite key: {c}")
for k in sorted(bib_keys - cited):
    warnings.append(f"Unused bib key: {k}")

# 5. \includegraphics{} targets
for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", all_tex):
    rel = m.group(1)
    full = os.path.join(PAPER_DIR, rel)
    # also try with .png extension
    if not os.path.exists(full) and not os.path.exists(full + ".png"):
        errors.append(f"Missing \\includegraphics target: {rel}")

# ── Report ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("LaTeX pre-flight check")
print("=" * 60)
if errors:
    print(f"\n  ERRORS ({len(errors)}):")
    for e in errors:
        print(f"    [ERROR] {e}")
else:
    print("\n  No errors found.")

if warnings:
    print(f"\n  WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"    [WARN]  {w}")
else:
    print("  No warnings.")

print("=" * 60)
sys.exit(1 if errors else 0)
