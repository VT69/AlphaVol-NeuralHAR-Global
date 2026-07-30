# coding: utf-8
"""
generate_tables.py
Produces all paper/tables/*.tex from the canonical CSVs (run_paper.py output).

Design decisions:
- Uses Styler.to_latex(hrules=True) for booktabs-style rules (pandas >= 1.3).
- Column headers with underscores → escaped (\_), Unicode → LaTeX equivalents.
- table4_dm_tests.csv is NOT rendered: it was produced by a buggy front-aligned
  code path in paper_tables.py and has been superseded by table2_dm_test.csv
  (canonical, from run_paper.py's end-aligned step_dm()).
- table5_mcs.csv IS rendered: it is written by paper_tables.py's table5_mcs(),
  which now uses correct end-alignment after the bug-fix applied in this session.
  table3_mcs.csv is also rendered as the authoritative 3-model MCS (run_paper.py).
"""
import os, re, sys
import pandas as pd
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT    = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(ROOT, "data", "paper_tables")
TEX_DIR = os.path.join(ROOT, "paper", "tables")
os.makedirs(TEX_DIR, exist_ok=True)

# ── Character sanitisation ────────────────────────────────────────────────────
_UNICODE_MAP = {
    '—': '--',     # em-dash
    '–': '--',     # en-dash
    '²': '$^{2}$',
    '³': '$^{3}$',
    'α': r'$\alpha$',
    'β': r'$\beta$',
    '≥': r'$\geq$',
    '≤': r'$\leq$',
    '×': r'$\times$',
    '\u2019': "'",   # right single quote
    '\u201c': "``",  # left double quote
    '\u201d': "''",  # right double quote
}

def sanitise(s: str) -> str:
    """Replace Unicode characters with LaTeX-safe equivalents."""
    for char, repl in _UNICODE_MAP.items():
        s = s.replace(char, repl)
    return s

def clean_colname(col: str) -> str:
    """Escape underscores in column headers that would end up outside math mode."""
    col = sanitise(col)
    # Wrap identifiers that look like variable names with underscores in \texttt{}
    if '_' in col and not col.startswith('$'):
        col = col.replace('_', r'\_')
    return col

def safe_fmt(x):
    if isinstance(x, float) and np.isnan(x):
        return ''
    if isinstance(x, float):
        return '%.4f' % x
    s = str(x)
    return sanitise(s)

# ── Core helpers ──────────────────────────────────────────────────────────────
def wrap_table(body: str, label: str, caption: str, note: str = '') -> str:
    parts = [
        "\\begin{table}[htbp]\n",
        "\\centering\n",
        f"\\caption{{{caption}}}\n",
        f"\\label{{tab:{label}}}\n",
        "\\resizebox{\\textwidth}{!}{%\n",
        body,
        "}\n",
    ]
    if note:
        parts.append(
            "\\begin{tablenotes}\\footnotesize\n"
            f"  \\item {note}\n"
            "\\end{tablenotes}\n"
        )
    parts.append("\\end{table}\n")
    return ''.join(parts)

def to_tex(df: pd.DataFrame, label: str, caption: str,
           note: str = '') -> str:
    # Clean column names
    df = df.copy()
    df.columns = [clean_colname(c) for c in df.columns]

    styled = df.style.format(safe_fmt)
    try:
        body = styled.to_latex(hrules=True, index=False)
        # Post-process: sanitise any remaining Unicode that leaked through cell values
        body = sanitise(body)
    except TypeError:
        # Older pandas fallback
        body = df.to_latex(index=False, escape=True)
        body = sanitise(body)
    return wrap_table(body, label, caption, note)

def save(df, filename, label, caption, note=''):
    tex = to_tex(df, label, caption, note)
    path = os.path.join(TEX_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(tex)
    print(f"  OK  {filename}")

SIG_NOTE = (r"Significance: $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$. "
            r"HAC standard errors.")

# ── Tables ────────────────────────────────────────────────────────────────────

# Table 1 — Descriptive statistics
df = pd.read_csv(os.path.join(CSV_DIR, "table1_descriptive_stats.csv"))
save(df, "table1_descriptive_stats.tex", "desc_stats",
     "Descriptive Statistics for Daily Log Realized Volatility")
print(df.to_string()); print()

# Table 1b — Full-sample OOS metrics (HAR/HAR-S, long window, run_paper.py)
df = pd.read_csv(os.path.join(CSV_DIR, "table1_oos_metrics.csv"))
save(df, "table1_oos_metrics.tex", "oos_full",
     "Full-Sample OOS Forecast Accuracy: HAR and HAR-S (Expanding Window, $N>2{,}400$)")
print(df.to_string()); print()

# Table 2a — In-sample coefficients
df = pd.read_csv(os.path.join(CSV_DIR, "table2_insample_coefs.csv"))
save(df, "table2_insample_coefs.tex", "insample",
     "In-Sample HAC-OLS Coefficients: HAR and HAR-S", note=SIG_NOTE)
print(df.to_string()); print()

# Table 2b — DM tests  *** SOLE AUTHORITATIVE DM TABLE ***
# Source: run_paper.py step_dm(), end-aligned (act[-min_n:])
# table4_dm_tests.csv is NOT used: it came from the front-aligned bug in
# notebooks/paper_tables.py and is superseded entirely by this table.
df = pd.read_csv(os.path.join(CSV_DIR, "table2_dm_test.csv"))
save(df, "table2_dm_test.tex", "dm_tests",
     r"Diebold--Mariano Test (HLN Correction): Challenger vs.\ HAR Benchmark "
     r"(End-Aligned Trailing Window, $N\approx275$--$287$)",
     note=(r"Positive DM statistic favours the challenger model. "
           r"`Better' column: A\,=\,HAR preferred, B\,=\,Challenger preferred. "
           r"All arrays aligned to the chronological end of the shorter series. "
           r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$."))
print(df.to_string()); print()

# Table 3 — OOS metrics, harmonised trailing window (all 3 models)
df = pd.read_csv(os.path.join(CSV_DIR, "table3_oos_metrics.csv"))
save(df, "table3_oos_metrics.tex", "oos_metrics",
     "OOS Forecast Accuracy: All Models on Harmonised Trailing Window")
print(df.to_string()); print()

# Table 3b — 3-model MCS (run_paper.py, end-aligned)
df = pd.read_csv(os.path.join(CSV_DIR, "table3_mcs.csv"))
save(df, "table3_mcs.tex", "mcs_full",
     r"Model Confidence Set: All Three Models ($\alpha=0.10$, QLIKE Loss, "
     r"End-Aligned Trailing Window)")
print(df.to_string()); print()

# Table 4 — Regime-conditional coefficients (run_paper.py)
df = pd.read_csv(os.path.join(CSV_DIR, "table4_regime_coefficients.csv"))
save(df, "table4_regime_coefficients.tex", "regime_coefs",
     "Regime-Conditional HAC-OLS Coefficients: HAR and HAR-S", note=SIG_NOTE)
print(df.to_string()); print()

# Table 5a — 2-model MCS: Neural-HAR vs HAR (paper_tables.py, fixed end-aligned)
df = pd.read_csv(os.path.join(CSV_DIR, "table5_mcs.csv"))
save(df, "table5_mcs.tex", "mcs",
     r"Model Confidence Set: Neural-HAR vs.\ HAR ($\alpha=0.10$, QLIKE Loss, "
     r"Trailing Window $N\approx275$--$287$)")
print(df.to_string()); print()

# Table 5b — Regime OOS metrics (run_paper.py)
df = pd.read_csv(os.path.join(CSV_DIR, "table5_regime_oos.csv"))
save(df, "table5_regime_oos.tex", "regime_oos",
     "Regime-Conditional OOS Forecast Accuracy: HAR and HAR-S")
print(df.to_string()); print()

# Table 6 — Economic significance backtest (run_paper.py)
df = pd.read_csv(os.path.join(CSV_DIR, "table6_backtest.csv"))
save(df, "table6_backtest.tex", "backtest",
     r"Economic Significance: Volatility-Targeting Backtest "
     r"(5\,bps Slippage, Quarter-Kelly, $\leq 5{\times}$ Leverage, "
     r"$N\approx275$--$287$)")
print(df.to_string()); print()

print("\nAll tables written to:", TEX_DIR)
