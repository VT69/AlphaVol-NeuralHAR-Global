# Decisions and binding requirements (human-in-the-loop log)

Every design decision and binding instruction from the author, in order. Later phases must satisfy every open requirement here. When an item is resolved, mark it with the commit that resolved it.

| # | Date | Source | Decision / requirement | Status |
|---|---|---|---|---|
| D1 | 2026-10-09 | Author | Work on a non-default branch. This session is bound to `claude/awesome-curie-6nitg4`, which is used instead of `research-audit`. | In effect |
| D2 | 2026-10-09 | Author | Phases 1–3 are read-only for project sources. Present BLOCKER/MAJOR fixes and wait for approval before editing. | In effect |
| D3 | 2026-10-09 | Author | **Do not change the volatility measure yet.** Document and recommend only (see `target_quality.md` §4). | In effect: awaiting decision on R1/R2/R3/R4 |
| D4 | 2026-10-09 | Author | Sentiment-coverage diagnosis is the top priority (done: `sentiment_coverage_diagnosis.md`). | Awaiting decision on re-fetch options A–D |
| **R1** | 2026-10-09 | Author | **Phase 5 requirement:** Neural-HAR must be **retrained with fixed seeds, ≥ 5 independent runs**, and compared against a **bias-corrected HAR** baseline with a **Diebold–Mariano test**. The current comparison is against a test-set-selected maximum (best of 3 architecture variants) and is **not valid**. | Open (Phase 5) |
| D5 | 2026-10-09 | Author | Deadline: given as the placeholder "[X]", so not known. Plan without a fixed date; ask. | Open |

## How Phase 5 will satisfy R1 (proposed; details in Phase 4 for approval)

1. **Seeds.** Seed `random`, `numpy`, `torch` (CPU), and the Optuna `TPESampler(seed=…)` per run. Set `torch.use_deterministic_algorithms(True)`. Run with seeds 0–4 at minimum, more if the compute budget allows. Report every run, not the best.
2. **Architecture choice fixed before the test window.** Pick GRN / Transformer / ensemble on validation loss only, record the choice, and never revisit it after looking at test losses. All three variants are reported as an ablation, labelled as such.
3. **Baselines on the identical trailing window.**
   (a) HAR as specified in the paper;
   (b) HAR + causal smearing bias correction (`scripts/audit_qlike_bias_check.py` logic);
   (c) HAR fitted directly on QLIKE (same loss as Neural-HAR);
   (d) Neural-HAR with exogenous inputs zeroed: the ablation that isolates the sentiment and non-linear contribution.
4. **Tests.** DM with HAC variance (h ≥ 2 given the lag structure, or h = 1 after the lag fix), QLIKE and MSE, per seed and on the seed-averaged forecast. Report the distribution of DM statistics across seeds and the share of seeds that are significant, plus MCS on the full set {(a)…(d), Neural-HAR}. Correct for multiple comparisons across assets × losses (Holm).
