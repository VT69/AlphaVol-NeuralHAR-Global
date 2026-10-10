# 03 — Methodology Audit (scoped, Step 1)

**Scope (per author brief of 2026-10-10):** the training loop, the train/val/test protocol, Optuna's use of validation data, how the reported variant was selected, early stopping, seeding, and the evaluation/metric code (QLIKE, MSE, DM, MCS). The current feature set is out of scope (it is being replaced in Step 2).

**Evidence script:** `scripts/audit_methodology_checks.py` → `analysis/artifacts/methodology_checks.json` (seeded; ~1 min). File:line references are to the files at commit `295157e`.

Severity: **BLOCKER** invalidates a reported result. **MAJOR** biases results or makes them irreproducible. **MINOR** is a deviation with small expected impact.

---

## 1. Verdict

The **metric code is numerically correct**. QLIKE and MSE match hand-computed values in all five implementations. DM matches an OLS reference to the 4 decimals the code rounds to. The MCS includes exactly the same models as `arch.bootstrap.MCS` in every case tested.

The **training protocol is not**. Walk-forward retraining is effectively frozen. For SPX/NIFTY the loss is clamped so that most training rows give no gradient. Nothing is seeded. Validation losses are reused optimistically. The reported architecture was chosen from three variants after each was evaluated on the test window. No held-out set was ever untouched.

## 2. Training loop and protocol

| # | Finding | Evidence | Severity |
|---|---|---|---|
| T1 | **Walk-forward fine-tuning runs at lr = 1e-6.** Pre-training steps `CosineAnnealingLR(T_max=50, eta_min=1e-6)` exactly 50 times, which ends at `eta_min`. The *same* optimizer is then reused for every 21-day fine-tune with no new scheduler. For Optuna's whole lr range [1e-4, 5e-3], the fine-tune rate is 1e-6, i.e. 1 % to 0.02 % of the tuned rate. The "expanding-window retraining" is therefore close to a no-op: the model is effectively the one trained once on train+val. | `train_neural_har.py:154-164` (scheduler), `:189-202` (fine-tune, no scheduler); `methodology_checks.json → lr_schedule` | MAJOR |
| T2 | **QLIKE loss clamps the prediction at [−10, 10], so rows below −10 give zero gradient.** For SPX/NIFTY, 90.8 % / 96.2 % of HAR-level forecasts and 61.0 % / 60.4 % of targets lie below −10. Gradient check: ∂loss/∂pred = 0.0 at pred = −11 (actual −12) vs 0.475 at pred = −9. Training is effectively restricted to the highest-variance days, and the model is never penalized for under-predicting variance below e^−10. The clamp is in the loss only; evaluation uses ±15 clipping, so train and test losses differ. BTC is unaffected (0 % below −10). | `neural_har.py:141-145`; `methodology_checks.json → qlike_clamp` | BLOCKER (SPX/NIFTY) |
| T3 | **No seeds anywhere in the live path.** torch, numpy, `random` and the Optuna TPE sampler are all unseeded. `torch.manual_seed(42)` exists only in dead code. Each run gives different hyperparameters, weights and forecasts, and no reported number can be regenerated. | `train_neural_har.py` (no `seed`); `:136` `create_study(direction="minimize")`; dead `neural_har.py:172` | BLOCKER (reproducibility) |
| T4 | **Optuna's use of validation data.** HPO (20 trials per architecture, not 40: `n_trials // 2`) trains on train and scores on val. Each trial reports the **best epoch's** val loss (early stopping, patience 5, ≤ 30 epochs). The final model is then trained for a **fixed 50 epochs** on train+val, so the epoch count that won HPO is never used. | `train_neural_har.py:87-134, 150-164, 267-276` | MAJOR |
| T5 | **Ensemble weights come from the HPO winners' best val losses** (`study.best_value`): the minimum over 20 trials × ≤ 30 epochs. That is an optimistically biased (winner's-curse) estimate, not the loss of the deployed models, which are retrained on train+val and never re-validated. | `train_neural_har.py:298-300` | MINOR |
| T6 | **Early stopping.** Used only inside HPO trials. The final training has none. The `EarlyStopping` class is dead code. | `neural_har.py:147-165` | MINOR |
| T7 | **Scaler** is fit on train only (correct: no leakage), never refit as the window expands, and applied to val/test. Exogenous NaNs are forward-filled (causal) and then **zero-filled** before the split. That causes no leakage, but it creates the time-indicator artefact shown in Phase 2. | `train_neural_har.py:230-232, 248-252` | MINOR (leakage); MAJOR (feature distortion: handled in Step 2) |
| T8 | **Minibatches are not shuffled** (`shuffle=False` for training). Every epoch ends on the most recent data, so the optimizer state is biased toward the end of the window. | `train_neural_har.py:153, 193, 261-262` | MINOR |
| T9 | **The HAR "prior" is trainable.** OLS weights initialize `har_linear`, but it is optimized jointly on QLIKE and can absorb the level shift (cf. Phase-1 A1). The ablation needed to separate "level" from "signal" was never run. | `neural_har.py:102, 119-123`; `train_neural_har.py:285` | MAJOR (interpretation) |
| T10 | **Test-row alignment is correct.** End-aligned Neural-HAR test rows and HAR forecasts refer to identical dates for all three assets (275 / 287 / 276). | `methodology_checks.json → alignment` | OK |

## 3. Variant selection and the held-out set

| # | Finding | Evidence | Severity |
|---|---|---|---|
| V1 | **The reported variant was chosen after test evaluation.** Three Neural-HAR variants were each run through the full pipeline including DM on the test window: DeepGRN (commit `e787251`, 15:27), FT-Transformer (`1a1455d`, 15:38) and the ensemble (`e27f6da`, 15:55), all on 2026-07-30, with result snapshots zipped after each. Test-window QLIKE DM statistics (BTC / SPX / NIFTY): DeepGRN −0.27 / +0.23 / +3.24; Transformer +0.83 / +5.04 / +2.35; Ensemble +1.98 / +5.05 / +2.44. The paper reports the ensemble, the variant with the largest BTC statistic and the only one significant for BTC (p = 0.048). | `methodology_checks.json → variants` (incl. `paper_reports_variant = ensemble_results`, `git_history`) | BLOCKER |
| V2 | **No genuine held-out set exists.** The test window was used for variant selection (V1), for the paper's narrative edits (`CHANGELOG_vs_internal_notes.md`), and for this audit's Phases 1–2. → Remedy: fresh sealed holdout (Step 4). | — | BLOCKER (remedied by R4) |
| V3 | **No cross-validation; single run.** One chronological split for HPO, and one model per architecture. There is no variance estimate across seeds or splits. | `train_neural_har.py:235-238` | MAJOR |

## 4. Metric and test implementations

### 4.1 QLIKE and MSE (hand-computed case)
Case: actual variance 2, predicted 1 (logs: ln 2, 0). By hand, QLIKE = 2 − ln 2 − 1 = **0.306853** and MSE (log scale) = (ln 2)² = **0.480453**.

| Implementation | Value | Match |
|---|---|---|
| `baseline_har.qlike_loss_numpy` | 0.306853 | ✓ |
| `regime_har.qlike_numpy` | 0.306853 | ✓ |
| `dm_test._loss_differential` (QLIKE) | 0.306853 | ✓ |
| `neural_har.qlike_loss` (torch) | 0.306853 | ✓ |
| `dm_test` MSE differential | 0.480453 | ✓ |
| QLIKE at pred = actual | 0.0 | ✓ (minimum) |

Five copies of the same loss with different clipping (±15 numpy, [−10, 10] torch on the prediction only) are a maintenance hazard, and the torch clamp is a bug (T2). → Single implementation in the rewrite.

### 4.2 Diebold–Mariano (`eval/dm_test.py`)

| Check | Repo | Reference | Note |
|---|---|---|---|
| h = 1, T = 300 (synthetic, seed 0) | 2.4793 | 2.479347 (OLS t-stat of d on a constant × HLN factor √((T−1)/T)) | Equal up to the repo's rounding to 4 dp |
| h = 2 | 2.5952 | 2.5991 (rectangular kernel, 1/T normalization, DM 1995); 2.6122 (statsmodels uniform HAC) | Small difference: the repo mixes `np.var(ddof=1)` with `np.cov` normalization |
| Identical forecasts | DM = 0, p = 1 | — | ✓ |
| p-value distribution | Normal: p = 0.0132 | HLN (1997) recommend Student-t(T−1): p = 0.0137 | MINOR |
| Sign | + ⇒ challenger better | — | Consistent with docs |
| `oos_r_squared` | 0.268489 | 0.268489 (hand) | ✓ The name suggests Campbell–Thompson, whose benchmark is the historical mean. Here the benchmark is HAR (a relative R²); label it as such. |

Issues: (D1) The paper's tests use `h = 1` although the information set was effectively 2-step (Phase 1). After the lag fix, h = 1 is right for a 1-step forecast, but loss differentials of volatility forecasts are serially correlated even at h = 1, so a **Newey–West (Bartlett) HAC** variance with a stated lag is the defensible choice. (D2) The rectangular kernel can yield a negative variance. The repo hides this behind a `max(V, 1e-20)` guard instead of failing. (D3) p-values are rounded to 4 dp, which produces the "p = 0.0000" in the paper. All three are MINOR now and fixed in the new harness.

### 4.3 Model Confidence Set (`eval/mcs_test.py`) vs `arch.bootstrap.MCS` (method R, circular block bootstrap, block 5)

| Case | Repo included | arch included | Repo p-values | arch p-values |
|---|---|---|---|---|
| Synthetic, clear ordering (worst/mid/best) | {best} | {best} | worst 0, mid 0, best 1 | worst 0, mid 0, best 1 |
| Committed HAR vs HAR-S QLIKE, BTC (275) | {HAR, HAR-S} | {HAR, HAR-S} | 1.0, 1.0 | HAR-S **0.300**, HAR 1.0 |
| … SPX (287) | {HAR, HAR-S} | {HAR, HAR-S} | 1.0, 1.0 | HAR **0.516**, HAR-S 1.0 |
| … NIFTY (276) | {HAR} | {HAR} | HAR-S 0.006 | HAR-S 0.005 |
| **Size** under equal predictive ability (3 correlated models, AR(1) ρ = 0.3 losses, T = 300, B = 300, 400 replications) | rejects 16.25 % | rejects 16.25 % | nominal 10 %, MC SE 1.5 pp | |

Findings:
* **M1 (OK).** Inclusion decisions agree with the reference implementation in every case, including a size study where both reject at an identical rate.
* **M2 (MAJOR, reporting).** For models that survive, the repo writes p = 1.0 "by convention" (`mcs_test.py:283-285`) instead of their MCS p-value. Paper tables therefore show p = 1.0 for every survivor, hiding e.g. HAR-S's 0.30 for BTC.
* **M3 (MAJOR, inference).** At T ≈ 300 with block length 5, **both** implementations over-reject: 16.25 % vs a nominal 10 %, about 4 MC standard errors. This is a property of the procedure at this sample size, not a code bug. The pre-registration must fix the block length by a data-driven rule and report MCS as supporting evidence, not as the primary test.
* **M4 (MINOR).** The elimination rule removes the model with the highest mean loss (`mcs_test.py:128-129`) instead of Hansen et al.'s e_R = argmax_i max_j t_ij. This made no difference in any case tested.
* **M5 (MINOR, unused).** The `TMax` variance `var_db[i,:].mean()/M` (`mcs_test.py:_T_max, 136, 181, 260`) is not the variance of d̄_i·. It is not used in the paper (TR only), but it is wrong if anyone switches.
* **M6 (MINOR).** MCS is run on QLIKE only, with B = 1000 and a fixed block of 5. Neither choice is justified in the paper.

## 5. Carried into the rewrite (Step 2 / harness)

| Finding | Remedy (logged in DECISIONS.md before implementation) |
|---|---|
| T1 | Fresh optimizer + scheduler per refit, or refit from scratch at each step (to decide in pre-registration, with the compute budget) |
| T2 | One QLIKE implementation, no prediction clamp in the loss (or a clamp far outside the data range), with a unit test of the gradient |
| T3 | Global seeding utility (random/numpy/torch/Optuna sampler) + `torch.use_deterministic_algorithms(True)`; ≥ 5 seeds (R1) |
| T4/T5 | Epoch count chosen on validation and reused; ensemble weights from out-of-sample validation predictions of the deployed models |
| V1–V3 | Pre-registration (Step 6), sealed holdout (Step 4), all variants reported |
| D1–D3 | DM with Newey–West HAC (lag stated), Student-t(T−1), unrounded p-values; Holm correction across assets × losses |
| M2–M6 | Use `arch.bootstrap.MCS` (maintained reference) with reported p-values for all models; block length by rule; MCS as secondary evidence |
