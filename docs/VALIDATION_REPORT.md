# Independent Model Validation Report
**Model:** Consumer Credit Risk PD Model v1.0 (LightGBM champion)
**Scope:** Full-scope initial validation prior to production deployment
**Framework:** Structured per SR 11-7 / OCC 2011-12 (conceptual soundness, ongoing monitoring, outcomes analysis)

---

## 1. Executive summary
The champion model is **fit for purpose** for approve/decline rank-ordering at the stated approval-rate target, subject to the conditions in §7. It delivers a statistically significant Gini improvement over the incumbent scorecard (non-overlapping bootstrap CIs; see `reports/RESULTS.md`), is materially better calibrated, passes the fair-lending test suite after the documented mitigation, and produces compliant adverse-action reason codes.

## 2. Conceptual soundness
- **Methodology.** Gradient-boosted trees are appropriate for tabular credit data with interaction-driven risk; the choice is benchmarked, not assumed: a WOE-scorecard, XGBoost, and an FT-Transformer neural network were trained as alternatives on identical splits (results in `reports/RESULTS.md`). The neural alternative matches the GBMs within bootstrap noise but does not surpass them, while adding training cost and opacity — its complexity is therefore not justified; documented as a governance finding, not a failure.
- **Data.** The development sample is a documented synthetic population (`src/creditrisk/data/simulate.py`) with a fully specified data-generating process. This is unusual relative to production practice and is the validation's primary limitation (§7), but it confers one advantage: fairness ground truth is known by construction, so the fairness test suite itself was validated (it correctly flags the injected bias and correctly passes the unbiased configuration).
- **Feature review.** All 16 features have monotone or economically interpretable relationships confirmed via SHAP dependence plots. One candidate feature was rejected on fair-lending grounds (§5).

## 3. Development evidence review
- Split hygiene confirmed: early stopping uses the validation set; the test set is untouched until final evaluation.
- Hyperparameters are modest and regularized; no evidence of test-set leakage; retraining with a different seed reproduces metrics within bootstrap noise.
- Reproducibility: single command (`python scripts/run_pipeline.py`), pinned dependencies, fixed seeds.

## 4. Outcomes analysis
- **Discrimination.** Champion Gini exceeds the incumbent by ≈ +0.08–0.10 with non-overlapping 95% bootstrap CIs (`reports/figures/gini_comparison.png`). KS and cumulative-capture (gains) corroborate the rank-ordering improvement.
- **Calibration.** Decile-level calibration is near the identity line (`reports/figures/calibration.png`); Brier score improves on the incumbent. Within-group calibration gaps are ≈ 0 for all protected groups.
- **Benchmarking.** XGBoost and the FT-Transformer match LightGBM within noise. The incumbent scorecard's deficit is structural (additivity cannot express the interaction effects present in the data), not a tuning artifact — supported by the scorecard's near-identical performance across bin counts, and independently confirmed by the extended model zoo (`reports/model_zoo/RESULTS.md`): an EBM (glass-box GAM with pairwise interactions) recovers most of the challenger lift, and grafting GBM-distilled SHAP-interaction cross terms onto the scorecard recovers a substantial fraction — both isolating *interactions* as the source of the gap. The zoo also quantifies the price of monotone constraints (a deployment option if reason-code stability is prioritized) and shows stacking adds little over the champion.
- **Sensitivity.** Business-impact estimates were re-run across LGD 0.45–0.65 and approval rates 60–80%; the loss-reduction conclusion is robust in sign and order of magnitude.

## 5. Fair lending review
- The naive challenger (including `geo_risk_index`) **fails** the four-fifths rule for the protected demographic group at the production threshold. Root cause: the feature is a regional default-rate index that encodes group membership beyond its legitimate signal — a proxy/redlining mechanism.
- Mitigation: feature exclusion. The champion passes AIR ≥ 0.80 on gender and demographic group with an immaterial Gini cost relative to the naive challenger, while retaining the full lift over the incumbent.
- Age-band approval-rate differences persist but track realized default rates (within-group calibration ≈ 0); treated as permissible risk differentiation under ECOA's empirically-derived-scorecard provisions, with ongoing monitoring.
- Residual risk: income measurement bias affecting one segment is partially unmitigated by feature exclusion (it enters through legitimately predictive ratios); flagged for data-quality remediation rather than model adjustment.

## 6. Explainability & adverse action
- SHAP additivity verified (contributions sum to the log-odds margin). Reason-code mapping covers all features; the top-4 extraction is deterministic and audit-logged.
- Sample notices (`reports/adverse_action_samples.txt`) reviewed for Reg B compliance-style language.

## 7. Conditions, limitations, and findings
| # | Severity | Finding | Required action |
|---|---|---|---|
| 1 | High | Development data is simulated | Validate on production or public real data (Home Credit companion pipeline) before any production use |
| 2 | Medium | Income measurement bias persists post-mitigation | Data-quality remediation plan; monitor segment-level calibration quarterly |
| 3 | Medium | PD calibration assumes stable macro conditions | Recalibration trigger in monitoring plan (PSI > 0.25 or default-rate drift > 20% relative) |
| 4 | Low | Reason-code granularity for correlated features | Periodic review of reason-code frequency distribution |

## 8. Validation outcome
**Approved with conditions** (findings 1–4 tracked to closure by Model Risk Management). Next full revalidation: 12 months from deployment or upon any monitoring trigger.
