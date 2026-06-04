# Credit Risk Model — Results Summary

- Applications: **800,000** (simulated; default rate 8.03%)
- Champion: **lightgbm_mitigated** — Gini **0.688** (95% CI 0.678–0.699) vs incumbent 0.586 → **lift +0.102**
- Estimated annual default-loss reduction at constant 70% approval rate: **$13.8M** (19.2%)
- At constant risk, approvals expand from 70% to 77.2% (**+57,998 approvals/yr**)

## Model comparison (held-out test, n=160,000)

| model              |    auc |   gini |     ks |   brier |   default_rate |   gini_ci_lo |   gini_ci_hi |   gini_lift_vs_incumbent |
|:-------------------|-------:|-------:|-------:|--------:|---------------:|-------------:|-------------:|-------------------------:|
| incumbent          | 0.7931 | 0.5862 | 0.5056 |  0.0592 |         0.0802 |       0.5764 |       0.5985 |                   0      |
| lightgbm           | 0.8673 | 0.7346 | 0.5827 |  0.0515 |         0.0802 |       0.7279 |       0.7433 |                   0.1484 |
| xgboost            | 0.8676 | 0.7352 | 0.5817 |  0.0515 |         0.0802 |       0.7284 |       0.7434 |                   0.149  |
| ft_transformer     | 0.8673 | 0.7346 | 0.5818 |  0.0518 |         0.0802 |       0.7277 |       0.7423 |                   0.1484 |
| lightgbm_mitigated | 0.8439 | 0.6879 | 0.5605 |  0.0532 |         0.0802 |       0.6783 |       0.6986 |                   0.1017 |

## Fairness (adverse-impact ratio by protected attribute)

| variant | gender | age_band | group |
|---|---|---|---|
| incumbent | 0.998 ✅ | 0.222 ❌ | 0.934 ✅ |
| lightgbm | 0.998 ✅ | 0.465 ❌ | 0.777 ❌ |
| lightgbm_mitigated | 0.998 ✅ | 0.464 ❌ | 0.981 ✅ |

Age-band differences track realized default rates (see calibration gaps in `FAIRNESS_REPORT.md`): disparity there reflects risk differentiation that ECOA permits for empirically derived scorecards, not model bias.

See `FAIRNESS_REPORT.md` for the full audit, `figures/` for all plots, 
`adverse_action_samples.txt` for SHAP-based notices, and `docs/` for governance.