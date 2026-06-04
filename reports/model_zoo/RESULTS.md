# Model Zoo — Extended Benchmark

All models on the **approved (mitigated) feature policy**, identical splits (800,000 rows). Uniform metrics incl. ECE and scoring latency.

| model               | family                |   gini |   gini_ci_lo |   gini_ci_hi |     ks |   brier |    ece |   latency_ms_per_1k |   gini_lift_vs_incumbent | note                                                      |
|:--------------------|:----------------------|-------:|-------------:|-------------:|-------:|--------:|-------:|--------------------:|-------------------------:|:----------------------------------------------------------|
| stacked_ensemble    | ensemble              | 0.689  |       0.6748 |       0.7039 | 0.5598 |  0.0531 | 0.0018 |                3.73 |                   0.1028 | logit-stack of lightgbm_champion, xgboost, catboost, tabm |
| catboost            | GBM                   | 0.6883 |       0.6746 |       0.7026 | 0.5587 |  0.0532 | 0.0013 |                0.69 |                   0.1021 |                                                           |
| lightgbm_champion   | GBM                   | 0.6879 |       0.6731 |       0.7039 | 0.5605 |  0.0532 | 0.0022 |                1.12 |                   0.1016 | deployed model (from run_pipeline)                        |
| xgboost             | GBM                   | 0.6869 |       0.6723 |       0.7016 | 0.5589 |  0.0532 | 0.0019 |                1    |                   0.1006 |                                                           |
| ft_transformer      | deep                  | 0.6847 |       0.6701 |       0.6996 | 0.5593 |  0.0533 | 0.0025 |                7.99 |                   0.0984 |                                                           |
| ebm                 | GAM (glass box)       | 0.6845 |       0.67   |       0.6983 | 0.5591 |  0.0532 | 0.0024 |                1.47 |                   0.0983 | 15 learned pairwise interaction terms                     |
| random_forest       | bagging               | 0.6809 |       0.6684 |       0.6963 | 0.5567 |  0.0537 | 0.0055 |                1.84 |                   0.0947 |                                                           |
| tabm                | deep                  | 0.6776 |       0.6642 |       0.693  | 0.5533 |  0.0539 | 0.0022 |                0.88 |                   0.0913 | BatchEnsemble MLP, k=8 members (Gorishniy 2024)           |
| mlp_resnet          | deep                  | 0.6736 |       0.6601 |       0.6893 | 0.5492 |  0.0546 | 0.0134 |                0.55 |                   0.0874 |                                                           |
| lightgbm_monotone   | GBM                   | 0.6349 |       0.6206 |       0.6512 | 0.5368 |  0.0539 | 0.0043 |                1.28 |                   0.0487 | 11 of 16 features constrained                             |
| scorecard_augmented | scorecard (glass box) | 0.6241 |       0.6109 |       0.6423 | 0.5216 |  0.0576 | 0.0147 |                0.62 |                   0.0378 | incumbent + 6 GBM-distilled cross terms                   |
| incumbent           | scorecard (glass box) | 0.5862 |       0.5687 |       0.6051 | 0.5056 |  0.0592 | 0.0184 |                0.4  |                   0      |                                                           |

## Findings

- **Glass-box interactions**: EBM recovers 97% of the champion's lift (trailing it by only 0.003 Gini) while every term remains a plottable shape function — strong evidence the portfolio's risk is pairwise-interaction-driven.
- **Distillation**: grafting 6 GBM-derived SHAP-interaction crosses onto the incumbent scorecard recovers 37% of the champion's lift (+0.038 Gini) with zero architecture change — a viable fallback if tree models were ever disallowed.
- **Price of monotonicity**: regulator-friendly monotone constraints on 11 of 16 features cost 0.053 Gini. Notably, the unconstrained features include the regime-dependent ones (e.g. num_credit_lines) where forcing a direction would be wrong.
- **Deep learning**: FT-Transformer / ResNet / TabM cluster with the GBMs; none separates from LightGBM beyond CI noise, at materially higher scoring latency.
- **Ensembling**: the stack adds +0.001 Gini over the champion — the families are largely capturing the same signal.
- **Calibration**: the champion is already well calibrated — an isotonic layer does not improve it (ECE 0.0022 → 0.0024), Brier 0.0532 → 0.0533; layer saved as `artifacts/champion_isotonic.joblib` for monitoring use (recalibration is the first response to drift per the monitoring plan).

Figures: `figures/model_zoo_gini.png`, `figures/gini_vs_latency.png`, `figures/ebm_terms.png`, `figures/calibration_isotonic.png`.