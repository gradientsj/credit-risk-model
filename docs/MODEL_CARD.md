# Model Card — Consumer Credit Risk Scoring Model v1.0

## Model details
| | |
|---|---|
| **Model name** | Consumer Credit Risk PD Model (champion) |
| **Version** | 1.0.0 |
| **Type** | Gradient-boosted decision trees (LightGBM), binary classifier |
| **Output** | Probability of default (PD) within the loan term; points-scaled score (PDO = 20, 600 ≡ 19:1 odds) |
| **Incumbent replaced** | Weight-of-evidence binned logistic-regression scorecard |
| **Owners** | Credit Risk Modeling (1st line); validated by Model Risk Management (2nd line) |
| **Artifacts** | `artifacts/champion.joblib`; metrics in `reports/metrics.json` |

## Intended use
- **Primary use**: rank-order unsecured personal-loan applicants by default risk to support approve/decline decisions at a portfolio-level approval-rate target.
- **Users**: automated decisioning system; credit officers for referrals.
- **Out of scope**: pricing without recalibration; collections prioritization; any use of the score for non-credit purposes; populations materially different from the development sample (e.g., secured lending).

## Training data
- 800,000 loan applications (simulated development sample; see `src/creditrisk/data/simulate.py` for the documented generating process). Default rate ≈ 8%.
- 70/10/20 train/validation/test split; validation used for early stopping only; all reported metrics are from the untouched 20% test set.
- **Protected attributes (gender, age band, demographic group) are present in the data for audit purposes only and are never model inputs.**

## Features (16)
Bureau-style characteristics (utilization, history length, delinquencies, inquiries, credit lines), affordability ratios (DTI, loan-to-income, expenses, savings), stability (employment type/tenure), and application terms (amount, purpose, housing).
**Excluded by governance decision**: `geo_risk_index` — a regional default-rate index that improved Gini but failed disparate-impact testing (see Fairness section and `reports/FAIRNESS_REPORT.md`).

## Performance (held-out test; see reports/metrics.json for exact values)
- Champion Gini ≈ 0.67 vs incumbent scorecard ≈ 0.57 — **lift ≈ +0.08–0.10 Gini** (bootstrap 95% CIs in `reports/RESULTS.md`).
- KS, Brier score and decile calibration reported in `reports/RESULTS.md` and `reports/figures/`.
- Business impact at a constant 70% approval rate: estimated multi-million-dollar annual reduction in default losses; at constant risk, approval rate expands materially (exact figures in `reports/metrics.json`, assumptions documented therein: LGD 0.55, annualized volume 800K).

## Fairness
- Audited across gender, age band, and demographic group at the production threshold: adverse-impact ratio (four-fifths rule), demographic parity, equal opportunity, equalized odds, within-group calibration.
- A candidate feature (`geo_risk_index`) was **rejected during development**: it raised Gini but pushed the protected group's AIR below 0.80. The champion passes the four-fifths rule on gender and demographic group; age-band differences track realized default rates (within-group calibration gaps ≈ 0), i.e., risk differentiation rather than model bias.
- Residual measurement bias (income under-reporting in one segment) is documented as a known limitation and monitored.

## Explainability
- Global: SHAP (TreeExplainer) summary and dependence plots (`reports/figures/`).
- Local: every decline produces ranked SHAP reason codes mapped to ECOA/Reg B adverse-action language (`reports/adverse_action_samples.txt`; served live by the scoring API).

## Limitations & ethical considerations
- Development data is simulated; real-data validation (Home Credit pipeline, `scripts/run_home_credit.py`) is a companion study, not a substitute for production backtesting.
- PD calibration assumes a stable macro environment; scores must be recalibrated, not just monitored, after material shifts.
- The model cannot detect bias mechanisms it has no signal for; the fairness suite tests enumerated protected attributes only.

## Maintenance
- Monitoring per `docs/MONITORING_PLAN.md` (PSI, Gini drift, fairness drift, calibration).
- Scheduled annual revalidation; event-driven revalidation triggers listed in the monitoring plan.
