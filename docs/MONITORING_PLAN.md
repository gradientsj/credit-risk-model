# Ongoing Monitoring Plan — Credit Risk PD Model v1.0

Owner: Credit Risk Modeling (1st line). Independent review of monitoring output: Model Risk Management (2nd line), quarterly.

## 1. Monitored dimensions, metrics, and thresholds

| Dimension | Metric | Frequency | Green | Amber (investigate) | Red (escalate) |
|---|---|---|---|---|---|
| Population stability | PSI of score distribution vs development | Monthly | < 0.10 | 0.10–0.25 | > 0.25 |
| Feature stability | PSI per feature (top 8 by SHAP importance) | Monthly | < 0.10 | 0.10–0.25 | > 0.25 |
| Discrimination | Gini on matured vintages (rolling 12m) | Quarterly | drop < 0.02 | 0.02–0.05 | > 0.05 |
| Calibration | Realized vs predicted default rate by decile | Quarterly | within CI | 10–20% relative drift | > 20% relative |
| Fair lending | AIR per protected attribute at production threshold | Quarterly | ≥ 0.90 | 0.80–0.90 | < 0.80 |
| Fair lending | Equal-opportunity gap (TPR for non-defaulters) | Quarterly | ≤ 0.03 | 0.03–0.05 | > 0.05 |
| Fair lending | Within-group calibration gap | Quarterly | ≤ 0.5pp | 0.5–1.0pp | > 1.0pp |
| Override rate | Manual overrides of model decision | Monthly | < 5% | 5–10% | > 10% |
| Reason codes | Distribution shift of top adverse-action codes | Quarterly | stable | new code in top-4 | dominant single code > 50% |

PSI implementation: `creditrisk.evaluation.psi` (10 quantile bins fixed at development).

## 2. Escalation and actions
- **Amber**: root-cause analysis within 10 business days; memo to model owner; continue monitoring at doubled frequency.
- **Red**: notify MRM and the risk committee within 5 business days. Actions in order of preference: (1) recalibrate intercept/slope on recent vintages; (2) retrain on refreshed window with full validation; (3) revert to incumbent scorecard (kept warm at `artifacts/incumbent_scorecard.joblib`) if the champion is unsafe.
- Any **fair-lending Red** triggers an immediate decisioning review and legal/compliance notification; the model may not continue auto-declining the affected segment without documented sign-off.

## 3. Event-driven revalidation triggers
- Macro shift: unemployment ± 2pp from development assumption, or policy rate ± 300bp.
- Product change: new channel, limit changes > 25%, or marketing into new segments.
- Data change: any upstream feature definition, bureau provider, or income-verification process change (note: income measurement bias is a tracked finding — any change to income verification requires fairness re-testing).
- Volume: applications per month outside 50–200% of development assumptions.

## 4. Vintage and outcome tracking
- Default outcomes mature at 12 months; monitoring uses a rolling matured-vintage window.
- Early-warning proxy: 60+ DPD at 3 months, correlated to terminal default during development; tracked monthly to give a leading indicator before vintages mature.

## 5. Reporting
- Monthly: automated dashboard (PSI, override rate, volume) to model owner.
- Quarterly: full monitoring pack (all metrics above, with trends and commentary) to the risk committee; includes a refreshed fairness audit using `creditrisk.fairness.audit`.
- Annually: full revalidation per `docs/VALIDATION_REPORT.md` scope.
