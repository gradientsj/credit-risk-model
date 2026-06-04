# Credit Risk Scoring Model

End-to-end credit risk modeling project: an 800K-application probability-of-default model with SHAP explainability, fair-lending testing against ground-truth injected bias, dollar-quantified business impact, model governance documentation, and a scoring API that returns ECOA-style adverse-action reason codes.

> 📰 **[Read the full write-up](ARTICLE.md)** — a long-form case study with every figure and result in context, from simulator design through the fairness audit to the twelve-model benchmark.

## Headline results (simulated 800K portfolio, held-out test)

| | Incumbent scorecard | Naive challenger | **Champion (mitigated)** |
|---|---|---|---|
| Gini | ~0.57 | ~0.72 | **~0.67 (+0.08–0.10 lift)** |
| Four-fifths rule (group) | pass | **FAIL (AIR < 0.80)** | pass (AIR ≈ 0.99) |
| Decision | retired | **rejected by fairness audit** | deployed |

- The naive challenger's extra Gini came partly from `geo_risk_index` — a regional default-rate index that silently encodes protected-group membership (a controlled, injected redlining mechanism). The fairness suite catches it; the champion excludes it and keeps the full lift over the incumbent.
- At a constant 70% approval rate, the champion cuts estimated annual default losses by several $M (exact figures with assumptions in `reports/metrics.json`); at constant risk it expands approvals materially — including for the previously under-approved segment.
- Exact numbers from the latest run: `reports/RESULTS.md`.

## Why simulated data?

The simulator (`src/creditrisk/data/simulate.py`) generates correlated bureau-style features from latent factors, with risk dominated by regime interactions (distressed revolvers vs transactors, leverage with vs without liquidity buffers) — structure an additive binned scorecard cannot express, which is what creates the challenger's headroom. Crucially, **protected attributes are causally inert by construction and bias is injected via two parameterized mechanisms** (a proxy feature and segment-level income under-reporting), so the fairness test suite is validated against known ground truth: it must fail the biased configuration and pass the clean one (covered in `tests/`).

A companion pipeline runs the same comparison on **real data** — Home Credit Default Risk, ~307K applications (`scripts/run_home_credit.py`, requires Kaggle credentials).

## Models compared

1. **WOE scorecard** (incumbent): quantile-binned weight-of-evidence + logistic regression, points-scaled (PDO 20).
2. **LightGBM** / **XGBoost** challengers with early stopping.
3. **FT-Transformer** implemented from scratch in PyTorch (feature tokenization + [CLS] + pre-norm transformer encoder, per Gorishniy et al. 2021), trained on GPU — included to *test* whether deep learning earns its complexity on tabular credit data (finding: it ties but does not beat the GBMs, which is itself a governance-relevant result).

### Extended model zoo (`scripts/run_model_zoo.py`)

Twelve variants benchmarked under the *approved feature policy* (mitigated set), each answering a specific question — results in `reports/model_zoo/RESULTS.md`:

| Model | Question it answers |
|---|---|
| CatBoost, Random Forest | GBM triad complete; is the lift boosting-specific or trees-generally? |
| **Monotone LightGBM** | the *price of monotonicity* — regulator-friendly directional constraints on 11/16 features (regime-dependent features deliberately left free) |
| **EBM** (InterpretML) | can a **glass-box** GAM with learned pairwise interactions recover the black-box lift? (Largely yes — evidence the risk is pairwise-interaction-driven) |
| **Augmented scorecard** | distillation: top SHAP-*interaction* pairs from the GBM become cross terms in the WOE scorecard — and they recover exactly the simulator's true regime interactions |
| MLP-ResNet, **TabM** (both from scratch) | modern tabular DL incl. the NeurIPS-2024 BatchEnsemble MLP |
| Stacked ensemble | the accuracy ceiling when families blend |
| Isotonic calibration | champion ECE/Brier before/after a monotone calibration layer |

All with uniform metrics (Gini + bootstrap CI, KS, Brier, **ECE**, **scoring latency**) and two extra figures: Gini-by-family and the accuracy-vs-latency frontier.

## Repository layout

```
src/creditrisk/
  data/simulate.py      # 800K simulator + controlled bias injection
  data/home_credit.py   # real-data pipeline (Kaggle)
  models/               # scorecard, LightGBM/XGBoost, FT-Transformer
  evaluation.py         # Gini/KS/Brier, bootstrap CIs, PSI, $-impact swap-set analysis
  fairness.py           # AIR/four-fifths, parity, equalized odds, calibration-by-group
  explain.py            # SHAP global + adverse-action reason codes (Reg B style)
  viz.py                # publication-style figures
scripts/run_pipeline.py     # one-command end-to-end run
scripts/run_home_credit.py  # real-data companion study
api/main.py                 # FastAPI scoring + reason codes
notebooks/                  # end-to-end walkthrough
docs/                       # MODEL_CARD, VALIDATION_REPORT (SR 11-7), MONITORING_PLAN
reports/                    # generated: RESULTS.md, FAIRNESS_REPORT.md, figures/, metrics.json
tests/                      # simulator ground truth + metrics + fairness-suite tests
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# full run (800K rows; ~15 min with GPU for the transformer)
python scripts/run_pipeline.py

# quick run
python scripts/run_pipeline.py --n 100000 --skip-nn

# tests
pytest tests/ -q

# scoring API
uvicorn api.main:app --port 8000   # then POST /score — see /docs
```

Example score request:

```bash
curl -s localhost:8000/score -H 'content-type: application/json' -d '{
  "age": 29, "annual_income": 38000, "employment_years": 1.5,
  "employment_type": "contract", "credit_history_months": 30,
  "num_credit_lines": 3, "revolving_utilization": 0.91, "debt_to_income": 0.52,
  "num_delinq_2y": 2, "inquiries_6m": 4, "savings_balance": 400,
  "monthly_expenses": 2100, "loan_amount": 18000,
  "loan_purpose": "small_business", "home_ownership": "rent"}'
```

A declined applicant gets ranked SHAP reason codes (e.g. *"R01 — Proportion of revolving balances to credit limits is too high"*) whose contributions sum to the actual score margin — faithful-by-construction adverse action.

## Governance

- `docs/MODEL_CARD.md` — intended use, data, performance, fairness, limitations
- `docs/VALIDATION_REPORT.md` — independent-validation write-up structured per SR 11-7 (conceptual soundness, outcomes analysis, fair-lending review, findings log)
- `docs/MONITORING_PLAN.md` — PSI/Gini/fairness drift thresholds, escalation paths, revalidation triggers
