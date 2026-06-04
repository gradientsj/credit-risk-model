#!/usr/bin/env python
"""Build notebooks/credit_risk_walkthrough.ipynb (execute with nbconvert after)."""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
md("""# Credit Risk Scoring — End-to-End Walkthrough

A probability-of-default model with the full governance loop a real lender requires:

1. **Data** — 800K simulated applications with correlated bureau-style features and *controlled bias injection* (so fairness tests have ground truth). A companion pipeline runs real Home Credit data.
2. **Models** — incumbent WOE scorecard vs LightGBM / XGBoost / FT-Transformer challengers.
3. **Evaluation** — Gini/KS with bootstrap CIs, calibration, cumulative capture.
4. **Fairness** — adverse-impact ratio (four-fifths rule), equal opportunity, calibration-by-group; a biased feature is caught and removed.
5. **Explainability** — SHAP global behavior + per-applicant adverse-action reason codes (ECOA/Reg B style).
6. **Business impact** — swap-set analysis: $ losses at constant volume, approval expansion at constant risk.

> This notebook runs on a 120K sample for speed; headline numbers from the full 800K run are loaded from `reports/metrics.json` at the end. Reproduce everything with `python scripts/run_pipeline.py`."""),

code("""import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from creditrisk import config as C, viz
from creditrisk.data.simulate import simulate_applicants
viz.set_style()

N = 120_000
df = simulate_applicants(N)
df.head()"""),

md("""## 1. The simulated portfolio

Features are noisy transforms of three latent factors (financial stability, credit experience, debt pressure), so the correlation structure resembles real bureau data. True default risk is dominated by **regime interactions** whose per-feature marginals roughly cancel — e.g. high utilization is dangerous for a distressed revolver but benign for a transactor. An additive binned scorecard structurally cannot express that; a GBM can. That's the challenger's headroom, by design."""),

code("""fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].hist(df["annual_income"]/1000, bins=80, color="#0173B2"); axes[0].set(title="Annual income ($k)", xlim=(0,150))
axes[1].hist(df["revolving_utilization"], bins=60, color="#DE8F05"); axes[1].set(title="Revolving utilization")
df.groupby(pd.cut(df["debt_to_income"], np.arange(0,0.9,0.1)), observed=True)["default"].mean().plot(
    kind="bar", ax=axes[2], color="#029E73", rot=45); axes[2].set(title="Default rate by DTI bin")
plt.tight_layout()
print(f"default rate: {df['default'].mean():.2%}")"""),

code("""corr = df[C.NUMERIC_FEATURES].corr()
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
plt.colorbar(im, shrink=0.8); ax.set_title("Feature correlations"); ax.grid(False)"""),

md("""### Fairness ground truth

Protected attributes are **causally inert by construction** — `true_pd` is independent of group and gender. Bias enters only through two parameterized mechanisms:

- **Proxy**: `geo_risk_index` (a regional default-rate index) is shifted upward for group B independent of repayment — the classic redlining mechanism;
- **Measurement**: group B's *reported* income understates true income.

Because the knobs are explicit, the fairness suite can be validated: it must flag the biased configuration and pass the clean one (`tests/test_pipeline.py`)."""),

code("""print("mean true PD by group :", df.groupby("group")["true_pd"].mean().round(4).to_dict())
print("mean true PD by gender:", df.groupby("gender")["true_pd"].mean().round(4).to_dict())
print("geo_risk_index by group:", df.groupby("group")["geo_risk_index"].mean().round(1).to_dict(),
      " <- injected proxy shift")
print("reported income by group:", df.groupby("group")["annual_income"].mean().round(0).to_dict(),
      " <- injected measurement bias")"""),

md("""## 2. Models: incumbent scorecard vs challengers

The incumbent is what most lenders actually run: coarse quantile bins → weight-of-evidence → logistic regression → points scaling (PDO 20). The challengers are LightGBM and XGBoost (and, in the full pipeline, an FT-Transformer trained from scratch — see `src/creditrisk/models/tabular_nn.py`).

The naive challenger uses **all** features, including the new `geo_risk_index` alternative-data feature."""),

code("""from creditrisk.models.scorecard import Scorecard
from creditrisk.models.gbm import train_lightgbm, train_xgboost, predict_gbm
from creditrisk.evaluation import evaluate_model, metrics_table, bootstrap_gini_ci

rng = np.random.default_rng(C.RANDOM_SEED)
idx = rng.permutation(len(df))
n_te, n_va = int(N*0.2), int(N*0.1)
test, valid, train = df.iloc[idx[:n_te]], df.iloc[idx[n_te:n_te+n_va]], df.iloc[idx[n_te+n_va:]]
y_tr, y_va, y_te = train["default"], valid["default"], test["default"].values

SC_NUM = ["annual_income","debt_to_income","revolving_utilization","credit_history_months",
          "num_delinq_2y","inquiries_6m","employment_years","loan_to_income"]
scorecard = Scorecard(n_bins=5).fit(train, y_tr, SC_NUM, ["home_ownership"])

feats = C.ALL_FEATURES
feats_fair = [f for f in feats if f != "geo_risk_index"]
lgbm       = train_lightgbm(train[feats], y_tr, valid[feats], y_va)
xgbm       = train_xgboost(train[feats], y_tr, valid[feats], y_va)
lgbm_fair  = train_lightgbm(train[feats_fair], y_tr, valid[feats_fair], y_va)

preds = {
    "incumbent":          scorecard.predict_proba(test),
    "lightgbm":           predict_gbm(lgbm, test[feats]),
    "xgboost":            predict_gbm(xgbm, test[feats]),
    "lightgbm_mitigated": predict_gbm(lgbm_fair, test[feats_fair]),
}
table = metrics_table([evaluate_model(y_te, p, m) for m, p in preds.items()])
table["gini_lift"] = (table["gini"] - table.loc["incumbent","gini"]).round(4)
table"""),

md("""The scorecard's WOE bins recover any *per-feature* shape, but the signal here lives in the joint distribution — hence the structural gap. Note the naive LightGBM beats the mitigated one: that extra Gini is exactly what the biased feature buys, and §4 shows why it has to go."""),

code("""from creditrisk.evaluation import gini as gini_fn
ginis = {m: float(table.loc[m, "gini"]) for m in preds}
viz.plot_roc(y_te, preds, ginis, ROOT/"reports/figures/nb_roc.png")
viz.plot_calibration(y_te, preds, ROOT/"reports/figures/nb_cal.png")
from IPython.display import Image, display
display(Image(str(ROOT/"reports/figures/nb_roc.png"), width=520))
display(Image(str(ROOT/"reports/figures/nb_cal.png"), width=520))"""),

md("""## 3. SHAP explainability and adverse action

Global: which features drive risk, and in which direction. Local: for every declined applicant, the top positive SHAP contributions become ranked **reason codes** — and because SHAP values sum to the actual score margin, the stated reasons are faithful to the decision (the regulatory requirement for adverse-action notices)."""),

code("""import shap
from creditrisk.explain import tree_shap, adverse_action_reasons, adverse_action_notice
from creditrisk.evaluation import approval_threshold

explainer, Xs, sv = tree_shap(lgbm_fair, test[feats_fair], sample=10_000)
shap.summary_plot(sv, Xs, max_display=12, show=False, plot_size=(7,5))
plt.title("SHAP summary — mitigated champion"); plt.show()"""),

code("""thr = approval_threshold(preds["lightgbm_mitigated"])
declined = np.where(preds["lightgbm_mitigated"] > thr)[0][0]
row = test.iloc[[declined]]
reasons = adverse_action_reasons(explainer, row[feats_fair])
print(adverse_action_notice(row["application_id"].iloc[0],
                            float(preds["lightgbm_mitigated"][declined]), thr, reasons))"""),

md("""## 4. Fairness audit — catching the injected bias

Decisions approve the lowest-PD applicants at a 70% approval rate. Per protected attribute we test the **adverse impact ratio** against the four-fifths rule, equal opportunity (TPR for non-defaulters), and within-group calibration.

Watch `group`: the naive challenger fails — its shiny new feature was pricing group membership."""),

code("""from creditrisk.fairness import audit
from creditrisk.evaluation import decisions

prot = test[C.PROTECTED_ATTRS].reset_index(drop=True)
summaries = {}
for variant in ["incumbent", "lightgbm", "lightgbm_mitigated"]:
    p = preds[variant]
    appr = decisions(p, approval_threshold(p))
    tables, summary = audit(y_te, p, appr, prot, C.PROTECTED_ATTRS)
    summaries[variant] = summary
    print(f"--- {variant}")
    print(summary[["min_air","demographic_parity_diff","equal_opportunity_diff",
                   "passes_four_fifths"]].round(3), "\\n")"""),

code("""groups_tbl = {}
for variant in ["incumbent","lightgbm","lightgbm_mitigated"]:
    p = preds[variant]
    appr = decisions(p, approval_threshold(p))
    tables, _ = audit(y_te, p, appr, prot, ["group"])
    groups_tbl[variant] = tables
viz.plot_fairness(groups_tbl, "group", ROOT/"reports/figures/nb_fair.png")
display(Image(str(ROOT/"reports/figures/nb_fair.png"), width=560))"""),

md("""Removing `geo_risk_index` restores group B's approval rate (AIR ≈ 0.98) at a modest Gini cost relative to the naive model — while keeping the full lift over the incumbent. Age-band disparities remain, but within-group calibration gaps are ≈ 0: those differences track realized default rates (risk differentiation ECOA permits for empirically derived systems), not model bias. The residual income-measurement bias is documented and monitored (`docs/MONITORING_PLAN.md`).

> The full pipeline also validates the *test suite itself*: with bias knobs at zero, the audit passes; at the default settings it must fail the naive model — and does."""),

md("""## 5. Business impact — swap-set analysis"""),

code("""from creditrisk.evaluation import business_impact
impact = business_impact(y_te, preds["incumbent"], preds["lightgbm_mitigated"],
                         test["loan_amount"].values)
print(f"At a constant {impact['approval_rate']:.0%} approval rate (annualized to "
      f"{C.ANNUAL_APPLICATIONS:,} applications, LGD {C.LGD}):")
print(f"  incumbent expected annual default losses : ${impact['incumbent_annual_loss']/1e6:,.1f}M")
print(f"  champion expected annual default losses  : ${impact['challenger_annual_loss']/1e6:,.1f}M")
print(f"  annual loss reduction                    : ${impact['annual_loss_reduction']/1e6:,.1f}M "
      f"({impact['loss_reduction_pct']:.1%})")
print(f"  swap-set: {impact['swap_in']:,} swapped in / {impact['swap_out']:,} swapped out")
print(f"At constant risk, approvals expand {impact['approval_rate']:.0%} -> "
      f"{impact['constant_risk_approval_rate']:.1%} "
      f"(+{impact['additional_approvals_annual']:,} approvals/yr)")"""),

md("""## 6. Full-scale (800K) results and governance artifacts"""),

code("""m = json.loads((ROOT/"reports/metrics.json").read_text())
print(f"run: {m['n_applications']:,} applications | champion: {m['champion']}")
print(f"Gini lift vs incumbent: +{m['gini_lift_vs_incumbent']}")
print(f"annual loss reduction : ${m['business_impact']['annual_loss_reduction']/1e6:.1f}M "
      f"({m['business_impact']['loss_reduction_pct']:.1%})")
print(f"approval expansion    : +{m['business_impact']['additional_approvals_annual']:,}/yr at constant risk")
pd.DataFrame(m["models"]).set_index("model")[["gini","ks","brier","gini_ci_lo","gini_ci_hi","gini_lift_vs_incumbent"]]"""),

code("""display(Image(str(ROOT/"reports/figures/gini_comparison.png"), width=560))
display(Image(str(ROOT/"reports/figures/adverse_impact_ratio.png"), width=560))
display(Image(str(ROOT/"reports/figures/nn_training.png"), width=480))"""),

md("""### Where to go next

- **Governance docs**: [`docs/MODEL_CARD.md`](../docs/MODEL_CARD.md) · [`docs/VALIDATION_REPORT.md`](../docs/VALIDATION_REPORT.md) (SR 11-7 structure) · [`docs/MONITORING_PLAN.md`](../docs/MONITORING_PLAN.md)
- **Generated reports**: [`reports/RESULTS.md`](../reports/RESULTS.md) · [`reports/FAIRNESS_REPORT.md`](../reports/FAIRNESS_REPORT.md) · [`reports/adverse_action_samples.txt`](../reports/adverse_action_samples.txt)
- **Scoring API**: `uvicorn api.main:app` → POST `/score` returns PD, points score, decision and reason codes
- **Real data**: `python scripts/run_home_credit.py` reruns the comparison on Home Credit Default Risk (~307K Kaggle applications)"""),
]

nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.10"}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "credit_risk_walkthrough.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"wrote {out}")
