#!/usr/bin/env python
"""End-to-end credit risk pipeline on simulated data.

Stages:
 1. Simulate 800K applications (correlated features, controlled bias injection)
 2. Train incumbent WOE scorecard and challengers (LightGBM, XGBoost, FT-Transformer)
 3. Evaluate discrimination (Gini/KS, bootstrap CIs) and calibration
 4. Fairness audit: incumbent vs naive challenger vs mitigated challenger
 5. SHAP explainability + sample adverse-action notices
 6. Business impact ($ loss reduction, approval expansion)
 7. Figures, RESULTS.md, metrics.json, serialized production artifacts

Usage: python scripts/run_pipeline.py [--n 800000] [--skip-nn]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditrisk import config as C
from creditrisk.data.simulate import simulate_applicants
from creditrisk.evaluation import (
    approval_threshold, bootstrap_gini_ci, business_impact, decisions,
    evaluate_model, metrics_table,
)
from creditrisk.explain import (
    adverse_action_notice, adverse_action_reasons, global_importance, tree_shap,
)
from creditrisk.fairness import audit, render_fairness_report
from creditrisk.models.gbm import predict_gbm, train_lightgbm, train_xgboost
from creditrisk.models.scorecard import Scorecard
from creditrisk import viz

# The incumbent is a classic bureau-characteristics scorecard; it predates the
# geo_risk_index alternative-data feature that the challenger initially adds.
SCORECARD_NUMERIC = [
    "annual_income", "debt_to_income", "revolving_utilization",
    "credit_history_months", "num_delinq_2y", "inquiries_6m",
    "employment_years", "loan_to_income",
]
SCORECARD_CATEGORICAL = ["home_ownership"]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=C.N_APPLICANTS)
    ap.add_argument("--skip-nn", action="store_true")
    args = ap.parse_args()

    for d in (C.DATA_DIR, C.FIGURES_DIR, C.MODELS_DIR, C.REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    viz.set_style()
    rng = np.random.default_rng(C.RANDOM_SEED)

    # ---- 1. data -------------------------------------------------------------
    log(f"Simulating {args.n:,} applications ...")
    df = simulate_applicants(args.n)
    df.to_parquet(C.DATA_DIR / "applications.parquet")
    log(f"  default rate {df['default'].mean():.3%}; "
        f"group B true-PD gap {df.groupby('group')['true_pd'].mean().diff().iloc[-1]:+.5f} "
        f"(ground truth: protected attrs are causally inert)")

    idx = rng.permutation(len(df))
    n_test = int(len(df) * 0.2)
    n_valid = int(len(df) * 0.1)
    test_i, valid_i, train_i = idx[:n_test], idx[n_test:n_test + n_valid], idx[n_test + n_valid:]
    train, valid, test = df.iloc[train_i], df.iloc[valid_i], df.iloc[test_i]
    y_tr, y_va, y_te = train[C.TARGET], valid[C.TARGET], test[C.TARGET].values
    loan_amt_te = test["loan_amount"].values
    log(f"  split: train {len(train):,} / valid {len(valid):,} / test {len(test):,}")

    feats = C.ALL_FEATURES
    feats_mitigated = [f for f in feats if f != "geo_risk_index"]

    preds: dict[str, np.ndarray] = {}
    results = []

    # ---- 2a. incumbent scorecard ----------------------------------------------
    log("Training incumbent WOE scorecard ...")
    scorecard = Scorecard(n_bins=5).fit(
        train, y_tr, SCORECARD_NUMERIC, SCORECARD_CATEGORICAL
    )
    preds["incumbent"] = scorecard.predict_proba(test)
    results.append(evaluate_model(y_te, preds["incumbent"], "incumbent"))

    # ---- 2b. LightGBM (naive: all features incl. proxy) -------------------------
    log("Training LightGBM challenger (all features) ...")
    lgbm = train_lightgbm(train[feats], y_tr, valid[feats], y_va)
    preds["lightgbm"] = predict_gbm(lgbm, test[feats])
    results.append(evaluate_model(y_te, preds["lightgbm"], "lightgbm"))

    # ---- 2c. XGBoost --------------------------------------------------------------
    log("Training XGBoost challenger ...")
    xgbm = train_xgboost(train[feats], y_tr, valid[feats], y_va)
    preds["xgboost"] = predict_gbm(xgbm, test[feats])
    results.append(evaluate_model(y_te, preds["xgboost"], "xgboost"))

    # ---- 2d. FT-Transformer ---------------------------------------------------------
    nn_history = None
    if not args.skip_nn:
        log("Training FT-Transformer (from scratch, GPU if available) ...")
        from creditrisk.models.tabular_nn import TabularNNClassifier
        nn = TabularNNClassifier(C.NUMERIC_FEATURES, C.CATEGORICAL_FEATURES, seed=C.RANDOM_SEED)
        nn.fit(train[feats], y_tr.values, valid[feats], y_va.values)
        preds["ft_transformer"] = nn.predict_proba(test[feats])
        results.append(evaluate_model(y_te, preds["ft_transformer"], "ft_transformer"))
        nn_history = nn.history_
        log(f"  best valid AUC {max(h['valid_auc'] for h in nn.history_):.4f} "
            f"({len(nn.history_)} epochs)")

    # ---- 2e. mitigated champion: LightGBM without proxy feature ----------------------
    log("Training mitigated LightGBM (geo_risk_index removed) ...")
    lgbm_fair = train_lightgbm(train[feats_mitigated], y_tr, valid[feats_mitigated], y_va)
    preds["lightgbm_mitigated"] = predict_gbm(lgbm_fair, test[feats_mitigated])
    results.append(evaluate_model(y_te, preds["lightgbm_mitigated"], "lightgbm_mitigated"))

    # ---- 3. evaluation table + CIs -----------------------------------------------------
    table = metrics_table(results)
    log("Bootstrapping Gini CIs ...")
    cis = {name: bootstrap_gini_ci(y_te, p) for name, p in preds.items()}
    table["gini_ci_lo"] = [round(cis[m][1], 4) for m in table.index]
    table["gini_ci_hi"] = [round(cis[m][2], 4) for m in table.index]
    table["gini_lift_vs_incumbent"] = (table["gini"] - table.loc["incumbent", "gini"]).round(4)
    print("\n" + table.to_string() + "\n")

    champion = "lightgbm_mitigated"
    gini_lift = float(table.loc[champion, "gini"] - table.loc["incumbent", "gini"])

    # ---- 4. fairness audit ----------------------------------------------------------------
    log("Running fairness audit (incumbent / naive challenger / mitigated champion) ...")
    protected_te = test[C.PROTECTED_ATTRS].reset_index(drop=True)
    audits, audit_tables, audit_summaries = {}, {}, {}
    for variant in ["incumbent", "lightgbm", champion]:
        p = preds[variant]
        thr = approval_threshold(p)
        appr = decisions(p, thr)
        tables, summary = audit(y_te, p, appr, protected_te, C.PROTECTED_ATTRS)
        audits[variant] = (tables, summary)
        audit_tables[variant], audit_summaries[variant] = tables, summary
        per_attr = ", ".join(
            f"{a}: AIR {summary.loc[a, 'min_air']:.3f} "
            f"({'pass' if summary.loc[a, 'passes_four_fifths'] else 'FAIL'})"
            for a in summary.index
        )
        log(f"  {variant}: {per_attr}")
    render_fairness_report(
        audits, C.REPORTS_DIR / "FAIRNESS_REPORT.md",
        context={
            "data": f"simulated, n={args.n:,}",
            "bias injected": f"proxy_strength={C.PROXY_BIAS_STRENGTH}, "
                             f"income_underreport={C.INCOME_UNDERREPORT_FRAC}",
            "ground truth": "protected attributes causally inert (true_pd independent of group)",
        },
    )

    # ---- 5. SHAP ---------------------------------------------------------------------------
    log("Computing SHAP values (champion) ...")
    explainer, X_shap, sv = tree_shap(lgbm_fair, test[feats_mitigated])
    gi = global_importance(sv, feats_mitigated)
    gi.to_csv(C.REPORTS_DIR / "shap_global_importance.csv", header=["mean_abs_shap"])

    import shap as shap_lib
    import matplotlib.pyplot as plt
    shap_lib.summary_plot(sv, X_shap, show=False, max_display=14, plot_size=(7, 5.5))
    plt.title("SHAP summary — champion model (LightGBM, mitigated)", fontsize=11)
    plt.savefig(C.FIGURES_DIR / "shap_summary.png", dpi=200, bbox_inches="tight"); plt.close()
    for feat in gi.index[:3]:
        shap_lib.dependence_plot(feat, sv, X_shap.assign(
            **{c: X_shap[c].astype("category").cat.codes for c in C.CATEGORICAL_FEATURES if c in X_shap}
        ), show=False)
        plt.savefig(C.FIGURES_DIR / f"shap_dependence_{feat}.png", dpi=200, bbox_inches="tight")
        plt.close()

    # sample adverse-action notices for 3 declined applicants. Development-
    # population medians anchor the reason-code consistency guard: a "too high"
    # reason is only stated for a value actually above the median (and vice
    # versa), so SHAP-real but linguistically-misleading reasons are suppressed.
    feature_medians = train[feats_mitigated].median(numeric_only=True)
    thr_champ = approval_threshold(preds[champion])
    declined_idx = np.where(preds[champion] > thr_champ)[0][:3]
    notices = []
    for i in declined_idx:
        row = test.iloc[[i]]
        reasons = adverse_action_reasons(
            explainer, row[feats_mitigated], reference_medians=feature_medians)
        notices.append(adverse_action_notice(
            row["application_id"].iloc[0], float(preds[champion][i]), thr_champ, reasons
        ))
    (C.REPORTS_DIR / "adverse_action_samples.txt").write_text("\n\n" + ("\n" + "=" * 72 + "\n").join(notices))

    # ---- 6. business impact ---------------------------------------------------------------------
    log("Computing business impact ...")
    impact = business_impact(y_te, preds["incumbent"], preds[champion], loan_amt_te)
    log(f"  annual loss reduction ${impact['annual_loss_reduction']/1e6:.1f}M "
        f"({impact['loss_reduction_pct']:.1%}); "
        f"constant-risk approvals +{impact['additional_approvals_annual']:,}/yr")

    # ---- 7. figures + reports ----------------------------------------------------------------------
    log("Rendering figures ...")
    main_preds = {k: preds[k] for k in ["incumbent", "lightgbm", "xgboost", "ft_transformer", champion] if k in preds}
    ginis = {k: float(table.loc[k, "gini"]) for k in main_preds}
    viz.plot_roc(y_te, main_preds, ginis, C.FIGURES_DIR / "roc_curves.png")
    viz.plot_calibration(y_te, main_preds, C.FIGURES_DIR / "calibration.png")
    viz.plot_gains(y_te, main_preds, C.FIGURES_DIR / "gains.png")
    viz.plot_gini_comparison(table.loc[list(main_preds)], cis, C.FIGURES_DIR / "gini_comparison.png")
    viz.plot_score_distribution(y_te, scorecard.score(test), C.FIGURES_DIR / "score_distribution.png")
    for attr in C.PROTECTED_ATTRS:
        viz.plot_fairness(audit_tables, attr, C.FIGURES_DIR / f"fairness_{attr}.png")
    viz.plot_air(audit_summaries, C.FIGURES_DIR / "adverse_impact_ratio.png")
    if nn_history:
        viz.plot_nn_history(nn_history, C.FIGURES_DIR / "nn_training.png")

    # metrics.json drives the docs and the API
    metrics = {
        "n_applications": args.n,
        "default_rate": float(df["default"].mean()),
        "champion": champion,
        "gini_lift_vs_incumbent": round(gini_lift, 4),
        "models": table.reset_index().to_dict(orient="records"),
        "business_impact": impact,
        "fairness": {v: s.reset_index().to_dict(orient="records") for v, (_, s) in audits.items()},
        "decision_threshold_pd": float(thr_champ),
        "approval_rate": C.APPROVAL_RATE,
    }
    (C.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # RESULTS.md
    lines = [
        "# Credit Risk Model — Results Summary",
        "",
        f"- Applications: **{args.n:,}** (simulated; default rate {df['default'].mean():.2%})",
        f"- Champion: **{champion}** — Gini **{table.loc[champion,'gini']:.3f}** "
        f"(95% CI {cis[champion][1]:.3f}–{cis[champion][2]:.3f}) vs incumbent "
        f"{table.loc['incumbent','gini']:.3f} → **lift +{gini_lift:.3f}**",
        f"- Estimated annual default-loss reduction at constant {C.APPROVAL_RATE:.0%} approval rate: "
        f"**${impact['annual_loss_reduction']/1e6:.1f}M** ({impact['loss_reduction_pct']:.1%})",
        f"- At constant risk, approvals expand from {C.APPROVAL_RATE:.0%} to "
        f"{impact['constant_risk_approval_rate']:.1%} "
        f"(**+{impact['additional_approvals_annual']:,} approvals/yr**)",
        "",
        "## Model comparison (held-out test, n={:,})".format(len(test)),
        "",
        table.round(4).to_markdown(),
        "",
        "## Fairness (adverse-impact ratio by protected attribute)",
        "",
        "| variant | " + " | ".join(C.PROTECTED_ATTRS) + " |",
        "|---|" + "---|" * len(C.PROTECTED_ATTRS),
    ]
    for v, (_, s) in audits.items():
        cells = " | ".join(
            f"{s.loc[a, 'min_air']:.3f} {'✅' if s.loc[a, 'passes_four_fifths'] else '❌'}"
            for a in C.PROTECTED_ATTRS
        )
        lines.append(f"| {v} | {cells} |")
    lines += [
        "",
        "Age-band differences track realized default rates (see calibration gaps in "
        "`FAIRNESS_REPORT.md`): disparity there reflects risk differentiation that "
        "ECOA permits for empirically derived scorecards, not model bias.",
    ]
    lines += [
        "",
        "See `FAIRNESS_REPORT.md` for the full audit, `figures/` for all plots, ",
        "`adverse_action_samples.txt` for SHAP-based notices, and `docs/` for governance.",
    ]
    (C.REPORTS_DIR / "RESULTS.md").write_text("\n".join(lines))

    # ---- artifacts for the API ------------------------------------------------------------------------
    joblib.dump({
        "model": lgbm_fair,
        "features": feats_mitigated,
        "threshold_pd": float(thr_champ),
        "approval_rate": C.APPROVAL_RATE,
        "feature_medians": feature_medians,  # reason-code consistency guard
        "version": "1.0.1",
    }, C.MODELS_DIR / "champion.joblib")
    joblib.dump(scorecard, C.MODELS_DIR / "incumbent_scorecard.joblib")
    log(f"Artifacts saved to {C.MODELS_DIR}")
    log("DONE")


if __name__ == "__main__":
    main()
