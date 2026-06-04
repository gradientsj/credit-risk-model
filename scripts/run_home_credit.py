#!/usr/bin/env python
"""Home Credit Default Risk pipeline (real Kaggle data, ~307K applications).

Prerequisites:
  1. Kaggle credentials at ~/.kaggle/kaggle.json (kaggle.com -> Account -> Create API Token)
  2. Accept the competition rules at
     https://www.kaggle.com/competitions/home-credit-default-risk

Runs the same comparison as the simulated pipeline on real data: LightGBM and
XGBoost challengers vs a logistic baseline, with SHAP explainability and a
gender fairness audit (the only protected field the dataset exposes).

Usage: python scripts/run_home_credit.py [--skip-download]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditrisk import config as C
from creditrisk import viz
from creditrisk.data import home_credit as hc
from creditrisk.evaluation import (
    approval_threshold, bootstrap_gini_ci, decisions, evaluate_model, metrics_table,
)
from creditrisk.explain import global_importance
from creditrisk.fairness import audit, render_fairness_report


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    out_dir = C.REPORTS_DIR / "home_credit"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    viz.set_style()

    if not args.skip_download:
        log("Downloading Home Credit data via Kaggle API ...")
        hc.download()

    log("Loading and engineering features ...")
    app = hc.load_applications()
    df = hc.engineer_features(app)
    log(f"  {len(df):,} applications, default rate {df['TARGET'].mean():.2%}")

    # gender 'XNA' rows are a handful; drop for a clean audit
    df = df[df["gender"].isin(["M", "F"])].reset_index(drop=True)
    feats = hc.HC_NUMERIC + hc.HC_CATEGORICAL
    for c in hc.HC_CATEGORICAL:
        df[c] = df[c].astype("category")

    rng = np.random.default_rng(C.RANDOM_SEED)
    idx = rng.permutation(len(df))
    n_test, n_valid = int(len(df) * 0.2), int(len(df) * 0.1)
    test, valid, train = (
        df.iloc[idx[:n_test]],
        df.iloc[idx[n_test:n_test + n_valid]],
        df.iloc[idx[n_test + n_valid:]],
    )
    y_tr, y_va, y_te = train["TARGET"], valid["TARGET"], test["TARGET"].values

    results, preds = [], {}

    log("Training logistic baseline (median-imputed, standardized) ...")
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    base = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    base.fit(train[hc.HC_NUMERIC], y_tr)
    preds["logistic_baseline"] = base.predict_proba(test[hc.HC_NUMERIC])[:, 1]
    results.append(evaluate_model(y_te, preds["logistic_baseline"], "logistic_baseline"))

    log("Training LightGBM ...")
    import lightgbm as lgb
    lgbm = lgb.LGBMClassifier(
        random_state=C.RANDOM_SEED,
        **{**C.TrainConfig().lgbm_params, "num_leaves": 127, "min_child_samples": 100},
    )
    lgbm.fit(
        train[feats], y_tr, eval_set=[(valid[feats], y_va)], eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    preds["lightgbm"] = lgbm.predict_proba(test[feats])[:, 1]
    results.append(evaluate_model(y_te, preds["lightgbm"], "lightgbm"))

    log("Training XGBoost ...")
    import xgboost as xgb
    xgbm = xgb.XGBClassifier(
        random_state=C.RANDOM_SEED, enable_categorical=True,
        early_stopping_rounds=100, eval_metric="auc", **C.TrainConfig().xgb_params,
    )
    xgbm.fit(train[feats], y_tr, eval_set=[(valid[feats], y_va)], verbose=False)
    preds["xgboost"] = xgbm.predict_proba(test[feats])[:, 1]
    results.append(evaluate_model(y_te, preds["xgboost"], "xgboost"))

    table = metrics_table(results)
    cis = {m: bootstrap_gini_ci(y_te, p) for m, p in preds.items()}
    print("\n" + table.to_string() + "\n")

    log("SHAP global importance (LightGBM) ...")
    import matplotlib.pyplot as plt
    import shap
    Xs = test[feats].sample(20_000, random_state=C.RANDOM_SEED)
    sv = shap.TreeExplainer(lgbm).shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[1]
    shap.summary_plot(sv, Xs, show=False, max_display=15, plot_size=(7, 6))
    plt.title("SHAP summary — Home Credit (LightGBM)")
    plt.savefig(fig_dir / "shap_summary.png", dpi=200, bbox_inches="tight"); plt.close()
    global_importance(sv, feats).to_csv(out_dir / "shap_global_importance.csv")

    log("Fairness audit (gender) ...")
    p = preds["lightgbm"]
    appr = decisions(p, approval_threshold(p))
    tables, summary = audit(y_te, p, appr, test[["gender"]].reset_index(drop=True), ["gender"])
    render_fairness_report(
        {"lightgbm": (tables, summary)}, out_dir / "FAIRNESS_REPORT.md",
        context={"data": "Home Credit Default Risk (real)",
                 "note": "gender is the only protected field the dataset exposes"},
    )

    viz.plot_roc(y_te, preds, {m: float(table.loc[m, "gini"]) for m in preds},
                 fig_dir / "roc_curves.png")
    viz.plot_calibration(y_te, preds, fig_dir / "calibration.png")
    viz.plot_gini_comparison(table, cis, fig_dir / "gini_comparison.png")
    (out_dir / "RESULTS.md").write_text(
        "# Home Credit Default Risk — Results\n\n" + table.round(4).to_markdown() + "\n"
    )
    log(f"Done. Reports in {out_dir}")


if __name__ == "__main__":
    main()
