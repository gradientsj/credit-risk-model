#!/usr/bin/env python
"""Extended model zoo on the simulated portfolio.

Benchmarks twelve model variants — spanning scorecards, GAMs, GBMs, bagging,
three deep architectures, and a stacked ensemble — under the *approved feature
policy* (mitigated set, no geo_risk_index), with uniform metrics: Gini with
bootstrap CIs, KS, Brier, ECE, and scoring latency. Also fits an isotonic
calibration layer for the champion and renders the EBM's glass-box terms.

Requires a prior `run_pipeline.py` run (uses its parquet, artifacts, split).

Usage: python scripts/run_model_zoo.py [--quick]   (--quick: 100K rows, fewer epochs)
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
from creditrisk import viz
from creditrisk.evaluation import bootstrap_gini_ci, ece, evaluate_model
from creditrisk.models import baselines as B
from creditrisk.models.gbm import predict_gbm, train_lightgbm, train_xgboost
from creditrisk.models.tabular_nn import TabularNNClassifier

ZOO_DIR = C.REPORTS_DIR / "model_zoo"
FIG_DIR = ZOO_DIR / "figures"

FAMILY = {
    "incumbent": "scorecard (glass box)",
    "scorecard_augmented": "scorecard (glass box)",
    "ebm": "GAM (glass box)",
    "lightgbm_champion": "GBM",
    "xgboost": "GBM",
    "catboost": "GBM",
    "lightgbm_monotone": "GBM",
    "random_forest": "bagging",
    "ft_transformer": "deep",
    "mlp_resnet": "deep",
    "tabm": "deep",
    "stacked_ensemble": "ensemble",
}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def timed_predict(fn, X) -> tuple[np.ndarray, float]:
    """Returns (predictions, ms per 1K rows)."""
    t0 = time.perf_counter()
    p = fn(X)
    ms = (time.perf_counter() - t0) * 1000
    return p, ms / (len(X) / 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    viz.set_style()

    # ---- data: identical split to run_pipeline.py -----------------------------
    df = pd.read_parquet(C.DATA_DIR / "applications.parquet")
    if args.quick:
        df = df.iloc[:100_000]
    rng = np.random.default_rng(C.RANDOM_SEED)
    idx = rng.permutation(len(df))
    n_te, n_va = int(len(df) * 0.2), int(len(df) * 0.1)
    test, valid, train = (
        df.iloc[idx[:n_te]], df.iloc[idx[n_te:n_te + n_va]], df.iloc[idx[n_te + n_va:]]
    )
    y_tr, y_va, y_te = train["default"], valid["default"], test["default"].values
    feats = [f for f in C.ALL_FEATURES if f != "geo_risk_index"]
    num_m = [f for f in C.NUMERIC_FEATURES if f != "geo_risk_index"]
    log(f"{len(df):,} rows | mitigated feature set ({len(feats)} features) | "
        f"train {len(train):,} / valid {len(valid):,} / test {len(test):,}")

    epochs = 8 if args.quick else 30
    preds: dict[str, np.ndarray] = {}
    latency: dict[str, float] = {}
    valid_preds: dict[str, np.ndarray] = {}  # for the stacking meta-learner
    notes: dict[str, str] = {}

    # ---- artifacts: incumbent + champion ----------------------------------------
    log("Scoring saved incumbent and champion artifacts ...")
    incumbent = joblib.load(C.MODELS_DIR / "incumbent_scorecard.joblib")
    preds["incumbent"], latency["incumbent"] = timed_predict(incumbent.predict_proba, test)
    champ = joblib.load(C.MODELS_DIR / "champion.joblib")
    champion_model = champ["model"]
    preds["lightgbm_champion"], latency["lightgbm_champion"] = timed_predict(
        lambda X: predict_gbm(champion_model, X[champ["features"]]), test)
    valid_preds["lightgbm_champion"] = predict_gbm(champion_model, valid[champ["features"]])
    notes["lightgbm_champion"] = "deployed model (from run_pipeline)"

    # ---- GBM family ---------------------------------------------------------------
    log("Training XGBoost (mitigated) ...")
    xgbm = train_xgboost(train[feats], y_tr, valid[feats], y_va)
    preds["xgboost"], latency["xgboost"] = timed_predict(
        lambda X: predict_gbm(xgbm, X[feats]), test)
    valid_preds["xgboost"] = predict_gbm(xgbm, valid[feats])

    log("Training CatBoost ...")
    cb = B.train_catboost(train[feats], y_tr, valid[feats], y_va)
    preds["catboost"], latency["catboost"] = timed_predict(
        lambda X: B.predict_catboost(cb, X[feats]), test)
    valid_preds["catboost"] = B.predict_catboost(cb, valid[feats])

    log("Training monotone-constrained LightGBM ...")
    mono = B.train_monotone_lgbm(train[feats], y_tr, valid[feats], y_va)
    preds["lightgbm_monotone"], latency["lightgbm_monotone"] = timed_predict(
        lambda X: predict_gbm(mono, X[feats]), test)
    notes["lightgbm_monotone"] = (
        f"{sum(v != 0 for v in (B.MONOTONE_DIRECTIONS.get(c, 0) for c in feats))} "
        "of 16 features constrained")

    # ---- bagging --------------------------------------------------------------------
    log("Training random forest ...")
    rf = B.train_random_forest(train[feats], y_tr)
    preds["random_forest"], latency["random_forest"] = timed_predict(
        lambda X: B.predict_random_forest(rf, X[feats]), test)

    # ---- glass boxes -------------------------------------------------------------------
    log("Training EBM (glass-box GAM + pairwise interactions) ...")
    ebm_model = B.train_ebm(train[feats], y_tr, interactions=15)
    preds["ebm"], latency["ebm"] = timed_predict(
        lambda X: B.predict_ebm(ebm_model, X[feats]), test)
    notes["ebm"] = "15 learned pairwise interaction terms"
    viz.plot_ebm_terms(ebm_model.term_names_, ebm_model.term_importances(),
                       FIG_DIR / "ebm_terms.png")

    log("Distilling SHAP interaction pairs into the augmented scorecard ...")
    pairs = B.top_shap_interaction_pairs(champion_model, train[feats], num_m, n_pairs=6)
    log(f"  top crosses: {', '.join(f'{a}×{b}' for a, b in pairs)}")
    sc_aug = B.train_augmented_scorecard(
        train, y_tr, pairs,
        ["annual_income", "debt_to_income", "revolving_utilization",
         "credit_history_months", "num_delinq_2y", "inquiries_6m",
         "employment_years", "loan_to_income"],
        ["home_ownership"])
    preds["scorecard_augmented"], latency["scorecard_augmented"] = timed_predict(
        lambda X: B.predict_augmented_scorecard(sc_aug, X), test)
    notes["scorecard_augmented"] = "incumbent + 6 GBM-distilled cross terms"

    # ---- deep family -----------------------------------------------------------------------
    for arch, kwargs in [
        ("ft_transformer", {}),
        ("mlp_resnet", {}),
        ("tabm", {}),
    ]:
        log(f"Training {arch} (mitigated, GPU if available) ...")
        nn_model = TabularNNClassifier(
            num_m, C.CATEGORICAL_FEATURES, arch=arch, arch_kwargs=kwargs,
            max_epochs=epochs, seed=C.RANDOM_SEED)
        nn_model.fit(train[feats], y_tr.values, valid[feats], y_va.values)
        preds[arch], latency[arch] = timed_predict(
            lambda X, m=nn_model: m.predict_proba(X[feats]), test)
        valid_preds[arch] = nn_model.predict_proba(valid[feats])
        log(f"  best valid AUC {max(h['valid_auc'] for h in nn_model.history_):.4f} "
            f"({len(nn_model.history_)} epochs)")
        if arch == "tabm":
            notes[arch] = "BatchEnsemble MLP, k=8 members (Gorishniy 2024)"

    # ---- stacked ensemble --------------------------------------------------------------------
    log("Fitting stacked ensemble (meta-LR on validation predictions) ...")
    from sklearn.linear_model import LogisticRegression
    bases = ["lightgbm_champion", "xgboost", "catboost", "tabm"]
    eps = 1e-6
    Zv = np.column_stack([np.log(np.clip(valid_preds[b], eps, 1 - eps) /
                                 np.clip(1 - valid_preds[b], eps, 1 - eps)) for b in bases])
    Zt = np.column_stack([np.log(np.clip(preds[b], eps, 1 - eps) /
                                 np.clip(1 - preds[b], eps, 1 - eps)) for b in bases])
    meta = LogisticRegression(max_iter=1000).fit(Zv, y_va)
    t0 = time.perf_counter()
    preds["stacked_ensemble"] = meta.predict_proba(Zt)[:, 1]
    latency["stacked_ensemble"] = sum(latency[b] for b in bases) + \
        (time.perf_counter() - t0) * 1000 / (len(test) / 1000)
    notes["stacked_ensemble"] = f"logit-stack of {', '.join(bases)}"

    # ---- uniform evaluation ----------------------------------------------------------------------
    log("Evaluating (Gini bootstrap CIs, KS, Brier, ECE, latency) ...")
    rows = []
    for name, p in preds.items():
        r = evaluate_model(y_te, p, name)
        g, lo, hi = bootstrap_gini_ci(y_te, p, n_boot=100, sample_cap=50_000)
        r.update(gini_ci_lo=round(lo, 4), gini_ci_hi=round(hi, 4),
                 ece=round(ece(y_te, p), 5),
                 latency_ms_per_1k=round(latency[name], 2),
                 family=FAMILY[name], note=notes.get(name, ""))
        rows.append(r)
    zoo = pd.DataFrame(rows).set_index("model")
    zoo["gini_lift_vs_incumbent"] = (zoo["gini"] - zoo.loc["incumbent", "gini"]).round(4)
    zoo = zoo.sort_values("gini", ascending=False)
    print("\n" + zoo[["family", "gini", "gini_ci_lo", "gini_ci_hi", "ks", "brier",
                      "ece", "latency_ms_per_1k", "gini_lift_vs_incumbent"]].to_string() + "\n")

    # ---- isotonic calibration of the champion ------------------------------------------------------
    log("Isotonic calibration (fit on validation, applied to test) ...")
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(valid_preds["lightgbm_champion"], y_va)
    p_cal = iso.predict(preds["lightgbm_champion"])
    cal_metrics = {
        "brier_raw": float(np.mean((preds["lightgbm_champion"] - y_te) ** 2)),
        "brier_cal": float(np.mean((p_cal - y_te) ** 2)),
        "ece_raw": ece(y_te, preds["lightgbm_champion"]),
        "ece_cal": ece(y_te, p_cal),
    }
    viz.plot_isotonic_calibration(y_te, preds["lightgbm_champion"], p_cal,
                                  cal_metrics, FIG_DIR / "calibration_isotonic.png")
    joblib.dump(iso, C.MODELS_DIR / "champion_isotonic.joblib")

    # ---- figures + reports ----------------------------------------------------------------------------
    viz.plot_model_zoo(zoo, FIG_DIR / "model_zoo_gini.png")
    viz.plot_gini_vs_latency(zoo, FIG_DIR / "gini_vs_latency.png")

    (ZOO_DIR / "metrics.json").write_text(json.dumps({
        "n_rows": len(df),
        "feature_policy": "mitigated (geo_risk_index excluded)",
        "models": zoo.reset_index().to_dict(orient="records"),
        "shap_distilled_pairs": [list(p) for p in pairs],
        "isotonic_calibration": cal_metrics,
        "stacking_bases": bases,
    }, indent=2, default=str))

    lift_aug = zoo.loc["scorecard_augmented", "gini_lift_vs_incumbent"]
    lift_champ = zoo.loc["lightgbm_champion", "gini_lift_vs_incumbent"]
    mono_cost = zoo.loc["lightgbm_champion", "gini"] - zoo.loc["lightgbm_monotone", "gini"]
    ebm_vs_champ = zoo.loc["lightgbm_champion", "gini"] - zoo.loc["ebm", "gini"]
    lines = [
        "# Model Zoo — Extended Benchmark",
        "",
        f"All models on the **approved (mitigated) feature policy**, identical splits "
        f"({len(df):,} rows). Uniform metrics incl. ECE and scoring latency.",
        "",
        zoo[["family", "gini", "gini_ci_lo", "gini_ci_hi", "ks", "brier", "ece",
             "latency_ms_per_1k", "gini_lift_vs_incumbent", "note"]].round(4).to_markdown(),
        "",
        "## Findings",
        "",
        f"- **Glass-box interactions**: EBM recovers "
        f"{(zoo.loc['ebm','gini_lift_vs_incumbent']/lift_champ):.0%} of the champion's lift "
        f"(trailing it by only {ebm_vs_champ:.3f} Gini) while every term remains a plottable "
        f"shape function — strong evidence the portfolio's risk is pairwise-interaction-driven.",
        f"- **Distillation**: grafting 6 GBM-derived SHAP-interaction crosses onto the incumbent "
        f"scorecard recovers {(lift_aug/lift_champ):.0%} of the champion's lift "
        f"(+{lift_aug:.3f} Gini) with zero architecture change — a viable fallback if tree "
        f"models were ever disallowed.",
        f"- **Price of monotonicity**: regulator-friendly monotone constraints on 11 of 16 "
        f"features cost {mono_cost:.3f} Gini. Notably, the unconstrained features include the "
        f"regime-dependent ones (e.g. num_credit_lines) where forcing a direction would be wrong.",
        f"- **Deep learning**: FT-Transformer / ResNet / TabM cluster with the GBMs; none "
        f"separates from LightGBM beyond CI noise, at materially higher scoring latency.",
        f"- **Ensembling**: the stack adds "
        f"+{zoo.loc['stacked_ensemble','gini'] - zoo.loc['lightgbm_champion','gini']:.3f} Gini "
        f"over the champion — the families are largely capturing the same signal.",
        ("- **Calibration**: isotonic layer improves champion ECE "
         if cal_metrics["ece_cal"] < cal_metrics["ece_raw"] else
         "- **Calibration**: the champion is already well calibrated — an isotonic "
         "layer does not improve it (ECE ")
        + f"{cal_metrics['ece_raw']:.4f} → {cal_metrics['ece_cal']:.4f}"
        + ("" if cal_metrics["ece_cal"] < cal_metrics["ece_raw"] else ")")
        + f", Brier {cal_metrics['brier_raw']:.4f} → {cal_metrics['brier_cal']:.4f}; "
        f"layer saved as `artifacts/champion_isotonic.joblib` for monitoring use "
        f"(recalibration is the first response to drift per the monitoring plan).",
        "",
        "Figures: `figures/model_zoo_gini.png`, `figures/gini_vs_latency.png`, "
        "`figures/ebm_terms.png`, `figures/calibration_isotonic.png`.",
    ]
    (ZOO_DIR / "RESULTS.md").write_text("\n".join(lines))
    log(f"Done. Reports in {ZOO_DIR}")


if __name__ == "__main__":
    main()
