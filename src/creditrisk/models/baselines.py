"""Extended model zoo: alternative challengers, each answering a specific
benchmarking or governance question.

* CatBoost            — completes the GBM triad (ordered boosting, native categoricals)
* Random forest       — is the lift from boosting specifically, or trees generally?
* Monotone LightGBM   — the *price of monotonicity*: regulators commonly require
                        PD monotone in utilization, delinquencies, DTI, etc., and
                        monotone models give stable adverse-action reason codes
* EBM (InterpretML)   — glass-box GAM with learned pairwise interactions; tests
                        whether an inherently interpretable model can recover the
                        black-box challenger's interaction lift
* Augmented scorecard — distillation: top SHAP interaction pairs from the GBM
                        become explicit cross-features in the WOE scorecard
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CATEGORICAL_FEATURES, RANDOM_SEED
from .scorecard import Scorecard

# Direction of risk wrt each numeric feature, where a direction is defensible a
# priori. Features whose true effect is regime-dependent (e.g. num_credit_lines:
# thin files are risky, seasoned multi-line borrowers are not) are left
# unconstrained — forcing a direction there is exactly the modeling error
# monotonicity reviews are meant to debate.
MONOTONE_DIRECTIONS = {
    "annual_income": -1,
    "loan_amount": +1,
    "loan_to_income": +1,
    "debt_to_income": +1,
    "revolving_utilization": +1,
    "credit_history_months": -1,
    "num_delinq_2y": +1,
    "inquiries_6m": +1,
    "employment_years": -1,
    "savings_balance": -1,
    "monthly_expenses": +1,
}


def _prep_cats(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def train_catboost(X_train, y_train, X_valid, y_valid, seed: int = RANDOM_SEED):
    from catboost import CatBoostClassifier

    cats = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        eval_metric="AUC", random_seed=seed, verbose=0,
        early_stopping_rounds=100, cat_features=cats, allow_writing_files=False,
    )
    Xt, Xv = X_train.copy(), X_valid.copy()
    for c in cats:
        Xt[c], Xv[c] = Xt[c].astype(str), Xv[c].astype(str)
    model.fit(Xt, y_train, eval_set=(Xv, y_valid))
    return model


def predict_catboost(model, X: pd.DataFrame) -> np.ndarray:
    X = X.copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype(str)
    return model.predict_proba(X)[:, 1]


def train_random_forest(X_train, y_train, seed: int = RANDOM_SEED):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=300, max_depth=16, min_samples_leaf=50,
        max_features="sqrt", n_jobs=-1, random_state=seed,
    )
    model.fit(pd.get_dummies(X_train, columns=CATEGORICAL_FEATURES), y_train)
    return model


def predict_random_forest(model, X: pd.DataFrame) -> np.ndarray:
    Xd = pd.get_dummies(X, columns=CATEGORICAL_FEATURES)
    Xd = Xd.reindex(columns=model.feature_names_in_, fill_value=0)
    return model.predict_proba(Xd)[:, 1]


def train_monotone_lgbm(X_train, y_train, X_valid, y_valid, seed: int = RANDOM_SEED):
    import lightgbm as lgb

    from ..config import TrainConfig

    constraints = [MONOTONE_DIRECTIONS.get(c, 0) for c in X_train.columns]
    params = {**TrainConfig().lgbm_params, "monotone_constraints": constraints}
    model = lgb.LGBMClassifier(random_state=seed, **params)
    model.fit(
        _prep_cats(X_train), y_train,
        eval_set=[(_prep_cats(X_valid), y_valid)], eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def train_ebm(X_train, y_train, seed: int = RANDOM_SEED, interactions: int = 15):
    """Explainable Boosting Machine: cyclic-boosted GAM + top-K learned pairwise
    interactions. Every term is a plottable shape function — glass box."""
    from interpret.glassbox import ExplainableBoostingClassifier

    model = ExplainableBoostingClassifier(
        interactions=interactions, outer_bags=4, inner_bags=0,
        random_state=seed, n_jobs=-1,
    )
    Xt = X_train.copy()
    for c in CATEGORICAL_FEATURES:
        if c in Xt.columns:
            Xt[c] = Xt[c].astype(str)
    model.fit(Xt, y_train)
    return model


def predict_ebm(model, X: pd.DataFrame) -> np.ndarray:
    X = X.copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype(str)
    return model.predict_proba(X)[:, 1]


# ---- augmented scorecard (SHAP-interaction distillation) ----------------------

def top_shap_interaction_pairs(
    gbm_model, X: pd.DataFrame, numeric: list[str],
    n_pairs: int = 6, sample: int = 2_000, seed: int = RANDOM_SEED,
) -> list[tuple[str, str]]:
    """Top numeric×numeric pairs by mean |SHAP interaction value| from a trained
    GBM — the candidate crosses to graft onto the scorecard."""
    import shap

    Xs = _prep_cats(X.sample(sample, random_state=seed))
    # the interaction path requires a float matrix; category codes are exactly
    # what LightGBM splits on, and only numeric×numeric pairs are kept anyway
    for c in Xs.columns:
        if str(Xs[c].dtype) == "category":
            Xs[c] = Xs[c].cat.codes
    inter = shap.TreeExplainer(gbm_model).shap_interaction_values(Xs)
    if isinstance(inter, list):
        inter = inter[1]
    strength = np.abs(inter).mean(axis=0)  # (F, F)
    cols = list(X.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if cols[i] in numeric and cols[j] in numeric:
                pairs.append((strength[i, j], cols[i], cols[j]))
    pairs.sort(reverse=True)
    return [(a, b) for _, a, b in pairs[:n_pairs]]


def add_cross_features(X: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    X = X.copy()
    for a, b in pairs:
        X[f"x_{a}__{b}"] = X[a] * X[b]
    return X


def train_augmented_scorecard(
    X_train, y_train, pairs: list[tuple[str, str]],
    numeric: list[str], categorical: list[str], n_bins: int = 5,
) -> Scorecard:
    """The incumbent's architecture, upgraded with GBM-distilled cross terms:
    how much of the challenger gap closes while staying a points-based,
    additive, fully explainable scorecard?"""
    Xa = add_cross_features(X_train, pairs)
    cross_cols = [f"x_{a}__{b}" for a, b in pairs]
    sc = Scorecard(n_bins=n_bins)
    sc.fit(Xa, y_train, numeric + cross_cols, categorical)
    sc.cross_pairs_ = pairs
    return sc


def predict_augmented_scorecard(sc: Scorecard, X: pd.DataFrame) -> np.ndarray:
    return sc.predict_proba(add_cross_features(X, sc.cross_pairs_))
