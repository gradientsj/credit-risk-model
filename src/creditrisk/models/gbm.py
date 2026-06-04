"""Gradient-boosted challenger models (LightGBM and XGBoost)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from ..config import CATEGORICAL_FEATURES, TrainConfig


def _prep(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns and X[col].dtype == object:
            X[col] = X[col].astype("category")
    return X


def train_lightgbm(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_valid: pd.DataFrame, y_valid: pd.Series,
    cfg: TrainConfig | None = None,
) -> lgb.LGBMClassifier:
    cfg = cfg or TrainConfig()
    model = lgb.LGBMClassifier(random_state=cfg.seed, **cfg.lgbm_params)
    model.fit(
        _prep(X_train), y_train,
        eval_set=[(_prep(X_valid), y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_valid: pd.DataFrame, y_valid: pd.Series,
    cfg: TrainConfig | None = None,
) -> xgb.XGBClassifier:
    cfg = cfg or TrainConfig()
    model = xgb.XGBClassifier(
        random_state=cfg.seed,
        enable_categorical=True,
        early_stopping_rounds=100,
        eval_metric="auc",
        **cfg.xgb_params,
    )
    model.fit(_prep(X_train), y_train, eval_set=[(_prep(X_valid), y_valid)], verbose=False)
    return model


def predict_gbm(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(_prep(X))[:, 1]
