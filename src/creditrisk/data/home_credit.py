"""Home Credit Default Risk (Kaggle) data pipeline.

Requires Kaggle API credentials in ``~/.kaggle/kaggle.json`` and acceptance of
the competition rules at
https://www.kaggle.com/competitions/home-credit-default-risk

Provides download, loading, and a compact feature-engineering pass over
``application_train`` plus aggregated ``bureau`` features — enough to support
the same scorecard-vs-GBM comparison and SHAP reporting as the simulated
pipeline, on real data.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR

HC_DIR = DATA_DIR / "home_credit"
COMPETITION = "home-credit-default-risk"


def download(force: bool = False) -> Path:
    """Download and extract the competition data via the Kaggle API."""
    HC_DIR.mkdir(parents=True, exist_ok=True)
    marker = HC_DIR / "application_train.csv"
    if marker.exists() and not force:
        return HC_DIR
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(HC_DIR)],
        check=True,
    )
    for z in HC_DIR.glob("*.zip"):
        with zipfile.ZipFile(z) as f:
            f.extractall(HC_DIR)
        z.unlink()
    return HC_DIR


def load_applications() -> pd.DataFrame:
    return pd.read_csv(HC_DIR / "application_train.csv")


def engineer_features(app: pd.DataFrame, with_bureau: bool = True) -> pd.DataFrame:
    """Compact, documented feature pass: domain ratios on the application table
    plus bureau aggregates. Keeps the feature count manageable so SHAP reports
    stay readable."""
    df = pd.DataFrame({"SK_ID_CURR": app["SK_ID_CURR"], "TARGET": app.get("TARGET")})

    # demographics / stability (protected-adjacent fields kept ONLY for audit)
    df["age_years"] = -app["DAYS_BIRTH"] / 365.25
    df["employment_years"] = (-app["DAYS_EMPLOYED"].replace(365243, np.nan)) / 365.25
    df["gender"] = app["CODE_GENDER"]  # audit only — never a model feature

    # core financials
    df["income_total"] = app["AMT_INCOME_TOTAL"]
    df["credit_amount"] = app["AMT_CREDIT"]
    df["annuity"] = app["AMT_ANNUITY"]
    df["goods_price"] = app["AMT_GOODS_PRICE"]
    df["credit_to_income"] = app["AMT_CREDIT"] / app["AMT_INCOME_TOTAL"]
    df["annuity_to_income"] = app["AMT_ANNUITY"] / app["AMT_INCOME_TOTAL"]
    df["credit_to_goods"] = app["AMT_CREDIT"] / app["AMT_GOODS_PRICE"]
    df["payment_years"] = app["AMT_CREDIT"] / app["AMT_ANNUITY"]

    # external scores — strongest known predictors in this competition
    for i in (1, 2, 3):
        df[f"ext_source_{i}"] = app[f"EXT_SOURCE_{i}"]
    df["ext_source_mean"] = app[[f"EXT_SOURCE_{i}" for i in (1, 2, 3)]].mean(axis=1)

    # documents / contact verifiability
    df["region_rating"] = app["REGION_RATING_CLIENT_W_CITY"]
    df["own_car"] = (app["FLAG_OWN_CAR"] == "Y").astype(int)
    df["own_realty"] = (app["FLAG_OWN_REALTY"] == "Y").astype(int)
    df["family_members"] = app["CNT_FAM_MEMBERS"]
    df["education"] = app["NAME_EDUCATION_TYPE"]
    df["income_type"] = app["NAME_INCOME_TYPE"]
    df["contract_type"] = app["NAME_CONTRACT_TYPE"]

    if with_bureau and (HC_DIR / "bureau.csv").exists():
        bureau = pd.read_csv(HC_DIR / "bureau.csv")
        agg = bureau.groupby("SK_ID_CURR").agg(
            bureau_loan_count=("SK_ID_BUREAU", "count"),
            bureau_active=("CREDIT_ACTIVE", lambda s: (s == "Active").sum()),
            bureau_overdue_max=("CREDIT_DAY_OVERDUE", "max"),
            bureau_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
            bureau_credit_sum=("AMT_CREDIT_SUM", "sum"),
        ).reset_index()
        agg["bureau_debt_ratio"] = agg["bureau_debt_sum"] / agg["bureau_credit_sum"].replace(0, np.nan)
        df = df.merge(agg, on="SK_ID_CURR", how="left")

    return df


HC_NUMERIC = [
    "age_years", "employment_years", "income_total", "credit_amount", "annuity",
    "credit_to_income", "annuity_to_income", "credit_to_goods", "payment_years",
    "ext_source_1", "ext_source_2", "ext_source_3", "ext_source_mean",
    "region_rating", "own_car", "own_realty", "family_members",
    "bureau_loan_count", "bureau_active", "bureau_overdue_max", "bureau_debt_ratio",
]
HC_CATEGORICAL = ["education", "income_type", "contract_type"]
