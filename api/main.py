"""FastAPI scoring service: PD, points-scaled score, decision, and SHAP-based
adverse-action reason codes for a single applicant.

Run:  uvicorn api.main:app --reload   (from the project root, venv active)
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creditrisk.config import (  # noqa: E402
    MODELS_DIR, SCORE_BASE_ODDS, SCORE_BASE_POINTS, SCORE_PDO,
)
from creditrisk.explain import adverse_action_reasons  # noqa: E402

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Champion model scoring with SHAP-based adverse-action reason codes",
    version="1.0.0",
)

_state: dict = {}


@app.on_event("startup")
def load_artifacts():
    import shap

    bundle = joblib.load(MODELS_DIR / "champion.joblib")
    _state["model"] = bundle["model"]
    _state["features"] = bundle["features"]
    _state["threshold"] = bundle["threshold_pd"]
    _state["version"] = bundle["version"]
    _state["explainer"] = shap.TreeExplainer(bundle["model"])


class Applicant(BaseModel):
    age: float = Field(..., ge=18, le=100)
    annual_income: float = Field(..., gt=0)
    employment_years: float = Field(..., ge=0)
    employment_type: str = Field(..., pattern="^(salaried|self_employed|contract)$")
    credit_history_months: float = Field(..., ge=0)
    num_credit_lines: int = Field(..., ge=0)
    revolving_utilization: float = Field(..., ge=0, le=1)
    debt_to_income: float = Field(..., ge=0, le=1)
    num_delinq_2y: int = Field(..., ge=0)
    inquiries_6m: int = Field(..., ge=0)
    savings_balance: float = Field(..., ge=0)
    monthly_expenses: float = Field(..., ge=0)
    loan_amount: float = Field(..., gt=0)
    loan_purpose: str
    home_ownership: str = Field(..., pattern="^(rent|own|mortgage)$")

    model_config = {"json_schema_extra": {"examples": [{
        "age": 34, "annual_income": 52000, "employment_years": 6.5,
        "employment_type": "salaried", "credit_history_months": 96,
        "num_credit_lines": 7, "revolving_utilization": 0.42,
        "debt_to_income": 0.31, "num_delinq_2y": 0, "inquiries_6m": 1,
        "savings_balance": 8000, "monthly_expenses": 2400,
        "loan_amount": 12000, "loan_purpose": "debt_consolidation",
        "home_ownership": "rent",
    }]}}


class ScoreResponse(BaseModel):
    probability_of_default: float
    credit_score: int
    decision: str
    threshold_pd: float
    model_version: str
    adverse_action_reasons: list[dict] | None


def _to_frame(applicant: Applicant) -> pd.DataFrame:
    row = applicant.model_dump()
    row["loan_to_income"] = row["loan_amount"] / max(row["annual_income"], 1_000)
    df = pd.DataFrame([row])
    for col in ("home_ownership", "loan_purpose", "employment_type"):
        df[col] = df[col].astype("category")
    return df[_state["features"]]


@app.get("/health")
def health():
    return {"status": "ok", "model_version": _state.get("version")}


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant):
    if "model" not in _state:
        raise HTTPException(503, "model not loaded")
    X = _to_frame(applicant)
    pd_hat = float(_state["model"].predict_proba(X)[:, 1][0])

    factor = SCORE_PDO / np.log(2)
    offset = SCORE_BASE_POINTS - factor * np.log(SCORE_BASE_ODDS)
    odds_good = (1 - pd_hat) / max(pd_hat, 1e-6)
    points = int(round(offset + factor * np.log(odds_good)))

    declined = pd_hat > _state["threshold"]
    reasons = None
    if declined:
        reasons = adverse_action_reasons(_state["explainer"], X)
        for r in reasons:  # JSON-safe
            r["applicant_value"] = str(r["applicant_value"])
    return ScoreResponse(
        probability_of_default=round(pd_hat, 6),
        credit_score=points,
        decision="declined" if declined else "approved",
        threshold_pd=round(_state["threshold"], 6),
        model_version=_state["version"],
        adverse_action_reasons=reasons,
    )
