"""SHAP explainability: global model behavior and per-applicant adverse-action
reason codes (ECOA / Regulation B style).

For declined applicants, the top positive SHAP contributions (features pushing
PD up) are mapped to standardized reason-code language — the basis of the
adverse-action notice a lender must send. SHAP's additivity makes the stated
reasons faithful to the actual score, which is the regulatory requirement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .config import CATEGORICAL_FEATURES

# Feature -> (reason code, adverse-action language). Loosely modeled on the
# FCRA/Reg B sample notice vocabulary.
REASON_CODES: dict[str, tuple[str, str]] = {
    "revolving_utilization": ("R01", "Proportion of revolving balances to credit limits is too high"),
    "debt_to_income": ("R02", "Debt obligations are too high relative to income"),
    "num_delinq_2y": ("R03", "Recent delinquency on prior obligations"),
    "credit_history_months": ("R04", "Length of credit history is insufficient"),
    "inquiries_6m": ("R05", "Too many recent inquiries for credit"),
    "annual_income": ("R06", "Income is insufficient for the amount of credit requested"),
    "loan_to_income": ("R07", "Amount requested is too high relative to income"),
    "loan_amount": ("R07", "Amount requested is too high relative to income"),
    "employment_years": ("R08", "Length of employment is insufficient"),
    "employment_type": ("R08", "Length or stability of employment is insufficient"),
    "num_credit_lines": ("R09", "Number or age of revolving accounts"),
    "savings_balance": ("R10", "Insufficient deposit or savings balances"),
    "monthly_expenses": ("R11", "Level of recurring monthly obligations"),
    "home_ownership": ("R12", "Residential status or housing obligation"),
    "loan_purpose": ("R13", "Purpose of the requested credit"),
    "geo_risk_index": ("R14", "Credit performance associated with applicant location"),
    "age": ("R15", "Insufficient credit file maturity"),
}


def _prep(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns and X[col].dtype == object:
            X[col] = X[col].astype("category")
    return X


def tree_shap(model, X: pd.DataFrame, sample: int | None = 20_000, seed: int = 42):
    """TreeExplainer SHAP values for a (Light)GBM model. Returns
    (explainer, X_sample, shap_values[n, features])."""
    Xs = X.sample(sample, random_state=seed) if sample and len(X) > sample else X
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(_prep(Xs))
    if isinstance(sv, list):  # older API returns [class0, class1]
        sv = sv[1]
    return explainer, Xs, sv


def global_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.Series:
    return (
        pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)
        .sort_values(ascending=False)
    )


def adverse_action_reasons(
    explainer,
    applicant: pd.DataFrame,
    n_reasons: int = 4,
) -> list[dict]:
    """Top-N reasons a single applicant's PD was pushed up, as reason codes."""
    sv = explainer.shap_values(_prep(applicant))
    if isinstance(sv, list):
        sv = sv[1]
    contrib = pd.Series(sv[0], index=applicant.columns)
    pushing_up = contrib[contrib > 0].sort_values(ascending=False).head(n_reasons)
    reasons = []
    for feat, val in pushing_up.items():
        code, text = REASON_CODES.get(feat, ("R99", f"Value of {feat}"))
        reasons.append({
            "feature": feat,
            "applicant_value": applicant.iloc[0][feat],
            "shap_contribution": float(val),
            "reason_code": code,
            "reason": text,
        })
    return reasons


def adverse_action_notice(
    application_id,
    pd_hat: float,
    threshold: float,
    reasons: list[dict],
) -> str:
    """Render a sample adverse-action notice for a declined applicant."""
    lines = [
        "ADVERSE ACTION NOTICE (sample, generated for model documentation)",
        f"Application ID: {application_id}",
        f"Decision: DECLINED  (model PD {pd_hat:.2%} vs cutoff {threshold:.2%})",
        "",
        "Principal reasons for adverse action (per ECOA / Regulation B):",
    ]
    for i, r in enumerate(reasons, 1):
        lines.append(f"  {i}. [{r['reason_code']}] {r['reason']}")
        lines.append(f"      (applicant value: {r['applicant_value']}, "
                     f"contribution to risk score: +{r['shap_contribution']:.4f})")
    lines += [
        "",
        "You have the right to a statement of specific reasons and to dispute "
        "the accuracy of information used in this decision.",
    ]
    return "\n".join(lines)
