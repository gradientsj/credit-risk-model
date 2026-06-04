"""Discrimination, calibration and business-impact evaluation.

Conventions: predictions are probabilities of default (PD). Decisions approve
the lowest-PD applicants up to a target approval rate, matching how a lender
swaps in a challenger at constant volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .config import ANNUAL_APPLICATIONS, APPROVAL_RATE, LGD


# ---- discrimination ----------------------------------------------------------

def gini(y_true, y_pred) -> float:
    return 2 * roc_auc_score(y_true, y_pred) - 1


def ks_statistic(y_true, y_pred) -> float:
    """Kolmogorov–Smirnov distance between score CDFs of goods and bads."""
    order = np.argsort(y_pred)
    y = np.asarray(y_true)[order]
    cum_bad = np.cumsum(y) / y.sum()
    cum_good = np.cumsum(1 - y) / (len(y) - y.sum())
    return float(np.max(np.abs(cum_bad - cum_good)))


def bootstrap_gini_ci(y_true, y_pred, n_boot: int = 200, seed: int = 42,
                      sample_cap: int = 100_000) -> tuple[float, float, float]:
    """Point estimate and percentile 95% CI for Gini (subsampled bootstrap)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = min(len(y_true), sample_cap)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), n)
        if y_true[idx].sum() in (0, n):
            continue
        stats.append(gini(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return gini(y_true, y_pred), float(lo), float(hi)


def ece(y_true, y_pred, n_bins: int = 10) -> float:
    """Expected calibration error over quantile bins of predicted PD."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    edges = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.digitize(y_pred, edges[1:-1])
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(y_pred[mask].mean() - y_true[mask].mean())
    return float(total)


def evaluate_model(y_true, y_pred, name: str = "") -> dict:
    return {
        "model": name,
        "auc": float(roc_auc_score(y_true, y_pred)),
        "gini": float(gini(y_true, y_pred)),
        "ks": ks_statistic(y_true, y_pred),
        "brier": float(brier_score_loss(y_true, y_pred)),
        "default_rate": float(np.mean(y_true)),
    }


# ---- decisioning ---------------------------------------------------------------

def approval_threshold(y_pred: np.ndarray, approval_rate: float = APPROVAL_RATE) -> float:
    """PD cutoff that approves the lowest-risk `approval_rate` share."""
    return float(np.quantile(y_pred, approval_rate))


def decisions(y_pred: np.ndarray, threshold: float) -> np.ndarray:
    """1 = approved, 0 = declined."""
    return (y_pred <= threshold).astype(int)


# ---- business impact -------------------------------------------------------------

def business_impact(
    y_true: np.ndarray,
    pd_incumbent: np.ndarray,
    pd_challenger: np.ndarray,
    loan_amount: np.ndarray,
    approval_rate: float = APPROVAL_RATE,
    lgd: float = LGD,
    annual_applications: int = ANNUAL_APPLICATIONS,
) -> dict:
    """Two standard swap-set analyses:

    1. **Constant volume**: hold the approval rate fixed; the challenger approves a
       better-selected book, so expected default losses fall.
    2. **Constant risk**: hold expected losses at the incumbent level; the
       challenger can approve more applicants for the same loss budget.

    Losses are estimated as realized_default × loan_amount × LGD on the approved
    book, scaled to annual application volume.
    """
    n = len(y_true)
    scale = annual_applications / n

    def book_loss(pd_model, rate):
        approved = decisions(pd_model, approval_threshold(pd_model, rate)).astype(bool)
        loss = float((y_true[approved] * loan_amount[approved] * lgd).sum())
        return approved, loss

    inc_approved, inc_loss = book_loss(pd_incumbent, approval_rate)
    cha_approved, cha_loss = book_loss(pd_challenger, approval_rate)

    # constant-risk: raise challenger approval rate until losses match incumbent
    rate_lo, rate_hi = approval_rate, 0.999
    for _ in range(40):
        mid = (rate_lo + rate_hi) / 2
        _, loss_mid = book_loss(pd_challenger, mid)
        if loss_mid < inc_loss:
            rate_lo = mid
        else:
            rate_hi = mid
    expanded_rate = rate_lo

    swap_in = int((cha_approved & ~inc_approved).sum())
    swap_out = int((inc_approved & ~cha_approved).sum())

    return {
        "approval_rate": approval_rate,
        "incumbent_annual_loss": inc_loss * scale,
        "challenger_annual_loss": cha_loss * scale,
        "annual_loss_reduction": (inc_loss - cha_loss) * scale,
        "loss_reduction_pct": 1 - cha_loss / inc_loss,
        "swap_in": swap_in,
        "swap_out": swap_out,
        "constant_risk_approval_rate": expanded_rate,
        "additional_approvals_annual": int((expanded_rate - approval_rate) * annual_applications),
        "assumptions": {
            "lgd": lgd,
            "annual_applications": annual_applications,
            "loss_basis": "realized defaults x loan amount x LGD on approved book",
        },
    }


# ---- stability (monitoring) -----------------------------------------------------

def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two score distributions."""
    edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(((a - e) * np.log(a / e)).sum())


def metrics_table(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results).set_index("model").round(4)
