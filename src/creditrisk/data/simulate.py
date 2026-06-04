"""Synthetic loan-application generator with controlled bias injection.

Design
------
Each applicant is driven by three latent factors:

* ``stability``  — financial stability (income level, savings, employment tenure)
* ``experience`` — credit experience (history length, number of lines)
* ``pressure``   — debt pressure (utilization, DTI, delinquencies, inquiries)

Observable features are noisy, correlated transforms of these latents, so the
feature correlation structure resembles real bureau data. The *true* default
probability is a nonlinear function of the latents and feature interactions —
deliberately beyond what a binned logistic scorecard can express, which is what
creates headroom for gradient-boosted challengers.

Fairness ground truth
---------------------
Protected attributes (gender, age band, demographic group) have **zero causal
effect** on default. Bias is then injected through two controlled mechanisms:

1. **Proxy bias** (``proxy_strength``): a ``geo_risk_index`` feature is shifted
   upward for group B independent of true risk. A model that uses it will
   systematically over-score group B's risk.
2. **Measurement bias** (``income_underreport``): group B's *reported* income is
   shrunk relative to true income (informal income, thin-file effects), so
   income-derived features understate group B's capacity to repay.

Because the injection is parameterized, the fairness test suite can be
validated against known ground truth: with knobs at 0 the disparity metrics
should pass; at the default settings they should fail and the audit must
catch it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (
    INCOME_UNDERREPORT_FRAC,
    N_APPLICANTS,
    PROXY_BIAS_STRENGTH,
    RANDOM_SEED,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_applicants(
    n: int = N_APPLICANTS,
    seed: int = RANDOM_SEED,
    proxy_strength: float = PROXY_BIAS_STRENGTH,
    income_underreport: float = INCOME_UNDERREPORT_FRAC,
    target_default_rate: float = 0.08,
) -> pd.DataFrame:
    """Generate ``n`` loan applications with features, protected attributes and
    a realized default outcome.

    Returns a DataFrame containing model features, protected attributes, the
    realized ``default`` label and ``true_pd`` (the causal default probability,
    useful for validating that protected attributes carry no signal).
    """
    rng = np.random.default_rng(seed)

    # ---- protected attributes (causally inert) ------------------------------
    gender = rng.choice(["female", "male"], size=n, p=[0.48, 0.52])
    group = rng.choice(["A", "B"], size=n, p=[0.85, 0.15])
    is_b = (group == "B").astype(float)

    age = np.clip(rng.gamma(shape=9.0, scale=4.6, size=n) + 18, 18, 85).round()
    age_band = pd.cut(
        age, bins=[17, 25, 35, 50, 65, 120],
        labels=["18-25", "26-35", "36-50", "51-65", "65+"],
    ).astype(str)

    # ---- latent factors ------------------------------------------------------
    stability = rng.normal(0, 1, n) + 0.35 * np.tanh((age - 40) / 15)
    experience = (
        0.45 * stability
        + 0.65 * np.tanh((age - 30) / 12)
        + rng.normal(0, 0.85, n)
    )
    pressure = -0.5 * stability + rng.normal(0, 0.9, n)

    # ---- observable features -------------------------------------------------
    true_income = np.exp(10.55 + 0.45 * stability + rng.normal(0, 0.25, n))
    # measurement bias: group B's *reported* income is shrunk; capacity to repay
    # (true_income) is unchanged.
    reported_shrink = 1.0 - income_underreport * is_b * rng.uniform(0.5, 1.5, n)
    annual_income = (true_income * reported_shrink).round(-2)

    employment_years = np.clip(
        np.maximum(age - 18 - rng.exponential(6, n), 0)
        * _sigmoid(0.8 * stability + rng.normal(0, 1, n)),
        0, 45,
    ).round(1)
    employment_type = np.where(
        rng.random(n) < _sigmoid(0.9 * stability - 0.3),
        "salaried",
        np.where(rng.random(n) < 0.55, "self_employed", "contract"),
    )

    credit_history_months = np.clip(
        12 * (age - 18) * _sigmoid(1.1 * experience) + rng.normal(0, 18, n),
        0, 600,
    ).round()
    num_credit_lines = np.clip(
        rng.poisson(np.exp(1.25 + 0.4 * experience)), 0, 40
    )
    revolving_utilization = np.clip(
        _sigmoid(0.9 * pressure - 0.35 * stability + rng.normal(0, 0.8, n)),
        0, 1.0,
    )
    debt_to_income = np.clip(
        0.28 + 0.13 * pressure - 0.05 * stability + rng.normal(0, 0.09, n),
        0.0, 0.95,
    )
    num_delinq_2y = rng.poisson(
        np.exp(-1.4 + 0.85 * pressure - 0.4 * experience).clip(max=4)
    ).clip(0, 20)
    inquiries_6m = rng.poisson(
        np.exp(-0.4 + 0.55 * pressure).clip(max=3)
    ).clip(0, 15)

    savings_balance = np.expm1(
        np.clip(8.2 + 1.1 * stability - 0.4 * pressure + rng.normal(0, 1.0, n), 0, 14)
    ).round(-1)
    monthly_expenses = (
        true_income / 12 * np.clip(0.45 + 0.1 * pressure + rng.normal(0, 0.08, n), 0.2, 0.95)
    ).round(-1)

    loan_amount = np.clip(
        true_income * np.clip(0.18 + 0.07 * pressure + rng.normal(0, 0.08, n), 0.03, 0.8),
        1_000, 100_000,
    ).round(-2)
    # observed feature uses *reported* income; the true risk equation below uses
    # true income, so the injected measurement bias stays causally inert.
    loan_to_income = loan_amount / annual_income.clip(min=1_000)
    true_lti = loan_amount / true_income.clip(min=1_000)
    loan_purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "auto", "medical", "small_business", "other"],
        size=n, p=[0.42, 0.16, 0.14, 0.09, 0.07, 0.12],
    )
    home_ownership = np.where(
        rng.random(n) < _sigmoid(0.8 * stability + 0.03 * (age - 35) - 0.2),
        np.where(rng.random(n) < 0.6, "mortgage", "own"),
        "rent",
    )

    # ---- true default probability (causal; no protected attributes) ----------
    # Main effects are kept modest and most of the explainable signal is routed
    # through feature *interactions*: a coarse-binned additive scorecard can
    # recover per-feature (even non-monotone) shapes via WOE bins, but it cannot
    # express interactions — that structural gap is the challenger's headroom.
    # Risk is dominated by *regime interactions* whose per-feature marginals are
    # designed to roughly cancel: high utilization is dangerous for a distressed
    # revolver but benign for a transactor; leverage is dangerous without a
    # savings buffer but absorbed by one. A coarse-binned additive scorecard can
    # recover any per-feature WOE shape, but when marginals net to ~zero the
    # signal lives in the joint distribution — structural headroom for the
    # gradient-boosted challenger.
    util_excess = np.maximum(revolving_utilization - 0.6, 0)
    high_dti = debt_to_income > 0.35
    low_savings = savings_balance < 2_000
    high_util = revolving_utilization > 0.65
    thin_file = (credit_history_months < 90) | (num_credit_lines < 5)
    sb = loan_purpose == "small_business"
    risk_signal = (
        -0.25 * stability
        - 0.12 * experience
        + 0.20 * pressure
        # distressed revolver vs transactor
        + 3.4 * util_excess * high_dti
        - 1.8 * util_excess * ~high_dti
        # leverage with vs without a liquidity buffer
        + 3.0 * true_lti * low_savings
        - 1.5 * true_lti * ~low_savings
        # debt pressure bites hardest when unstable, is deliberate when stable
        + 1.9 * pressure * np.maximum(-stability, 0)
        - 0.9 * pressure * np.maximum(stability, 0)
        # credit-seeking while delinquent vs isolated events
        + 1.5 * (num_delinq_2y >= 2) * (inquiries_6m >= 3)
        - 0.4 * (num_delinq_2y >= 2) * (inquiries_6m < 3)
        # new ventures vs established operators
        + 1.2 * sb * (employment_years < 4)
        - 0.5 * sb * (employment_years >= 4)
        + 0.9 * (employment_type == "contract") * (debt_to_income > 0.4)
        # thin file + maxed cards vs seasoned revolvers
        + 1.6 * high_util * thin_file
        - 0.7 * high_util * ~thin_file
        - 0.6 * (revolving_utilization < 0.25) * thin_file
    )
    # unobservable heterogeneity, split into a *regional* economic shock and a
    # purely idiosyncratic component
    regional_shock = rng.normal(0, 1.3, n)
    risk = risk_signal + regional_shock + rng.normal(0, 2.9, n)

    # proxy bias: geo_risk_index mimics a *regional realized default-rate index*
    # (an alternative-data feature the challenger model adds; the incumbent
    # scorecard predates it). It carries genuinely unique signal — the regional
    # shock, invisible in applicant-level features — so any model that wants
    # the Gini must lean on it. The injected group-B shift is unrelated to
    # repayment: the classic redlining mechanism, where an on-average-predictive
    # index silently prices group membership.
    geo_risk_index = np.clip(
        50
        + 5.5 * regional_shock                # unique legitimate signal
        + 1.5 * risk_signal                   # redundant legitimate signal
        + 28.0 * proxy_strength * is_b        # injected group proxy
        + rng.normal(0, 4, n),
        0, 100,
    ).round(1)
    # calibrate intercept to hit the target portfolio default rate
    lo, hi = -8.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(risk + mid).mean() > target_default_rate:
            hi = mid
        else:
            lo = mid
    true_pd = _sigmoid(risk + (lo + hi) / 2)
    default = rng.binomial(1, true_pd)

    df = pd.DataFrame({
        "application_id": np.arange(1, n + 1),
        # protected attributes (audit only)
        "gender": gender,
        "group": group,
        "age_band": age_band,
        # features
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "employment_type": employment_type,
        "credit_history_months": credit_history_months,
        "num_credit_lines": num_credit_lines,
        "revolving_utilization": revolving_utilization.round(4),
        "debt_to_income": debt_to_income.round(4),
        "num_delinq_2y": num_delinq_2y,
        "inquiries_6m": inquiries_6m,
        "savings_balance": savings_balance,
        "monthly_expenses": monthly_expenses,
        "loan_amount": loan_amount,
        "loan_to_income": loan_to_income.round(4),
        "loan_purpose": loan_purpose,
        "home_ownership": home_ownership,
        "geo_risk_index": geo_risk_index,
        # outcome
        "true_pd": true_pd.round(6),
        "default": default,
    })
    return df


if __name__ == "__main__":
    df = simulate_applicants(50_000)
    print(df.head())
    print(f"default rate: {df['default'].mean():.4f}")
    print(df.groupby("group")["true_pd"].mean())
