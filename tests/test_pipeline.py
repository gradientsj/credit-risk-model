"""Unit tests: simulator ground truth, scorecard, metrics, fairness suite.

Run from the project root:  pytest tests/ -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditrisk.data.simulate import simulate_applicants
from creditrisk.evaluation import (
    approval_threshold, business_impact, decisions, gini, ks_statistic, psi,
)
from creditrisk.fairness import audit, group_metrics
from creditrisk.models.scorecard import Scorecard


@pytest.fixture(scope="module")
def df():
    return simulate_applicants(60_000, seed=7)


def test_simulator_shape_and_rate(df):
    assert len(df) == 60_000
    assert 0.06 < df["default"].mean() < 0.10
    assert df["application_id"].is_unique


def test_protected_attrs_causally_inert(df):
    """true_pd must be (statistically) independent of protected attributes."""
    overall = df["true_pd"].mean()
    for attr in ("gender", "group"):
        gaps = df.groupby(attr)["true_pd"].mean() - overall
        assert gaps.abs().max() < 0.01, f"{attr} leaks into true_pd: {gaps.to_dict()}"


def test_bias_knobs_zero_removes_group_signal():
    """With knobs at 0, geo_risk_index must not differ by group beyond noise."""
    clean = simulate_applicants(60_000, seed=7, proxy_strength=0.0, income_underreport=0.0)
    gap = clean.groupby("group")["geo_risk_index"].mean().diff().iloc[-1]
    assert abs(gap) < 0.5
    biased = simulate_applicants(60_000, seed=7)  # default knobs
    gap_b = biased.groupby("group")["geo_risk_index"].mean().diff().iloc[-1]
    assert gap_b > 10, "proxy injection should shift group B's geo index"


def test_scorecard_discriminates_and_scales(df):
    train, test = df.iloc[:40_000], df.iloc[40_000:]
    sc = Scorecard(n_bins=5).fit(
        train, train["default"],
        ["revolving_utilization", "debt_to_income", "num_delinq_2y", "annual_income"],
        ["home_ownership"],
    )
    p = sc.predict_proba(test)
    assert gini(test["default"], p) > 0.3
    scores = sc.score(test)
    # higher points must mean lower risk
    assert test["default"][scores >= np.median(scores)].mean() < \
           test["default"][scores < np.median(scores)].mean()


def test_metrics_sanity():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.1, 20_000)
    perfect, random_p = y + rng.normal(0, 1e-6, len(y)), rng.random(len(y))
    assert gini(y, perfect) > 0.99
    assert abs(gini(y, random_p)) < 0.05
    assert ks_statistic(y, perfect) > 0.99
    assert psi(random_p, random_p) < 1e-6
    assert psi(random_p, random_p * 0.5 + 0.5) > 0.25


def test_approval_threshold_hits_rate():
    p = np.random.default_rng(1).random(50_000)
    thr = approval_threshold(p, 0.7)
    assert abs(decisions(p, thr).mean() - 0.7) < 0.01


def test_business_impact_positive_for_better_model(df):
    """A model closer to true_pd must reduce losses at constant volume."""
    y = df["default"].values
    noisy = df["true_pd"].values + np.random.default_rng(2).normal(0, 0.08, len(df))
    impact = business_impact(y, noisy, df["true_pd"].values, df["loan_amount"].values)
    assert impact["annual_loss_reduction"] > 0
    assert impact["constant_risk_approval_rate"] > impact["approval_rate"]


def test_fairness_audit_detects_injected_bias(df):
    """Scoring WITH the proxy feature must show worse group AIR than scoring
    against true_pd (the unbiased benchmark)."""
    y = df["default"].values
    prot = df[["group"]].reset_index(drop=True)

    # biased score: true risk + proxy contamination
    biased = df["true_pd"].values + 0.004 * (df["geo_risk_index"].values - 50)
    fair = df["true_pd"].values

    air = {}
    for name, p in [("biased", biased), ("fair", fair)]:
        appr = decisions(p, approval_threshold(p))
        _, summary = audit(y, p, appr, prot, ["group"])
        air[name] = summary.loc["group", "min_air"]
    assert air["biased"] < air["fair"] - 0.05
    assert air["fair"] > 0.9


def test_group_metrics_columns(df):
    y = df["default"].values
    p = df["true_pd"].values
    appr = decisions(p, approval_threshold(p))
    gm = group_metrics(y, p, appr, df["gender"].reset_index(drop=True))
    assert {"approval_rate", "air", "tpr_qualified", "calibration_gap"} <= set(gm.columns)
    assert gm["share"].sum() == pytest.approx(1.0)
