"""Tests for the extended model zoo: architectures, constraints, calibration."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditrisk.evaluation import ece
from creditrisk.models.baselines import MONOTONE_DIRECTIONS, add_cross_features


def test_ece_sanity():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.2, 50_000)
    y = rng.binomial(1, p)
    assert ece(y, p) < 0.01            # perfectly calibrated by construction
    assert ece(y, p * 3) > 0.05        # systematically inflated


def test_monotone_directions_are_signs():
    assert set(MONOTONE_DIRECTIONS.values()) <= {-1, 1}
    # regime-dependent features must stay unconstrained
    assert "num_credit_lines" not in MONOTONE_DIRECTIONS
    assert "age" not in MONOTONE_DIRECTIONS


def test_add_cross_features():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    Xc = add_cross_features(X, [("a", "b")])
    assert list(Xc["x_a__b"]) == [3.0, 8.0]
    assert list(X.columns) == ["a", "b"]  # input not mutated


@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(3)
    n = 4_000
    X = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "c1": rng.choice(["a", "b", "c"], n),
    })
    # main effect + interaction: learnable by every arch in a few CPU epochs
    logits = 1.4 * X["f1"] + 1.0 * X["f1"] * (X["f2"] > 0) - 2.0
    y = rng.binomial(1, 1 / (1 + np.exp(-logits)))
    return X, pd.Series(y)


@pytest.mark.parametrize("arch", ["ft_transformer", "mlp_resnet", "tabm"])
def test_nn_architectures_train_and_predict(tiny, arch):
    from creditrisk.models.tabular_nn import TabularNNClassifier
    from sklearn.metrics import roc_auc_score

    X, y = tiny
    model = TabularNNClassifier(
        ["f1", "f2"], ["c1"], arch=arch,
        arch_kwargs={"d": 32, "n_layers": 2, "k": 4} if arch == "tabm"
        else ({"d": 32, "n_blocks": 2} if arch == "mlp_resnet"
              else {"d_token": 16, "n_layers": 1, "n_heads": 4}),
        max_epochs=6, patience=6, batch_size=512, device="cpu",
    )
    model.fit(X.iloc[:3000], y.iloc[:3000].values, X.iloc[3000:], y.iloc[3000:].values)
    p = model.predict_proba(X.iloc[3000:])
    assert p.shape == (1000,)
    assert (0 <= p).all() and (p <= 1).all()
    # interaction-driven target: any working learner beats coin flip
    assert roc_auc_score(y.iloc[3000:], p) > 0.55


def test_monotone_lgbm_respects_constraint():
    """PD must be non-decreasing in a +1-constrained feature, all else fixed."""
    from creditrisk.models.baselines import train_monotone_lgbm
    rng = np.random.default_rng(5)
    n = 20_000
    X = pd.DataFrame({
        "debt_to_income": rng.uniform(0, 0.9, n),
        "annual_income": rng.lognormal(10.5, 0.4, n),
    })
    pd_true = 1 / (1 + np.exp(-(2.5 * X["debt_to_income"] - np.log(X["annual_income"]) + 8)))
    y = pd.Series(rng.binomial(1, pd_true))
    model = train_monotone_lgbm(X.iloc[:16000], y.iloc[:16000], X.iloc[16000:], y.iloc[16000:])
    grid = pd.DataFrame({
        "debt_to_income": np.linspace(0, 0.9, 50),
        "annual_income": np.full(50, 40_000.0),
    })
    p = model.predict_proba(grid)[:, 1]
    assert (np.diff(p) >= -1e-9).all()
