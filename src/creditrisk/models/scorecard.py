"""Incumbent model: a classic WOE-binned logistic-regression scorecard.

This mirrors the scorecards most lenders still run in production: each
characteristic is coarse-binned, bins are encoded as weight-of-evidence (WOE),
and a logistic regression is fit on the WOE values. Scores are scaled to the
conventional points/PDO system so the output reads like a bureau score.

Its structural limits (coarse bins, additivity — no interactions) are exactly
why a gradient-boosted challenger finds Gini headroom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..config import SCORE_BASE_ODDS, SCORE_BASE_POINTS, SCORE_PDO


class WOEBinner:
    """Quantile-based WOE binning for numeric and categorical features."""

    def __init__(self, n_bins: int = 5, min_bin_frac: float = 0.02):
        self.n_bins = n_bins
        self.min_bin_frac = min_bin_frac
        self.numeric_edges_: dict[str, np.ndarray] = {}
        self.woe_maps_: dict[str, dict] = {}
        self.iv_: dict[str, float] = {}

    @staticmethod
    def _woe_table(binned: pd.Series, y: pd.Series) -> tuple[dict, float]:
        tab = pd.crosstab(binned, y)
        for col in (0, 1):
            if col not in tab:
                tab[col] = 0
        good = (tab[0] + 0.5) / (tab[0].sum() + 0.5)
        bad = (tab[1] + 0.5) / (tab[1].sum() + 0.5)
        woe = np.log(good / bad)
        iv = float(((good - bad) * woe).sum())
        return woe.to_dict(), iv

    def fit(self, X: pd.DataFrame, y: pd.Series, numeric: list[str], categorical: list[str]):
        for col in numeric:
            edges = np.unique(
                np.quantile(X[col].dropna(), np.linspace(0, 1, self.n_bins + 1))
            )
            edges[0], edges[-1] = -np.inf, np.inf
            self.numeric_edges_[col] = edges
            binned = pd.cut(X[col], edges, duplicates="drop")
            self.woe_maps_[col], self.iv_[col] = self._woe_table(binned, y)
        for col in categorical:
            self.woe_maps_[col], self.iv_[col] = self._woe_table(X[col], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col, edges in self.numeric_edges_.items():
            binned = pd.cut(X[col], edges, duplicates="drop")
            out[col] = binned.map(self.woe_maps_[col]).astype(float).fillna(0.0)
        for col, mapping in self.woe_maps_.items():
            if col in self.numeric_edges_:
                continue
            out[col] = X[col].map(mapping).astype(float).fillna(0.0)
        return pd.DataFrame(out, index=X.index)


class Scorecard:
    """WOE binning + logistic regression + points scaling."""

    def __init__(self, n_bins: int = 5, C: float = 1.0):
        self.binner = WOEBinner(n_bins=n_bins)
        self.lr = LogisticRegression(C=C, max_iter=2000)
        self.features_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, numeric: list[str], categorical: list[str]):
        self.features_ = numeric + categorical
        self.binner.fit(X, y, numeric, categorical)
        woe = self.binner.transform(X[self.features_])
        self.lr.fit(woe, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        woe = self.binner.transform(X[self.features_])
        return self.lr.predict_proba(woe)[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Convert PD to conventional credit-score points (higher = better)."""
        pd_hat = np.clip(self.predict_proba(X), 1e-6, 1 - 1e-6)
        odds_good = (1 - pd_hat) / pd_hat
        factor = SCORE_PDO / np.log(2)
        offset = SCORE_BASE_POINTS - factor * np.log(SCORE_BASE_ODDS)
        return np.round(offset + factor * np.log(odds_good)).astype(int)

    def information_values(self) -> pd.Series:
        return pd.Series(self.binner.iv_).sort_values(ascending=False)
