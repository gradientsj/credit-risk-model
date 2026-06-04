"""Publication-quality figures for the model report.

Style follows common conventions from ML/quant-finance papers: colorblind-safe
palette, vector-friendly sizing, annotated key numbers, no chartjunk.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

PALETTE = {
    "incumbent": "#888888",
    "lightgbm": "#0173B2",
    "xgboost": "#DE8F05",
    "ft_transformer": "#029E73",
    "lightgbm_mitigated": "#CC78BC",
}
_FALLBACK = ["#CC78BC", "#CA9161", "#949494"]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
    })


def _color(name: str, i: int = 0) -> str:
    return PALETTE.get(name, _FALLBACK[i % len(_FALLBACK)])


def plot_roc(y_true, preds: dict[str, np.ndarray], ginis: dict[str, float], path: Path):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for i, (name, p) in enumerate(preds.items()):
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, lw=1.8, color=_color(name, i),
                label=f"{name}  (Gini = {ginis[name]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC — incumbent scorecard vs challengers (held-out test)")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.savefig(path); plt.close(fig)


def plot_calibration(y_true, preds: dict[str, np.ndarray], path: Path):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for i, (name, p) in enumerate(preds.items()):
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", ms=3.5, lw=1.4, color=_color(name, i), label=name)
    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5, label="perfect calibration")
    ax.set(xlabel="Mean predicted PD (decile)", ylabel="Realized default rate",
           title="Calibration (quantile bins, held-out test)")
    ax.legend(fontsize=8.5)
    fig.savefig(path); plt.close(fig)


def plot_score_distribution(y_true, scores: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    bins = np.linspace(scores.min(), scores.max(), 60)
    ax.hist(scores[y_true == 0], bins=bins, density=True, alpha=0.6,
            color="#0173B2", label="non-defaulters")
    ax.hist(scores[y_true == 1], bins=bins, density=True, alpha=0.6,
            color="#D55E00", label="defaulters")
    ax.set(xlabel="Credit score (points)", ylabel="Density",
           title="Score separation by outcome (challenger, points-scaled)")
    ax.legend()
    fig.savefig(path); plt.close(fig)


def plot_gains(y_true, preds: dict[str, np.ndarray], path: Path):
    """Cumulative bad-capture curve: % of all defaulters caught in worst-scored x%."""
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    n = len(y_true)
    for i, (name, p) in enumerate(preds.items()):
        order = np.argsort(-p)  # riskiest first
        capture = np.cumsum(np.asarray(y_true)[order]) / np.sum(y_true)
        ax.plot(np.arange(1, n + 1) / n, capture, lw=1.6, color=_color(name, i), label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="random")
    ax.set(xlabel="Share of applications (ranked riskiest first)",
           ylabel="Share of defaulters captured",
           title="Cumulative default capture (gains)")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.savefig(path); plt.close(fig)


def plot_gini_comparison(results: pd.DataFrame, cis: dict[str, tuple], path: Path):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    names = list(results.index)
    g = results["gini"].values
    yerr = np.array([[g[i] - cis[n][1], cis[n][2] - g[i]] for i, n in enumerate(names)]).T
    colors = [_color(n, i) for i, n in enumerate(names)]
    bars = ax.bar(names, g, yerr=yerr, capsize=4, color=colors, width=0.6)
    for bar, val in zip(bars, g):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}",
                ha="center", fontsize=9)
    lift = g[1:] - g[0]
    ax.set(ylabel="Gini (95% bootstrap CI)",
           title=f"Discrimination vs incumbent (best lift: +{lift.max():.3f} Gini)")
    ax.set_ylim(0, max(g) * 1.18)
    plt.setp(ax.get_xticklabels(), rotation=12)
    fig.savefig(path); plt.close(fig)


def plot_fairness(audit_tables: dict[str, dict[str, pd.DataFrame]], attr: str, path: Path):
    """Grouped bars: approval rate by protected group, one cluster per variant."""
    variants = list(audit_tables.keys())
    groups = list(audit_tables[variants[0]][attr].index)
    x = np.arange(len(groups))
    width = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for i, v in enumerate(variants):
        vals = audit_tables[v][attr]["approval_rate"].values
        ax.bar(x + i * width, vals, width, label=v, color=_FALLBACK[i % 3] if v not in PALETTE else PALETTE[v])
        for xi, val in zip(x + i * width, vals):
            ax.text(xi, val + 0.005, f"{val:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels(groups)
    ax.set(ylabel="Approval rate", title=f"Approval rate by {attr} — bias mitigation effect")
    ax.legend(fontsize=8.5)
    fig.savefig(path); plt.close(fig)


def plot_air(summaries: dict[str, pd.DataFrame], path: Path, air_threshold: float = 0.8):
    """Minimum adverse-impact ratio per protected attribute, per model variant."""
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    variants = list(summaries.keys())
    attrs = list(summaries[variants[0]].index)
    x = np.arange(len(attrs))
    width = 0.8 / len(variants)
    for i, v in enumerate(variants):
        vals = summaries[v]["min_air"].values
        ax.bar(x + i * width, vals, width, label=v,
               color=PALETTE.get(v, _FALLBACK[i % 3]))
    ax.axhline(air_threshold, color="#D55E00", ls="--", lw=1.2,
               label=f"four-fifths rule ({air_threshold:.2f})")
    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels(attrs)
    ax.set(ylabel="Minimum adverse impact ratio", ylim=(0, 1.12),
           title="Adverse impact ratio vs four-fifths rule")
    ax.legend(fontsize=8)
    fig.savefig(path); plt.close(fig)


def plot_nn_history(history: list[dict], path: Path):
    h = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(5.2, 3.8))
    ax1.plot(h["epoch"], h["train_loss"], color="#0173B2", lw=1.5, label="train loss")
    ax1.set(xlabel="Epoch", ylabel="Train BCE loss")
    ax2 = ax1.twinx()
    ax2.plot(h["epoch"], h["valid_auc"], color="#029E73", lw=1.5, label="valid AUC")
    ax2.set_ylabel("Validation AUC")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8.5, loc="center right")
    ax1.set_title("FT-Transformer training")
    fig.savefig(path); plt.close(fig)
