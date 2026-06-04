"""Fairness testing across protected classes.

Metrics computed per protected attribute, at the production decision threshold:

* **Approval rate** by group, and **Adverse Impact Ratio** (AIR) vs the most-
  approved group — the EEOC "four-fifths rule" (AIR >= 0.80) is the screening
  standard regulators and fair-lending audits apply first.
* **Demographic parity difference** — max gap in approval rates.
* **Equal opportunity difference** — max gap in TPR for *non-defaulters*
  (qualified applicants approved), the metric most aligned with credit access.
* **Equalized odds** — also checks FPR gaps (defaulters approved).
* **Within-group calibration** — mean predicted PD vs realized default rate per
  group; mis-calibration against a group means systematic over-pricing of risk.

The simulator injects bias with known mechanisms (proxy feature, income
under-reporting), so these tests have ground truth: they must FAIL on the
biased model and PASS (or materially improve) after mitigation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AIR_THRESHOLD = 0.80          # four-fifths rule
PARITY_TOLERANCE = 0.05       # max approval-rate / TPR gap treated as pass


def group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    approved: np.ndarray,
    protected: pd.Series,
) -> pd.DataFrame:
    """Per-group confusion and rate metrics. `approved`=1 is the favorable outcome;
    a 'qualified' applicant is a non-defaulter (y_true=0)."""
    rows = []
    for g, idx in protected.groupby(protected).groups.items():
        loc = protected.index.get_indexer(idx)
        yt, ap, pd_hat = y_true[loc], approved[loc], y_pred[loc]
        qualified, unqualified = yt == 0, yt == 1
        rows.append({
            "group": g,
            "n": len(loc),
            "share": len(loc) / len(protected),
            "approval_rate": ap.mean(),
            "tpr_qualified": ap[qualified].mean() if qualified.any() else np.nan,
            "fpr_defaulters": ap[unqualified].mean() if unqualified.any() else np.nan,
            "mean_pd_pred": pd_hat.mean(),
            "realized_default_rate": yt.mean(),
            "calibration_gap": pd_hat.mean() - yt.mean(),
        })
    df = pd.DataFrame(rows).set_index("group")
    df["air"] = df["approval_rate"] / df["approval_rate"].max()
    return df


def fairness_summary(gm: pd.DataFrame, attribute: str) -> dict:
    dp_diff = float(gm["approval_rate"].max() - gm["approval_rate"].min())
    eo_diff = float(gm["tpr_qualified"].max() - gm["tpr_qualified"].min())
    fpr_diff = float(gm["fpr_defaulters"].max() - gm["fpr_defaulters"].min())
    min_air = float(gm["air"].min())
    return {
        "attribute": attribute,
        "min_air": min_air,
        "demographic_parity_diff": dp_diff,
        "equal_opportunity_diff": eo_diff,
        "fpr_diff": fpr_diff,
        "max_abs_calibration_gap": float(gm["calibration_gap"].abs().max()),
        "passes_four_fifths": min_air >= AIR_THRESHOLD,
        "passes_equal_opportunity": eo_diff <= PARITY_TOLERANCE,
    }


def audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    approved: np.ndarray,
    df_protected: pd.DataFrame,
    attributes: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Run the full audit. Returns per-attribute group tables and a summary."""
    tables, summaries = {}, []
    for attr in attributes:
        gm = group_metrics(y_true, y_pred, approved, df_protected[attr])
        tables[attr] = gm
        summaries.append(fairness_summary(gm, attr))
    return tables, pd.DataFrame(summaries).set_index("attribute")


def render_fairness_report(
    audits: dict[str, tuple[dict[str, pd.DataFrame], pd.DataFrame]],
    out_path,
    context: dict | None = None,
) -> str:
    """Write a Markdown fairness report comparing model variants.

    `audits` maps model-variant name -> (tables, summary) from `audit()`.
    """
    lines = [
        "# Fairness & Disparate Impact Report",
        "",
        "Decision rule: approve lowest-PD applicants at the production approval rate.",
        f"Pass criteria: AIR >= {AIR_THRESHOLD} (four-fifths rule); "
        f"equal-opportunity gap <= {PARITY_TOLERANCE}.",
        "",
    ]
    if context:
        lines += [f"- **{k}**: {v}" for k, v in context.items()] + [""]
    for variant, (tables, summary) in audits.items():
        lines += [f"## Model variant: {variant}", "", summary.round(4).to_markdown(), ""]
        for attr, gm in tables.items():
            lines += [f"### By {attr}", "", gm.round(4).to_markdown(), ""]
    text = "\n".join(lines)
    out_path = str(out_path)
    with open(out_path, "w") as f:
        f.write(text)
    return text
