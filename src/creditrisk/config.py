"""Central configuration for the credit risk pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_SEED = 42

# --- Simulation -------------------------------------------------------------
N_APPLICANTS = 800_000

# Bias-injection knobs (see data/simulate.py for mechanics). Setting both to 0
# produces an unbiased world; defaults inject measurable, controlled bias so the
# fairness test suite has ground truth to detect.
PROXY_BIAS_STRENGTH = 0.8        # how strongly geo_risk_index encodes group B
INCOME_UNDERREPORT_FRAC = 0.20   # group B reported income shrinkage (measurement bias)

# --- Modeling ---------------------------------------------------------------
TARGET = "default"
PROTECTED_ATTRS = ["gender", "age_band", "group"]  # audited, never used as features

NUMERIC_FEATURES = [
    "annual_income",
    "loan_amount",
    "loan_to_income",
    "debt_to_income",
    "revolving_utilization",
    "credit_history_months",
    "num_credit_lines",
    "num_delinq_2y",
    "inquiries_6m",
    "employment_years",
    "savings_balance",
    "monthly_expenses",
    "geo_risk_index",   # the injected proxy feature — kept so the audit can catch it
    "age",
]
CATEGORICAL_FEATURES = ["home_ownership", "loan_purpose", "employment_type"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Decisioning / business impact ------------------------------------------
APPROVAL_RATE = 0.70          # incumbent portfolio approval rate to match
LGD = 0.55                    # loss given default (unsecured personal lending)
ANNUAL_APPLICATIONS = 800_000 # annualized application volume for $ impact

# --- Scorecard scaling --------------------------------------------------------
SCORE_BASE_POINTS = 600
SCORE_BASE_ODDS = 19.0   # odds of good:bad at base points
SCORE_PDO = 20.0         # points to double the odds


@dataclass
class TrainConfig:
    test_size: float = 0.2
    valid_size: float = 0.1          # carved from train for early stopping
    seed: int = RANDOM_SEED
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "verbose": -1,
    })
    xgb_params: dict = field(default_factory=lambda: {
        "objective": "binary:logistic",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "n_jobs": -1,
    })
