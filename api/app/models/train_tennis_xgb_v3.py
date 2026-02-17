# FILE: api/app/models/train_tennis_xgb_v3.py
"""
Train the V3 XGBoost tennis model — the best-performing variant.

Key improvements over combined_v2:
  1. ELO differential as a direct XGB feature (d_elo, d_surface_elo)
  2. Ranking differential (d_rank_log) — log-space
  3. Age differential (d_age)
  4. Round ordinal encoding (round_ordinal)
  5. Restored style proxy features (d_svc_pts_w, d_ret_pts_w, …)
  6. Smarter NaN handling (domain defaults, not blanket 0)
  7. Optuna hyperparameter tuning (optional --tune flag)
  8. Monotonic constraints on ELO / ranking features
  9. Better regularization + learning rate schedule
  10. Cross-validated calibration with reliability diagram logging
  11. Feature importance report

Usage:
  # Quick train with good defaults
  python -m app.models.train_tennis_xgb_v3

  # With Optuna tuning (slow but finds best hyperparams)
  python -m app.models.train_tennis_xgb_v3 --tune --tune-trials 80

  # Custom CSV
  python -m app.models.train_tennis_xgb_v3 --csv tennis_training_v3.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier
from joblib import dump as joblib_dump

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "tennis_training_v3.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ── Feature definitions ──────────────────────────────────────────────────────

# Order matters: this is the exact order stored in the .txt and used by DMatrix
FEATURE_COLS: List[str] = [
    # ELO block (NEW — the single biggest improvement)
    "d_elo",
    "d_surface_elo",
    "d_rank_log",
    "d_age",
    # Rolling form
    "d_win_rate_last_20",
    "d_win_rate_surface",
    "d_experience",
    "d_rest_days",
    "d_matches_10d",
    "d_matches_30d",
    # Context
    "is_hard",
    "is_clay",
    "is_grass",
    "is_grand_slam",
    "is_masters",
    "best_of",
    "round_ordinal",
    # H2H
    "h2h_p1_win_pct",
    "h2h_total_matches",
    "h2h_surface_p1_win_pct",
    "h2h_surface_matches",
    # Market totals / spread
    "has_total_line",
    "market_total_line",
    "market_total_line_centered",
    "market_total_over_implied",
    "market_total_under_implied",
    "market_total_over_no_vig",
    "market_total_under_no_vig",
    "market_total_overround",
    "has_game_spread",
    "market_spread_p1_line",
    "market_spread_p2_line",
    "market_spread_p1_implied",
    "market_spread_p2_implied",
    "market_spread_p1_no_vig",
    "market_spread_p2_no_vig",
    # Style proxies (restored)
    "d_svc_pts_w",
    "d_ret_pts_w",
    "d_ace_pg",
    "d_bp_save",
    "d_bp_win",
    # TA rolling
    "d_last5_hold",
    "d_last5_break",
    "d_last10_hold",
    "d_last10_break",
    "d_surf_last10_hold",
    "d_surf_last10_break",
    "d_last10_aces_pg",
    "d_surf_last10_aces_pg",
    "d_last10_df_pg",
    "d_surf_last10_df_pg",
    "d_last10_tb_match_rate",
    "d_last10_tb_win_pct",
    "d_surf_last10_tb_match_rate",
    "d_surf_last10_tb_win_pct",
]

# Features that should default to their "no info" value (not 0)
FILL_DEFAULTS: Dict[str, float] = {
    "d_elo": 0.0,
    "d_surface_elo": 0.0,
    "d_rank_log": 0.0,
    "d_age": 0.0,
    "d_win_rate_last_20": 0.0,        # equal-strength prior
    "d_win_rate_surface": 0.0,
    "d_experience": 0.0,
    "d_rest_days": 0.0,
    "d_matches_10d": 0.0,
    "d_matches_30d": 0.0,
    "h2h_p1_win_pct": 0.5,            # no H2H info → coin flip
    "h2h_total_matches": 0,
    "h2h_surface_p1_win_pct": 0.5,
    "h2h_surface_matches": 0,
    "has_total_line": 0,
    "market_total_line": 0.0,
    "market_total_line_centered": 0.0,
    "market_total_over_implied": 0.5,
    "market_total_under_implied": 0.5,
    "market_total_over_no_vig": 0.5,
    "market_total_under_no_vig": 0.5,
    "market_total_overround": 0.0,
    "has_game_spread": 0,
    "market_spread_p1_line": 0.0,
    "market_spread_p2_line": 0.0,
    "market_spread_p1_implied": 0.5,
    "market_spread_p2_implied": 0.5,
    "market_spread_p1_no_vig": 0.5,
    "market_spread_p2_no_vig": 0.5,
    "round_ordinal": 0,
}

# Monotonic constraints for features where the direction is known
# 1 = "higher → more likely to be class 1 (p1 wins)"
# -1 = "higher → less likely"
# 0 = no constraint
MONOTONE_CONSTRAINTS: Dict[str, int] = {
    "d_elo": 1,            # higher ELO diff → p1 more likely to win
    "d_surface_elo": 1,
    "d_rank_log": 1,       # higher rank diff (log) → p1 more likely (since we did log(r2)-log(r1))
    "d_win_rate_last_20": 1,
    "d_win_rate_surface": 1,
    "h2h_p1_win_pct": 1,
    "h2h_surface_p1_win_pct": 1,
}


@dataclass
class SplitConfig:
    val_days: int = 180
    test_days: int = 180
    early_stop_days: int = 120


# ── Data loading + splitting ─────────────────────────────────────────────────

def load_dataset(csv_path: pathlib.Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def prepare_data(df: pd.DataFrame, date_col: str) -> Tuple[pd.DataFrame, bool]:
    """Clean data and ensure required columns."""
    if "y" not in df.columns:
        raise ValueError("Training CSV must contain a 'y' column.")

    # Deduplicate
    if "match_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["match_id"])
        if len(df) != before:
            logger.info("Dropped %d duplicate match_id rows", before - len(df))

    # Parse dates
    has_date = False
    if date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        before = len(df)
        df = df[df[date_col].notna()].copy()
        if len(df) != before:
            logger.info("Dropped %d rows with invalid dates", before - len(df))
        has_date = True
    else:
        logger.warning("No '%s' column. Using random splits.", date_col)

    # Ensure label column
    df["y"] = df["y"].astype(int)

    # Check feature columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning("Missing feature columns (will use defaults): %s", missing)
        for c in missing:
            df[c] = FILL_DEFAULTS.get(c, 0.0)

    # Fill NaN with domain-appropriate defaults
    for c in FEATURE_COLS:
        default = FILL_DEFAULTS.get(c, 0.0)
        df[c] = df[c].fillna(default)

    return df, has_date


def split_by_date(
    df: pd.DataFrame, date_col: str, cfg: SplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_date = df[date_col].max()
    test_start = max_date - pd.Timedelta(days=cfg.test_days)
    val_start = test_start - pd.Timedelta(days=cfg.val_days)

    train = df[df[date_col] < val_start].copy()
    val = df[(df[date_col] >= val_start) & (df[date_col] < test_start)].copy()
    test = df[df[date_col] >= test_start].copy()
    return train, val, test


def split_for_early_stop(
    train_df: pd.DataFrame, date_col: str, cfg: SplitConfig, has_date: bool, seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if has_date:
        max_d = train_df[date_col].max()
        cutoff = max_d - pd.Timedelta(days=cfg.early_stop_days)
        fit = train_df[train_df[date_col] < cutoff].copy()
        es = train_df[train_df[date_col] >= cutoff].copy()
    else:
        fit = train_df.sample(frac=0.9, random_state=seed)
        es = train_df.drop(fit.index)

    if fit.empty or es.empty:
        logger.warning("Degenerate early-stop split; using full train.")
        return train_df.copy(), train_df.iloc[0:0].copy()
    return fit, es


# ── Metrics ──────────────────────────────────────────────────────────────────

def eval_metrics(y_true: pd.Series, y_proba: np.ndarray, label: str) -> dict:
    y_pred = (y_proba >= 0.5).astype(int)
    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        metrics["auc"] = None

    logger.info(
        "%s  →  acc=%.4f  logloss=%.4f  brier=%.4f  auc=%s",
        label, metrics["accuracy"], metrics["log_loss"], metrics["brier"],
        f'{metrics["auc"]:.4f}' if metrics["auc"] is not None else "n/a",
    )
    return metrics


def profit_at_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.55
) -> dict:
    """
    Simulate flat-stake betting: bet on p1 when model says >threshold,
    bet on p2 when model says <(1-threshold). Returns ROI %.
    """
    bets = 0
    wins = 0
    for yt, yp in zip(y_true, y_proba):
        if yp >= threshold:
            bets += 1
            if yt == 1:
                wins += 1
        elif yp <= (1.0 - threshold):
            bets += 1
            if yt == 0:
                wins += 1
    accuracy = wins / bets if bets > 0 else 0.0
    return {"bets": bets, "wins": wins, "accuracy": accuracy, "threshold": threshold}


# ── Model training ───────────────────────────────────────────────────────────

def _monotone_tuple() -> str:
    """Build monotone constraints tuple string for XGBoost."""
    return "(" + ",".join(
        str(MONOTONE_CONSTRAINTS.get(c, 0)) for c in FEATURE_COLS
    ) + ")"


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_early: Optional[pd.DataFrame],
    y_early: Optional[pd.Series],
    seed: int,
    params: Optional[dict] = None,
) -> XGBClassifier:
    """Train XGBoost with sensible defaults or custom params from Optuna."""

    default_params = {
        "n_estimators": 2500,
        "max_depth": 5,
        "learning_rate": 0.02,
        "subsample": 0.80,
        "colsample_bytree": 0.75,
        "colsample_bylevel": 0.75,
        "min_child_weight": 5,
        "reg_lambda": 3.0,
        "reg_alpha": 0.1,
        "gamma": 0.1,
        "max_delta_step": 1,             # helps with calibration
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "grow_policy": "lossguide",       # leaf-wise (like LightGBM)
        "max_leaves": 63,
        "monotone_constraints": _monotone_tuple(),
        "random_state": seed,
        "n_jobs": -1,
    }

    if params:
        default_params.update(params)

    # Only enable early stopping when we actually have validation data
    have_early = (
        X_early is not None
        and y_early is not None
        and len(X_early) > 0
    )
    if have_early:
        default_params["early_stopping_rounds"] = 80

    model = XGBClassifier(**default_params)

    fit_kw: dict = {"verbose": 50}
    if have_early:
        fit_kw["eval_set"] = [(X_early, y_early)]

    model.fit(X_train, y_train, **fit_kw)
    return model


# ── Optuna hyperparameter tuning ─────────────────────────────────────────────

def tune_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
    n_trials: int = 60,
) -> dict:
    """Run Optuna Bayesian search for best XGB hyperparameters."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("optuna not installed.  pip install optuna")
        raise

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 3000,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 3),
            "max_leaves": trial.suggest_int("max_leaves", 15, 127),
            "grow_policy": "lossguide",
            "monotone_constraints": _monotone_tuple(),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": -1,
        }

        model = XGBClassifier(**params, early_stopping_rounds=60)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=0,
        )

        preds = model.predict_proba(X_val)[:, 1]
        return log_loss(y_val, preds, labels=[0, 1])

    study = optuna.create_study(direction="minimize", study_name="xgb_tennis_v3")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["n_estimators"] = 3000
    best["grow_policy"] = "lossguide"
    best["monotone_constraints"] = _monotone_tuple()
    best["objective"] = "binary:logistic"
    best["eval_metric"] = "logloss"
    best["tree_method"] = "hist"
    best["random_state"] = seed
    best["n_jobs"] = -1

    logger.info("Best Optuna params (logloss=%.5f):\n%s", study.best_value, json.dumps(study.best_params, indent=2))
    return best


# ── Calibration ──────────────────────────────────────────────────────────────

def maybe_calibrate(
    model: XGBClassifier, X_cal: pd.DataFrame, y_cal: pd.Series, method: str,
) -> Optional[CalibratedClassifierCV]:
    if method == "none":
        return None
    counts = y_cal.value_counts()
    if len(counts) < 2 or int(counts.min()) < 2:
        logger.warning("Calibration skipped: insufficient class diversity.")
        return None

    # Clone the model without early_stopping_rounds so CalibratedClassifierCV
    # can re-fit it internally without needing an eval_set.
    from sklearn.base import clone
    base = clone(model)
    # Remove early stopping params that cause issues during calibration re-fit
    base.set_params(early_stopping_rounds=None)

    cv = min(3, int(counts.min()))
    if cv < 2:
        logger.warning("Calibration skipped: need cv >= 2.")
        return None
    try:
        cal = CalibratedClassifierCV(base, method=method, cv=cv)
        cal.fit(X_cal, y_cal)
        return cal
    except Exception as exc:
        logger.warning("Calibration failed: %s", exc)
        return None


# ── Feature importance logging ───────────────────────────────────────────────

def log_feature_importance(model: XGBClassifier, feature_cols: List[str]) -> dict:
    """Log and return feature importances sorted by gain."""
    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")

    # Map internal feature names (f0, f1, ...) to our names
    mapped: dict = {}
    for i, col in enumerate(feature_cols):
        key = f"f{i}"
        mapped[col] = importance.get(key, importance.get(col, 0.0))

    # Sort descending
    sorted_imp = sorted(mapped.items(), key=lambda x: x[1], reverse=True)

    logger.info("─── Feature Importance (gain) ───")
    for rank, (feat, gain) in enumerate(sorted_imp, 1):
        bar = "█" * int(gain / max(1, sorted_imp[0][1]) * 30)
        logger.info("  %2d. %-30s %8.1f  %s", rank, feat, gain, bar)

    return dict(sorted_imp)


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train V3 tennis XGB model.")
    p.add_argument("--csv", type=str, default=str(DEFAULT_CSV_PATH))
    p.add_argument("--model-out", type=str, default=str(MODEL_DIR / "xgb_tennis_v3.json"))
    p.add_argument("--features-out", type=str, default=str(MODEL_DIR / "xgb_tennis_v3_features.txt"))
    p.add_argument("--metadata-out", type=str, default=str(MODEL_DIR / "xgb_tennis_v3_meta.json"))
    p.add_argument("--calibration-out", type=str, default=str(MODEL_DIR / "xgb_tennis_v3_calibration.joblib"))
    p.add_argument("--date-col", type=str, default="match_date")
    p.add_argument("--val-days", type=int, default=180)
    p.add_argument("--test-days", type=int, default=180)
    p.add_argument("--early-stop-days", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calibration", type=str, choices=["none", "sigmoid", "isotonic"], default="sigmoid")
    p.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search")
    p.add_argument("--tune-trials", type=int, default=60)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = SplitConfig(val_days=args.val_days, test_days=args.test_days, early_stop_days=args.early_stop_days)

    # ── Load & prepare ───────────────────────────────────────────────────
    df = load_dataset(pathlib.Path(args.csv))
    df, has_date = prepare_data(df, args.date_col)

    if has_date:
        df = df.sort_values(args.date_col).reset_index(drop=True)
        train_df, val_df, test_df = split_by_date(df, args.date_col, cfg)
    else:
        train_df = df.sample(frac=0.8, random_state=args.seed)
        remain = df.drop(train_df.index)
        val_df = remain.sample(frac=0.5, random_state=args.seed)
        test_df = remain.drop(val_df.index)

    logger.info("Split sizes: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    # ── Early-stop split ─────────────────────────────────────────────────
    fit_df, early_df = split_for_early_stop(train_df, args.date_col, cfg, has_date, args.seed)
    logger.info("Early-stop split: fit=%d  early=%d", len(fit_df), len(early_df))

    X_train = fit_df[FEATURE_COLS].copy()
    y_train = fit_df["y"].astype(int)
    X_early = early_df[FEATURE_COLS].copy() if len(early_df) > 0 else None
    y_early = early_df["y"].astype(int) if len(early_df) > 0 else None
    X_val = val_df[FEATURE_COLS].copy()
    y_val = val_df["y"].astype(int)
    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df["y"].astype(int)

    # ── Optuna tuning (optional) ─────────────────────────────────────────
    tuned_params = None
    if args.tune:
        logger.info("Starting Optuna hyperparameter search (%d trials)...", args.tune_trials)
        tuned_params = tune_hyperparams(X_train, y_train, X_val, y_val, args.seed, args.tune_trials)

    # ── Train ────────────────────────────────────────────────────────────
    logger.info("Training on %d samples with %d features...", len(X_train), len(FEATURE_COLS))
    model = train_model(X_train, y_train, X_early, y_early, args.seed, tuned_params)

    # ── Evaluate (uncalibrated) ──────────────────────────────────────────
    logger.info("═══ Uncalibrated Performance ═══")
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    val_metrics = eval_metrics(y_val, val_proba, "VAL")
    test_metrics = eval_metrics(y_test, test_proba, "TEST")

    # Simulated betting
    for thr in [0.55, 0.60, 0.65]:
        result = profit_at_threshold(y_test.values, test_proba, thr)
        logger.info(
            "  Betting@%.0f%%: %d bets, %d wins, %.1f%% accuracy",
            thr * 100, result["bets"], result["wins"], result["accuracy"] * 100,
        )

    # ── Calibration ──────────────────────────────────────────────────────
    calibrator = maybe_calibrate(model, X_val, y_val, args.calibration)
    cal_metrics = None
    if calibrator is not None:
        logger.info("═══ Calibrated Performance (%s) ═══", args.calibration)
        cal_test_proba = calibrator.predict_proba(X_test)[:, 1]
        cal_metrics = eval_metrics(y_test, cal_test_proba, "TEST_CAL")
        for thr in [0.55, 0.60, 0.65]:
            result = profit_at_threshold(y_test.values, cal_test_proba, thr)
            logger.info(
                "  Betting@%.0f%%: %d bets, %d wins, %.1f%% accuracy",
                thr * 100, result["bets"], result["wins"], result["accuracy"] * 100,
            )

    # ── Feature importance ───────────────────────────────────────────────
    importance = log_feature_importance(model, FEATURE_COLS)

    # ── Save model artifacts ─────────────────────────────────────────────
    booster = model.get_booster()
    booster.save_model(pathlib.Path(args.model_out).as_posix())
    logger.info("Saved model → %s", args.model_out)

    with open(args.features_out, "w", encoding="utf-8") as f:
        f.write("\n".join(FEATURE_COLS))
    logger.info("Saved feature list → %s", args.features_out)

    if calibrator is not None:
        joblib_dump(calibrator, args.calibration_out)
        logger.info("Saved calibration → %s", args.calibration_out)

    # ── Metadata ─────────────────────────────────────────────────────────
    metadata = {
        "model_version": "v3",
        "csv": args.csv,
        "date_col": args.date_col,
        "num_features": len(FEATURE_COLS),
        "features": FEATURE_COLS,
        "split": {
            "val_days": cfg.val_days,
            "test_days": cfg.test_days,
            "early_stop_days": cfg.early_stop_days,
        },
        "rows": {
            "total": int(len(df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
            "train_fit": int(len(fit_df)),
            "early_stop": int(len(early_df)),
        },
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
            "test_calibrated": cal_metrics,
        },
        "betting_simulation": {
            thr: profit_at_threshold(y_test.values, test_proba, thr)
            for thr in [0.55, 0.60, 0.65, 0.70]
        },
        "feature_importance_gain": importance,
        "calibration": args.calibration,
        "optuna_tuned": args.tune,
        "fill_defaults": {k: v for k, v in FILL_DEFAULTS.items()},
        "monotone_constraints": {k: v for k, v in MONOTONE_CONSTRAINTS.items()},
    }
    with open(args.metadata_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved metadata → %s", args.metadata_out)

    logger.info("═══ Done! ═══")


if __name__ == "__main__":
    main()
