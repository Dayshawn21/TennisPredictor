# FILE: api/app/models/train_tennis_xgb_simple_v2.py
"""
Train a v2 XGBoost model using simple rolling features with leakage-safe splits.

Key upgrades vs v1:
- Time-based splits (train/val/test) using match_date
- No symmetric leakage across splits
- Early stopping on a holdout derived from the training window
- Optional probability calibration saved separately
"""

from __future__ import annotations
import argparse
import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump as joblib_dump

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "tennis_training_simple.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "d_win_rate_last_20",
    "d_win_rate_surface",
    "d_experience",
    "d_rest_days",
    "d_matches_10d",
    "d_matches_30d",
    "is_hard",
    "is_clay",
    "is_grass",
    "is_grand_slam",
    "is_masters",
    "best_of",
    "d_svc_pts_w",
    "d_ret_pts_w",
    "d_ace_pg",
    "d_bp_save",
    "d_bp_win",
]


@dataclass
class SplitConfig:
    val_days: int = 180
    test_days: int = 180
    early_stop_days: int = 120


def load_dataset(csv_path: pathlib.Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def ensure_date_column(df: pd.DataFrame, date_col: str) -> Tuple[pd.DataFrame, bool]:
    if date_col not in df.columns:
        logger.warning("No %s column found. Falling back to random splits.", date_col)
        return df, False
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    before = len(df)
    df = df[df[date_col].notna()].copy()
    if len(df) != before:
        logger.info("Dropped %d rows with invalid %s", before - len(df), date_col)
    return df, True


def create_symmetric_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    base = df.copy()
    base["y"] = base["y"].astype(int)

    flipped = df.copy()
    for col in ["d_win_rate_last_20", "d_win_rate_surface", "d_experience"]:
        if col in flipped.columns:
            flipped[col] = -flipped[col]
    flipped["y"] = 1 - base["y"]

    combined = pd.concat([base, flipped], ignore_index=True)
    X = combined[FEATURE_COLS].copy()
    y = combined["y"].astype(int)
    return X, y


def split_by_date(
    df: pd.DataFrame,
    date_col: str,
    cfg: SplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_date = df[date_col].max()
    test_start = max_date - pd.Timedelta(days=cfg.test_days)
    val_start = test_start - pd.Timedelta(days=cfg.val_days)

    train = df[df[date_col] < val_start].copy()
    val = df[(df[date_col] >= val_start) & (df[date_col] < test_start)].copy()
    test = df[df[date_col] >= test_start].copy()

    return train, val, test


def split_for_early_stop(
    train_df: pd.DataFrame,
    date_col: str,
    cfg: SplitConfig,
    has_date: bool,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if has_date:
        max_date = train_df[date_col].max()
        early_start = max_date - pd.Timedelta(days=cfg.early_stop_days)
        fit_df = train_df[train_df[date_col] < early_start].copy()
        early_df = train_df[train_df[date_col] >= early_start].copy()
    else:
        fit_df = train_df.sample(frac=0.9, random_state=seed)
        early_df = train_df.drop(fit_df.index)

    if fit_df.empty or early_df.empty:
        logger.warning("Early-stop split degenerate; using full train for fit and empty early set.")
        fit_df = train_df.copy()
        early_df = train_df.iloc[0:0].copy()

    return fit_df, early_df


def eval_metrics(y_true: pd.Series, y_proba: np.ndarray, label: str) -> dict:
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_proba),
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, y_proba)
    except ValueError:
        metrics["auc"] = None

    logger.info("%s metrics:", label)
    logger.info("  accuracy: %.4f", metrics["accuracy"])
    logger.info("  log_loss: %.4f", metrics["log_loss"])
    logger.info("  brier:    %.4f", metrics["brier"])
    if metrics["auc"] is not None:
        logger.info("  auc:      %.4f", metrics["auc"])
    else:
        logger.info("  auc:      n/a (single class)")
    return metrics


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_early: Optional[pd.DataFrame],
    y_early: Optional[pd.Series],
    seed: int,
) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=1500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )

    eval_set = None
    if X_early is not None and y_early is not None and len(X_early) > 0:
        eval_set = [(X_early, y_early)]

    try:
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=50 if eval_set else None,
            verbose=50,
        )
    except TypeError:
        try:
            callbacks = None
            if eval_set:
                from xgboost.callback import EarlyStopping
                callbacks = [EarlyStopping(rounds=50, save_best=True)]
            model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                callbacks=callbacks,
                verbose=50,
            )
        except TypeError:
            # Old XGBoost: no early_stopping_rounds or callbacks
            model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=50,
            )
    return model


def maybe_calibrate(
    model: XGBClassifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    method: str,
) -> Optional[CalibratedClassifierCV]:
    if method == "none":
        return None
    counts = y_cal.value_counts()
    if len(counts) < 2:
        logger.warning("Calibration skipped: only one class present in calibration set.")
        return None
    min_class = int(counts.min())
    if min_class < 2:
        logger.warning("Calibration skipped: not enough samples per class (min=%d).", min_class)
        return None
    try:
        calibrator = CalibratedClassifierCV(model, method=method, cv="prefit")
        calibrator.fit(X_cal, y_cal)
        return calibrator
    except Exception:
        # Older sklearn: no "prefit" support; fall back to internal CV calibration
        cv = min(3, min_class)
        if cv < 2:
            logger.warning("Calibration skipped: cv=%d too small.", cv)
            return None
        calibrator = CalibratedClassifierCV(model, method=method, cv=cv)
        calibrator.fit(X_cal, y_cal)
        return calibrator


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train v2 simple tennis XGB model.")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--model-out", type=str, default=str(MODEL_DIR / "xgb_tennis_simple_v2.json"))
    parser.add_argument("--features-out", type=str, default=str(MODEL_DIR / "xgb_tennis_simple_v2_features.txt"))
    parser.add_argument("--metadata-out", type=str, default=str(MODEL_DIR / "xgb_tennis_simple_v2_meta.json"))
    parser.add_argument("--calibration-out", type=str, default=str(MODEL_DIR / "xgb_tennis_simple_v2_calibration.joblib"))
    parser.add_argument("--date-col", type=str, default="match_date")
    parser.add_argument("--val-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=180)
    parser.add_argument("--early-stop-days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration", type=str, choices=["none", "sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--no-symmetric", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = SplitConfig(val_days=args.val_days, test_days=args.test_days, early_stop_days=args.early_stop_days)

    df = load_dataset(pathlib.Path(args.csv))
    if "y" not in df.columns:
        raise ValueError("Training CSV must contain a y column.")

    if "match_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["match_id"])
        if len(df) != before:
            logger.info("Dropped %d duplicate match_id rows", before - len(df))

    df, has_date = ensure_date_column(df, args.date_col)

    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column: {col}")

    df = df.copy()
    df["y"] = df["y"].astype(int)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    if has_date:
        df = df.sort_values(args.date_col).reset_index(drop=True)
        train_df, val_df, test_df = split_by_date(df, args.date_col, cfg)
    else:
        train_df = df.sample(frac=0.8, random_state=args.seed)
        remain_df = df.drop(train_df.index)
        val_df = remain_df.sample(frac=0.5, random_state=args.seed)
        test_df = remain_df.drop(val_df.index)

    logger.info("Split sizes: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    fit_df, early_df = split_for_early_stop(train_df, args.date_col, cfg, has_date, args.seed)
    logger.info("Early-stop split: fit=%d early=%d", len(fit_df), len(early_df))

    if args.no_symmetric:
        X_train = fit_df[FEATURE_COLS].copy()
        y_train = fit_df["y"].astype(int)
    else:
        X_train, y_train = create_symmetric_dataset(fit_df)

    X_train = X_train.fillna(0.0)

    if len(early_df) > 0:
        X_early = early_df[FEATURE_COLS].copy().fillna(0.0)
        y_early = early_df["y"].astype(int)
    else:
        X_early = None
        y_early = None

    logger.info("Training samples: %d", len(X_train))
    model = train_model(X_train, y_train, X_early, y_early, seed=args.seed)

    X_val = val_df[FEATURE_COLS].copy().fillna(0.0)
    y_val = val_df["y"].astype(int)
    X_test = test_df[FEATURE_COLS].copy().fillna(0.0)
    y_test = test_df["y"].astype(int)

    logger.info("Uncalibrated performance:")
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    val_metrics = eval_metrics(y_val, val_proba, "val")
    test_metrics = eval_metrics(y_test, test_proba, "test")

    calibrator = maybe_calibrate(model, X_val, y_val, args.calibration)
    cal_metrics = None
    if calibrator is not None:
        logger.info("Calibrated performance (%s):", args.calibration)
        cal_test_proba = calibrator.predict_proba(X_test)[:, 1]
        cal_metrics = eval_metrics(y_test, cal_test_proba, "test_calibrated")

    booster = model.get_booster()
    booster.save_model(pathlib.Path(args.model_out).as_posix())
    logger.info("Saved model to %s", args.model_out)

    with open(args.features_out, "w", encoding="utf-8") as f:
        f.write("\n".join(FEATURE_COLS))
    logger.info("Saved feature list to %s", args.features_out)

    if calibrator is not None:
        joblib_dump(calibrator, args.calibration_out)
        logger.info("Saved calibration model to %s", args.calibration_out)

    metadata = {
        "csv": args.csv,
        "date_col": args.date_col,
        "split": {
            "val_days": cfg.val_days,
            "test_days": cfg.test_days,
            "early_stop_days": cfg.early_stop_days,
        },
        "rows": {
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
        "features": FEATURE_COLS,
        "symmetric": not args.no_symmetric,
        "calibration": args.calibration,
    }
    with open(args.metadata_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", args.metadata_out)


if __name__ == "__main__":
    main()
