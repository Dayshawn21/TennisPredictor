from __future__ import annotations

import os
import pathlib
import logging

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Where the CSV from tennis_training_dataset_ta.py lives
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]  # points to /api
CSV_PATH = PROJECT_ROOT / "tennis_training_ta.csv"

# Where to save the trained model
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgb_tennis_ta.json"


# These are the "diff" features we really care about for prediction
FEATURE_COLS = [
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


def load_base_dataset(csv_path: pathlib.Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def build_symmetric_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Our CSV was built from the perspective where p1 is the actual winner.
    That means y is effectively always 1 in the base data.

    To train a proper model, we:
      - Row A: keep as-is (p1 is winner) => label 1
      - Row B: mirror perspective (swap players, flip diff features) => label 0

    This makes the problem symmetric and gives us both positive and negative examples.
    """
    # Base (winner perspective)
    base = df.copy()
    base["y"] = 1  # p1 wins

    # Flipped (loser perspective)
    flipped = df.copy()

    # Flip sign of all diff features
    for col in FEATURE_COLS:
        flipped[col] = -flipped[col]

    flipped["y"] = 0  # p1 (in flipped view) is actually the loser

    # Combine
    full = pd.concat([base, flipped], ignore_index=True)

    # Keep only rows where we have at least some features (NaNs are okay, XGB can handle them)
    X = full[FEATURE_COLS]
    y = full["y"].astype(int)

    logger.info("Built symmetric dataset with %d rows (%d original * 2)", len(full), len(df))
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # XGBoost params tuned to be sane defaults
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",   # fast on CPU
        random_state=42,
        n_jobs=4,
    )

    logger.info("Training XGBoost model on %d rows, validating on %d rows", len(X_train), len(X_val))
    model.fit(X_train, y_train)

    # Evaluation
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_val_proba)
    acc = accuracy_score(y_val, y_val_pred)
    ll = log_loss(y_val, y_val_proba)

    logger.info("Validation AUC:  %.4f", auc)
    logger.info("Validation ACC:  %.4f", acc)
    logger.info("Validation LogLoss: %.4f", ll)

    # Top feature importances
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    logger.info("Top feature importances:")
    for idx in order:
        logger.info("  %-30s %.4f", FEATURE_COLS[idx], importances[idx])

    return model


def main():
    df = load_base_dataset(CSV_PATH)
    X, y = build_symmetric_dataset(df)

    model = train_model(X, y)

    # Save the underlying Booster instead of the sklearn wrapper
    booster = model.get_booster()
    booster.save_model(MODEL_PATH.as_posix())
    logger.info("Saved booster model to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
