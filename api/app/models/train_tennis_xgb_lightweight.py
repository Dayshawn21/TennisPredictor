# FILE 3: api/app/models/train_tennis_xgb_lightweight.py
"""
Train lightweight XGBoost model using ONLY features available in 2026:
- ELO differentials
- Surface indicators
- Tournament tier
- Best-of format

This model can be used when detailed match stats are unavailable.
"""

from __future__ import annotations
import pathlib
import logging
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "tennis_training_lightweight.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgb_tennis_lightweight.json"

# Features we'll use
FEATURE_COLS = [
    "d_elo",
    "d_surface_elo",
    "d_elo_rank",
    "d_age",
    "is_hard",
    "is_clay",
    "is_grass",
    "is_grand_slam",
    "is_masters",
    "best_of",
    "d_rest_days",
    "d_matches_10d",
]


def load_dataset(csv_path: pathlib.Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer differential features.
    """
    # ELO differences (already calculated in extraction)
    # Just ensure they exist
    if "d_elo" not in df.columns and "p1_elo" in df.columns:
        df["d_elo"] = df["p1_elo"] - df["p2_elo"]
    
    if "d_surface_elo" not in df.columns and "p1_surface_elo" in df.columns:
        df["d_surface_elo"] = df["p1_surface_elo"] - df["p2_surface_elo"]
    
    # Fill missing values
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    
    return df


def train_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    
    logger.info("Training XGBoost on %d rows, validating on %d rows",
                len(X_train), len(X_val))
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    # Evaluation
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_val, y_val_proba)
    acc = accuracy_score(y_val, y_val_pred)
    ll = log_loss(y_val, y_val_proba)
    brier = brier_score_loss(y_val, y_val_proba)
    
    logger.info("=" * 60)
    logger.info("VALIDATION METRICS:")
    logger.info("  AUC:         %.4f", auc)
    logger.info("  Accuracy:    %.4f", acc)
    logger.info("  Log Loss:    %.4f", ll)
    logger.info("  Brier Score: %.4f", brier)
    logger.info("=" * 60)
    
    # Cross-validation score
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring="roc_auc", n_jobs=-1
    )
    logger.info("5-Fold CV AUC: %.4f (±%.4f)", cv_scores.mean(), cv_scores.std())
    
    # Feature importances
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    logger.info("\nFeature Importances:")
    for idx in order:
        if idx < len(FEATURE_COLS):
            logger.info("  %-20s %.4f", FEATURE_COLS[idx], importances[idx])
    
    return model


def main():
    df = load_dataset(CSV_PATH)
    df = engineer_features(df)
    
    # Filter to rows with required features
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    logger.info("Using %d features: %s", len(available_cols), available_cols)
    
    X = df[available_cols].copy()
    y = df["y"].astype(int)
    
    # Remove rows with too many missing values
    X = X.fillna(0.0)
    
    logger.info("Training on %d samples", len(X))
    
    model = train_model(X, y)
    
    # Save model
    booster = model.get_booster()
    booster.save_model(MODEL_PATH.as_posix())
    logger.info("Saved model to %s", MODEL_PATH)
    
    # Save feature list
    features_path = MODEL_DIR / "xgb_tennis_lightweight_features.txt"
    with open(features_path, "w") as f:
        f.write("\n".join(available_cols))
    logger.info("Saved feature list to %s", features_path)


if __name__ == "__main__":
    main()