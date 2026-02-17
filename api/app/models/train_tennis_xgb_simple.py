# FILE: api/app/models/train_tennis_xgb_simple.py
"""
Train lightweight XGBoost model using simple rolling features.
No ELO required - uses win rates and form instead.
"""

from __future__ import annotations
import pathlib
import logging
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "tennis_training_simple.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgb_tennis_simple.json"

# Features we'll use
FEATURE_COLS = [
    "d_win_rate_last_20",
    "d_win_rate_surface",
    "d_experience",
    "is_hard",
    "is_clay",
    "is_grass",
    "is_grand_slam",
    "is_masters",
    "best_of",
]


def load_dataset(csv_path: pathlib.Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def create_symmetric_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create symmetric dataset by flipping each match.
    Row A: P1 vs P2 with P1 features => label = 1 if P1 won
    Row B: P2 vs P1 with flipped features => label = 1 if P2 won (flipped from A)
    
    This balances the dataset and makes the model perspective-invariant.
    """
    logger.info("Creating symmetric dataset...")
    
    # Original perspective
    base = df.copy()
    base['y'] = df['y'].astype(int)
    
    # Flipped perspective
    flipped = df.copy()
    
    # Flip all differential features
    for col in ['d_win_rate_last_20', 'd_win_rate_surface', 'd_experience']:
        if col in flipped.columns:
            flipped[col] = -flipped[col]
    
    # Flip label
    flipped['y'] = 1 - df['y'].astype(int)
    
    # Combine
    combined = pd.concat([base, flipped], ignore_index=True)
    
    logger.info("Created %d training examples from %d matches", len(combined), len(df))
    
    X = combined[FEATURE_COLS].copy()
    y = combined['y'].astype(int)
    
    logger.info("Label distribution: %s", y.value_counts().to_dict())
    
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    
    model = XGBClassifier(
        n_estimators=200,
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
    
    # Feature importances
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    logger.info("\nFeature Importances:")
    for idx in order:
        if idx < len(FEATURE_COLS):
            logger.info("  %-25s %.4f", FEATURE_COLS[idx], importances[idx])
    
    return model


def main():
    df = load_dataset(CSV_PATH)
    
    # Create symmetric dataset
    X, y = create_symmetric_dataset(df)
    
    # Train model
    model = train_model(X, y)
    
    # Save model
    booster = model.get_booster()
    booster.save_model(MODEL_PATH.as_posix())
    logger.info("✅ Saved model to %s", MODEL_PATH)
    
    # Save feature list
    features_path = MODEL_DIR / "xgb_tennis_simple_features.txt"
    with open(features_path, "w") as f:
        f.write("\n".join(FEATURE_COLS))
    logger.info("✅ Saved feature list to %s", features_path)


if __name__ == "__main__":
    main()