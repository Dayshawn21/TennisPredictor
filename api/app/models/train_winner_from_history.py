from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from typing import List, Tuple

import pandas as pd
from sqlalchemy import text

from app.db_session import engine

# If you don't have xgboost installed, install it in your venv:
# pip install xgboost
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score


@dataclass
class TrainConfig:
    min_rows: int = int(os.getenv("MIN_ROWS", "200"))  # require enough finished matches
    test_frac: float = float(os.getenv("TEST_FRAC", "0.2"))
    model_out: str = os.getenv("MODEL_OUT", "api/app/models/tennis/xgb_winner_history.json")
    cols_out: str = os.getenv("COLS_OUT", "api/app/models/tennis/winner_feature_cols.json")


SQL = """
SELECT
  match_id,
  match_date,
  tour,
  surface,
  d_elo_overall,
  d_elo_surface,
  p1_win
FROM predictions_history
WHERE p1_win IS NOT NULL
  AND d_elo_overall IS NOT NULL
ORDER BY match_date ASC, match_id ASC
"""


def norm_surface(surface: str | None) -> str:
    s = (surface or "").lower()
    if "hard" in s:
        return "hard"
    if "clay" in s:
        return "clay"
    if "grass" in s:
        return "grass"
    return "other"


def time_split(df: pd.DataFrame, test_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split: last X% of rows are test."""
    n = len(df)
    cut = max(1, int(n * (1.0 - test_frac)))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def make_features(df: pd.DataFrame, feature_cols: List[str] | None = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build training features:
      - numeric: d_elo_overall, d_elo_surface (fallback to 0 if missing)
      - categorical: tour, surface_norm (one-hot)
    """
    work = df.copy()
    work["surface_norm"] = work["surface"].apply(norm_surface)

    # numeric
    work["d_elo_surface"] = work["d_elo_surface"].fillna(0.0)

    # one-hot
    X = pd.get_dummies(
        work[["d_elo_overall", "d_elo_surface", "tour", "surface_norm"]],
        columns=["tour", "surface_norm"],
        dummy_na=False,
    )

    # lock columns for consistent inference later
    if feature_cols is not None:
        for c in feature_cols:
            if c not in X.columns:
                X[c] = 0
        X = X[feature_cols]
        return X, feature_cols

    cols = list(X.columns)
    return X, cols


def main() -> None:
    cfg = TrainConfig()

    with engine.begin() as conn:
        df = pd.read_sql(text(SQL), conn)

    if df.empty:
        raise RuntimeError("No finished rows found in predictions_history (p1_win is NULL for all rows).")

    if len(df) < cfg.min_rows:
        raise RuntimeError(
            f"Not enough training rows yet: {len(df)}. "
            f"Need at least {cfg.min_rows}. Keep persisting daily + backfilling winners."
        )

    train_df, test_df = time_split(df, cfg.test_frac)
    if test_df.empty:
        raise RuntimeError("Test split is empty. Reduce TEST_FRAC or add more rows.")

    X_train, feat_cols = make_features(train_df)
    y_train = train_df["p1_win"].astype(int).values

    X_test, _ = make_features(test_df, feature_cols=feat_cols)
    y_test = test_df["p1_win"].astype(int).values

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )

    model.fit(X_train, y_train)

    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)

    auc = roc_auc_score(y_test, p)
    ll = log_loss(y_test, p)
    acc = accuracy_score(y_test, pred)

    print("=== Winner model trained from predictions_history ===")
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"Test AUC:   {auc:.4f}")
    print(f"Test LL:    {ll:.4f}")
    print(f"Test Acc:   {acc:.4f}")
    print(f"Feature cols: {len(feat_cols)}")

    # Ensure output folder exists
    os.makedirs(os.path.dirname(cfg.model_out), exist_ok=True)

    # Save model (xgboost json)
    model.get_booster().save_model(cfg.model_out)

    # Save columns used
    with open(cfg.cols_out, "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, indent=2)

    print(f"Saved model -> {cfg.model_out}")
    print(f"Saved cols  -> {cfg.cols_out}")


if __name__ == "__main__":
    main()
