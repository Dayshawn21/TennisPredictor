from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional
import pathlib
from functools import lru_cache

import numpy as np
from xgboost import Booster, DMatrix
from pydantic import BaseModel, Field


class EloPrediction(BaseModel):
    # Identifiers / metadata
    match_id: Optional[Any] = None
    match_key: Optional[str] = None
    match_date: Optional[date] = None
    tour: Optional[str] = None
    tournament: Optional[str] = None
    round: Optional[str] = None
    surface: Optional[str] = None

    p1_name: Optional[str] = None
    p2_name: Optional[str] = None
    p1_ta_id: Optional[int] = None
    p2_ta_id: Optional[int] = None

    # Winner probs / odds
    p1_win_prob: Optional[float] = None
    p2_win_prob: Optional[float] = None
    p1_fair_american: Optional[int] = None
    p2_fair_american: Optional[int] = None

    # Missing data info
    missing_reason: Optional[str] = None
    predicted_winner: Optional[str] = None
    projected_total_games: Optional[float] = None
    projected_spread_p1: Optional[float] = None
    projected_sets: Optional[str] = None
    best_of: Optional[int] = None
    # Debug inputs
    inputs: Dict[str, Any] = Field(default_factory=dict)

   

class EloPredictionsResponse(BaseModel):
    as_of: date
    source: str
    cached: bool
    count: int
    items: List[EloPrediction]


# Reuse the same PROJECT_ROOT + model path as in train_tennis_xgb.py
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]  # points to /api
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_tennis_ta.json"

# These must match the training script
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


class TennisXgbPredictor:
    def __init__(self, booster: Booster) -> None:
        self._booster = booster

    def predict_proba(self, features: Dict[str, float]) -> float:
        # Build feature row in the correct column order, defaulting missing to 0.0
        row = [float(features.get(col, 0.0)) for col in FEATURE_COLS]
        dmat = DMatrix(np.array([row], dtype=float), feature_names=FEATURE_COLS)
        proba = float(self._booster.predict(dmat)[0])
        return proba


@lru_cache(maxsize=1)
def get_default_tennis_predictor() -> TennisXgbPredictor:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"XGBoost tennis model not found at {MODEL_PATH}")

    booster = Booster()
    booster.load_model(MODEL_PATH.as_posix())
    return TennisXgbPredictor(booster)
