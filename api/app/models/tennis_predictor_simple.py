# FILE: api/app/models/tennis_predictor_simple.py
"""
Simple tennis predictor using rolling features (no ELO required).
"""

from __future__ import annotations
import pathlib
from functools import lru_cache
import numpy as np
from xgboost import Booster, DMatrix
from typing import Dict

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_tennis_simple.json"

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


class SimpleTennisPredictor:
    def __init__(self, booster: Booster) -> None:
        self._booster = booster

    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        Predict P1 win probability.
        
        Args:
            features: Dict with keys matching FEATURE_COLS
        
        Returns:
            P1 win probability (0-1)
        """
        row = [float(features.get(col, 0.0)) for col in FEATURE_COLS]
        dmat = DMatrix(np.array([row], dtype=float), feature_names=FEATURE_COLS)
        proba = float(self._booster.predict(dmat)[0])
        return proba


@lru_cache(maxsize=1)
def get_simple_predictor() -> SimpleTennisPredictor:
    """Load the trained simple predictor model."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Simple model not found at {MODEL_PATH}")
    
    booster = Booster()
    booster.load_model(MODEL_PATH.as_posix())
    return SimpleTennisPredictor(booster)