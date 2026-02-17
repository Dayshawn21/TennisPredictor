# app/models/odds.py
from __future__ import annotations

from typing import Optional, Tuple
import math


def prob_to_american(p: Optional[float]) -> Optional[int]:
    if p is None or p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        return int(-round((p / (1 - p)) * 100))
    return int(round(((1 - p) / p) * 100))


def american_to_implied_prob(odds: int) -> float:
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)


def no_vig_two_way(p1: Optional[float], p2: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if p1 is None or p2 is None:
        return None, None
    s = p1 + p2
    if s <= 0:
        return None, None
    return p1 / s, p2 / s


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def blend_probs_logit(model_p: Optional[float], market_p: Optional[float], w_model: float = 0.65) -> Optional[float]:
    if model_p is None and market_p is None:
        return None
    if model_p is None:
        return market_p
    if market_p is None:
        return model_p
    w = min(max(w_model, 0.0), 1.0)
    return _sigmoid(w * _logit(model_p) + (1 - w) * _logit(market_p))
