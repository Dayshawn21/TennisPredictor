# api/app/services/tennis/elo_predictor_enhanced.py
"""
Enhanced ELO predictor (surface + best-of aware), with stable compatibility exports.
"""

from __future__ import annotations

from typing import Dict, Any
import math


def _elo_win_prob(p1_elo: float, p2_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((p2_elo - p1_elo) / 400.0))


def _best_of_adjust(p: float, best_of: int) -> float:
    if best_of <= 3:
        # Best-of-3: P(win >=2 of 3)
        return (p * p) * (3.0 - 2.0 * p)
    # Best-of-5: P(win >=3 of 5)
    q = 1.0 - p
    return (10.0 * (p**3) * (q**2)) + (5.0 * (p**4) * q) + (p**5)


def predict_match_enhanced(match_data: Dict[str, Any]) -> Dict[str, Any]:
    p1_elo = match_data.get("p1_elo")
    p2_elo = match_data.get("p2_elo")
    if p1_elo is None or p2_elo is None:
        return {
            "method": "elo_enhanced",
            "p1_win_prob": None,
            "p2_win_prob": None,
            "predicted_winner": None,
            "missing_reason": "missing_elo",
        }

    p1_elo = float(p1_elo)
    p2_elo = float(p2_elo)
    base = _elo_win_prob(p1_elo, p2_elo)

    surface = (match_data.get("surface") or "").lower()
    p1_helo = match_data.get("p1_helo")
    p2_helo = match_data.get("p2_helo")
    p1_celo = match_data.get("p1_celo")
    p2_celo = match_data.get("p2_celo")
    p1_gelo = match_data.get("p1_gelo")
    p2_gelo = match_data.get("p2_gelo")

    surf_adj = None
    if "hard" in surface and p1_helo is not None and p2_helo is not None:
        surf_adj = _elo_win_prob(float(p1_helo), float(p2_helo))
    elif "clay" in surface and p1_celo is not None and p2_celo is not None:
        surf_adj = _elo_win_prob(float(p1_celo), float(p2_celo))
    elif "grass" in surface and p1_gelo is not None and p2_gelo is not None:
        surf_adj = _elo_win_prob(float(p1_gelo), float(p2_gelo))

    p_match = (0.75 * base + 0.25 * surf_adj) if surf_adj is not None else base
    best_of = int(match_data.get("best_of") or 3)
    p_match = _best_of_adjust(p_match, best_of)

    p1_prob = max(0.001, min(0.999, float(p_match)))
    p2_prob = 1.0 - p1_prob
    winner = "p1" if p1_prob >= 0.5 else "p2"
    return {
        "method": "elo_enhanced",
        "p1_win_prob": p1_prob,
        "p2_win_prob": p2_prob,
        "predicted_winner": winner,
        "missing_reason": None,
    }


def predict_match_elo(match_data: Dict[str, Any]) -> Dict[str, Any]:
    return predict_match_enhanced(match_data)


def predict_match(match_data: Dict[str, Any]) -> Dict[str, Any]:
    return predict_match_enhanced(match_data)
