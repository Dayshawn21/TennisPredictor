from __future__ import annotations

import math
import os
from typing import Optional

from app.models.ely import elo_prob, pick_surface_elo
from app.models.odds import prob_to_american


# -----------------------------
# Helpers
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _round_american(odds: Optional[int], step: int) -> Optional[int]:
    if odds is None:
        return None
    if step <= 1:
        return int(odds)
    rounded = int(round(float(odds) / float(step)) * step)
    if rounded == 0:
        return step if odds > 0 else -step
    return rounded


def norm_surface(surface: str | None) -> str:
    s = (surface or "").lower()
    if "hard" in s:
        return "hard"
    if "clay" in s:
        return "clay"
    if "grass" in s:
        return "grass"
    return "hard"


def infer_best_of(row: dict) -> int:
    """
    Prefer explicit best_of if present.
    Otherwise infer Grand Slam for ATP as Bo5; WTA remains Bo3.
    """
    bo = row.get("best_of") or row.get("bestOf")
    if bo is not None:
        try:
            return int(bo)
        except Exception:
            pass

    tour = (row.get("tour") or "").upper()
    tname = (row.get("tournament") or "").lower()

    is_slam = any(
        k in tname
        for k in ["australian open", "french open", "roland garros", "wimbledon", "us open"]
    )

    if is_slam and tour == "ATP":
        return 5
    return 3


# -----------------------------
# Spread from Elo
# -----------------------------
def project_spread_games(best_of: int, d_elo_used: float) -> float:
    """
    Map Elo diff -> expected game spread for p1.

    Bo3: 100 Elo ≈ ~2 games
    Bo5: 100 Elo ≈ ~3 games
    Negative => p1 favored.
    """
    if best_of == 5:
        spread = -(d_elo_used / 33.3)  # 100 -> -3.0
        spread = clamp(spread, -12.5, 12.5)
    else:
        spread = -(d_elo_used / 50.0)  # 100 -> -2.0
        spread = clamp(spread, -7.5, 7.5)

    return round(spread, 1)


# -----------------------------
# BO3 Joint logic (sets + totals consistent)
# -----------------------------
def project_three_set_prob_bo3(p1_prob: float, spread_p1: float) -> float:
    """
    Probability match goes 3 sets.
    - Closer (p near 0.50) => higher p3
    - Bigger |spread| => lower p3
    """
    closeness = 1.0 - abs(p1_prob - 0.5) / 0.5  # 0..1
    spread_pen = abs(spread_p1)

    x = (2.2 * (closeness - 0.6)) - (0.55 * (spread_pen - 1.5))
    p3 = logistic(x)
    return clamp(p3, 0.10, 0.60)


def expected_games_2_0_bo3(surface: str | None, spread_p1: float) -> float:
    total = 19.2
    total -= clamp(abs(spread_p1) * 0.45, 0.0, 4.0)

    surf = norm_surface(surface)
    if surf == "hard":
        total += 0.2
    elif surf == "clay":
        total += 0.7
    elif surf == "grass":
        total -= 0.3

    return clamp(total, 16.0, 22.5)


def expected_games_2_1_bo3(surface: str | None) -> float:
    total = 28.0
    surf = norm_surface(surface)
    if surf == "hard":
        total += 0.2
    elif surf == "clay":
        total += 0.8
    elif surf == "grass":
        total -= 0.4
    return clamp(total, 24.0, 34.0)


def project_total_games_bo3_joint(p3: float, surface: str | None, spread_p1: float) -> float:
    g20 = expected_games_2_0_bo3(surface, spread_p1)
    g21 = expected_games_2_1_bo3(surface)
    total = (1.0 - p3) * g20 + p3 * g21
    return round(total, 1)


def project_sets_bo3_joint(p1_prob: float, p3: float, spread_p1: float) -> str:
    # Big spread => straight sets most likely
    if spread_p1 <= -3.5:
        return "2-0"
    if spread_p1 >= 3.5:
        return "0-2"

    if p3 >= 0.45:
        return "2-1" if p1_prob >= 0.5 else "1-2"
    return "2-0" if p1_prob >= 0.5 else "0-2"


# -----------------------------
# BO5 Joint logic (3-0 / 3-1 / 3-2 + totals consistent)
# -----------------------------
def project_long_match_prob_bo5(p1_prob: float, spread_p1: float) -> tuple[float, float]:
    """
    Return (p4, p5) where:
      p4 = probability match goes 4 sets
      p5 = probability match goes 5 sets

    Closer + smaller spread => more sets.
    """
    closeness = 1.0 - abs(p1_prob - 0.5) / 0.5  # 0..1
    spread_pen = abs(spread_p1)

    # Prob of reaching at least 4 sets
    x4 = (2.0 * (closeness - 0.55)) - (0.45 * (spread_pen - 2.0))
    p4plus = clamp(logistic(x4), 0.15, 0.75)

    # Prob of reaching 5 sets (harder than 4)
    x5 = (2.1 * (closeness - 0.65)) - (0.55 * (spread_pen - 1.5))
    p5 = clamp(logistic(x5), 0.05, 0.45)

    # Ensure p5 <= p4plus
    p5 = min(p5, p4plus)

    # Convert to exact p4 and p5 (p4 = 4 sets but not 5)
    p4 = p4plus - p5
    return (p4, p5)


def expected_games_3_0_bo5(surface: str | None, spread_p1: float) -> float:
    """
    Typical 3-0 totals are often ~27–33 depending on closeness.
    Bigger spread => fewer games.
    """
    total = 30.0
    total -= clamp(abs(spread_p1) * 0.55, 0.0, 6.0)

    surf = norm_surface(surface)
    if surf == "hard":
        total += 0.4
    elif surf == "clay":
        total += 1.2
    elif surf == "grass":
        total -= 0.6

    return clamp(total, 26.0, 38.0)


def expected_games_3_1_bo5(surface: str | None) -> float:
    total = 40.0
    surf = norm_surface(surface)
    if surf == "hard":
        total += 0.5
    elif surf == "clay":
        total += 1.5
    elif surf == "grass":
        total -= 0.8
    return clamp(total, 34.0, 52.0)


def expected_games_3_2_bo5(surface: str | None) -> float:
    total = 46.5
    surf = norm_surface(surface)
    if surf == "hard":
        total += 0.6
    elif surf == "clay":
        total += 1.8
    elif surf == "grass":
        total -= 1.0
    return clamp(total, 38.0, 60.0)


def project_total_games_bo5_joint(p4: float, p5: float, surface: str | None, spread_p1: float) -> float:
    g30 = expected_games_3_0_bo5(surface, spread_p1)
    g31 = expected_games_3_1_bo5(surface)
    g32 = expected_games_3_2_bo5(surface)

    # probabilities: p30 = 1 - (p4+p5)
    p30 = max(0.0, 1.0 - (p4 + p5))
    total = p30 * g30 + p4 * g31 + p5 * g32
    return round(total, 1)


def project_sets_bo5_joint(p1_prob: float, p4: float, p5: float, spread_p1: float) -> str:
    """
    Most-likely sets label for Bo5.
    """
    # Big favorite/dog => 3-0 / 0-3
    if spread_p1 <= -6.0:
        return "3-0"
    if spread_p1 >= 6.0:
        return "0-3"

    # Otherwise decide between 3-1 vs 3-2 vs 3-0 using p4/p5
    # If p5 is high, prefer 3-2; else if p4 is decent, prefer 3-1; else 3-0.
    if p5 >= 0.22:
        return "3-2" if p1_prob >= 0.5 else "2-3"
    if p4 >= 0.30:
        return "3-1" if p1_prob >= 0.5 else "1-3"
    return "3-0" if p1_prob >= 0.5 else "0-3"


# -----------------------------
# Main predictor used by /predictions/today
# -----------------------------
def predict_match(row: dict) -> dict:
    # Pick the best Elo for the match surface (falls back to overall)
    p1_used = pick_surface_elo(row, "p1")
    p2_used = pick_surface_elo(row, "p2")

    best_of = infer_best_of(row)

    # Production-safe: if either side has no rating, return null outputs
    if p1_used is None or p2_used is None:
        reason = row.get("missing_reason") or "missing Elo rating for one/both players"
        return {
            "match_id": row.get("match_id"),
            "match_key": row.get("match_key"),
            "match_date": row.get("match_date"),
            "tour": row.get("tour"),
            "tournament": row.get("tournament"),
            "round": row.get("round"),
            "surface": row.get("surface"),
            "p1_name": row.get("p1_name"),
            "p2_name": row.get("p2_name"),
            "p1_ta_id": row.get("p1_ta_id"),
            "p2_ta_id": row.get("p2_ta_id"),
            "p1_win_prob": None,
            "p2_win_prob": None,
            "p1_fair_american": None,
            "p2_fair_american": None,
            "missing_reason": reason,
            "predicted_winner": None,
            "projected_total_games": None,
            "projected_spread_p1": None,
            "projected_sets": None,
            "best_of": best_of,
            "inputs": {
                "p1_elo_used": p1_used,
                "p2_elo_used": p2_used,
                "p1_elo_overall": row.get("p1_elo"),
                "p2_elo_overall": row.get("p2_elo"),
                "p1_elo_hard": row.get("p1_helo"),
                "p2_elo_hard": row.get("p2_helo"),
                "p1_elo_clay": row.get("p1_celo"),
                "p2_elo_clay": row.get("p2_celo"),
                "p1_elo_grass": row.get("p1_gelo"),
                "p2_elo_grass": row.get("p2_gelo"),
            },
        }

    # Win probability comes directly from Elo (this is the main Elo effect)
    p1_prob = float(elo_prob(p1_used, p2_used))
    p2_prob = 1.0 - p1_prob

    # Elo diff -> spread (this is the other Elo effect)
    d_elo_used = float(p1_used) - float(p2_used)
    projected_spread_p1 = project_spread_games(best_of, d_elo_used)

    # Joint projections (sets + totals consistent)
    surface = row.get("surface")

    if best_of == 5:
        p4, p5 = project_long_match_prob_bo5(p1_prob, projected_spread_p1)
        projected_total_games = project_total_games_bo5_joint(p4, p5, surface, projected_spread_p1)
        projected_sets = project_sets_bo5_joint(p1_prob, p4, p5, projected_spread_p1)
    else:
        p3 = project_three_set_prob_bo3(p1_prob, projected_spread_p1)
        projected_total_games = project_total_games_bo3_joint(p3, surface, projected_spread_p1)
        projected_sets = project_sets_bo3_joint(p1_prob, p3, projected_spread_p1)

    predicted_winner = "p1" if p1_prob >= 0.5 else "p2"

    try:
        fair_step = int(os.getenv("FAIR_ODDS_ROUND_STEP", "1"))
    except Exception:
        fair_step = 1

    return {
        "match_id": row.get("match_id"),
        "match_key": row.get("match_key"),
        "match_date": row.get("match_date"),
        "tour": row.get("tour"),
        "tournament": row.get("tournament"),
        "round": row.get("round"),
        "surface": surface,
        "p1_name": row.get("p1_name"),
        "p2_name": row.get("p2_name"),
        "p1_ta_id": row.get("p1_ta_id"),
        "p2_ta_id": row.get("p2_ta_id"),
        "p1_win_prob": round(p1_prob, 4),
        "p2_win_prob": round(p2_prob, 4),
        "p1_fair_american": _round_american(prob_to_american(p1_prob), fair_step),
        "p2_fair_american": _round_american(prob_to_american(p2_prob), fair_step),
        "missing_reason": row.get("missing_reason"),

        # ✅ outputs you want
        "predicted_winner": predicted_winner,
        "projected_total_games": projected_total_games,
        "projected_spread_p1": projected_spread_p1,
        "projected_sets": projected_sets,
        "best_of": best_of,

        "inputs": {
            "p1_elo_used": p1_used,
            "p2_elo_used": p2_used,
            "p1_elo_overall": row.get("p1_elo"),
            "p2_elo_overall": row.get("p2_elo"),
            "p1_elo_hard": row.get("p1_helo"),
            "p2_elo_hard": row.get("p2_helo"),
            "p1_elo_clay": row.get("p1_celo"),
            "p2_elo_clay": row.get("p2_celo"),
            "p1_elo_grass": row.get("p1_gelo"),
            "p2_elo_grass": row.get("p2_gelo"),
        },
    }
