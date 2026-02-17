from __future__ import annotations

import math
from typing import Any, Dict

from fastapi import HTTPException


def poisson_cdf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k >= 0 else 0.0
    s = 0.0
    exp_neg = math.exp(-lam)
    for i in range(0, k + 1):
        s += exp_neg * (lam ** i) / math.factorial(i)
    return min(1.0, max(0.0, s))


def tier(p: float) -> str:
    if p >= 0.70:
        return "LOCK ðŸ”¥"
    if p >= 0.62:
        return "STRONG"
    if p >= 0.55:
        return "LEAN"
    return "PASS"


def over_under_probs(line: float, expected: float) -> tuple[float, float]:
    over_k = int(math.floor(line)) + 1
    under_k = over_k - 1
    p_under = poisson_cdf(under_k, expected)
    p_over = 1.0 - p_under
    return p_over, p_under


def blend_surface(overall: float, surface: float, surf_w: float = 0.6) -> float:
    return surf_w * surface + (1.0 - surf_w) * overall


def predict_prop_payload(req: Dict[str, Any]) -> Dict[str, Any]:
    prop_type = req.get("prop_type")

    if prop_type == "aces":
        needed = [
            req.get("p_last10_aces_pg"),
            req.get("p_surf_last10_aces_pg"),
            req.get("opp_last10_aces_allowed_pg"),
            req.get("opp_surf_last10_aces_allowed_pg"),
        ]
        if any(v is None for v in needed):
            raise HTTPException(
                status_code=422,
                detail="aces requires p_last10_aces_pg, p_surf_last10_aces_pg, opp_last10_aces_allowed_pg, opp_surf_last10_aces_allowed_pg",
            )
        player_aces_pg = blend_surface(req["p_last10_aces_pg"], req["p_surf_last10_aces_pg"], surf_w=0.6)
        opp_allow_aces_pg = blend_surface(req["opp_last10_aces_allowed_pg"], req["opp_surf_last10_aces_allowed_pg"], surf_w=0.6)
        expected = (0.65 * player_aces_pg + 0.35 * opp_allow_aces_pg) * req["expected_games"]

    elif prop_type == "break_points_won":
        if req.get("expected_return_games") is None:
            raise HTTPException(422, detail="break_points_won requires expected_return_games")
        needed = [
            req.get("p_last10_bp_won_prg"),
            req.get("p_surf_last10_bp_won_prg"),
            req.get("opp_last10_bp_won_allowed_psg"),
            req.get("opp_surf_last10_bp_won_allowed_psg"),
        ]
        if any(v is None for v in needed):
            raise HTTPException(
                status_code=422,
                detail="break_points_won requires p_last10_bp_won_prg, p_surf_last10_bp_won_prg, opp_last10_bp_won_allowed_psg, opp_surf_last10_bp_won_allowed_psg",
            )
        player_bp_won_prg = blend_surface(req["p_last10_bp_won_prg"], req["p_surf_last10_bp_won_prg"], surf_w=0.6)
        opp_bp_won_allowed_psg = blend_surface(req["opp_last10_bp_won_allowed_psg"], req["opp_surf_last10_bp_won_allowed_psg"], surf_w=0.6)
        expected = (0.60 * player_bp_won_prg + 0.40 * opp_bp_won_allowed_psg) * req["expected_return_games"]

    else:
        raise HTTPException(status_code=400, detail="Unsupported prop_type")

    expected = max(0.0, float(expected))
    p_over, p_under = over_under_probs(req["line"], expected)
    conf = tier(max(p_over, p_under))
    return {
        "player": req["player"],
        "prop_type": req["prop_type"],
        "line": req["line"],
        "expected": round(expected, 3),
        "p_over": round(p_over, 4),
        "p_under": round(p_under, 4),
        "confidence_tier": conf,
    }
