"""
Service-game simulator for tennis totals / sets.
Approach A: use estimated service hold probabilities to Monte Carlo simulate
set scores, total games, and number of sets.

Notes:
- This sim is intentionally "game-level" (not point-level). It's fast and stable.
- Tie-breaks are modeled as a single Bernoulli event with probability = p_tb (default p1 win prob).
- Total games counts a 7-6 set as 13 games (tie-break counts as one game), matching common markets.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, Optional, Tuple

_DEFAULT_LINES_BO3 = [16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5]
_DEFAULT_LINES_BO5 = [31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _seed_from_match_id(match_id: Optional[str]) -> int:
    if not match_id:
        return 1337
    h = hashlib.md5(match_id.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def _pct_to_prob(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    # if already probability
    if 0.0 <= v <= 1.0:
        return v
    # else treat as percent
    if 0.0 <= v <= 100.0:
        return v / 100.0
    return None


def estimate_hold_probs(match_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate per-service-game hold probability for p1 and p2.

    Preferred inputs (float or numeric-like):
      - p1_service_hold_pct
      - p2_service_hold_pct
      - p1_opponent_hold_pct  (how often opponents hold vs p1)
      - p2_opponent_hold_pct

    If opponent_hold is present, we blend:
      p1_hold ~= avg(p1_service_hold, p2_opponent_hold)
      p2_hold ~= avg(p2_service_hold, p1_opponent_hold)

    Returns (p1_hold, p2_hold) or (None, None) if insufficient data.
    """
    tour = (match_data.get("tour") or "").upper()

    p1_srv = _pct_to_prob(match_data.get("p1_service_hold_pct"))
    p2_srv = _pct_to_prob(match_data.get("p2_service_hold_pct"))
    p1_opp_hold = _pct_to_prob(match_data.get("p1_opponent_hold_pct"))
    p2_opp_hold = _pct_to_prob(match_data.get("p2_opponent_hold_pct"))

    if p1_srv is None or p2_srv is None:
        return (None, None)

    if p1_opp_hold is not None and p2_opp_hold is not None:
        p1_hold = 0.5 * p1_srv + 0.5 * p2_opp_hold
        p2_hold = 0.5 * p2_srv + 0.5 * p1_opp_hold
    else:
        p1_hold, p2_hold = p1_srv, p2_srv

    # sane clamps
    if tour == "ATP":
        p1_hold = _clamp(p1_hold, 0.60, 0.97)
        p2_hold = _clamp(p2_hold, 0.60, 0.97)
    else:
        p1_hold = _clamp(p1_hold, 0.45, 0.92)
        p2_hold = _clamp(p2_hold, 0.45, 0.92)

    return (p1_hold, p2_hold)


def _simulate_set(
    rng: random.Random,
    hold_p1: float,
    hold_p2: float,
    first_server: int,
    p1_tb: float,
) -> Tuple[int, int, int]:
    """
    Simulate one set at game-level.
    Returns (p1_games, p2_games, next_server).
    next_server is who would serve the next game if match continued (1 or 2).
    """
    g1 = 0
    g2 = 0
    server = first_server  # 1=p1, 2=p2

    def other(s: int) -> int:
        return 2 if s == 1 else 1

    while True:
        # normal set end
        if (g1 >= 6 or g2 >= 6) and abs(g1 - g2) >= 2:
            return g1, g2, server  # server already points to next game server

        # tiebreak at 6-6
        if g1 == 6 and g2 == 6:
            tb_first_server = server
            if rng.random() < p1_tb:
                g1 += 1
            else:
                g2 += 1
            # next set: first server is the receiver of first TB server
            return g1, g2, other(tb_first_server)

        # service game
        p_hold = hold_p1 if server == 1 else hold_p2
        if rng.random() < p_hold:
            if server == 1:
                g1 += 1
            else:
                g2 += 1
        else:
            if server == 1:
                g2 += 1
            else:
                g1 += 1

        server = other(server)


def simulate_match_totals(
    match_id: Optional[str],
    best_of: int,
    hold_p1: float,
    hold_p2: float,
    p1_tb: float,
    n_sims: int = 6000,
) -> Dict[str, Any]:
    rng = random.Random(_seed_from_match_id(match_id))
    sets_to_win = (best_of // 2) + 1

    p1_wins = 0
    total_games_sum = 0.0

    sets_hist: Dict[int, int] = {}
    score_hist: Dict[str, int] = {}
    games_hist: Dict[int, int] = {}

    for _ in range(max(1, int(n_sims))):
        server = 1 if rng.random() < 0.5 else 2  # coin toss

        p1_sets = 0
        p2_sets = 0
        games_total = 0

        while p1_sets < sets_to_win and p2_sets < sets_to_win:
            g1, g2, next_server = _simulate_set(rng, hold_p1, hold_p2, server, p1_tb)
            games_total += (g1 + g2)
            server = next_server

            if g1 > g2:
                p1_sets += 1
            else:
                p2_sets += 1

        if p1_sets > p2_sets:
            p1_wins += 1

        total_games_sum += games_total
        sets_played = p1_sets + p2_sets
        sets_hist[sets_played] = sets_hist.get(sets_played, 0) + 1
        score_key = f"{p1_sets}-{p2_sets}"
        score_hist[score_key] = score_hist.get(score_key, 0) + 1
        games_hist[games_total] = games_hist.get(games_total, 0) + 1

    sims = float(max(1, int(n_sims)))
    exp_games = total_games_sum / sims
    p1_win_prob = p1_wins / sims

    lines = _DEFAULT_LINES_BO5 if best_of == 5 else _DEFAULT_LINES_BO3
    over_probs: Dict[str, float] = {}
    for ln in lines:
        over_probs[str(ln)] = sum(cnt for g, cnt in games_hist.items() if g > ln) / sims

    def quantile(q: float) -> float:
        target = q * sims
        run = 0.0
        for g in sorted(games_hist.keys()):
            run += games_hist[g]
            if run >= target:
                return float(g)
        return float(max(games_hist.keys()))

    return {
        "sims": int(sims),
        "hold_p1": float(hold_p1),
        "hold_p2": float(hold_p2),
        "p1_win_prob_sim": float(p1_win_prob),
        "p2_win_prob_sim": float(1.0 - p1_win_prob),
        "expected_total_games": float(exp_games),
        "total_games_p25": quantile(0.25),
        "total_games_p50": quantile(0.50),
        "total_games_p75": quantile(0.75),
        "sets_probs": {str(k): v / sims for k, v in sorted(sets_hist.items())},
        "score_probs": {k: v / sims for k, v in sorted(score_hist.items())},
        "over_probs": over_probs,
    }


def predict_match_totals(
    match_data: Dict[str, Any],
    p1_win_prob_hint: Optional[float] = None,
    n_sims: int = 6000,
) -> Optional[Dict[str, Any]]:
    best_of = int(match_data.get("best_of") or 3)
    hold_p1, hold_p2 = estimate_hold_probs(match_data)
    if hold_p1 is None or hold_p2 is None:
        return None

    p1_tb = p1_win_prob_hint if (p1_win_prob_hint is not None and 0 < p1_win_prob_hint < 1) else 0.5
    p1_tb = _clamp(float(p1_tb), 0.05, 0.95)

    return simulate_match_totals(
        match_id=str(match_data.get("match_id") or ""),
        best_of=best_of,
        hold_p1=float(hold_p1),
        hold_p2=float(hold_p2),
        p1_tb=p1_tb,
        n_sims=int(n_sims),
    )
