import math
from functools import lru_cache

def clamp(x, lo=1e-6, hi=1-1e-6):
    return max(lo, min(hi, x))

def logit(p):
    p = clamp(p)
    return math.log(p/(1-p))

def logistic(z):
    return 1/(1+math.exp(-z))

def combine_point_probs(service_pts_w: float, opp_return_pts_w: float) -> float:
    # A serving vs B: use A serve points won and (1 - B return points won)
    s = clamp(service_pts_w)
    o = clamp(1 - opp_return_pts_w)
    return clamp(logistic((logit(s) + logit(o)) / 2))

def hold_prob_from_point(p: float) -> float:
    p = clamp(p)
    q = 1 - p
    first = p**4 * (1 + 4*q + 10*q*q)
    deuce_reach = 20 * (p**3) * (q**3)
    deuce_win = (p*p) / (1 - 2*p*q)
    return clamp(first + deuce_reach * deuce_win)

def tiebreak_win_prob(pAserve: float, pBserve: float, A_serves_first: bool) -> float:
    pAserve = clamp(pAserve)
    pBserve = clamp(pBserve)

    def server_for_point(k: int) -> str:
        if k == 0:
            return "A" if A_serves_first else "B"
        block = (k - 1) // 2
        if block % 2 == 0:
            return "B" if A_serves_first else "A"
        return "A" if A_serves_first else "B"

    @lru_cache(None)
    def dp(a: int, b: int, k: int) -> float:
        if a >= 7 and a - b >= 2:
            return 1.0
        if b >= 7 and b - a >= 2:
            return 0.0

        srv = server_for_point(k)
        p = pAserve if srv == "A" else (1 - pBserve)  # A wins point prob
        return p * dp(a+1, b, k+1) + (1-p) * dp(a, b+1, k+1)

    return dp(0, 0, 0)

def set_score_distribution(holdA: float, holdB: float, pAserve: float, pBserve: float, A_serves_first: bool):
    holdA, holdB = clamp(holdA), clamp(holdB)

    @lru_cache(None)
    def dp(a: int, b: int, next_server: int):
        # next_server: 0=A serves next game, 1=B serves next game
        if (a >= 6 or b >= 6) and abs(a - b) >= 2:
            return {(a, b): 1.0}
        if a == 7 or b == 7:
            return {(a, b): 1.0}

        if a == 6 and b == 6:
            tbA = tiebreak_win_prob(pAserve, pBserve, A_serves_first=(next_server == 0))
            return {(7, 6): tbA, (6, 7): 1 - tbA}

        out = {}
        if next_server == 0:
            p_hold = holdA
            win_state = dp(a+1, b, 1)
            lose_state = dp(a, b+1, 1)
        else:
            p_hold = holdB
            win_state = dp(a, b+1, 0)
            lose_state = dp(a+1, b, 0)

        for k, v in win_state.items():
            out[k] = out.get(k, 0.0) + p_hold * v
        for k, v in lose_state.items():
            out[k] = out.get(k, 0.0) + (1 - p_hold) * v
        return out

    first_server = 0 if A_serves_first else 1
    return dp(0, 0, first_server)

def summarize_set(dist):
    p_set_win = sum(p for (ga, gb), p in dist.items() if ga > gb)
    e_games = sum((ga + gb) * p for (ga, gb), p in dist.items())
    return p_set_win, e_games
