def american_to_prob(odds: int) -> float:
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)

def prob_to_american(p: float) -> int:
    p = max(1e-6, min(0.999999, p))
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    else:
        return int(round(100 * (1 - p) / p))

def devig_two_way(p_home_raw: float, p_away_raw: float) -> tuple[float, float]:
    # Normalize so the two implied probs sum to 1
    s = p_home_raw + p_away_raw
    if s <= 0:
        return 0.5, 0.5
    return p_home_raw / s, p_away_raw / s
