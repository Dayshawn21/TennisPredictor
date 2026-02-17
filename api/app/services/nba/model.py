from .features import get_player_recent_stats, get_opp_defense, get_team_context
from ...core.pmf_ev import discrete_normal_pmf, ev_per_dollar, kelly_fraction

def score_points_stub(line: float, american_odds: int):
    """Stub function using hardcoded values for testing without database"""
    mu, sigma = 27.8, 5.2
    kmin, kmax = int(line) - 7, int(line) + 7
    pmf = discrete_normal_pmf(mu, sigma, kmin, kmax)
    mode_k = max(pmf, key=lambda k: pmf[k])
    ev_stat = sum(k*v for k, v in pmf.items())
    p_over = sum(v for k, v in pmf.items() if k > line)

    def q(prob):
        acc = 0.0
        for k, v in pmf.items():
            acc += v
            if acc >= prob:
                return int(k)
        return int(kmax)

    pi68 = [q(0.16), q(0.84)]
    pi90 = [q(0.05), q(0.95)]
    ev = ev_per_dollar(american_odds, p_over)
    kelly = kelly_fraction(american_odds, p_over)
    conf = max(1, min(10, int((p_over - 0.50)/0.03) + 5))

    return ev_stat, mode_k, pi68, pi90, p_over, ev, kelly, conf, {str(k): round(v, 4) for k, v in pmf.items()}

async def score_points_model(p, db):
    # 1) player form
    player_stats = await get_player_recent_stats(p.player_id, db)
    if not player_stats:
        return None

    # 2) opponent defense for this position
    opp = await get_opp_defense(p.opp_team_id, p.position, db)
    if not opp:
        return None

    # 3) opponent team context (pace/def)
    ctx = await get_team_context(p.opp_team_id, db)
    if not ctx:
        return None

    # --- simple heuristic model you can replace with ML ---
    # baseline from recent points
    mu = player_stats["avg_points"]

    # adjust for pace (centered on 100) and opponent allowed vs position (lower = tougher)
    pace_adj = ctx["pace_l10"] / 100.0
    pos_adj = 100.0 / max(1.0, opp["allowed_pts_l10"]) * 22.0  # scale to typical SF baseline (~22 pts)
    mu = mu * 0.6 + (mu * pace_adj) * 0.2 + pos_adj * 0.2

    sigma = max(3.0, mu * 0.18)

    # PMF & summaries
    kmin, kmax = int(p.line) - 7, int(p.line) + 7
    pmf = discrete_normal_pmf(mu, sigma, kmin, kmax)
    mode_k = max(pmf, key=lambda k: pmf[k])
    ev_stat = sum(k * v for k, v in pmf.items())
    p_over = sum(v for k, v in pmf.items() if k > p.line)

    # intervals
    def q(prob):
        acc = 0.0
        for k, v in pmf.items():
            acc += v
            if acc >= prob:
                return int(k)
        return int(kmax)
    pi68, pi90 = [q(0.16), q(0.84)], [q(0.05), q(0.95)]

    ev = ev_per_dollar(p.american_odds, p_over)
    kelly = kelly_fraction(p.american_odds, p_over)
    conf = max(1, min(10, int((p_over - 0.50) / 0.03) + 5))

    return ev_stat, mode_k, pi68, pi90, p_over, ev, kelly, conf, {str(k): round(v, 4) for k, v in pmf.items()}
