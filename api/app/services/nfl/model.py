import math
from ...core.pmf_ev import discretized_lognormal_pmf, poisson_pmf, ev_per_dollar, kelly_fraction
from .features import (
    recent_qb_form, opponent_pass_allowed_pg,
    recent_rb_form, opponent_rush_allowed_pg,
    recent_wr_form
)

LEAGUE_PASS_ALLOWED = 230.0
LEAGUE_RUSH_ALLOWED = 115.0  # league-ish average rush yards allowed
YDS_SIGMA_LN = 0.28          # dispersion for yards (tune/learn later)

def _mu_sigma_from_mean(mu_linear: float, sigma_ln: float = YDS_SIGMA_LN):
    mu_ln = math.log(max(1e-6, mu_linear)) - 0.5 * (sigma_ln ** 2)
    return mu_ln, sigma_ln

async def score_passing_yards(p, db):
    form = await recent_qb_form(p.player_id, db)
    if not form: return None
    opp_allowed = await opponent_pass_allowed_pg(p.opp_team_id, db)
    mu_linear = form["avg_pass_yds"] * (opp_allowed / LEAGUE_PASS_ALLOWED)
    mu_linear = max(120.0, min(mu_linear, 400.0))
    mu_ln, sigma_ln = _mu_sigma_from_mean(mu_linear)
    kmin, kmax = max(0, int(p.line) - 125), int(p.line) + 125
    pmf = discretized_lognormal_pmf(mu_ln, sigma_ln, kmin, kmax)
    return _summarize_pmf(pmf, p.line, p.american_odds)

async def score_rushing_yards(p, db):
    form = await recent_rb_form(p.player_id, db)
    if not form: return None
    opp_allowed = await opponent_rush_allowed_pg(p.opp_team_id, db)
    # scale recent avg by opponent rush allowed vs league baseline
    mu_linear = form["avg_rush_yds"] * (opp_allowed / LEAGUE_RUSH_ALLOWED)
    mu_linear = max(20.0, min(mu_linear, 180.0))
    mu_ln, sigma_ln = _mu_sigma_from_mean(mu_linear, sigma_ln=0.32)  # rushing slightly more skew
    kmin, kmax = max(0, int(p.line) - 80), int(p.line) + 80
    pmf = discretized_lognormal_pmf(mu_ln, sigma_ln, kmin, kmax)
    return _summarize_pmf(pmf, p.line, p.american_odds)

async def score_receiving_yards(p, db):
    form = await recent_wr_form(p.player_id, db)
    if not form: return None
    # proxy opponent via pass allowed (same table as QB)
    opp_allowed = await opponent_pass_allowed_pg(p.opp_team_id, db)
    mu_linear = form["avg_rec_yds"] * (opp_allowed / LEAGUE_PASS_ALLOWED)
    mu_linear = max(15.0, min(mu_linear, 200.0))
    mu_ln, sigma_ln = _mu_sigma_from_mean(mu_linear, sigma_ln=0.30)
    kmin, kmax = max(0, int(p.line) - 80), int(p.line) + 80
    pmf = discretized_lognormal_pmf(mu_ln, sigma_ln, kmin, kmax)
    return _summarize_pmf(pmf, p.line, p.american_odds)

async def score_receptions(p, db):
    form = await recent_wr_form(p.player_id, db)
    if not form: return None
    # Estimate catch rate & targets → lambda for Poisson
    targets = max(0.0, form["avg_targets"])
    catch_rate = 0.60 if targets == 0 else min(0.85, max(0.4, form["avg_receptions"] / max(1.0, targets)))
    lam = max(0.8, min(12.0, targets * catch_rate))
    kmin, kmax = 0, max(12, int(p.line) + 8)
    pmf = poisson_pmf(lam, kmin, kmax)
    return _summarize_pmf(pmf, p.line, p.american_odds)

# ---------- shared summarizer ----------
def _summarize_pmf(pmf, line, american_odds):
    mode_k = max(pmf, key=lambda k: pmf[k])
    ev_stat = sum(k * v for k, v in pmf.items())
    p_over = sum(v for k, v in pmf.items() if k > line)
    def q(prob):
        acc = 0.0
        for k, v in pmf.items():
            acc += v
            if acc >= prob:
                return int(k)
        return int(max(pmf))
    pi68, pi90 = [q(0.16), q(0.84)], [q(0.05), q(0.95)]
    ev = ev_per_dollar(american_odds, p_over)
    kelly = kelly_fraction(american_odds, p_over)
    conf = max(1, min(10, int((p_over - 0.50) / 0.03) + 5))
    return ev_stat, mode_k, pi68, pi90, p_over, ev, kelly, conf, {str(k): round(v, 4) for k, v in pmf.items()}
