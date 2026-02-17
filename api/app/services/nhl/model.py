from ...core.pmf_ev import poisson_pmf, ev_per_dollar, kelly_fraction

def score_sog_stub(line: float, american_odds: int):
    # TODO: replace with real λ from your model (expected Shots on Goal)
    lmbda = 3.2  # typical SOG expectation for a top-6 forward

    kmin, kmax = max(0, int(line) - 6), int(line) + 6
    pmf = poisson_pmf(lmbda, kmin, kmax)

    mode_k = max(pmf, key=lambda k: pmf[k])
    ev_stat = sum(k*v for k, v in pmf.items())
    p_over = sum(v for k, v in pmf.items() if k > line)

    def q(p):
        acc=0.0
        for k,v in pmf.items():
            acc += v
            if acc >= p: return int(k)
        return int(kmax)

    pi68, pi90 = [q(0.16), q(0.84)], [q(0.05), q(0.95)]
    ev = ev_per_dollar(american_odds, p_over)
    kelly = kelly_fraction(american_odds, p_over)
    conf = max(1, min(10, int((p_over - 0.50)/0.03) + 5))
    return ev_stat, mode_k, pi68, pi90, p_over, ev, kelly, conf, {str(k): round(v,4) for k,v in pmf.items()}
