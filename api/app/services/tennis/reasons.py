def fmt(x, d=1):
    return None if x is None else round(float(x), d)

def build_aces_reasons(player: str,
                       opponent: str,
                       line: float,
                       proj: float,
                       player_avg: float | None,
                       opp_allowed: float | None,
                       h2h_avg: float | None = None,
                       surface: str | None = None):
    reasons = []

    if surface:
        reasons.append(f"Surface: {surface} (used for surface-specific averages).")

    if player_avg is not None:
        reasons.append(f"{player} averages about {fmt(player_avg)} aces per match{(' on ' + surface) if surface else ''}.")

    if opp_allowed is not None:
        reasons.append(f"{opponent} concedes about {fmt(opp_allowed)} aces per match to opponents{(' on ' + surface) if surface else ''}.")

    if h2h_avg is not None:
        reasons.append(f"In their H2H sample, {player} averages about {fmt(h2h_avg)} aces vs {opponent}.")

    edge = proj - line
    if edge >= 0:
        reasons.append(f"Projection: {fmt(proj)} vs line {line} (+{fmt(edge)}). That supports the OVER.")
    else:
        reasons.append(f"Projection: {fmt(proj)} vs line {line} ({fmt(edge)}). That supports the UNDER.")

    return reasons
