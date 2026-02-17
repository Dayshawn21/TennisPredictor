def r(x, d=1):
    return None if x is None else round(float(x), d)

def build_breaks_reasons(player: str,
                         opponent: str,
                         line: float,
                         proj_breaks: float,
                         proj_chances: float,
                         player_bp_won_avg: float | None,
                         player_bp_chances_avg: float | None,
                         opp_bp_faced_avg: float | None,
                         opp_bp_saved_rate: float | None = None,
                         h2h_breaks_avg: float | None = None,
                         surface: str | None = None):
    reasons = []

    if surface:
        reasons.append(f"Surface: {surface} (used for surface-specific break-point averages).")

    if player_bp_chances_avg is not None and player_bp_won_avg is not None:
        reasons.append(
            f"{player} creates about {r(player_bp_chances_avg)} break-point chances per match and converts about {r(player_bp_won_avg)} breaks on average."
        )

    if opp_bp_faced_avg is not None:
        reasons.append(
            f"{opponent} allows about {r(opp_bp_faced_avg)} break-point chances per match (opponents generate chances vs them)."
        )

    reasons.append(
        f"Projected chances in this matchup: {r(proj_chances)}; projected breaks (converted): {r(proj_breaks)}."
    )

    if opp_bp_saved_rate is not None:
        reasons.append(f"{opponent} saves about {r(opp_bp_saved_rate*100,0)}% of break points, which affects conversion.")

    if h2h_breaks_avg is not None:
        reasons.append(f"H2H context: {player} averages about {r(h2h_breaks_avg)} breaks vs {opponent} (when they’ve played).")

    edge = proj_breaks - line
    if edge >= 0:
        reasons.append(f"Projection {r(proj_breaks)} vs line {line} (+{r(edge)}). That supports the OVER.")
    else:
        reasons.append(f"Projection {r(proj_breaks)} vs line {line} ({r(edge)}). That supports the UNDER.")

    return reasons
