def project_breaks(player_bp_won_avg: float,
                   player_bp_chances_avg: float,
                   opp_bp_faced_avg: float,
                   opp_bp_saved_rate: float | None = None,
                   h2h_breaks_avg: float | None = None,
                   surface_factor: float = 0.0) -> dict:
    """
    Project BREAKS (break points converted / breaks of serve).

    Intuition:
    - Player creates chances (bp_chances_avg)
    - Opponent allows chances (opp_bp_faced_avg)
    - Conversion depends on player conversion and opponent save rate
    """

    # Blend expected chances created in this matchup
    exp_chances = 0.55 * player_bp_chances_avg + 0.45 * opp_bp_faced_avg

    # Player conversion rate (won/chances) from their averages
    conv_rate = 0.0
    if player_bp_chances_avg > 0:
        conv_rate = player_bp_won_avg / player_bp_chances_avg

    # Clamp to realistic tennis ranges
    conv_rate = max(0.15, min(conv_rate, 0.60))

    # Adjust for opponent save ability if provided
    if opp_bp_saved_rate is not None:
        # If opponent saves a lot, conversion should drop
        # Example: saved_rate=0.70 -> multiplier ~0.85
        conv_rate *= max(0.70, min(1.05, 1.0 - (opp_bp_saved_rate - 0.60)))

    proj_breaks = exp_chances * conv_rate + surface_factor

    # Optional H2H nudge
    if h2h_breaks_avg is not None:
        proj_breaks = 0.85 * proj_breaks + 0.15 * h2h_breaks_avg

    return {
        "projected_breaks": max(0.0, proj_breaks),
        "projected_bp_chances": max(0.0, exp_chances),
        "conversion_rate": conv_rate
    }
