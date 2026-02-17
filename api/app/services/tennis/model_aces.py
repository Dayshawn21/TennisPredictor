def project_aces(player_avg, opp_allowed, h2h_avg=None, surface_factor=0.0):
    """
    Weighted blend:
      - player form (surface avg)
      - opponent allowed (aces conceded)
      - optional H2H
      - optional surface factor (small)
    """
    w_player = 0.50
    w_opp = 0.35
    w_h2h = 0.15 if h2h_avg is not None else 0.0

    base = (w_player * player_avg) + (w_opp * opp_allowed)
    if h2h_avg is not None:
        base += w_h2h * h2h_avg

    return max(0.0, base + surface_factor)
