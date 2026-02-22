from __future__ import annotations

from app.routers.tennis_predictions_today_enhanced import _pick_summary


def _base_kwargs() -> dict:
    return {
        "pick_side": "p1",
        "p1_name": "Player A",
        "p2_name": "Player B",
        "p1_prob": 0.66,
        "p2_prob": 0.34,
        "p1_nv": 0.54,
        "p2_nv": 0.46,
        "p1_odds": -120,
        "p2_odds": 100,
        "p1_wr20": 0.72,
        "p2_wr20": 0.54,
        "p1_surf_wr": 0.70,
        "p2_surf_wr": 0.56,
        "p1_svc_pts_w": 0.65,
        "p2_svc_pts_w": 0.61,
        "p1_ret_pts_w": 0.40,
        "p2_ret_pts_w": 0.36,
        "d_rest_days": 3.0,
        "d_matches_10d": -3.0,
        "h2h_p1_win_pct": 0.70,
        "h2h_total": 6,
        "style_summary": {"p1": {"label": "all_court"}, "p2": {"label": "baseliner"}},
        "surface": "hard",
        "p1_elo": 1900.0,
        "p2_elo": 1780.0,
        "p1_rank": 8,
        "p2_rank": 22,
        "h2h_p1_wins": 4,
        "h2h_p2_wins": 2,
        "h2h_surface_p1_wins": 3,
        "h2h_surface_p2_wins": 1,
        "h2h_surface_total": 4,
        "individual_predictions": [
            {"name": "elo", "p1_prob": 0.67},
            {"name": "xgb", "p1_prob": 0.65},
            {"name": "market_no_vig", "p1_prob": 0.62},
        ],
        "p1_ace_pg": 7.2,
        "p2_ace_pg": 4.8,
        "p1_bp_save": 0.66,
        "p2_bp_save": 0.58,
        "p1_bp_win": 0.44,
        "p2_bp_win": 0.36,
        "p1_matches_played": 24,
        "p2_matches_played": 24,
        "elo_used_median": False,
    }


def test_pick_summary_strong_value_and_agreement():
    s = _pick_summary(**_base_kwargs())
    assert s is not None
    assert "High conviction" in s
    assert "market edge" in s
    assert "Key factors:" in s
    assert "Risks:" not in s


def test_pick_summary_small_edge_and_model_split_downgrades_conviction():
    k = _base_kwargs()
    k["p1_prob"] = 0.56
    k["p1_nv"] = 0.53
    k["individual_predictions"] = [
        {"name": "elo", "p1_prob": 0.70},
        {"name": "xgb", "p1_prob": 0.48},
        {"name": "market_no_vig", "p1_prob": 0.53},
    ]
    s = _pick_summary(**k)
    assert s is not None
    assert "Lean" in s or "Moderate conviction" in s
    assert "models split" in s


def test_pick_summary_data_gaps_warns_and_stays_compact():
    k = _base_kwargs()
    k["p1_elo"] = None
    k["p2_elo"] = None
    k["elo_used_median"] = True
    k["p1_matches_played"] = 8
    k["p2_matches_played"] = 8
    k["p1_wr20"] = None
    k["p2_wr20"] = None
    s = _pick_summary(**k)
    assert s is not None
    assert "ELO is estimated (median fill)" in s
    assert "small sample" in s


def test_pick_summary_contradictory_signals_contains_factor_and_risk():
    k = _base_kwargs()
    k["p1_prob"] = 0.61
    k["p1_nv"] = 0.53
    k["p1_surf_wr"] = 0.48
    k["p2_surf_wr"] = 0.63
    s = _pick_summary(**k)
    assert s is not None
    assert "Key factors:" in s
    assert "Risks:" in s


def test_pick_summary_suppresses_micro_deltas():
    k = _base_kwargs()
    k["p1_svc_pts_w"] = 0.611
    k["p2_svc_pts_w"] = 0.610
    k["p1_ret_pts_w"] = 0.371
    k["p2_ret_pts_w"] = 0.370
    k["p1_bp_save"] = 0.600
    k["p2_bp_save"] = 0.598
    k["p1_bp_win"] = 0.410
    k["p2_bp_win"] = 0.408
    s = _pick_summary(**k)
    assert s is not None
    assert "serve edge" not in s
    assert "return edge" not in s


def test_pick_summary_caps_key_factors_and_risks():
    k = _base_kwargs()
    k["p1_prob"] = 0.52
    k["p1_nv"] = 0.56
    k["individual_predictions"] = [
        {"name": "elo", "p1_prob": 0.72},
        {"name": "xgb", "p1_prob": 0.48},
    ]
    k["p1_elo"] = 1750
    k["p2_elo"] = 1860
    k["p1_surf_wr"] = 0.44
    k["p2_surf_wr"] = 0.67
    k["d_rest_days"] = -4
    k["d_matches_10d"] = 4
    k["p1_matches_played"] = 7
    k["elo_used_median"] = True
    s = _pick_summary(**k)
    assert s is not None

    key_chunk = s.split("Key factors:", 1)[1].split(".", 1)[0] if "Key factors:" in s else ""
    risk_chunk = s.split("Risks:", 1)[1].split(".", 1)[0] if "Risks:" in s else ""

    key_count = 0 if not key_chunk.strip() else len([x for x in key_chunk.split(";") if x.strip()])
    risk_count = 0 if not risk_chunk.strip() else len([x for x in risk_chunk.split(";") if x.strip()])

    assert key_count <= 3
    assert risk_count <= 2
