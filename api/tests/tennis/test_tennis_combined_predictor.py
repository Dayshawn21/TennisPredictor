from __future__ import annotations

from app.services.tennis import combined_predictor as cp


def _base_match() -> dict:
    return {
        "p1_elo": 1800,
        "p2_elo": 1700,
        "surface": "hard",
        "best_of": 3,
        "p1_helo": 1810,
        "p2_helo": 1710,
        "p1_odds_american": -130,
        "p2_odds_american": 110,
        "has_fresh_odds": True,
        "high_overround": False,
        "odds_age_minutes": 15.0,
        "market_overround_main": 0.04,
        # style / rolling
        "p1_svc_pts_w": 0.64,
        "p2_svc_pts_w": 0.61,
        "p1_ret_pts_w": 0.39,
        "p2_ret_pts_w": 0.37,
        "p1_win_rate_last_20": 0.75,
        "p2_win_rate_last_20": 0.55,
        "p1_win_rate_surface": 0.72,
        "p2_win_rate_surface": 0.58,
        # TA presence marker
        "d_last10_hold": 0.03,
    }


def test_dynamic_weights_base_case(monkeypatch):
    monkeypatch.setattr(cp, "predict_match_elo", lambda md: {"p1_win_prob": 0.6})
    monkeypatch.setattr(cp, "predict_match_xgb", lambda md: {"p1_win_prob": 0.55})

    out = cp.predict_match_combined(_base_match())
    w = out["effective_weights"]
    assert out["method"] == "combined_ensemble"
    assert set(w.keys()) == {"elo", "xgb", "market_no_vig"}
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert round(w["elo"], 3) == 0.4
    assert round(w["xgb"], 3) == 0.4
    assert round(w["market_no_vig"], 3) == 0.2


def test_dynamic_weights_stale_odds_downweights_market(monkeypatch):
    monkeypatch.setattr(cp, "predict_match_elo", lambda md: {"p1_win_prob": 0.6})
    monkeypatch.setattr(cp, "predict_match_xgb", lambda md: {"p1_win_prob": 0.55})

    md = _base_match()
    md["has_fresh_odds"] = False
    out = cp.predict_match_combined(md)
    w = out["effective_weights"]
    assert "stale_odds" in out["predictor_quality_flags"]
    assert w["market_no_vig"] < 0.2


def test_dynamic_weights_missing_market(monkeypatch):
    monkeypatch.setattr(cp, "predict_match_elo", lambda md: {"p1_win_prob": 0.6})
    monkeypatch.setattr(cp, "predict_match_xgb", lambda md: {"p1_win_prob": 0.55})

    md = _base_match()
    md["p1_odds_american"] = None
    md["p2_odds_american"] = None
    out = cp.predict_match_combined(md)
    w = out["effective_weights"]
    assert "market_missing" in (out.get("missing_reason") or "")
    assert "market_no_vig" not in w
    assert set(w.keys()) == {"elo", "xgb"}


def test_dynamic_weights_elo_fallback_and_xgb_partial(monkeypatch):
    monkeypatch.setattr(cp, "predict_match_elo", lambda md: {"p1_win_prob": 0.6})
    monkeypatch.setattr(cp, "predict_match_xgb", lambda md: {"p1_win_prob": 0.55})

    md = _base_match()
    md["elo_used_median"] = True
    md.pop("d_last10_hold", None)  # TA group missing alone should be tolerated now
    md["p1_win_rate_last_20"] = None  # missing rolling -> XGB multiplier 0.8
    out = cp.predict_match_combined(md)
    w = out["effective_weights"]
    raw = {x["name"]: x["weight"] for x in out["individual_predictions"]}
    assert "elo_fallback_used" in out["predictor_quality_flags"]
    assert "xgb_feature_groups_partial" in out["predictor_quality_flags"]
    assert w["elo"] < 0.4
    assert raw["xgb"] < 0.40
