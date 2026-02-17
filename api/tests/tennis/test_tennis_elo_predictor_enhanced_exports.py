from __future__ import annotations

from app.services.tennis import elo_predictor_enhanced as mod


def test_compat_exports_call_same_logic():
    md = {
        "p1_elo": 1800,
        "p2_elo": 1700,
        "surface": "hard",
        "best_of": 3,
        "p1_helo": 1810,
        "p2_helo": 1710,
    }
    out_a = mod.predict_match_enhanced(md)
    out_b = mod.predict_match_elo(md)
    out_c = mod.predict_match(md)
    assert out_a == out_b == out_c
    assert out_a["method"] == "elo_enhanced"
    assert out_a["p1_win_prob"] is not None
