from __future__ import annotations

from app.routers.tennis_predictions_today_enhanced import _tweet_text


def test_tweet_text_standard_includes_core_fields():
    t = _tweet_text(
        pick_side="p1",
        p1_name="Player A",
        p2_name="Player B",
        p1_odds=-120,
        p2_odds=100,
        p1_prob=0.66,
        p2_prob=0.34,
        p1_nv=0.54,
        p2_nv=0.46,
        pick_summary="Player A over Player B. High conviction. Key factors: market edge +12.0pp; ELO +120.",
    )
    assert t is not None
    assert "Player A over Player B (-120)" in t
    assert "Model 66.0% vs Mkt 54.0%" in t
    assert "Edge +12.0pp" in t
    assert "high conviction" in t.lower()


def test_tweet_text_handles_missing_market_cleanly():
    t = _tweet_text(
        pick_side="p2",
        p1_name="Player A",
        p2_name="Player B",
        p1_odds=-110,
        p2_odds=105,
        p1_prob=0.48,
        p2_prob=0.52,
        p1_nv=None,
        p2_nv=None,
        pick_summary=None,
    )
    assert t is not None
    assert "None" not in t
    assert "nan" not in t.lower()
    assert "Model 52.0%" in t


def test_tweet_text_handles_missing_edge():
    t = _tweet_text(
        pick_side="p1",
        p1_name="Player A",
        p2_name="Player B",
        p1_odds=None,
        p2_odds=None,
        p1_prob=None,
        p2_prob=None,
        p1_nv=None,
        p2_nv=None,
        pick_summary="Player A over Player B. Moderate conviction.",
    )
    assert t is not None
    assert "Edge" not in t
    assert "None" not in t


def test_tweet_text_is_capped_to_280_chars():
    long_name_1 = "Alexandria-Cassandra The First of Longname Province and Beyond"
    long_name_2 = "Bartholomew-Jonathan The Second of Extra Extended Match Card"
    long_summary = (
        "Key factors: market edge +10.2pp; stronger serve (66.0% vs 61.1% pts won); "
        "return pressure (40.2% vs 36.8% pts won); hot form L20 (75% vs 52%). "
        "Risks: models split (elo 71% vs xgb 49%)."
    )
    t = _tweet_text(
        pick_side="p1",
        p1_name=long_name_1,
        p2_name=long_name_2,
        p1_odds=-135,
        p2_odds=115,
        p1_prob=0.64,
        p2_prob=0.36,
        p1_nv=0.53,
        p2_nv=0.47,
        pick_summary=long_summary,
    )
    assert t is not None
    assert len(t) <= 280


def test_tweet_text_is_single_line_and_ascii_punctuation_safe():
    t = _tweet_text(
        pick_side="p1",
        p1_name="Player A",
        p2_name="Player B",
        p1_odds=-120,
        p2_odds=100,
        p1_prob=0.61,
        p2_prob=0.39,
        p1_nv=0.56,
        p2_nv=0.44,
        pick_summary="Player A over Player B.\nKey factors: market edge +5.0pp.",
    )
    assert t is not None
    assert "\n" not in t and "\r" not in t
    assert "->" not in t
    assert "—" not in t
