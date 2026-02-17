from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.tennis_auto import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/tennis")
    return TestClient(app)


def test_predict_auto_success(monkeypatch):
    async def _fake_predict(req, db):
        return {
            "player1": "A",
            "player2": "B",
            "surface": "hard",
            "match_date": "2026-01-12",
            "winner_side": 1,
            "pick": "player1",
            "p_player1_model": 0.6,
            "p_player2_model": 0.4,
            "p_player1": 0.6,
            "p_player2": 0.4,
            "confidence_tier": "LEAN",
            "edge_pct": 10.0,
            "h2h_p1_wins": 0,
            "h2h_p2_wins": 0,
            "h2h_surface_matches": 0,
            "h2h_applied": False,
            "h2h_adjustment": 0.0,
            "model": "xgb_tennis_ta.json",
            "features": {"d_last5_hold": 0.1},
        }

    monkeypatch.setattr("app.routers.tennis_auto.predict_auto_service", _fake_predict)
    client = _client()
    resp = client.post(
        "/tennis/predict-auto",
        json={"player1": "A", "player2": "B", "surface": "hard"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pick"] == "player1"
    assert "features" in data


def test_predict_auto_validation_error():
    client = _client()
    resp = client.post("/tennis/predict-auto", json={"player1": "A"})
    assert resp.status_code == 422


def test_get_player_stats_success(monkeypatch):
    async def _fake_stats(player, surface, as_of, db):
        return {
            "player": player,
            "as_of": "2026-01-12",
            "snapshots": {"hard": {"last10_hold": 0.7}},
        }

    monkeypatch.setattr("app.routers.tennis_auto.get_player_stats_service", _fake_stats)
    client = _client()
    resp = client.get("/tennis/player/stats?player=Sebastian%20Baez&surface=hard")
    assert resp.status_code == 200
    data = resp.json()
    assert data["player"] == "Sebastian Baez"
    assert "snapshots" in data
