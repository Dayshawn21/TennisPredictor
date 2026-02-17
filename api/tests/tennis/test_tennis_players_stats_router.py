from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.tennis_predictions_today_enhanced import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/tennis")
    return TestClient(app)


def test_players_stats_success(monkeypatch):
    async def _fake_stats(*, player, tour, surface, debug, impl):
        return {"player": player, "tour": tour, "surface_requested": surface, "match_status": {}, "resolution": {}}

    monkeypatch.setattr(
        "app.routers.tennis_predictions_today_enhanced.get_player_full_stats_service",
        _fake_stats,
    )
    client = _client()
    resp = client.get("/tennis/players/stats?player=Carlos%20Alcaraz&tour=ATP&surface=hard")
    assert resp.status_code == 200
    data = resp.json()
    assert data["player"] == "Carlos Alcaraz"


def test_players_stats_validation_error():
    client = _client()
    resp = client.get("/tennis/players/stats")
    assert resp.status_code == 422
