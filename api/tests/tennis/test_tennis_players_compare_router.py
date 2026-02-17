from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.tennis_predictions_today_enhanced import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/tennis")
    return TestClient(app)


def test_players_compare_success(monkeypatch):
    async def _fake_compare(*, player1, player2, tour, surface, event_id, debug, impl):
        return {
            "meta": {"tour": tour, "surface_requested": surface, "as_of": "2026-01-12", "input_names": {"player1": player1, "player2": player2}},
            "players": {"left": {}, "right": {}},
            "compare": {},
            "h2h": {"overall": {"p1_wins": 0, "p2_wins": 0, "total": 0}, "surface": {"p1_wins": 0, "p2_wins": 0, "total": 0}, "record": {"overall": "0-0", "surface": "0-0"}, "last_10_matchups": [], "event_id": event_id},
            "last_10_matches": {"left": [], "right": []},
            "last_10_resolution": {"left": {}, "right": {}},
            "last_10_matchups": [],
            "quality": {},
            "debug": None,
        }

    monkeypatch.setattr(
        "app.routers.tennis_predictions_today_enhanced.get_players_compare_service",
        _fake_compare,
    )
    client = _client()
    resp = client.get("/tennis/players/compare?player1=Rafael%20Nadal&player2=Novak%20Djokovic")
    assert resp.status_code == 200
    data = resp.json()
    assert data["meta"]["input_names"]["player1"] == "Rafael Nadal"


def test_players_compare_validation_error():
    client = _client()
    resp = client.get("/tennis/players/compare?player1=Rafael%20Nadal")
    assert resp.status_code == 422
