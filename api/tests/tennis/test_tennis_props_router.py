from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.tennis_props import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/tennis")
    return TestClient(app)


def test_props_aces_success_shape():
    client = _client()
    payload = {
        "player": "player1",
        "prop_type": "aces",
        "line": 8.5,
        "expected_games": 22.0,
        "p_last10_aces_pg": 0.34,
        "p_surf_last10_aces_pg": 0.36,
        "opp_last10_aces_allowed_pg": 0.29,
        "opp_surf_last10_aces_allowed_pg": 0.31,
    }
    resp = client.post("/tennis/props", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"player", "prop_type", "line", "expected", "p_over", "p_under", "confidence_tier"}
    assert data["prop_type"] == "aces"


def test_props_break_points_requires_expected_return_games():
    client = _client()
    payload = {
        "player": "player2",
        "prop_type": "break_points_won",
        "line": 3.5,
        "expected_games": 22.0,
        "p_last10_bp_won_prg": 0.12,
        "p_surf_last10_bp_won_prg": 0.14,
        "opp_last10_bp_won_allowed_psg": 0.10,
        "opp_surf_last10_bp_won_allowed_psg": 0.11,
    }
    resp = client.post("/tennis/props", json=payload)
    assert resp.status_code == 422
    assert resp.json()["detail"] == "break_points_won requires expected_return_games"


def test_props_pydantic_validation_error():
    client = _client()
    resp = client.post("/tennis/props", json={"player": "player1"})
    assert resp.status_code == 422
