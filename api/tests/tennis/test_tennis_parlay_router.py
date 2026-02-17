from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.tennis_predictions_today_enhanced import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/tennis")
    return TestClient(app)


def test_parlay_suggested_success(monkeypatch):
    async def _fake_parlay(
        *,
        legs,
        top_n,
        days_ahead,
        include_incomplete,
        min_parlay_payout,
        min_edge,
        min_ev,
        min_leg_odds,
        candidate_pool,
        max_overlap,
        objective,
        impl,
    ):
        return []

    monkeypatch.setattr(
        "app.routers.tennis_predictions_today_enhanced.get_suggested_parlay_service",
        _fake_parlay,
    )
    client = _client()
    resp = client.get("/tennis/parlay/suggested")
    assert resp.status_code == 200
    assert resp.json() == []


def test_parlay_suggested_validation_error():
    client = _client()
    resp = client.get("/tennis/parlay/suggested?legs=1")
    assert resp.status_code == 422
