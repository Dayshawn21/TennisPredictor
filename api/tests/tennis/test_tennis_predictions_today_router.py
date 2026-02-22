from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.tennis_predictions_today_enhanced import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/tennis")
    return TestClient(app)


def test_predictions_today_enhanced_success(monkeypatch):
    async def _fake_enhanced(*, days_ahead, include_incomplete, bust_cache, impl, **kwargs):
        return {
            "as_of": "2026-01-12",
            "source": "tennis",
            "cached": False,
            "count": 1,
            "items": [
                {
                    "match_id": "1",
                    "inputs": {"tweet_text": "Player A over Player B (-120) | Model 66.0%"},
                }
            ],
        }

    monkeypatch.setattr(
        "app.routers.tennis_predictions_today_enhanced.get_predictions_today_enhanced_service",
        _fake_enhanced,
    )
    client = _client()
    resp = client.get("/tennis/predictions/today/enhanced")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"as_of", "source", "cached", "count", "items"}
    assert data["items"][0]["inputs"]["tweet_text"].startswith("Player A over Player B")


def test_predictions_today_success(monkeypatch):
    async def _fake_today(*, days_ahead, include_incomplete, bust_cache, impl):
        return {
            "as_of": "2026-01-12",
            "source": "tennis",
            "cached": False,
            "count": 1,
            "items": [],
        }

    monkeypatch.setattr(
        "app.routers.tennis_predictions_today_enhanced.get_predictions_today_service",
        _fake_today,
    )
    client = _client()
    resp = client.get("/tennis/predictions/today")
    assert resp.status_code == 200
    assert resp.json()["count"] == 1
