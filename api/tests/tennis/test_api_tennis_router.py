from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.api_tennis_router import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_ingest_fixtures_success(monkeypatch):
    async def _fake_ingest(start, end):
        return {"start": "2026-01-10", "end": "2026-01-11", "counts": {"ATP": 1, "WTA": 2, "TOTAL": 3}}

    monkeypatch.setattr("app.routers.api_tennis_router.ingest_fixtures_service", _fake_ingest)
    client = _client()
    resp = client.post("/tennis/api-tennis/fixtures/ingest?start=2026-01-10&end=2026-01-11")
    assert resp.status_code == 200
    data = resp.json()
    assert data["start"] == "2026-01-10"
    assert data["end"] == "2026-01-11"
    assert data["counts"]["TOTAL"] == 3


def test_ingest_fixtures_validation_error():
    client = _client()
    resp = client.post("/tennis/api-tennis/fixtures/ingest")
    assert resp.status_code == 422


def test_ingest_fixtures_daily_success(monkeypatch):
    async def _fake_daily(forward_days, backfill_days):
        return {
            "today": "2026-01-12",
            "upcoming": {"start": "2026-01-12", "end": "2026-01-19", "counts": {"ATP": 2, "WTA": 1, "TOTAL": 3}},
            "backfill": {"start": "2026-01-05", "end": "2026-01-11", "counts": {"ATP": 1, "WTA": 1, "TOTAL": 2}},
            "combined_counts": {"ATP": 3, "WTA": 2, "TOTAL": 5},
        }

    monkeypatch.setattr("app.routers.api_tennis_router.ingest_fixtures_daily_service", _fake_daily)
    client = _client()
    resp = client.post("/tennis/api-tennis/fixtures/ingest-daily?forward_days=7&backfill_days=7")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"today", "upcoming", "backfill", "combined_counts"}
    assert data["combined_counts"]["TOTAL"] == 5


def test_ingest_fixtures_daily_validation_error():
    client = _client()
    resp = client.post("/tennis/api-tennis/fixtures/ingest-daily?forward_days=31")
    assert resp.status_code == 422
