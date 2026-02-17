from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.tennis_admin_elo import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_elo_health_success(monkeypatch):
    async def _fake_health(match_date):
        return {
            "as_of": "2026-01-12",
            "counts": {"ok": 5, "missing": 1},
            "rows": [
                {
                    "match_date": "2026-01-12",
                    "tour": "ATP",
                    "tournament_name": "AO",
                    "tournament_round": "R1",
                    "player1_name": "A",
                    "player1_id": 1,
                    "player2_name": "B",
                    "player2_id": 2,
                    "status": "scheduled",
                    "p1_elo": 2000,
                    "p2_elo": 1900,
                    "p1_elo_source": "ta",
                    "p2_elo_source": "ta",
                    "elo_status": "ok",
                }
            ],
        }

    monkeypatch.setattr("app.routers.tennis_admin_elo.get_elo_health", _fake_health)
    client = _client()
    resp = client.get("/tennis/admin/elo-health?match_date=2026-01-12")
    assert resp.status_code == 200
    data = resp.json()
    assert data["as_of"] == "2026-01-12"
    assert "counts" in data
    assert isinstance(data["rows"], list)


def test_elo_health_validation_error():
    client = _client()
    resp = client.get("/tennis/admin/elo-health")
    assert resp.status_code == 422


def test_fix_fixtures_mapping_success(monkeypatch):
    async def _fake_fix(match_date):
        return {
            "match_date": "2026-01-12",
            "rows_updated": 4,
            "before": {"fixtures_on_date": 10, "bad_p1": 2, "bad_p2": 2},
            "after": {"fixtures_on_date": 10, "bad_p1": 0, "bad_p2": 0},
        }

    monkeypatch.setattr("app.routers.tennis_admin_elo.fix_fixtures_mapping_service", _fake_fix)
    client = _client()
    resp = client.post("/tennis/admin/fix-fixtures-mapping?match_date=2026-01-12")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"match_date", "rows_updated", "before", "after"}
    assert data["rows_updated"] == 4


def test_fix_fixtures_mapping_validation_error():
    client = _client()
    resp = client.post("/tennis/admin/fix-fixtures-mapping")
    assert resp.status_code == 422
