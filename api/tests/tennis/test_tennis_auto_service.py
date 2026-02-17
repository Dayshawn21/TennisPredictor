from __future__ import annotations

import asyncio
from datetime import date

from fastapi import HTTPException

from app.schemas.tennis_auto import TennisPredictAutoRequest
from app.services.tennis import auto_prediction_service as svc


class _FakeDb:
    pass


def test_predict_auto_player_not_found(monkeypatch):
    async def _fake_find_player_id(db, name):
        return None

    monkeypatch.setattr(svc, "find_player_id", _fake_find_player_id)

    req = TennisPredictAutoRequest(player1="A", player2="B", surface="hard")
    try:
        asyncio.run(svc.predict_auto(req=req, db=_FakeDb()))
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 404
        assert "Player not found in tennis_players" in exc.detail


def test_predict_auto_snapshot_missing(monkeypatch):
    async def _fake_find_player_id(db, name):
        return 1 if name == "A" else 2

    async def _fake_load_snapshot(db, player_id, surface, as_of):
        return None

    monkeypatch.setattr(svc, "find_player_id", _fake_find_player_id)
    monkeypatch.setattr(svc, "load_snapshot", _fake_load_snapshot)

    req = TennisPredictAutoRequest(player1="A", player2="B", surface="hard")
    try:
        asyncio.run(svc.predict_auto(req=req, db=_FakeDb()))
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 404
        assert "No snapshot found for player_id=1" in exc.detail


def test_predict_auto_model_failure(monkeypatch):
    async def _fake_find_player_id(db, name):
        return 1 if name == "A" else 2

    async def _fake_load_snapshot(db, player_id, surface, as_of):
        return {
            "last5_hold": 1.0,
            "last5_break": 1.0,
            "last10_hold": 1.0,
            "last10_break": 1.0,
            "surf_last10_hold": 1.0,
            "surf_last10_break": 1.0,
            "last10_aces_pg": 1.0,
            "surf_last10_aces_pg": 1.0,
            "last10_df_pg": 1.0,
            "surf_last10_df_pg": 1.0,
            "last10_tb_match_rate": 1.0,
            "last10_tb_win_pct": 1.0,
            "surf_last10_tb_match_rate": 1.0,
            "surf_last10_tb_win_pct": 1.0,
        }

    async def _fake_h2h(**kwargs):
        return 0, 0, 0

    class _BadPredictor:
        def predict_proba(self, features):
            raise RuntimeError("boom")

    monkeypatch.setattr(svc, "find_player_id", _fake_find_player_id)
    monkeypatch.setattr(svc, "load_snapshot", _fake_load_snapshot)
    monkeypatch.setattr(svc, "get_h2h_from_db", _fake_h2h)
    monkeypatch.setattr(svc, "get_default_tennis_predictor", lambda: _BadPredictor())

    req = TennisPredictAutoRequest(player1="A", player2="B", surface="hard")
    try:
        asyncio.run(svc.predict_auto(req=req, db=_FakeDb()))
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 500
        assert exc.detail["message"] == "Model prediction failed."


def test_get_player_stats_surface_success(monkeypatch):
    async def _fake_find_player_id(db, name):
        return 7

    async def _fake_load_snapshot_by_surface(db, player_id, surface, as_of):
        return {"last10_hold": 0.72}

    monkeypatch.setattr(svc, "find_player_id", _fake_find_player_id)
    monkeypatch.setattr(svc, "load_snapshot_by_surface", _fake_load_snapshot_by_surface)

    result = asyncio.run(svc.get_player_stats(player="Sebastian Baez", surface="hard", as_of=date(2026, 1, 12), db=_FakeDb()))
    assert result.player == "Sebastian Baez"
    assert "hard" in result.snapshots
