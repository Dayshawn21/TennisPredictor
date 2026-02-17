from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.tennis_auto_repo import (
    find_player_id,
    get_h2h_from_db,
    load_snapshot,
    load_snapshot_by_surface,
)
from app.models.tennis_predictor import FEATURE_COLS, get_default_tennis_predictor
from app.schemas.tennis_auto import (
    TennisPlayerStatsResponse,
    TennisPredictAutoRequest,
    TennisPredictAutoResponse,
)


def confidence_tier(p_pick: float) -> str:
    if p_pick >= 0.70:
        return "LOCK ðŸ”¥"
    if p_pick >= 0.62:
        return "STRONG"
    if p_pick >= 0.55:
        return "LEAN"
    return "PASS"


def normalize_surface(raw: str) -> str:
    s = raw.strip().lower()
    if s in {"hard", "hardcourt", "outdoor hard", "indoor hard", "indoor", "carpet"}:
        return "hard"
    if s in {"clay", "red clay", "green clay"}:
        return "clay"
    if s in {"grass"}:
        return "grass"
    return "hard"


def apply_h2h_adjustment(
    p_player1: float,
    p1_wins: int,
    p2_wins: int,
    surface_matches: Optional[int] = None,
) -> Tuple[float, bool, float]:
    total = p1_wins + p2_wins
    if total < 3:
        return p_player1, False, 0.0
    if surface_matches is not None and surface_matches < 2:
        return p_player1, False, 0.0

    edge = (p1_wins - p2_wins) / total
    delta = 0.03 * edge
    p_adj = p_player1 + delta
    p_adj = min(0.95, max(0.05, p_adj))
    return p_adj, True, float(p_adj - p_player1)


def build_diff_features(p1: Dict[str, Any], p2: Dict[str, Any]) -> Dict[str, float]:
    mapping = {
        "d_last5_hold": ("last5_hold", "last5_hold"),
        "d_last5_break": ("last5_break", "last5_break"),
        "d_last10_hold": ("last10_hold", "last10_hold"),
        "d_last10_break": ("last10_break", "last10_break"),
        "d_surf_last10_hold": ("surf_last10_hold", "surf_last10_hold"),
        "d_surf_last10_break": ("surf_last10_break", "surf_last10_break"),
        "d_last10_aces_pg": ("last10_aces_pg", "last10_aces_pg"),
        "d_surf_last10_aces_pg": ("surf_last10_aces_pg", "surf_last10_aces_pg"),
        "d_last10_df_pg": ("last10_df_pg", "last10_df_pg"),
        "d_surf_last10_df_pg": ("surf_last10_df_pg", "surf_last10_df_pg"),
        "d_last10_tb_match_rate": ("last10_tb_match_rate", "last10_tb_match_rate"),
        "d_last10_tb_win_pct": ("last10_tb_win_pct", "last10_tb_win_pct"),
        "d_surf_last10_tb_match_rate": ("surf_last10_tb_match_rate", "surf_last10_tb_match_rate"),
        "d_surf_last10_tb_win_pct": ("surf_last10_tb_win_pct", "surf_last10_tb_win_pct"),
    }
    out: Dict[str, float] = {}
    for feat in FEATURE_COLS:
        k1, k2 = mapping[feat]
        v1 = p1.get(k1)
        v2 = p2.get(k2)
        out[feat] = (float(v1) if v1 is not None else 0.0) - (float(v2) if v2 is not None else 0.0)
    return out


async def predict_auto(req: TennisPredictAutoRequest, db: AsyncSession) -> TennisPredictAutoResponse:
    as_of = req.match_date or date.today()
    surface = normalize_surface(req.surface)

    p1_canonical_id = await find_player_id(db, req.player1)
    p2_canonical_id = await find_player_id(db, req.player2)
    if p1_canonical_id is None:
        raise HTTPException(status_code=404, detail=f"Player not found in tennis_players: '{req.player1}'. Add them to tennis_players first.")
    if p2_canonical_id is None:
        raise HTTPException(status_code=404, detail=f"Player not found in tennis_players: '{req.player2}'. Add them to tennis_players first.")

    p1_snap = await load_snapshot(db, p1_canonical_id, surface, as_of)
    p2_snap = await load_snapshot(db, p2_canonical_id, surface, as_of)
    if p1_snap is None:
        raise HTTPException(status_code=404, detail=f"No snapshot found for player_id={p1_canonical_id} surface='{surface}' as_of<={as_of}.")
    if p2_snap is None:
        raise HTTPException(status_code=404, detail=f"No snapshot found for player_id={p2_canonical_id} surface='{surface}' as_of<={as_of}.")

    features = build_diff_features(p1_snap, p2_snap)
    predictor = get_default_tennis_predictor()
    try:
        p_player1_model = float(predictor.predict_proba(features))
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": "Model prediction failed.", "error": str(e)})

    p_player1_model = max(0.0, min(1.0, p_player1_model))
    p_player2_model = 1.0 - p_player1_model

    p1_wins, p2_wins, surf_matches = await get_h2h_from_db(
        db=db,
        p1_canonical_id=p1_canonical_id,
        p2_canonical_id=p2_canonical_id,
        surface=surface,
        as_of=as_of,
    )

    p_player1 = p_player1_model
    h2h_applied = False
    h2h_delta = 0.0
    if (p1_wins + p2_wins) > 0:
        p_player1, h2h_applied, h2h_delta = apply_h2h_adjustment(
            p_player1=p_player1_model,
            p1_wins=p1_wins,
            p2_wins=p2_wins,
            surface_matches=surf_matches,
        )
    p_player2 = 1.0 - p_player1

    winner_side = 1 if p_player1 >= 0.5 else 2
    pick = f"player{winner_side}"
    p_pick = p_player1 if winner_side == 1 else p_player2
    conf = confidence_tier(p_pick)
    edge_pct = round((p_pick - 0.50) * 100.0, 2)

    return TennisPredictAutoResponse(
        player1=req.player1,
        player2=req.player2,
        surface=surface,
        match_date=as_of,
        winner_side=winner_side,
        pick=pick,
        p_player1_model=round(p_player1_model, 4),
        p_player2_model=round(p_player2_model, 4),
        p_player1=round(p_player1, 4),
        p_player2=round(p_player2, 4),
        confidence_tier=conf,
        edge_pct=edge_pct,
        h2h_p1_wins=p1_wins,
        h2h_p2_wins=p2_wins,
        h2h_surface_matches=surf_matches,
        h2h_applied=h2h_applied,
        h2h_adjustment=round(h2h_delta, 4),
        model="xgb_tennis_ta.json",
        features=features,
    )


async def get_player_stats(
    player: str,
    surface: Optional[str],
    as_of: Optional[date],
    db: AsyncSession,
) -> TennisPlayerStatsResponse:
    as_of_date = as_of or date.today()
    player_id = await find_player_id(db, player)
    if player_id is None:
        raise HTTPException(status_code=404, detail=f"Player not found in tennis_players: '{player}'. Add them to tennis_players first.")

    if surface:
        s = normalize_surface(surface)
        snap = await load_snapshot_by_surface(db, player_id, s, as_of_date)
        if not snap:
            return TennisPlayerStatsResponse(player=player, as_of=as_of_date, snapshots={s: {}})
        clean = {k: (float(v) if v is not None else None) for k, v in snap.items()}
        return TennisPlayerStatsResponse(player=player, as_of=as_of_date, snapshots={s: clean})

    surfaces = ["hard", "clay", "grass"]
    snapshots: Dict[str, Dict[str, Optional[float]]] = {}
    for s in surfaces:
        snap = await load_snapshot_by_surface(db, player_id, s, as_of_date)
        snapshots[s] = {k: (float(v) if v is not None else None) for k, v in snap.items()} if snap else {}
    return TennisPlayerStatsResponse(player=player, as_of=as_of_date, snapshots=snapshots)
