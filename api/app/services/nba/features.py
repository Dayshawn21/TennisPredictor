from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from .models_db import NBATeamCtx, NBAOppVsPos, NBAPlayerBox

# last N games averages for a player
async def get_player_recent_stats(player_id: int, db: AsyncSession, n: int = 5):
    q = await db.execute(
        select(
            func.avg(NBAPlayerBox.pts).label("avg_pts"),
            func.avg(NBAPlayerBox.usage_rate).label("avg_usage"),
            func.avg(NBAPlayerBox.minutes).label("avg_minutes")
        )
        .where(NBAPlayerBox.player_id == player_id)
        .order_by(NBAPlayerBox.game_id.desc())
        .limit(n)
    )
    row = q.fetchone()
    if not row or row[0] is None:
        return None
    return {"avg_points": float(row.avg_pts), "avg_usage": float(row.avg_usage or 0), "avg_minutes": float(row.avg_minutes or 0)}

# opponent defense vs position (as_of = today fallback to most recent)
async def get_opp_defense(opp_team_id: int, position: str, db: AsyncSession):
    q = await db.execute(
        select(NBAOppVsPos).where(
            NBAOppVsPos.team_id == opp_team_id,
            NBAOppVsPos.position == position
        ).order_by(NBAOppVsPos.as_of.desc()).limit(1)
    )
    opp = q.scalar_one_or_none()
    if not opp:
        return None
    return {
        "allowed_pts_l10": float(opp.allowed_pts_l10 or 0),
        "allowed_reb_l10": float(opp.allowed_reb_l10 or 0),
        "allowed_ast_l10": float(opp.allowed_ast_l10 or 0),
        "allowed_3pm_l10": float(opp.allowed_3pm_l10 or 0),
    }

# team context (pace/defense) — use most recent row
async def get_team_context(team_id: int, db: AsyncSession):
    q = await db.execute(
        select(NBATeamCtx).where(NBATeamCtx.team_id == team_id).order_by(NBATeamCtx.as_of.desc()).limit(1)
    )
    ctx = q.scalar_one_or_none()
    if not ctx:
        return None
    return {
        "pace_l10": float(ctx.pace_l10 or 100.0),
        "def_rating_l10": float(ctx.def_rating_l10 or 112.0),
        "off_rating_l10": float(ctx.off_rating_l10 or 114.0),
    }
