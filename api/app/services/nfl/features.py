from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from .models_db import NFLPlayerStats, NFLTeamCtx, NFLOddsProp

# recent averages for passing yards & attempts (last N games)
async def recent_qb_form(player_id: int, db: AsyncSession, n: int = 4):
    # First get the last N games, then average them
    subq = select(NFLPlayerStats).where(
        NFLPlayerStats.player_id == player_id
    ).order_by(desc(NFLPlayerStats.game_id)).limit(n).subquery()
    
    q = await db.execute(
        select(
            func.avg(subq.c.pass_yds).label("avg_pass_yds"),
            func.avg(subq.c.attempts_pass).label("avg_att")
        )
    )
    row = q.fetchone()
    if not row or row[0] is None:
        return None
    return {
        "avg_pass_yds": float(row.avg_pass_yds or 0.0),
        "avg_att": float(row.avg_att or 0.0),
    }

# opponent context (most recent as_of row)
async def opponent_pass_allowed_pg(opp_team_id: int, db: AsyncSession):
    q = await db.execute(
        select(NFLTeamCtx).where(NFLTeamCtx.team_id == opp_team_id)
                          .order_by(desc(NFLTeamCtx.as_of))
                          .limit(1)
    )
    ctx = q.scalar_one_or_none()
    if not ctx:
        return 230.0  # fallback league-ish average
    return float(ctx.pass_yds_allowed_pg or 230.0)

# recent RB form (rushing yards & attempts)
async def recent_rb_form(player_id: int, db: AsyncSession, n: int = 4):
    # First get the last N games, then average them
    subq = select(NFLPlayerStats).where(
        NFLPlayerStats.player_id == player_id
    ).order_by(desc(NFLPlayerStats.game_id)).limit(n).subquery()
    
    q = await db.execute(
        select(
            func.avg(subq.c.rush_yds).label("avg_rush_yds"),
            func.avg(subq.c.attempts_rush).label("avg_att")
        )
    )
    row = q.fetchone()
    if not row or row[0] is None:
        return None
    return {
        "avg_rush_yds": float(row.avg_rush_yds or 0.0),
        "avg_att": float(row.avg_att or 0.0),
    }

# opponent rush defense
async def opponent_rush_allowed_pg(opp_team_id: int, db: AsyncSession):
    q = await db.execute(
        select(NFLTeamCtx).where(NFLTeamCtx.team_id == opp_team_id)
                          .order_by(desc(NFLTeamCtx.as_of))
                          .limit(1)
    )
    ctx = q.scalar_one_or_none()
    if not ctx:
        return 115.0  # fallback league-ish average
    return float(ctx.rush_yds_allowed_pg or 115.0)

# recent WR form (receiving yards, receptions, targets)
async def recent_wr_form(player_id: int, db: AsyncSession, n: int = 4):
    # First get the last N games, then average them
    subq = select(NFLPlayerStats).where(
        NFLPlayerStats.player_id == player_id
    ).order_by(desc(NFLPlayerStats.game_id)).limit(n).subquery()
    
    q = await db.execute(
        select(
            func.avg(subq.c.rec_yds).label("avg_rec_yds"),
            func.avg(subq.c.receptions).label("avg_receptions"),
            func.avg(subq.c.targets).label("avg_targets")
        )
    )
    row = q.fetchone()
    if not row or row[0] is None:
        return None
    return {
        "avg_rec_yds": float(row.avg_rec_yds or 0.0),
        "avg_receptions": float(row.avg_receptions or 0.0),
        "avg_targets": float(row.avg_targets or 0.0),
    }


async def get_odds_by_prop_id(prop_id: str, db: AsyncSession):
    q = await db.execute(select(NFLOddsProp).where(NFLOddsProp.prop_id == prop_id))
    return q.scalar_one_or_none()
