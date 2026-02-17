from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from .models_db import NBATeamCtx, NBAGame
from ...core.odds import american_to_prob, prob_to_american, devig_two_way

HOME_COURT_ELO = 2.5   # points-ish advantage to home team (tunable)
K_SCALE = 6.5          # logistic slope (bigger -> steeper curve)

@dataclass
class GameOdds:
    game_id: int
    book: str
    moneyline_home: int
    moneyline_away: int
    spread_home: float | None
    total_points: float | None

async def fetch_games_with_odds(date_utc: str, book: str | None, db: AsyncSession) -> list[tuple[NBAGame, GameOdds]]:
    # Pull games on that date
    gq = await db.execute(
        select(NBAGame).where(func.date(NBAGame.date_utc) == date_utc)
    )
    games = gq.scalars().all()
    if not games:
        return []

    # For simplicity, grab the latest odds row per game/book using a subquery
    results: list[tuple[NBAGame, GameOdds]] = []
    for g in games:
        # latest odds for this game (either any book or the chosen book)
        if book:
            sql = """
            SELECT game_id, book, moneyline_home, moneyline_away, spread_home, total_points
            FROM nba_game_odds
            WHERE game_id = %s AND book = %s
            ORDER BY ts DESC
            LIMIT 1
            """
            params = (g.game_id, book)
        else:
            sql = """
            SELECT game_id, book, moneyline_home, moneyline_away, spread_home, total_points
            FROM nba_game_odds
            WHERE game_id = %s
            ORDER BY ts DESC
            LIMIT 1
            """
            params = (g.game_id,)

        rows = (await db.execute(sql, params)).all()
        if not rows:
            continue
        r = rows[0]
        results.append((
            g,
            GameOdds(
                game_id=int(r[0]), book=str(r[1]),
                moneyline_home=int(r[2]), moneyline_away=int(r[3]),
                spread_home=float(r[4]) if r[4] is not None else None,
                total_points=float(r[5]) if r[5] is not None else None
            )
        ))
    return results

async def latest_team_ctx(team_id: int, db: AsyncSession) -> dict | None:
    q = await db.execute(
        select(NBATeamCtx).where(NBATeamCtx.team_id == team_id).order_by(desc(NBATeamCtx.as_of)).limit(1)
    )
    ctx = q.scalar_one_or_none()
    if not ctx:
        return None
    return {
        "off": float(ctx.off_rating_l10 or 114.0),
        "def": float(ctx.def_rating_l10 or 112.0),
        "pace": float(ctx.pace_l10 or 99.5),
    }

def logistic_win_prob(point_diff: float) -> float:
    # Convert expected point diff to win prob via logistic curve
    # probability(home wins) = 1 / (1 + exp(-diff / K_SCALE))
    import math
    return 1.0 / (1.0 + math.exp(-point_diff / K_SCALE))

def blend_probs(model_p: float, market_p: float, w_market: float = 0.35) -> float:
    # Simple blend: use some weight for market (contains injury/news)
    return (1 - w_market) * model_p + w_market * market_p

def edge_confidence(edge_bp: float) -> int:
    # basis points (e.g., 0.03 = 3%), map to 1..10 roughly
    return max(1, min(10, int(5 + (edge_bp / 0.02))))

async def score_moneyline_for_game(game: NBAGame, odds: GameOdds, db: AsyncSession):
    # team contexts
    home_ctx = await latest_team_ctx(game.home_team_id, db)
    away_ctx = await latest_team_ctx(game.away_team_id, db)
    if not home_ctx or not away_ctx:
        return None

    # Simple "rating": OFF - DEF
    home_rating = home_ctx["off"] - home_ctx["def"]
    away_rating = away_ctx["off"] - away_ctx["def"]

    # expected point diff (home positive), add home-court
    point_diff = (home_rating - away_rating) + HOME_COURT_ELO

    model_home_p = logistic_win_prob(point_diff)

    # market implied (de-vigged)
    p_home_raw = american_to_prob(odds.moneyline_home)
    p_away_raw = american_to_prob(odds.moneyline_away)
    p_home_mkt, p_away_mkt = devig_two_way(p_home_raw, p_away_raw)

    # blended fair
    p_home_fair = blend_probs(model_home_p, p_home_mkt, w_market=0.35)
    p_away_fair = 1.0 - p_home_fair

    # fair odds
    home_fair_american = prob_to_american(p_home_fair)
    away_fair_american = prob_to_american(p_away_fair)

    # edges vs book
    edge_home = p_home_fair - p_home_mkt
    edge_away = p_away_fair - p_away_mkt

    # choose side with bigger positive edge
    if edge_home >= edge_away:
        pick = "HOME"
        rec_prob = p_home_fair
        book_odds = odds.moneyline_home
        fair_odds = home_fair_american
        edge = edge_home
    else:
        pick = "AWAY"
        rec_prob = p_away_fair
        book_odds = odds.moneyline_away
        fair_odds = away_fair_american
        edge = edge_away

    conf = edge_confidence(edge)
    return {
        "game_id": game.game_id,
        "book": odds.book,
        "home_team_id": game.home_team_id,
        "away_team_id": game.away_team_id,
        "moneyline_home": odds.moneyline_home,
        "moneyline_away": odds.moneyline_away,
        "model_home_prob": round(model_home_p, 4),
        "market_home_prob": round(p_home_mkt, 4),
        "blended_home_prob": round(p_home_fair, 4),
        "pick": pick,
        "pick_probability": round(rec_prob, 4),
        "fair_odds": fair_odds,
        "edge": round(edge, 4),
        "confidence": conf,
        "spread_home": odds.spread_home,
        "total_points": odds.total_points
    }
