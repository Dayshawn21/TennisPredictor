from fastapi import APIRouter, Depends, HTTPException
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from app.db_session import get_db
from ..schemas import PropReq, Rec
from ..services.nfl.model import (
    score_passing_yards, score_rushing_yards, score_receiving_yards, score_receptions
)

router = APIRouter()

def _infer_market(p: PropReq) -> str:
    if p.market: 
        return p.market.upper()
    pos = (p.position or "").upper()
    if pos == "QB": return "PLAYER_PASS_YDS"
    if pos in ("RB","FB"): return "PLAYER_RUSH_YDS"
    if pos in ("WR","TE"): return "PLAYER_REC_YDS"
    return "PLAYER_PASS_YDS"  # default

def _label_for(market: str, p: PropReq) -> str:
    lbl = {
        "PLAYER_PASS_YDS": "Passing Yards",
        "PLAYER_RUSH_YDS": "Rushing Yards",
        "PLAYER_REC_YDS":  "Receiving Yards",
        "PLAYER_RECEPTIONS": "Receptions"
    }.get(market, "Stat")
    unit = "" if market == "PLAYER_RECEPTIONS" else " "
    return f"{p.player} {lbl} O/U{unit}{p.line}"

@router.post("/props/recommendations", response_model=List[Rec])
async def recommendations(props: List[PropReq], db: AsyncSession = Depends(get_db)):
    recs: List[Rec] = []
    for p in props:
        # auto-fill from odds table if missing
        if (p.market is None) or (p.line is None) or (p.american_odds is None) or (p.game_id is None):
            odds = await get_odds_by_prop_id(p.prop_id, db)
            if odds:
                p.market = p.market or odds.market
                p.line = p.line if p.line is not None else float(odds.line)
                p.american_odds = p.american_odds if p.american_odds is not None else int(odds.american_odds)
                p.game_id = p.game_id or int(odds.game_id)
                p.player_id = p.player_id or int(odds.player_id)

        market = (p.market or "").upper()
        if market in ("", "PLAYER_PASS_YDS",) or (market == "" and (p.position or "").upper() == "QB"):
            result = await score_passing_yards(p, db)
            label = f"{p.player} Passing Yards O/U {p.line}"
        elif market == "PLAYER_RUSH_YDS":
            result = await score_rushing_yards(p, db)
            label = f"{p.player} Rushing Yards O/U {p.line}"
        elif market == "PLAYER_REC_YDS":
            result = await score_receiving_yards(p, db)
            label = f"{p.player} Receiving Yards O/U {p.line}"
        elif market == "PLAYER_RECEPTIONS":
            result = await score_receptions(p, db)
            label = f"{p.player} Receptions O/U {p.line}"
        else:
            # fallback: treat unknown as passing yards
            result = await score_passing_yards(p, db)
            label = f"{p.player} Passing Yards O/U {p.line}"

        if not result:
            continue
        ev_stat, mode_k, pi68, pi90, p_over, ev, kelly, conf, pmf = result
        recs.append(Rec(
            prop_id=p.prop_id, label=label,
            predicted_value=round(ev_stat, 1), mode_exact=int(mode_k),
            pi68=pi68, pi90=pi90, p_over_line=round(p_over, 3),
            ev_per_dollar=round(ev, 4), kelly=round(kelly, 4),
            confidence=conf, pmf_window=pmf,
            american_odds=p.american_odds or 0, line=p.line or 0.0,
            reasons=["Auto-filled odds", "Recent form", "Opponent context"]
        ))
    return recs