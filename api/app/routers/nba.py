from fastapi import APIRouter, Depends, Query
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.db_session import get_db
from ..schemas import PropReq, Rec
from ..services.nba.model import score_points_stub
from ..services.nba.moneyline import fetch_games_with_odds, score_moneyline_for_game

router = APIRouter(prefix="/nba", tags=["nba"])

@router.post("/props/recommendations", response_model=List[Rec])
async def recommendations(props: List[PropReq], db: AsyncSession = Depends(get_db)) -> List[Rec]:
    recs: List[Rec] = []
    for p in props:
        ev_stat, mode_k, pi68, pi90, p_over, ev, kelly, conf, pmf = score_points_stub(p.line, p.american_odds)

        pmf_list = [(int(k), float(v)) for k, v in pmf.items()]

        conf_labels = ["Very Low", "Low", "Below Avg", "Average", "Above Avg", "Good", "High", "Very High", "Elite", "Max"]
        conf_str = conf_labels[conf - 1] if 1 <= conf <= 10 else "Average"

        recs.append(Rec(
            prop_id=p.prop_id,
            label=f"{p.player} Points O/U {p.line}",
            predicted_value=round(ev_stat, 1),
            mode_exact=int(mode_k),
            pi68=tuple(pi68),
            pi90=tuple(pi90),
            p_over_line=round(p_over, 3),
            ev_per_dollar=round(ev, 4),
            kelly=round(kelly, 4),
            confidence=conf_str,
            pmf_window=pmf_list,
            american_odds=p.american_odds,
            line=p.line,
            reasons=["Recent form", "Opp vs-position", "Pace context"],
        ))
    return recs

@router.get("/games/moneyline")
async def moneyline_picks(
    date: str = Query(..., description="UTC date YYYY-MM-DD"),
    book: Optional[str] = Query(None, description="Sportsbook name filter"),
    db: AsyncSession = Depends(get_db),
):
    pairs = await fetch_games_with_odds(date, book, db)
    out = []
    for g, o in pairs:
        scored = await score_moneyline_for_game(g, o, db)
        if scored:
            out.append(scored)
    out.sort(key=lambda x: x["edge"], reverse=True)
    return out
