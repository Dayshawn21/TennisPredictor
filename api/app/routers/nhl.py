from fastapi import APIRouter, Depends
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from app.db_session import get_db
from ..schemas import PropReq, Rec
from ..services.nhl.model import score_sog_stub

router = APIRouter()

@router.post("/props/recommendations", response_model=List[Rec])
async def recommendations(props: List[PropReq], db: AsyncSession = Depends(get_db)):
    # Assume "line" is SOG line for now; you'll branch by market later
    confidence_labels = ["Very Low", "Low", "Below Avg", "Average", "Above Avg", "Good", "High", "Very High", "Elite", "Max"]
    recs: List[Rec] = []
    for p in props:
        ev_stat, mode_k, pi68, pi90, p_over, ev, kelly, conf, pmf = score_sog_stub(p.line, p.american_odds)
        # Convert pmf dict to list of tuples
        pmf_list = list(pmf.items())
        # Convert confidence int to string label
        conf_str = confidence_labels[conf - 1] if 1 <= conf <= 10 else "Average"
        recs.append(Rec(
            prop_id=p.prop_id,
            label=f"{p.player} SOG O/U {p.line}",
            predicted_value=round(ev_stat,1),
            mode_exact=int(mode_k),
            pi68=pi68, pi90=pi90,
            p_over_line=round(p_over,3),
            ev_per_dollar=round(ev,4),
            kelly=round(kelly,4),
            confidence=conf_str,
            pmf_window=pmf_list,
            american_odds=p.american_odds,
            line=p.line,
            reasons=["TOI + line assignment","Opp shot suppression","PP usage"]
        ))
    return recs
