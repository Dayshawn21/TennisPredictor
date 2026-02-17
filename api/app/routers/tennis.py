# app/api/routers/tennis_predictions_today.py
from __future__ import annotations

import datetime as dt
from datetime import date, timedelta
from itertools import combinations
from typing import Optional

from fastapi import APIRouter, Query
from sqlalchemy import text
from pydantic import BaseModel, Field

from app.db_session import engine
from app.schemas.tennis_predictions import EloPredictionsResponse
from app.routers.tennis_predictions_today_enhanced import predictions_today

router = APIRouter(tags=["Tennis"])


class OddsRow(BaseModel):
    match_id: str
    match_date: str
    tour: str | None = None
    tournament: str | None = None
    round: str | None = None
    surface: str | None = None
    p1_name: str | None = None
    p2_name: str | None = None
    p1_odds_american: int | None = None
    p2_odds_american: int | None = None
    odds_fetched_at: str | None = None


# -----------------------------
# ✅ Legacy route (optional)
# -----------------------------
@router.get(
    "/predictions/today/legacy",
    response_model=EloPredictionsResponse,
    deprecated=True,
)
async def predictions_today_legacy(
    days_ahead: int = Query(0, ge=0, le=14),
    include_incomplete: bool = Query(True),
    bust_cache: bool = Query(False),
):
    """
    Deprecated legacy route kept for backwards compatibility.
    Canonical route is now GET /predictions/today (served by enhanced router).
    """
    return await predictions_today_enhanced(
        days_ahead=days_ahead,
        include_incomplete=include_incomplete,
        bust_cache=bust_cache,
    )


@router.get("/odds/today", response_model=list[OddsRow])
async def odds_today(days_ahead: int = Query(0, ge=0, le=14)):
    sql = text(
        """
        select
          match_id,
          match_date,
          upper(tour) as tour,
          tournament,
          "round" as round,
          surface,
          p1_name,
          p2_name,
          p1_odds_american,
          p2_odds_american,
          odds_fetched_at
        from tennis_matches
        where match_date = any(:dates)
          and upper(tour) in ('ATP','WTA')
        order by match_date, tournament, p1_name;
        """
    )

    today = date.today()
    dates = [today + timedelta(days=i) for i in range(days_ahead + 1)]

    async with engine.begin() as conn:
        res = await conn.execute(sql, {"dates": dates})
        rows = res.mappings().all()

    out: list[OddsRow] = []
    for r in rows:
        out.append(
            OddsRow(
                match_id=str(r["match_id"]),
                match_date=r["match_date"].isoformat(),
                tour=r.get("tour"),
                tournament=r.get("tournament"),
                round=r.get("round"),
                surface=r.get("surface"),
                p1_name=r.get("p1_name"),
                p2_name=r.get("p2_name"),
                p1_odds_american=r.get("p1_odds_american"),
                p2_odds_american=r.get("p2_odds_american"),
                odds_fetched_at=(r["odds_fetched_at"].isoformat() if r.get("odds_fetched_at") else None),
            )
        )
    return out


# -------- parlay models/helpers (keep yours as-is) --------
class ParlayLeg(BaseModel):
    match_id: str
    pick: str
    odds_american: int
    model_prob: float
    no_vig_prob: float
    edge: float
    summary: str | None = None


class ParlaySuggestion(BaseModel):
    legs: list[ParlayLeg] = Field(default_factory=list)
    parlay_decimal: float = 0.0
    parlay_american: int | None = None
    win_prob: float = 0.0
    ev: float = 0.0


def _american_to_decimal(odds: int) -> float:
    if odds == 0:
        return 0.0
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))


def _decimal_to_american(dec: float) -> Optional[int]:
    if dec is None or dec <= 1.0:
        return None
    profit = dec - 1.0
    if profit >= 1.0:
        return int(round(profit * 100))
    return int(round(-100.0 / profit))


def _is_minus_125_or_better(odds: int, min_odds: int = -125) -> bool:
    if odds == 0:
        return False
    if odds > 0:
        return True
    return odds >= min_odds


def _parse_iso_utc(s: str) -> dt.datetime | None:
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    except Exception:
        return None


@router.get("/parlay/suggested", response_model=list[ParlaySuggestion])
async def suggested_parlay(
    legs: int = Query(2, ge=2, le=4),
    top_n: int = Query(3, ge=1, le=10),
    days_ahead: int = Query(0, ge=0, le=14),
    include_incomplete: bool = Query(True),
    min_parlay_payout: int = Query(-125),
    min_edge: float | None = Query(None),
    min_ev: float | None = Query(None),
    min_leg_odds: int | None = Query(None),
    candidate_pool: int = Query(80, ge=10, le=200),
    max_overlap: int = Query(0, ge=0, le=3),
    objective: str = Query("win_prob"),
):
    # ✅ always use enhanced
    data = await predictions_today(
        days_ahead=days_ahead, include_incomplete=include_incomplete, bust_cache=True
    )

    items = data["items"] if isinstance(data, dict) else data.items
    now = dt.datetime.now(dt.timezone.utc)

    candidates: list[ParlayLeg] = []

    for it in items:
        d = it if isinstance(it, dict) else it.model_dump()

        start_iso = d.get("match_start_utc") or d.get("matchStartUtc")
        start_dt = _parse_iso_utc(start_iso) if start_iso else None
        if start_dt is None or start_dt <= now:
            continue

        pick_side = d.get("predicted_winner_blended") or d.get("predicted_winner")
        if pick_side not in ("p1", "p2"):
            continue

        if pick_side == "p1":
            odds = d.get("p1_market_odds_american")
            mp = d.get("p1_blended_win_prob") or d.get("p1_win_prob")
            nv = d.get("p1_market_no_vig_prob")
            pick_name = d.get("p1_name")
        else:
            odds = d.get("p2_market_odds_american")
            mp = d.get("p2_blended_win_prob") or d.get("p2_win_prob")
            nv = d.get("p2_market_no_vig_prob")
            pick_name = d.get("p2_name")

        inputs = d.get("inputs") if isinstance(d, dict) else None
        pick_summary = None
        if isinstance(inputs, dict):
            pick_summary = inputs.get("pick_summary") or inputs.get("pickSummary")

        if odds is None or mp is None or nv is None:
            continue

        odds = int(odds)
        mp = float(mp)
        nv = float(nv)

        if min_leg_odds is not None and not _is_minus_125_or_better(odds, min_odds=int(min_leg_odds)):
            continue

        edge = mp - nv
        if min_edge is not None and edge < float(min_edge):
            continue

        dec = _american_to_decimal(odds)
        leg_ev = mp * (dec - 1.0) - (1.0 - mp)
        if min_ev is not None and leg_ev < float(min_ev):
            continue

        candidates.append(
            ParlayLeg(
                match_id=str(d.get("match_id")),
                pick=str(pick_name or pick_side),
                odds_american=odds,
                model_prob=mp,
                no_vig_prob=nv,
                edge=edge,
                summary=str(pick_summary) if pick_summary else None,
            )
        )

    candidates.sort(
        key=lambda c: (c.model_prob * (_american_to_decimal(c.odds_american) - 1.0) - (1.0 - c.model_prob)),
        reverse=True,
    )
    candidates = candidates[:candidate_pool]

    all_suggestions: list[ParlaySuggestion] = []
    for combo in combinations(candidates, legs):
        if len({c.match_id for c in combo}) != len(combo):
            continue

        win_prob = 1.0
        dec = 1.0
        for c in combo:
            win_prob *= c.model_prob
            dec *= _american_to_decimal(c.odds_american)

        parlay_american = _decimal_to_american(dec)
        if parlay_american is None:
            continue

        if not _is_minus_125_or_better(parlay_american, min_odds=min_parlay_payout):
            continue

        ev = win_prob * (dec - 1.0) - (1.0 - win_prob)

        all_suggestions.append(
            ParlaySuggestion(
                legs=list(combo),
                parlay_decimal=dec,
                parlay_american=parlay_american,
                win_prob=win_prob,
                ev=ev,
            )
        )

    obj = (objective or "win_prob").lower()
    if obj == "ev":
        all_suggestions.sort(key=lambda s: (s.ev, s.win_prob), reverse=True)
    else:
        all_suggestions.sort(key=lambda s: (s.win_prob, s.ev), reverse=True)

    selected: list[ParlaySuggestion] = []
    for s in all_suggestions:
        if len(selected) >= top_n:
            break

        s_ids = {l.match_id for l in s.legs}
        ok = True
        for prev in selected:
            prev_ids = {l.match_id for l in prev.legs}
            if len(s_ids & prev_ids) > max_overlap:
                ok = False
                break

        if ok:
            selected.append(s)

    return selected
