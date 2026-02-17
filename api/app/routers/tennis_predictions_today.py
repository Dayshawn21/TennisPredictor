# app/api/routers/tennis_predictions_today.py
from __future__ import annotations

import datetime as dt
import os
from datetime import date, timedelta
from itertools import combinations
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from itertools import combinations

from app.db_session import engine
from app.models.odds import american_to_implied_prob, no_vig_two_way, blend_probs_logit, prob_to_american
from app.schemas.tennis_predictions import EloPrediction, EloPredictionsResponse
from app.services.tennis.elo_predictor import predict_match
from app.utils.ttl_cache import AsyncTTLCache

router = APIRouter(tags=["Tennis"])

# cache variants separately by query params
_pred_caches: dict[str, AsyncTTLCache[dict]] = {}


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


class ParlayLeg(BaseModel):
    match_id: str
    pick: str
    odds_american: int
    model_prob: float
    no_vig_prob: float
    edge: float


class ParlaySuggestion(BaseModel):
    legs: list[ParlayLeg] = Field(default_factory=list)
    parlay_decimal: float = 0.0
    parlay_american: int | None = None
    win_prob: float = 0.0
    ev: float = 0.0

def _get_pred_cache(days_ahead: int, include_incomplete: bool) -> AsyncTTLCache[dict]:
    key = f"{int(days_ahead)}:{int(include_incomplete)}"
    if key not in _pred_caches:
        _pred_caches[key] = AsyncTTLCache(ttl_seconds=600)
    return _pred_caches[key]


def _as_float(x) -> Optional[float]:
    return float(x) if x is not None else None


def _dt_to_iso_z(x) -> Optional[str]:
    """Serialize datetime to ISO string in UTC with Z suffix."""
    if x is None or not isinstance(x, dt.datetime):
        return None
    if x.tzinfo is None:
        x = x.replace(tzinfo=dt.timezone.utc)
    else:
        x = x.astimezone(dt.timezone.utc)
    return x.isoformat().replace("+00:00", "Z")


def _round_american(odds: Optional[int], step: int) -> Optional[int]:
    if odds is None:
        return None
    if step <= 1:
        return int(odds)
    rounded = int(round(float(odds) / float(step)) * step)
    if rounded == 0:
        return step if odds > 0 else -step
    return rounded


def _american_to_decimal(odds: int) -> float:
    if odds == 0:
        return 0.0
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / (-odds))


def _decimal_to_american(dec: float) -> Optional[int]:
    if dec is None or dec <= 1.0:
        return None
    profit = dec - 1.0
    if dec >= 2.0:
        return int(round(profit * 100))
    return int(round(-100.0 / profit))

def _is_minus_125_or_better(odds: int, min_odds: int = -125) -> bool:
    # ✅ allow + odds, and allow favorites no worse than -125
    if odds == 0:
        return False
    if odds > 0:
        return True
    return odds >= min_odds  # -123 ✅, -145 ❌


def pick_surface_elo(
    surface: str,
    overall: Optional[float],
    hard: Optional[float],
    clay: Optional[float],
    grass: Optional[float],
) -> Optional[float]:
    s = (surface or "").lower()
    if "hard" in s:
        return hard if hard is not None else overall
    if "clay" in s:
        return clay if clay is not None else overall
    if "grass" in s:
        return grass if grass is not None else overall
    return overall


SOFASCORE_MATCHES_ELO_SQL = text(
    r"""
    WITH matches AS (
      SELECT
        m.match_id,
        m.match_key,
        m.match_date,
        m.match_start_utc,
        upper(m.tour) AS tour,
        m.tournament AS tournament,
        m."round" AS round,
        COALESCE(NULLIF(m.surface,''), 'unknown') AS surface,
        m.status,
        m.score_raw,
        m.p1_name AS p1_name,
        m.p2_name AS p2_name,
        m.p1_canonical_id AS p1_player_id,
        m.p2_canonical_id AS p2_player_id,
        m.p1_odds_american,
        m.p2_odds_american,
        m.odds_fetched_at,
        CASE
          WHEN upper(m.tour) = 'ATP'
           AND (
                m.tournament ILIKE '%Australian Open%'
             OR m.tournament ILIKE '%Roland Garros%'
             OR m.tournament ILIKE '%French Open%'
             OR m.tournament ILIKE '%Wimbledon%'
             OR m.tournament ILIKE '%US Open%'
           )
           AND COALESCE(m."round",'') !~* 'qual'
          THEN 5
          ELSE 3
        END AS best_of
      FROM tennis_matches m
      WHERE m.match_date = ANY(:dates)
        AND upper(m.tour) IN ('ATP','WTA')
        AND (
          :include_incomplete = true
          OR COALESCE(lower(m.status),'') IN ('finished','completed','ended')
        )
    ),
    mapped AS (
      SELECT
        x.*,
        NULLIF(trim(ta1.source_player_id), '')::bigint AS p1_ta_id,
        NULLIF(trim(ta2.source_player_id), '')::bigint AS p2_ta_id
      FROM matches x
      LEFT JOIN tennis_player_sources ta1
        ON ta1.player_id = x.p1_player_id
       AND ta1.source = CASE
         WHEN x.tour = 'ATP' THEN 'tennisabstract_elo_atp'
         ELSE 'tennisabstract_elo_wta'
       END
      LEFT JOIN tennis_player_sources ta2
        ON ta2.player_id = x.p2_player_id
       AND ta2.source = CASE
         WHEN x.tour = 'ATP' THEN 'tennisabstract_elo_atp'
         ELSE 'tennisabstract_elo_wta'
       END
    ),
    fatigue AS (
      SELECT
        m.*,

        p1_last.last_match_date AS p1_last_match_date,
        p1_last.rest_days       AS p1_rest_days,
        p1_last.went_distance   AS p1_last_went_distance,

        p1_wl.matches_10d       AS p1_matches_10d,
        p1_wl.sets_10d          AS p1_sets_10d,

        p2_last.last_match_date AS p2_last_match_date,
        p2_last.rest_days       AS p2_rest_days,
        p2_last.went_distance   AS p2_last_went_distance,

        p2_wl.matches_10d       AS p2_matches_10d,
        p2_wl.sets_10d          AS p2_sets_10d

      FROM mapped m

      LEFT JOIN LATERAL (
        SELECT
          pm.match_date AS last_match_date,
          (m.match_date - pm.match_date)::int AS rest_days,
          (sp.sets_played = bo.best_of_pm) AS went_distance
        FROM tennis_matches pm
        CROSS JOIN LATERAL (
          SELECT
            CASE
              WHEN COALESCE(pm.score_raw,'') ~ '^\s*\d+\s*-\s*\d+\s*$' THEN
                split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 1)::int +
                split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 2)::int
              ELSE
                (SELECT count(*)
                 FROM unnest(regexp_split_to_array(COALESCE(pm.score_raw,''), '\s+')) t(tok)
                 WHERE regexp_replace(tok, '\(.*\)', '', 'g') ~ '^\d+\-\d+$'
                )
            END AS sets_played
        ) sp
        CROSS JOIN LATERAL (
          SELECT
            CASE
              WHEN upper(pm.tour) = 'ATP'
               AND (
                    pm.tournament ILIKE '%Australian Open%'
                 OR pm.tournament ILIKE '%Roland Garros%'
                 OR pm.tournament ILIKE '%French Open%'
                 OR pm.tournament ILIKE '%Wimbledon%'
                 OR pm.tournament ILIKE '%US Open%'
               )
               AND COALESCE(pm."round",'') !~* 'qual'
              THEN 5 ELSE 3
            END AS best_of_pm
        ) bo
        WHERE m.p1_player_id IS NOT NULL
          AND pm.match_id <> m.match_id
          AND (pm.p1_canonical_id = m.p1_player_id OR pm.p2_canonical_id = m.p1_player_id)
          AND pm.match_date <= m.match_date
          AND COALESCE(lower(pm.status),'') IN ('finished','completed','ended')
          AND COALESCE(pm.score_raw,'') <> ''
        ORDER BY pm.match_date DESC
        LIMIT 1
      ) p1_last ON TRUE

      LEFT JOIN LATERAL (
        SELECT
          count(*)::int AS matches_10d,
          COALESCE(sum(sp.sets_played), 0)::int AS sets_10d
        FROM tennis_matches pm
        CROSS JOIN LATERAL (
          SELECT
            CASE
              WHEN COALESCE(pm.score_raw,'') ~ '^\s*\d+\s*-\s*\d+\s*$' THEN
                split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 1)::int +
                split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 2)::int
              ELSE
                (SELECT count(*)
                 FROM unnest(regexp_split_to_array(COALESCE(pm.score_raw,''), '\s+')) t(tok)
                 WHERE regexp_replace(tok, '\(.*\)', '', 'g') ~ '^\d+\-\d+$'
                )
            END AS sets_played
        ) sp
        WHERE m.p1_player_id IS NOT NULL
          AND pm.match_id <> m.match_id
          AND (pm.p1_canonical_id = m.p1_player_id OR pm.p2_canonical_id = m.p1_player_id)
          AND pm.match_date >= (m.match_date - interval '10 days')
          AND pm.match_date <= m.match_date
          AND COALESCE(lower(pm.status),'') IN ('finished','completed','ended')
          AND COALESCE(pm.score_raw,'') <> ''
      ) p1_wl ON TRUE

      LEFT JOIN LATERAL (
        SELECT
          pm.match_date AS last_match_date,
          (m.match_date - pm.match_date)::int AS rest_days,
          (sp.sets_played = bo.best_of_pm) AS went_distance
        FROM tennis_matches pm
        CROSS JOIN LATERAL (
          SELECT
            CASE
              WHEN COALESCE(pm.score_raw,'') ~ '^\s*\d+\s*-\s*\d+\s*$' THEN
                split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 1)::int +
                split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 2)::int
              ELSE
                (SELECT count(*)
                 FROM unnest(regexp_split_to_array(COALESCE(pm.score_raw,''), '\s+')) t(tok)
                 WHERE regexp_replace(tok, '\(.*\)', '', 'g') ~ '^\d+\-\d+$'
                )
            END AS sets_played
        ) sp
        CROSS JOIN LATERAL (
          SELECT
            CASE
              WHEN upper(pm.tour) = 'ATP'
               AND (
                    pm.tournament ILIKE '%Australian Open%'
                 OR pm.tournament ILIKE '%Roland Garros%'
                 OR pm.tournament ILIKE '%French Open%'
                 OR pm.tournament ILIKE '%Wimbledon%'
                 OR pm.tournament ILIKE '%US Open%'
               )
               AND COALESCE(pm."round",'') !~* 'qual'
              THEN 5 ELSE 3
            END AS best_of_pm
        ) bo
        WHERE m.p2_player_id IS NOT NULL
          AND pm.match_id <> m.match_id
          AND (pm.p1_canonical_id = m.p2_player_id OR pm.p2_canonical_id = m.p2_player_id)
          AND pm.match_date <= m.match_date
          AND COALESCE(lower(pm.status),'') IN ('finished','completed','ended')
          AND COALESCE(pm.score_raw,'') <> ''
        ORDER BY pm.match_date DESC
        LIMIT 1
      ) p2_last ON TRUE

      LEFT JOIN LATERAL (
        SELECT
          count(*)::int AS matches_10d,
          COALESCE(sum(sp.sets_played), 0)::int AS sets_10d
        FROM tennis_matches pm
        CROSS JOIN LATERAL (
          SELECT
            CASE
              WHEN COALESCE(pm.score_raw,'') ~ '^\s*\d+\s*-\s*\d+\s*$' THEN
                split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 1)::int +
                split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 2)::int
              ELSE
                (SELECT count(*)
                 FROM unnest(regexp_split_to_array(COALESCE(pm.score_raw,''), '\s+')) t(tok)
                 WHERE regexp_replace(tok, '\(.*\)', '', 'g') ~ '^\d+\-\d+$'
                )
            END AS sets_played
        ) sp
        WHERE m.p2_player_id IS NOT NULL
          AND pm.match_id <> m.match_id
          AND (pm.p1_canonical_id = m.p2_player_id OR pm.p2_canonical_id = m.p2_player_id)
          AND pm.match_date >= (m.match_date - interval '10 days')
          AND pm.match_date <= m.match_date
          AND COALESCE(lower(pm.status),'') IN ('finished','completed','ended')
          AND COALESCE(pm.score_raw,'') <> ''
      ) p2_wl ON TRUE
    ),
    asof AS (
      SELECT
        d.match_date,
        d.tour,
        max(s.as_of_date) AS as_of_date
      FROM (SELECT DISTINCT match_date, tour FROM fatigue) d
      JOIN tennisabstract_elo_snapshots s
        ON upper(s.tour) = d.tour
       AND s.as_of_date <= d.match_date
      GROUP BY d.match_date, d.tour
    ),
    snap AS (
      SELECT DISTINCT ON (upper(s.tour), s.as_of_date, s.player_id)
        upper(s.tour) AS tour,
        s.as_of_date,
        s.player_id,
        s.elo,
        s.helo,
        s.celo,
        s.gelo,
        s.created_at,
        s.player_name
      FROM tennisabstract_elo_snapshots s
      JOIN asof a
        ON upper(s.tour) = a.tour
       AND s.as_of_date = a.as_of_date
      WHERE s.player_id IS NOT NULL
      ORDER BY
        upper(s.tour),
        s.as_of_date,
        s.player_id,
        (s.elo  IS NOT NULL) DESC,
        (s.helo IS NOT NULL) DESC,
        (s.celo IS NOT NULL) DESC,
        (s.gelo IS NOT NULL) DESC,
        s.created_at DESC,
        s.player_name ASC
    ),
    med AS (
      SELECT
        tour,
        as_of_date,
        (percentile_cont(0.5) WITHIN GROUP (ORDER BY elo::double precision))::float8  AS med_elo,
        (percentile_cont(0.5) WITHIN GROUP (ORDER BY helo::double precision))::float8 AS med_helo,
        (percentile_cont(0.5) WITHIN GROUP (ORDER BY celo::double precision))::float8 AS med_celo,
        (percentile_cont(0.5) WITHIN GROUP (ORDER BY gelo::double precision))::float8 AS med_gelo
      FROM snap
      GROUP BY 1,2
    ),
    final_rows AS (
      SELECT
        m.*,
        a.as_of_date,

        s1.elo::float8  AS p1_elo_raw,
        s1.helo::float8 AS p1_helo_raw,
        s1.celo::float8 AS p1_celo_raw,
        s1.gelo::float8 AS p1_gelo_raw,

        s2.elo::float8  AS p2_elo_raw,
        s2.helo::float8 AS p2_helo_raw,
        s2.celo::float8 AS p2_celo_raw,
        s2.gelo::float8 AS p2_gelo_raw,

        md.med_elo,
        md.med_helo,
        md.med_celo,
        md.med_gelo

      FROM fatigue m
      LEFT JOIN asof a
        ON a.match_date = m.match_date AND a.tour = m.tour
      LEFT JOIN med md
        ON md.tour = m.tour AND md.as_of_date = a.as_of_date
      LEFT JOIN snap s1
        ON s1.tour = m.tour
       AND s1.as_of_date = a.as_of_date
       AND s1.player_id = m.p1_ta_id
      LEFT JOIN snap s2
        ON s2.tour = m.tour
       AND s2.as_of_date = a.as_of_date
       AND s2.player_id = m.p2_ta_id
    ),
    dedup AS (
      SELECT DISTINCT ON (match_id) *
      FROM final_rows
      ORDER BY
        match_id,
        (p1_player_id IS NOT NULL AND p2_player_id IS NOT NULL) DESC,
        (as_of_date IS NOT NULL) DESC,
        (p1_ta_id IS NOT NULL AND p2_ta_id IS NOT NULL) DESC,
        (p1_elo_raw IS NOT NULL AND p2_elo_raw IS NOT NULL) DESC
    )
    SELECT *
    FROM dedup
    ORDER BY match_date, tour, tournament, p1_name;
    """
)


@router.get("/predictions/today", response_model=EloPredictionsResponse)
async def predictions_today(
    days_ahead: int = Query(0, ge=0, le=14),
    include_incomplete: bool = Query(True),
    bust_cache: bool = Query(False),
):
    today = date.today()
    dates = [today + timedelta(days=i) for i in range(days_ahead + 1)]

    cache = _get_pred_cache(days_ahead, include_incomplete)
    if not bust_cache:
        cached = await cache.get()
        if cached:
            cached["cached"] = True
            return cached

    async with engine.begin() as conn:
        res = await conn.execute(
            SOFASCORE_MATCHES_ELO_SQL,
            {"dates": dates, "include_incomplete": include_incomplete},
        )
        rows = res.mappings().all()

    items: List[EloPrediction] = []
    w_model = float(os.getenv("ODDS_BLEND_WEIGHT", "0.65"))
    try:
        fair_step = int(os.getenv("FAIR_ODDS_ROUND_STEP", "1"))
    except Exception:
        fair_step = 1

    for r in rows:
        surface = r.get("surface") or "unknown"

        med_elo = _as_float(r.get("med_elo"))
        med_h = _as_float(r.get("med_helo"))
        med_c = _as_float(r.get("med_celo"))
        med_g = _as_float(r.get("med_gelo"))

        p1_overall_raw = _as_float(r.get("p1_elo_raw"))
        p2_overall_raw = _as_float(r.get("p2_elo_raw"))

        p1_overall = p1_overall_raw if p1_overall_raw is not None else med_elo
        p2_overall = p2_overall_raw if p2_overall_raw is not None else med_elo

        p1_hard_raw = _as_float(r.get("p1_helo_raw"))
        p2_hard_raw = _as_float(r.get("p2_helo_raw"))
        p1_clay_raw = _as_float(r.get("p1_celo_raw"))
        p2_clay_raw = _as_float(r.get("p2_celo_raw"))
        p1_grass_raw = _as_float(r.get("p1_gelo_raw"))
        p2_grass_raw = _as_float(r.get("p2_gelo_raw"))

        p1_hard = p1_hard_raw if p1_hard_raw is not None else (med_h if med_h is not None else p1_overall)
        p2_hard = p2_hard_raw if p2_hard_raw is not None else (med_h if med_h is not None else p2_overall)
        p1_clay = p1_clay_raw if p1_clay_raw is not None else (med_c if med_c is not None else p1_overall)
        p2_clay = p2_clay_raw if p2_clay_raw is not None else (med_c if med_c is not None else p2_overall)
        p1_grass = p1_grass_raw if p1_grass_raw is not None else (med_g if med_g is not None else p1_overall)
        p2_grass = p2_grass_raw if p2_grass_raw is not None else (med_g if med_g is not None else p2_overall)

        p1_used = pick_surface_elo(surface, p1_overall, p1_hard, p1_clay, p1_grass)
        p2_used = pick_surface_elo(surface, p2_overall, p2_hard, p2_clay, p2_grass)

        missing_canonical = (r.get("p1_player_id") is None or r.get("p2_player_id") is None)
        missing_elo = (p1_used is None or p2_used is None)

        missing_reason = None
        if missing_canonical:
            missing_reason = "Missing canonical player_id(s) on tennis_matches (alias resolver pending)."
        elif missing_elo:
            missing_reason = "Missing Elo rating(s): no snapshot + no median fallback available for this tour/date."
        elif r.get("p1_ta_id") is None or r.get("p2_ta_id") is None:
            missing_reason = (
                "Known players but TA Elo mapping missing "
                "(tennis_player_sources source=tennisabstract_elo_atp/tennisabstract_elo_wta)."
            )

        if missing_canonical or missing_elo:
            p1_prob = p2_prob = None
            p1_fair = p2_fair = None
            predicted_winner = None
            proj_total = proj_spread = None
            proj_sets = None
            inputs_used = {"p1_elo_used": None, "p2_elo_used": None}
        else:
            pred = predict_match(
                {
                    "best_of": int(r["best_of"]),
                    "surface": surface,
                    "p1_elo": float(p1_used),
                    "p2_elo": float(p2_used),
                }
            )
            p1_prob = float(pred.get("p1_win_prob")) if pred.get("p1_win_prob") is not None else None
            p2_prob = float(pred.get("p2_win_prob")) if pred.get("p2_win_prob") is not None else None

            p1_fair = _round_american(prob_to_american(p1_prob), fair_step)
            p2_fair = _round_american(prob_to_american(p2_prob), fair_step)

            predicted_winner = pred.get("predicted_winner")
            proj_total = pred.get("projected_total_games")
            proj_spread = pred.get("projected_spread_p1")
            proj_sets = pred.get("projected_sets")

            inputs_used = {"p1_elo_used": float(p1_used), "p2_elo_used": float(p2_used)}

        p1_mkt_odds = r.get("p1_odds_american")
        p2_mkt_odds = r.get("p2_odds_american")

        p1_imp = american_to_implied_prob(int(p1_mkt_odds)) if p1_mkt_odds is not None else None
        p2_imp = american_to_implied_prob(int(p2_mkt_odds)) if p2_mkt_odds is not None else None
        p1_nv, p2_nv = no_vig_two_way(p1_imp, p2_imp)

        p1_blend = blend_probs_logit(p1_prob, p1_nv, w_model=w_model)
        p2_blend = (1.0 - p1_blend) if p1_blend is not None else None
        predicted_blend = None
        if p1_blend is not None:
            predicted_blend = "p1" if p1_blend >= 0.5 else "p2"

        items.append(
            EloPrediction(
                match_id=str(r["match_id"]),
                match_key=r.get("match_key"),
                match_date=r.get("match_date"),
                match_start_utc=_dt_to_iso_z(r.get("match_start_utc")),

                tour=r.get("tour"),
                tournament=r.get("tournament"),
                round=r.get("round"),
                surface=surface,
                best_of=int(r["best_of"]) if r.get("best_of") is not None else None,

                p1_name=r.get("p1_name"),
                p2_name=r.get("p2_name"),
                p1_player_id=r.get("p1_player_id"),
                p2_player_id=r.get("p2_player_id"),
                p1_ta_id=r.get("p1_ta_id"),
                p2_ta_id=r.get("p2_ta_id"),

                p1_win_prob=p1_prob,
                p2_win_prob=p2_prob,
                p1_fair_american=p1_fair,
                p2_fair_american=p2_fair,

                p1_market_odds_american=p1_mkt_odds,
                p2_market_odds_american=p2_mkt_odds,
                odds_fetched_at=_dt_to_iso_z(r.get("odds_fetched_at")),

                p1_market_implied_prob=p1_imp,
                p2_market_implied_prob=p2_imp,
                p1_market_no_vig_prob=p1_nv,
                p2_market_no_vig_prob=p2_nv,

                p1_blended_win_prob=p1_blend,
                p2_blended_win_prob=p2_blend,
                predicted_winner_blended=predicted_blend,

                missing_reason=missing_reason,
                inputs={
                    **inputs_used,
                    "p1_elo_overall": p1_overall,
                    "p2_elo_overall": p2_overall,
                    "p1_elo_hard": p1_hard,
                    "p2_elo_hard": p2_hard,
                    "p1_elo_clay": p1_clay,
                    "p2_elo_clay": p2_clay,
                    "p1_elo_grass": p1_grass,
                    "p2_elo_grass": p2_grass,
                    "p1_rest_days": r.get("p1_rest_days"),
                    "p2_rest_days": r.get("p2_rest_days"),
                    "p1_last_match_went_distance": r.get("p1_last_went_distance"),
                    "p2_last_match_went_distance": r.get("p2_last_went_distance"),
                    "p1_matches_last_10d": r.get("p1_matches_10d"),
                    "p2_matches_last_10d": r.get("p2_matches_10d"),
                    "p1_sets_last_10d": r.get("p1_sets_10d"),
                    "p2_sets_last_10d": r.get("p2_sets_10d"),
                },

                predicted_winner=predicted_winner,
                projected_total_games=proj_total,
                projected_spread_p1=proj_spread,
                projected_sets=proj_sets,
            )
        )

    resp = EloPredictionsResponse(
        as_of=today,
        source=(
            "tennis_matches + tennisabstract_elo_snapshots (overall+surface) "
            "+ fatigue + elo_predictor "
            "+ TA mapping via tennisabstract_elo_atp/tennisabstract_elo_wta "
            "+ sofascore odds (market + no-vig + blend)"
        ),
        cached=False,
        count=len(items),
        items=items,
    )

    payload = resp.model_dump(mode="json")

    if not bust_cache:
        await cache.set(payload)

    return payload


@router.get("/odds/today", response_model=list[OddsRow])
async def odds_today(days_ahead: int = Query(0, ge=0, le=14)):
    sql = text("""
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
    """)

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



