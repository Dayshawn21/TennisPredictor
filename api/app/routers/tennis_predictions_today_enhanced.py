# api/app/routers/tennis_predictions_today_enhanced.py
"""
Enhanced tennis predictions endpoint using combined predictor.
Combines ELO + XGBoost + Market Odds.
Adds DB-driven alias expansion for rolling stats lookup via tennis_player_aliases.
"""

from __future__ import annotations

import csv
import math
import os
import json
import time
import logging
from difflib import SequenceMatcher
from pathlib import Path
from functools import lru_cache
import datetime as dt
import re
import unicodedata
from datetime import date, timedelta
from typing import Optional, Dict, Any, List, Tuple
from itertools import combinations
from pydantic import BaseModel, Field

from fastapi import APIRouter, Query, HTTPException
from sqlalchemy import text

from app.db_session import engine
from app.db.repositories.tennis_predictions_repo import (
    batch_alias_variants as repo_batch_alias_variants,
    batch_fetch_rolling_stats as repo_batch_fetch_rolling_stats,
    fetch_ta_feature_diffs,
    fetch_h2h_raw_payload,
    fetch_matches_elo_light_rows,
    get_h2h_from_db as repo_get_h2h_from_db,
    get_last_matchups_from_db as repo_get_last_matchups_from_db,
    get_player_last_matches_from_db as repo_get_player_last_matches_from_db,
    resolve_canonical_player_id as repo_resolve_canonical_player_id,
)
from app.models.odds import (
    prob_to_american as prob_to_american_model,
    american_to_implied_prob as american_to_implied_prob_model,
    no_vig_two_way as no_vig_two_way_model,
    blend_probs_logit,
)
from app.schemas.tennis_predictions import EloPrediction, EloPredictionsResponse, H2HMatch
from app.services.tennis.combined_predictor import predict_match_combined, predict_batch_combined_async
from app.services.tennis.player_stats_service import (
    build_player_stats_payload as build_player_stats_payload_service,
    build_players_compare_payload,
    build_players_compare_context,
    build_players_compare_response,
    get_player_full_stats as get_player_full_stats_service,
    get_players_compare as get_players_compare_service,
    validate_players_compare_inputs,
)
from app.services.tennis.predictions_service import (
    get_debug_stats_lookup as get_debug_stats_lookup_service,
    get_h2h_matches as get_h2h_matches_service,
    get_predictions_today as get_predictions_today_service,
    get_predictions_today_enhanced as get_predictions_today_enhanced_service,
    get_suggested_parlay as get_suggested_parlay_service,
)
from app.utils.ttl_cache import AsyncTTLCache


__version__ = "18"

router = APIRouter(tags=["Tennis Enhanced"])
logger = logging.getLogger(__name__)

SOFASCORE_MATCHES_ELO_SQL_LIGHT = text(
    r"""
    WITH matches AS (
      SELECT
        m.match_id,
        m.match_key,
        m.sofascore_event_id,
        m.match_date,
        m.match_start_utc,
        upper(m.tour) AS tour,
        m.tournament AS tournament,
        m."round" AS round,
        COALESCE(NULLIF(m.surface,''), 'unknown') AS surface,
        m.status,
        m.p1_name AS p1_name,
        m.p2_name AS p2_name,
        m.p1_canonical_id AS p1_player_id,
        m.p2_canonical_id AS p2_player_id,
        m.h2h_p1_wins,
        m.h2h_p2_wins,
        m.h2h_total_matches,
        m.h2h_surface_p1_wins,
        m.h2h_surface_p2_wins,
        m.h2h_surface_matches,
        m.p1_odds_american,
        m.p2_odds_american,
        m.odds_fetched_at,
        m.sofascore_total_games_line,
        m.sofascore_total_games_over_american,
        m.sofascore_total_games_under_american,
        m.sofascore_spread_p1_line,
        m.sofascore_spread_p2_line,
        m.sofascore_spread_p1_odds_american,
        m.sofascore_spread_p2_odds_american,
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
        AND COALESCE(lower(m.status), '') NOT LIKE 'cancel%'
        AND COALESCE(lower(m.status), '') NOT LIKE 'walkover%'
        AND (
          :include_incomplete = true
          OR COALESCE(lower(m.status),'') IN ('finished','completed','ended')
        )
    ),
    mapped AS (
      SELECT
        x.*,
        CASE
          WHEN trim(ta1.source_player_id) ~ '^[0-9]+$'
          THEN trim(ta1.source_player_id)::bigint
          ELSE NULL
        END AS p1_ta_id,
        CASE
          WHEN trim(ta2.source_player_id) ~ '^[0-9]+$'
          THEN trim(ta2.source_player_id)::bigint
          ELSE NULL
        END AS p2_ta_id
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
        c1.last_match_date AS p1_last_match_date,
        c1.rest_days       AS p1_rest_days,
        c1.went_distance   AS p1_last_went_distance,
        c1.matches_10d     AS p1_matches_10d,
        c1.sets_10d        AS p1_sets_10d,
        c2.last_match_date AS p2_last_match_date,
        c2.rest_days       AS p2_rest_days,
        c2.went_distance   AS p2_last_went_distance,
        c2.matches_10d     AS p2_matches_10d,
        c2.sets_10d        AS p2_sets_10d
      FROM mapped m
      LEFT JOIN tennis_player_fatigue_cache c1
        ON c1.player_id = m.p1_player_id AND c1.as_of_date = m.match_date
      LEFT JOIN tennis_player_fatigue_cache c2
        ON c2.player_id = m.p2_player_id AND c2.as_of_date = m.match_date
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
        s.official_rank,
        s.age,
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
        s1.official_rank::int AS p1_rank,
        s2.official_rank::int AS p2_rank,
        s1.age::float8 AS p1_age,
        s2.age::float8 AS p2_age,
        l1.elo::float8  AS p1_elo_latest,
        l2.elo::float8  AS p2_elo_latest,
        md.med_elo,
        md.med_helo,
        md.med_celo,
        md.med_gelo
      FROM fatigue m
      LEFT JOIN asof a
        ON a.match_date = m.match_date AND a.tour = m.tour
      LEFT JOIN tennisabstract_elo_medians md
        ON md.tour = m.tour AND md.as_of_date = a.as_of_date
      LEFT JOIN snap s1
        ON s1.tour = m.tour
       AND s1.as_of_date = a.as_of_date
       AND s1.player_id = m.p1_ta_id
      LEFT JOIN snap s2
        ON s2.tour = m.tour
       AND s2.as_of_date = a.as_of_date
       AND s2.player_id = m.p2_ta_id
      LEFT JOIN tennisabstract_elo_latest l1
        ON upper(l1.tour) = m.tour
       AND l1.player_id = m.p1_ta_id
      LEFT JOIN tennisabstract_elo_latest l2
        ON upper(l2.tour) = m.tour
       AND l2.player_id = m.p2_ta_id
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

H2H_RAW_SQL = text(
    """
    SELECT payload
    FROM sofascore_event_h2h_raw
    WHERE event_id = :event_id
    LIMIT 1
    """
)

_pred_caches: dict[str, AsyncTTLCache[dict]] = {}


def _get_pred_cache(
    days_ahead: int,
    include_incomplete: bool,
    min_edge: float = 0.025,
    max_odds_age_min: int = 180,
    max_overround: float = 0.08,
) -> AsyncTTLCache[dict]:
    key = (
        f"enhanced_{days_ahead}_{include_incomplete}"
        f"_edge{float(min_edge):.4f}"
        f"_age{int(max_odds_age_min)}"
        f"_ovr{float(max_overround):.4f}"
    )
    if key not in _pred_caches:
        ttl_sec = 300 if include_incomplete else 3600
        _pred_caches[key] = AsyncTTLCache(ttl_seconds=ttl_sec)
    return _pred_caches[key]


def _dt_to_iso_z(x) -> Optional[str]:
    if x is None or not isinstance(x, dt.datetime):
        return None
    if x.tzinfo is None:
        x = x.replace(tzinfo=dt.timezone.utc)
    else:
        x = x.astimezone(dt.timezone.utc)
    return x.isoformat().replace("+00:00", "Z")


def _odds_age_minutes(odds_fetched_at: Any, now_utc: Optional[dt.datetime] = None) -> Optional[float]:
    if odds_fetched_at is None:
        return None
    now_utc = now_utc or dt.datetime.now(dt.timezone.utc)
    ts = odds_fetched_at
    if isinstance(ts, str):
        s = ts.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            ts = dt.datetime.fromisoformat(s)
        except Exception:
            return None
    if not isinstance(ts, dt.datetime):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    else:
        ts = ts.astimezone(dt.timezone.utc)
    delta_min = (now_utc - ts).total_seconds() / 60.0
    return max(0.0, float(delta_min))


def _american_to_decimal(odds: Optional[int]) -> Optional[float]:
    if odds is None:
        return None
    try:
        o = int(odds)
    except Exception:
        return None
    if o > 0:
        return 1.0 + (float(o) / 100.0)
    if o < 0:
        return 1.0 + (100.0 / float(-o))
    return None


def _kelly_fraction(prob: Optional[float], american_odds: Optional[int], cap: float = 0.02) -> Optional[float]:
    if prob is None or american_odds is None:
        return None
    dec = _american_to_decimal(american_odds)
    if dec is None or dec <= 1.0:
        return None
    p = max(0.0, min(1.0, float(prob)))
    b = dec - 1.0
    q = 1.0 - p
    raw = (b * p - q) / b
    if raw <= 0:
        return 0.0
    return float(min(cap, raw))


def _prob_to_american(p: Optional[float]) -> Optional[int]:
    # Keep consistent rounding with the shared odds utility.
    return prob_to_american_model(p)


def _round_american(odds: Optional[int], step: int) -> Optional[int]:
    if odds is None:
        return None
    if step <= 1:
        return int(odds)
    rounded = int(round(float(odds) / float(step)) * step)
    if rounded == 0:
        return step if odds > 0 else -step
    return rounded


def _derive_model_game_spread_lines(p1_win_prob: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    """
    Derive game spread lines from match win probability.
    Positive internal margin means p1 is stronger by games.
    """
    if p1_win_prob is None:
        return None, None
    try:
        p = float(p1_win_prob)
    except Exception:
        return None, None
    if not (0.0 < p < 1.0):
        return None, None
    eps = 1e-6
    p = max(eps, min(1.0 - eps, p))
    margin_games = 2.9 * math.log(p / (1.0 - p))
    margin_games = max(-6.5, min(6.5, margin_games))
    margin_games = round(margin_games * 2.0) / 2.0
    p1_line = -margin_games
    p2_line = margin_games
    return float(p1_line), float(p2_line)


def _safe_get(d: dict, *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _normalize_surface_family(surface: Optional[str]) -> str:
    s = (surface or "").lower()
    if "clay" in s:
        return "clay"
    if "grass" in s:
        return "grass"
    if "hard" in s:
        return "hard"
    return "unknown"


def _norm_name_simple(name: Optional[str]) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _format_h2h_score(ev: dict) -> Optional[str]:
    if not isinstance(ev, dict):
        return None
    hs = ev.get("homeScore") or {}
    aws = ev.get("awayScore") or {}
    sets: list[str] = []
    for key in ("period1", "period2", "period3", "period4", "period5"):
        h = hs.get(key)
        a = aws.get(key)
        if h is None or a is None:
            continue
        sets.append(f"{h}-{a}")
    if sets:
        return " ".join(sets)
    if hs.get("display") is not None and aws.get("display") is not None:
        return f"{hs.get('display')}-{aws.get('display')}"
    if hs.get("current") is not None and aws.get("current") is not None:
        return f"{hs.get('current')}-{aws.get('current')}"
    return None


def _format_db_score(row: dict[str, Any]) -> Optional[str]:
    score = row.get("score")
    if score is not None:
        s = str(score).strip()
        if s:
            return s
    score_raw = row.get("score_raw")
    if score_raw is not None:
        s = str(score_raw).strip()
        if s:
            return s
    return None


def _infer_winner_side_from_score(score_text: Optional[str]) -> Optional[str]:
    if not score_text:
        return None
    s = str(score_text).strip()
    if not s:
        return None
    parts = re.findall(r"(\d+)\s*-\s*(\d+)", s)
    if not parts:
        return None
    p1_sets = 0
    p2_sets = 0
    for a_str, b_str in parts:
        try:
            a = int(a_str)
            b = int(b_str)
        except Exception:
            continue
        if a > b:
            p1_sets += 1
        elif b > a:
            p2_sets += 1
    if p1_sets > p2_sets:
        return "p1"
    if p2_sets > p1_sets:
        return "p2"
    return None


def _extract_h2h_matches(payload: dict, limit: int = 5) -> list[dict]:
    events = _safe_get(payload, "events", "events", default=[]) or []
    if not isinstance(events, list):
        return []
    rows: list[dict] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        ts = ev.get("startTimestamp")
        date_str = None
        try:
            if ts:
                date_str = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).date().isoformat()
        except Exception:
            date_str = None
        rows.append(
            {
                "date": date_str,
                "tournament": _safe_get(ev, "tournament", "name", default=None),
                "round": _safe_get(ev, "roundInfo", "name", default=None),
                "surface": ev.get("groundType") or _safe_get(ev, "tournament", "uniqueTournament", "groundType", default=None),
                "home": _safe_get(ev, "homeTeam", "name", default=None),
                "away": _safe_get(ev, "awayTeam", "name", default=None),
                "score": _format_h2h_score(ev),
                "winner_code": ev.get("winnerCode"),
            }
        )
    rows.sort(key=lambda x: x.get("date") or "", reverse=True)
    return rows[: max(0, limit)]


async def _get_h2h_from_db(
    conn,
    p1_canonical_id: Optional[int],
    p2_canonical_id: Optional[int],
    p1_name: Optional[str],
    p2_name: Optional[str],
    surface: Optional[str],
    as_of: date,
    lookback_years: int = 5,
) -> Tuple[int, int, int, int, int, int]:
    return await repo_get_h2h_from_db(
        conn=conn,
        p1_canonical_id=p1_canonical_id,
        p2_canonical_id=p2_canonical_id,
        p1_name=p1_name,
        p2_name=p2_name,
        surface=surface,
        as_of=as_of,
        lookback_years=lookback_years,
        normalize_surface_family=_normalize_surface_family,
        norm_name_simple=_norm_name_simple,
    )


async def _get_last_matchups_from_db(
    conn,
    p1_canonical_id: Optional[int],
    p2_canonical_id: Optional[int],
    p1_name: Optional[str],
    p2_name: Optional[str],
    limit: int = 10,
) -> list[dict[str, Any]]:
    return await repo_get_last_matchups_from_db(
        conn=conn,
        p1_canonical_id=p1_canonical_id,
        p2_canonical_id=p2_canonical_id,
        p1_name=p1_name,
        p2_name=p2_name,
        limit=limit,
        norm_name_simple=_norm_name_simple,
        infer_winner_side_from_score=_infer_winner_side_from_score,
        format_db_score=_format_db_score,
    )


async def _get_player_last_matches_from_db(
    conn,
    player_name: Optional[str],
    *,
    canonical_player_id: Optional[int] = None,
    name_candidates: Optional[list[str]] = None,
    limit: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return await repo_get_player_last_matches_from_db(
        conn=conn,
        player_name=player_name,
        canonical_player_id=canonical_player_id,
        name_candidates=name_candidates,
        limit=limit,
        norm_name_simple=_norm_name_simple,
        infer_winner_side_from_score=_infer_winner_side_from_score,
        format_db_score=_format_db_score,
    )


async def _resolve_canonical_player_id(
    conn,
    player_name: Optional[str],
    tour: str,
) -> tuple[Optional[int], dict[str, Any]]:
    return await repo_resolve_canonical_player_id(
        conn=conn,
        player_name=player_name,
        name_norm_stats=_norm_name_stats,
        norm_name_simple=_norm_name_simple,
        name_variants_for_stats=_name_variants_for_stats,
    )


def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _norm_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    name = _strip_accents(name.strip())
    name = re.sub(r"\s+", " ", name)
    return name.lower()


# Normalization for stats CSV keys (strips punctuation to match CSV keying).
def _norm_name_stats(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    name = _strip_accents(name.strip())
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or None


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _name_variants_for_stats(pname: str) -> list[str]:
    tokens = [t for t in pname.split() if t]
    if not tokens:
        return [pname]
    variants: list[str] = []
    seen: set[str] = set()

    def add(v: str) -> None:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            variants.append(v)

    add(pname)

    # drop suffixes like "jr", "iii"
    tokens_no_suffix = [t for t in tokens if t not in _NAME_SUFFIXES]
    if tokens_no_suffix and tokens_no_suffix != tokens:
        add(" ".join(tokens_no_suffix))

    # drop single-letter initials
    tokens_no_init = [t for t in tokens_no_suffix if len(t) > 1]
    if tokens_no_init and tokens_no_init != tokens_no_suffix:
        add(" ".join(tokens_no_init))

    # first + last only
    if len(tokens_no_init) >= 2:
        add(f"{tokens_no_init[0]} {tokens_no_init[-1]}")

    return variants


# --- DB-driven alias expansion (YOUR Step 1 + Step 2) -----------------

_ALIAS_CANON_SQL = text(
    """
    SELECT canonical_player_id
    FROM public.tennis_player_aliases
    WHERE alias_name_norm = :alias_norm
      AND is_pending = false
      AND canonical_player_id IS NOT NULL
    LIMIT 1
    """
)

_ALIAS_VARIANTS_SQL = text(
    """
    SELECT alias_name_norm
    FROM public.tennis_player_aliases
    WHERE canonical_player_id = :cid
      AND is_pending = false
    """
)


async def _alias_variants_for_name(conn, base_norm: str) -> List[str]:
    """
    Returns: [base_norm] + all confirmed alias_name_norm values for same canonical_player_id (if known).
    """
    variants = {base_norm}

    res = await conn.execute(_ALIAS_CANON_SQL, {"alias_norm": base_norm})
    row = res.first()
    cid = row[0] if row else None

    if cid is not None:
        res2 = await conn.execute(_ALIAS_VARIANTS_SQL, {"cid": cid})
        for r in res2.fetchall():
            if r and r[0]:
                variants.add(str(r[0]))

    return list(variants)


async def _fetch_rolling_for_player(
    conn,
    player_name: Optional[str],
    _variants_cache: dict[str, List[str]],
) -> Optional[Dict[str, Any]]:
    """
    Fetch rolling stats row for a player using normalized matching + DB alias variants.
    Returns dict with keys matching the table columns or None.
    """
    base = _norm_name(player_name)
    if not base:
        return None

    if base in _variants_cache:
        variants = _variants_cache[base]
    else:
        variants = await _alias_variants_for_name(conn, base)
        _variants_cache[base] = variants

    q = text(
        r"""
        SELECT
            win_rate_last_20,
            matches_played,
            hard_win_rate_last_10,
            clay_win_rate_last_10,
            grass_win_rate_last_10
        FROM tennis_player_rolling_stats
        WHERE lower(regexp_replace(trim(player_name), '\s+', ' ', 'g')) = :nm
        LIMIT 1
        """
    )

    for v in variants:
        res = await conn.execute(q, {"nm": v})
        row = res.mappings().first()
        if row:
            return dict(row)

    return None


# --- Batch rolling stats fetch (performance) ----------------------------

async def _batch_alias_variants(
    conn,
    base_norms: list[str],
) -> dict[str, list[str]]:
    return await repo_batch_alias_variants(conn=conn, base_norms=base_norms)


async def _batch_fetch_rolling_stats(
    conn,
    variants_by_base: dict[str, list[str]],
) -> dict[str, dict]:
    return await repo_batch_fetch_rolling_stats(conn=conn, variants_by_base=variants_by_base)


def _rolling_for_player_from_batch(
    player_name: Optional[str],
    variants_by_base: dict[str, list[str]],
    rolling_by_norm: dict[str, dict],
) -> Optional[Dict[str, Any]]:
    """Return rolling stats dict for player_name using pre-fetched batch maps."""
    base = _norm_name(player_name)
    if not base:
        return None
    variants = variants_by_base.get(base) or [base]
    for v in variants:
        rr = rolling_by_norm.get(v)
        if rr:
            rr2 = dict(rr)
            rr2.pop("player_name_norm", None)
            return rr2
    return None

# -----------------------------
# Parlay models/helpers + route
# -----------------------------

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


def _parse_american_odds(odds) -> Optional[int]:
    if odds is None:
        return None
    if isinstance(odds, bool):
        return None
    if isinstance(odds, int):
        return int(odds)
    if isinstance(odds, float):
        if math.isnan(odds) or math.isinf(odds):
            return None
        return int(round(odds))
    s = str(odds).strip().upper()
    if not s:
        return None
    if s in {"EVEN", "EV", "PK", "PICK", "PICKEM", "PK.", "PK'"}:
        return 100
    s = s.replace("+", "").replace("−", "-").replace("–", "-")
    try:
        if "." in s:
            return int(round(float(s)))
        return int(s)
    except Exception:
        return None



# Forward declaration for type checkers / linters (overwritten by the actual endpoint below)
predictions_today_enhanced = None  # type: ignore

async def _suggested_parlay_impl(
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
    # always use enhanced predictions
    data = await predictions_today_enhanced(
        days_ahead=days_ahead, include_incomplete=include_incomplete, bust_cache=False
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
            pick_name = d.get("p1_name") or d.get("p1Name")
        else:
            odds = d.get("p2_market_odds_american")
            mp = d.get("p2_blended_win_prob") or d.get("p2_win_prob")
            nv = d.get("p2_market_no_vig_prob")
            pick_name = d.get("p2_name") or d.get("p2Name")

        inputs = d.get("inputs") if isinstance(d, dict) else None
        pick_summary = None
        if isinstance(inputs, dict):
            pick_summary = inputs.get("pick_summary") or inputs.get("pickSummary")

        if odds is None or mp is None or nv is None:
            continue

        odds = _parse_american_odds(odds)
        if odds is None:
            continue
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
                match_id=str(d.get("match_id") or d.get("matchId")),
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
    return await get_suggested_parlay_service(
        legs=legs,
        top_n=top_n,
        days_ahead=days_ahead,
        include_incomplete=include_incomplete,
        min_parlay_payout=min_parlay_payout,
        min_edge=min_edge,
        min_ev=min_ev,
        min_leg_odds=min_leg_odds,
        candidate_pool=candidate_pool,
        max_overlap=max_overlap,
        objective=objective,
        impl=_suggested_parlay_impl,
    )

# -----------------------------
# Totals/Sets simulator (A–C)
# -----------------------------
# Uses player_surface_stats.csv (built from Tennis Abstract Match Stats exports).
# Output is attached under inputs["totals_sets_sim"] for each match.

def _clamp(x: float, lo: float = 1e-6, hi: float = 1 - 1e-6) -> float:
    return max(lo, min(hi, x))
def _clamp01(x: float) -> float:
    """Clamp a value into [0, 1]. Used for probabilities."""
    try:
        v = float(x)
    except Exception:
        return 0.5
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v




def _logit(p: float) -> float:
    p = _clamp(p)
    return math.log(p / (1.0 - p))


def _logistic(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _combine_point_probs(service_pts_w: float, opp_return_pts_w: float) -> float:
    """Approx P(server wins a point on serve) for a matchup.

    Uses logit-mean of:
      - server's service points won %
      - (1 - returner's return points won %)
    """
    s = _clamp(float(service_pts_w))
    o = _clamp(1.0 - float(opp_return_pts_w))
    return _clamp(_logistic((_logit(s) + _logit(o)) / 2.0))


def _hold_prob_from_point(p: float) -> float:
    """P(server holds a service game) from point-win prob p (includes deuce)."""
    p = _clamp(p)
    q = 1.0 - p
    first = p**4 * (1 + 4 * q + 10 * q * q)
    deuce_reach = 20 * (p**3) * (q**3)
    deuce_win = (p * p) / (1 - 2 * p * q)
    return _clamp(first + deuce_reach * deuce_win)


def _tiebreak_win_prob(pAserve: float, pBserve: float, A_serves_first: bool) -> float:
    """Probability that player A wins a tiebreak.

    pAserve: P(A wins point | A serves)
    pBserve: P(B wins point | B serves)
    A_serves_first: whether A serves first point of the tiebreak

    Notes:
    - Tiebreak is first to 7, win by 2.
    - Serving order: 1 point by first server, then 2 points by the other, then 2 by the first, repeating.
    - This implementation avoids unbounded recursion by using an exact closed-form for the "deuce" region (6-6+),
      where the score difference is a random walk with absorbing boundaries at +/-2.
    """
    # Clamp inputs defensively
    pAserve = _clamp01(pAserve)
    pBserve = _clamp01(pBserve)

    # Convenience: P(A wins point) when server is A vs B
    p_A_on_Aserve = pAserve
    p_A_on_Bserve = 1.0 - pBserve

    def server_for_point(k: int) -> str:
        """Return 'A' or 'B' for point number k (0-indexed) in the tiebreak."""
        if k == 0:
            return "A" if A_serves_first else "B"
        block = (k - 1) // 2  # blocks of 2 points after first
        if block % 2 == 0:
            return "B" if A_serves_first else "A"
        return "A" if A_serves_first else "B"

    def pA_wins_point_at(k: int) -> float:
        srv = server_for_point(k)
        return p_A_on_Aserve if srv == "A" else p_A_on_Bserve

    def deuce_prob(diff: int, k_mod4: int) -> float:
        """Exact win probability for A from scores where both players have >=6 points.

        diff = a - b in {-1,0,1} for non-terminal states (terminal is +/-2).
        k_mod4 = (a + b) % 4 indicates who serves next.
        """
        # Terminal by 2
        if diff >= 2:
            return 1.0
        if diff <= -2:
            return 0.0

        # Next server depends on k mod 4
        # For A_serves_first:
        #   k mod4: 0->A, 1->B, 2->B, 3->A
        # For B_serves_first it flips.
        p0 = p_A_on_Aserve  # A serves next
        p1 = p_A_on_Bserve  # B serves next

        # When tie (diff==0), k is even => k_mod4 in {0,2}. Win prob from tie is identical across these mods.
        # Let x = P(A eventually wins | tie with k even). Derivation over the 2-point cycles gives:
        # x = (p0*p1) / (1 - (p0 + p1 - 2*p0*p1))
        c = p0 + p1 - 2.0 * p0 * p1
        denom = 1.0 - c
        if abs(denom) < 1e-12:
            # Extremely degenerate case (numerical); fall back to 0.5
            x = 0.5
        else:
            x = (p0 * p1) / denom

        km = k_mod4 % 4
        if diff == 0:
            return _clamp01(x)

        # diff = +1 or -1 can only occur when k is odd => km in {1,3}
        if diff == 1:
            # If B serves next (km==1): win now with p1, else return to tie with (1-p1)
            if km == 1:
                return _clamp01(p1 + (1.0 - p1) * x)
            # If A serves next (km==3)
            return _clamp01(p0 + (1.0 - p0) * x)

        # diff == -1
        if km == 1:
            return _clamp01(p1 * x)  # must win next point to return to tie
        return _clamp01(p0 * x)

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(a: int, b: int) -> float:
        # Terminal (win by 2, at least 7)
        if (a >= 7 or b >= 7) and abs(a - b) >= 2:
            return 1.0 if a > b else 0.0

        # Deuce region (6-6 and beyond): use exact closed form to avoid infinite recursion
        if a >= 6 and b >= 6:
            return deuce_prob(a - b, (a + b) % 4)

        k = a + b
        p = pA_wins_point_at(k)
        return p * dp(a + 1, b) + (1.0 - p) * dp(a, b + 1)

    return _clamp01(dp(0, 0))


def _set_score_distribution(
    holdA: float, holdB: float, pAserve: float, pBserve: float, A_serves_first: bool
) -> dict[tuple[int, int], float]:
    """Game-level DP with server alternation and TB at 6-6."""
    holdA = _clamp(holdA)
    holdB = _clamp(holdB)
    start_server = 0 if A_serves_first else 1  # 0=A serves next game; 1=B serves next game

    states: dict[tuple[int, int, int], float] = {(0, 0, start_server): 1.0}
    finals: dict[tuple[int, int], float] = {}

    while states:
        (a, b, srv), prob = states.popitem()

        if (a >= 6 or b >= 6) and abs(a - b) >= 2:
            finals[(a, b)] = finals.get((a, b), 0.0) + prob
            continue
        if a == 7 or b == 7:
            finals[(a, b)] = finals.get((a, b), 0.0) + prob
            continue

        if a == 6 and b == 6:
            tbA = _tiebreak_win_prob(pAserve, pBserve, A_serves_first=(srv == 0))
            finals[(7, 6)] = finals.get((7, 6), 0.0) + prob * tbA
            finals[(6, 7)] = finals.get((6, 7), 0.0) + prob * (1.0 - tbA)
            continue

        if srv == 0:
            p_hold = holdA
            win_state = (a + 1, b, 1)  # A holds
            lose_state = (a, b + 1, 1)  # A broken
        else:
            p_hold = holdB
            win_state = (a, b + 1, 0)  # B holds -> B gets game
            lose_state = (a + 1, b, 0)  # B broken -> A gets game

        states[win_state] = states.get(win_state, 0.0) + prob * p_hold
        states[lose_state] = states.get(lose_state, 0.0) + prob * (1.0 - p_hold)

    return finals


def _summarize_set(dist: dict[tuple[int, int], float]) -> dict[str, float]:
    """Summarize a set score distribution.

    Returns:
        dict with:
          - p_set_win_A: probability A wins the set
          - expected_games: expected games in the set
          - expected_set_games: alias for expected_games
    """
    p_set_win = sum(p for (ga, gb), p in dist.items() if ga > gb)
    e_games = sum((ga + gb) * p for (ga, gb), p in dist.items())
    e_games = float(e_games)
    return {
        "p_set_win_A": float(p_set_win),
        "expected_games": e_games,
        "expected_set_games": e_games,
    }


def _surface_key(surface: str) -> str:
    s = (surface or "").lower()
    if "hard" in s:
        return "hard"
    if "clay" in s:
        return "clay"
    if "grass" in s:
        return "grass"
    # Default to hard when unknown to avoid falling into "all"/missing stats.
    return "hard"


# ---- Player surface stats provider (CSV -> dict) ----
_PLAYER_STATS: Optional[dict[tuple[str, str, str, str], dict[str, Any]]] = None
_PLAYER_STATS_ID: Optional[dict[tuple[str, str, str, int], dict[str, Any]]] = None
_PLAYER_STATS_ID_AVG: Optional[dict[tuple[str, str, str], dict[str, Any]]] = None
_PLAYER_STATS_AVG: Optional[dict[tuple[str, str, str], dict[str, Any]]] = None
_PLAYER_STATS_PATH: Optional[Path] = None
_PLAYER_STATS_NAMES_TS: Optional[dict[tuple[str, str], set[str]]] = None
_PLAYER_STATS_TOKEN_INDEX_TS: Optional[dict[tuple[str, str], dict[str, set[str]]]] = None


def _name_tokens(name_norm: Optional[str]) -> list[str]:
    if not name_norm:
        return []
    return [t for t in str(name_norm).split() if t]


def _name_tokens_meaningful(name_norm: Optional[str]) -> list[str]:
    return [t for t in _name_tokens(name_norm) if len(t) > 1]


def _build_token_index(names: set[str]) -> dict[str, set[str]]:
    idx: dict[str, set[str]] = {}
    for nm in names:
        for tok in _name_tokens_meaningful(nm):
            idx.setdefault(tok, set()).add(nm)
    return idx


def _name_match_score(query_norm: str, cand_norm: str) -> float:
    seq = SequenceMatcher(None, query_norm, cand_norm).ratio()
    q = set(_name_tokens_meaningful(query_norm))
    c = set(_name_tokens_meaningful(cand_norm))
    tok = (2.0 * len(q & c) / (len(q) + len(c))) if q and c else 0.0
    q_last = _name_tokens(query_norm)[-1] if _name_tokens(query_norm) else ""
    c_last = _name_tokens(cand_norm)[-1] if _name_tokens(cand_norm) else ""
    bonus = 0.06 if q_last and c_last and q_last == c_last else 0.0
    return float(0.65 * seq + 0.35 * tok + bonus)


def _resolve_name_match(
    player_name: str,
    pool: set[str],
    token_index: Optional[dict[str, set[str]]] = None,
) -> dict[str, Any]:
    base = _norm_name_stats(player_name) or _norm_name(player_name) or ""
    out: dict[str, Any] = {
        "input": player_name,
        "base_norm": base,
        "chosen_name": None,
        "method": "not_found",
        "score": None,
        "ambiguous": False,
        "quality_flags": [],
        "top_candidates": [],
        "reason": None,
    }
    if not base or not pool:
        out["reason"] = "no_input_or_pool"
        return out

    toks = _name_tokens(base)
    if any(len(t) == 1 for t in toks):
        out["quality_flags"].append("name_abbrev_detected")

    tried: set[str] = set()
    chain: list[tuple[str, str]] = []

    def add_candidate(nm: Optional[str], method: str) -> None:
        if not nm:
            return
        if nm in tried:
            return
        tried.add(nm)
        chain.append((nm, method))

    add_candidate(base, "exact_norm")
    for v in _name_variants_for_stats(base):
        if v != base:
            add_candidate(v, "variant")
    for v in _name_variants_for_stats(base):
        parts = _name_tokens(v)
        if len(parts) >= 2:
            add_candidate(" ".join([parts[-1]] + parts[:-1]), "swapped")

    # initials expansion: "j sinner" -> names in pool with same last name and first startswith("j")
    if len(toks) >= 2 and len(toks[0]) == 1:
        initial = toks[0]
        last = toks[-1]
        for cand in pool:
            cp = _name_tokens(cand)
            if len(cp) >= 2 and cp[-1] == last and cp[0].startswith(initial):
                add_candidate(cand, "initials_expansion")

    for nm, method in chain:
        if nm in pool:
            out["chosen_name"] = nm
            out["method"] = method
            out["score"] = 1.0
            out["reason"] = "matched_in_chain"
            return out

    # Fuzzy token candidate pool
    meaningful = _name_tokens_meaningful(base)
    candidates: set[str] = set()
    if token_index is not None:
        for tok in meaningful:
            for cand in token_index.get(tok, set()):
                candidates.add(cand)
    if not candidates:
        candidates = set(pool)

    scored: list[tuple[float, str]] = []
    for cand in candidates:
        scored.append((_name_match_score(base, cand), cand))
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[:8]
    out["top_candidates"] = [{"name": nm, "score": round(sc, 4)} for sc, nm in top]
    if not top:
        out["reason"] = "no_candidates"
        return out

    top_score, top_name = top[0]
    second_score = top[1][0] if len(top) > 1 else 0.0
    min_score = 0.86
    min_gap = 0.03

    if top_score >= min_score and (top_score - second_score >= min_gap or top_score >= 0.93):
        out["chosen_name"] = top_name
        out["method"] = "fuzzy_token"
        out["score"] = float(top_score)
        out["reason"] = "matched_fuzzy"
        out["quality_flags"].append("fuzzy_match_used")
        return out

    if top_score >= 0.82:
        out["ambiguous"] = True
        out["reason"] = "ambiguous_fuzzy_match"
        out["quality_flags"].append("ambiguous_name_match")
        return out

    out["reason"] = "low_fuzzy_score"
    return out
def _match_prob_from_set_prob(p_set: float, best_of: int) -> float:
    """Convert per-set win probability to match win probability for best-of-N sets."""
    p = _clamp01(p_set)
    if best_of == 3:
        # P(win in 2) + P(win in 3)
        return (p * p) * (3.0 - 2.0 * p)
    if best_of == 5:
        # sum_{k=3..5} C(5,k) p^k (1-p)^(5-k) = p^3 (10 - 15p + 6p^2)
        return (p ** 3) * (10.0 - 15.0 * p + 6.0 * (p ** 2))
    # generic fallback
    n = (best_of // 2) + 1
    out = 0.0
    q = 1.0 - p
    for k in range(n, best_of + 1):
        out += math.comb(best_of, k) * (p ** k) * (q ** (best_of - k))
    return _clamp01(out)


def _invert_match_to_set_prob(p_match: float, best_of: int) -> float:
    """Invert match win prob -> per-set win prob via binary search."""
    target = _clamp01(p_match)
    lo, hi = 1e-6, 1.0 - 1e-6
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _match_prob_from_set_prob(mid, best_of) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _base_serve_point_win(tour: str, surface: str) -> float:
    """Reasonable tour/surface baseline for service point-win probability."""
    t = (tour or "").upper()
    s = (surface or "").lower()
    base = 0.64 if t == "ATP" else 0.60
    if "grass" in s:
        base += 0.008
    elif "clay" in s:
        base -= 0.008
    if "indoor" in s:
        base += 0.004
    return _clamp(base, 0.52, 0.72)


def _calibrate_serve_points_to_set_prob(
    base_avg: float, p_set_target: float, tour: str, surface: str
) -> Tuple[float, float, float, float]:
    """
    Option C:
    When player surface stats are missing (or even when present), calibrate the
    serve point-win rates so the implied per-set win probability matches the
    model's match win probability (converted -> per-set).

    Returns (pAserve, pBserve, p_set_est, delta_used)
    """
    base = _clamp(base_avg, 0.52, 0.72)
    if not (0.0 < base < 1.0):
        base = _base_serve_point_win(tour, surface)

    target = _clamp01(p_set_target)
    # If target ~ 0.5, keep symmetric
    if abs(target - 0.5) < 1e-6:
        pA = base
        pB = base
        holdA = _hold_prob_from_point(pA)
        holdB = _hold_prob_from_point(pB)
        _setA = _set_score_distribution(holdA, holdB, pA, pB, True)
        _setB = _set_score_distribution(holdA, holdB, pA, pB, False)
        _setDist = {k: 0.5 * (_setA.get(k, 0.0) + _setB.get(k, 0.0)) for k in (set(_setA) | set(_setB))}
        p_set = _summarize_set(_setDist).get("p_set_win_A", 0.5)
        return pA, pB, p_set, 0.0

    sign = 1.0 if target >= 0.5 else -1.0
    target_t = sign * (target - 0.5)  # >= 0

    lo, hi = 0.0, 0.25
    best = (base, base, 0.5, 0.0, 1e9)  # pA, pB, p_set, d, err

    for _ in range(55):
        d = (lo + hi) / 2.0
        pA = _clamp(base + sign * d, 0.48, 0.80)
        pB = _clamp(base - sign * d, 0.48, 0.80)
        holdA = _hold_prob_from_point(pA)
        holdB = _hold_prob_from_point(pB)
        _setA = _set_score_distribution(holdA, holdB, pA, pB, True)
        _setB = _set_score_distribution(holdA, holdB, pA, pB, False)
        _setDist = {k: 0.5 * (_setA.get(k, 0.0) + _setB.get(k, 0.0)) for k in (set(_setA) | set(_setB))}
        p_set = _summarize_set(_setDist).get("p_set_win_A", 0.5)
        val_t = sign * (p_set - 0.5)  # monotone increasing in d
        err = val_t - target_t

        if abs(err) < best[4]:
            best = (pA, pB, p_set, d, abs(err))

        if err < 0:
            lo = d
        else:
            hi = d

    return best[0], best[1], best[2], best[3]



def _find_player_stats_path() -> Optional[Path]:
    """Locate the consolidated player surface stats CSV.

    We prefer a single consolidated file (player_surface_stats.csv), but also support
    alternate filenames so you can drop in a rebuilt export without changing code.
    """
    env = os.getenv("PLAYER_SURFACE_STATS_PATH")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))

    here = Path(__file__).resolve()
    candidates += [
        # most common
        Path.cwd() / "player_surface_stats.csv",
        Path.cwd() / "app" / "data" / "player_surface_stats.csv",
        here.parents[1] / "data" / "player_surface_stats.csv",  # api/app/data

        # alternates you may generate locally
        Path.cwd() / "player_surface_stats_final.csv",
        Path.cwd() / "player_surface_stats_fixed.csv",
        Path.cwd() / "app" / "data" / "player_surface_stats_final.csv",
        Path.cwd() / "app" / "data" / "player_surface_stats_fixed.csv",
        here.parents[1] / "data" / "player_surface_stats_final.csv",
        here.parents[1] / "data" / "player_surface_stats_fixed.csv",
        # clay-section export
        Path.cwd() / "player_surface_stats_with_clay_section.csv",
        Path.cwd() / "app" / "data" / "player_surface_stats_with_clay_section.csv",
        here.parents[1] / "data" / "player_surface_stats_with_clay_section.csv",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def _load_player_stats_if_needed() -> None:
    """Load player surface stats into in-memory indexes.

    Expected CSV columns (your player_surface_stats_final.csv matches this):
      - tour, window, surface, player
      - svc_service_pts_win_pct, ret_return_pts_win_pct
      - svc_matches, ret_matches
      - svc_aces_per_game, ret_opp_aces_per_game (optional but recommended)
      - svc_bp_save_pct, ret_bp_win_pct (optional but recommended)

    Note: these *are already percentages/rates*, so we do NOT divide by match count.
    """
    global _PLAYER_STATS, _PLAYER_STATS_AVG, _PLAYER_STATS_ID, _PLAYER_STATS_ID_AVG, _PLAYER_STATS_PATH
    global _PLAYER_STATS_NAMES_TS, _PLAYER_STATS_TOKEN_INDEX_TS
    if _PLAYER_STATS is not None:
        # If we previously loaded successfully, keep cache.
        if _PLAYER_STATS_PATH is not None:
            return
        # If we cached an empty/missing state (e.g., file unavailable at startup),
        # retry on subsequent calls so a later-mounted file is picked up.
        if isinstance(_PLAYER_STATS, dict) and len(_PLAYER_STATS) > 0:
            return

    path = _find_player_stats_path()
    if path is None:
        _PLAYER_STATS = {}
        _PLAYER_STATS_AVG = {}
        _PLAYER_STATS_ID = {}
        _PLAYER_STATS_ID_AVG = {}
        _PLAYER_STATS_PATH = None
        _PLAYER_STATS_NAMES_TS = {}
        _PLAYER_STATS_TOKEN_INDEX_TS = {}
        return
    _PLAYER_STATS_PATH = path
    logger.info("player_surface_stats: loading from %s", path)

    idx_name: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    idx_id: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    names_ts: dict[tuple[str, str], set[str]] = {}

    # Averages keyed by (tour, window, surface) to support fallbacks.
    sums: dict[tuple[str, str, str], dict[str, float]] = {}
    cnts: dict[tuple[str, str, str], dict[str, int]] = {}

    def to_float(v) -> Optional[float]:
        try:
            if v is None:
                return None
            sv = str(v).strip()
            if sv == "" or sv.lower() == "nan":
                return None
            return float(sv)
        except Exception:
            return None

    def to_int(v) -> int:
        try:
            if v is None:
                return 0
            sv = str(v).strip()
            if sv == "" or sv.lower() == "nan":
                return 0
            return int(float(sv))
        except Exception:
            return 0

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            tour = str(r.get("tour") or "").upper().strip()
            window = str(r.get("window") or "").lower().strip()  # e.g. "12 month" / "all time"
            surface = str(r.get("surface") or "").lower().strip()
            player = _norm_name_stats(r.get("player") or r.get("player_name") or r.get("name")) or ""

            # Optional stable identifier (if present in the stats file)
            ta_raw = (
                r.get("ta_id")
                or r.get("player_id")
                or r.get("tennisabstract_id")
                or r.get("tennis_abstract_id")
                or r.get("taId")
            )
            ta_id: Optional[int] = None
            try:
                if ta_raw not in (None, ""):
                    ta_id = int(float(str(ta_raw).strip()))
            except Exception:
                ta_id = None

            if not tour or not window or not surface or not player:
                continue

            # Rates/pcts (already scaled to [0,1] where appropriate)
            svc_pts = to_float(r.get("svc_service_pts_win_pct"))
            ret_pts = to_float(r.get("ret_return_pts_win_pct"))
            svc_dfs_pg = to_float(r.get("svc_dfs_per_game"))
            svc_first_pct = to_float(r.get("svc_first_serve_pct"))
            svc_first_win = to_float(r.get("svc_first_serve_win_pct"))
            svc_second_win = to_float(r.get("svc_second_serve_win_pct"))
            ret_first_win = to_float(r.get("ret_first_return_win_pct"))
            ret_second_win = to_float(r.get("ret_second_return_win_pct"))
            ret_opp_hold_pct = to_float(r.get("ret_opp_hold_pct"))

            # Match-counts (for reliability decisions)
            svc_n = to_int(r.get("svc_matches"))
            ret_n = to_int(r.get("ret_matches"))

            # Optional prop-related rates
            svc_aces_pg = to_float(r.get("svc_aces_per_game"))
            ret_opp_aces_pg = to_float(r.get("ret_opp_aces_per_game"))
            svc_bp_save_pct = to_float(r.get("svc_bp_save_pct"))
            ret_bp_win_pct = to_float(r.get("ret_bp_win_pct"))
            svc_hold_pct = to_float(r.get("svc_hold_pct"))
            ret_opp_hold_pct = to_float(r.get("ret_opp_hold_pct"))

            stats_row: dict[str, Any] = {
                "svc_pts": svc_pts,
                "ret_pts": ret_pts,
                "svc_n": svc_n,
                "ret_n": ret_n,
                "svc_aces_pg": svc_aces_pg,
                "svc_dfs_pg": svc_dfs_pg,
                "svc_first_pct": svc_first_pct,
                "svc_first_win": svc_first_win,
                "svc_second_win": svc_second_win,
                "ret_first_win": ret_first_win,
                "ret_second_win": ret_second_win,
                "ret_opp_aces_pg": ret_opp_aces_pg,
                "svc_bp_save_pct": svc_bp_save_pct,
                "ret_bp_win_pct": ret_bp_win_pct,
                "svc_hold_pct": svc_hold_pct,
                "ret_opp_hold_pct": ret_opp_hold_pct,
            }

            idx_name[(tour, window, surface, player)] = stats_row
            if ta_id is not None:
                idx_id[(tour, window, surface, ta_id)] = stats_row
            names_ts.setdefault((tour, surface), set()).add(player)

            key = (tour, window, surface)
            if key not in sums:
                sums[key] = {}
                cnts[key] = {}
            for k2 in (
                "svc_pts",
                "ret_pts",
                "svc_aces_pg",
                "svc_dfs_pg",
                "svc_first_pct",
                "svc_first_win",
                "svc_second_win",
                "ret_first_win",
                "ret_second_win",
                "ret_opp_aces_pg",
                "svc_bp_save_pct",
                "ret_bp_win_pct",
                "svc_hold_pct",
                "ret_opp_hold_pct",
            ):
                v2 = stats_row.get(k2)
                if v2 is None:
                    continue
                sums[key][k2] = sums[key].get(k2, 0.0) + float(v2)
                cnts[key][k2] = cnts[key].get(k2, 0) + 1

    avgs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for k in sums.keys():
        avgs[k] = {}
        for metric, total in sums[k].items():
            c = cnts[k].get(metric, 0)
            if c > 0:
                avgs[k][metric] = float(total) / float(c)

    _PLAYER_STATS = idx_name
    _PLAYER_STATS_ID = idx_id
    _PLAYER_STATS_AVG = avgs
    _PLAYER_STATS_ID_AVG = avgs
    _PLAYER_STATS_NAMES_TS = names_ts
    _PLAYER_STATS_TOKEN_INDEX_TS = {k: _build_token_index(v) for k, v in names_ts.items()}
    logger.info(
        "player_surface_stats: loaded %d name-keys, %d id-keys, %d avg-keys  sample_keys=%s",
        len(idx_name), len(idx_id), len(avgs),
        list(idx_name.keys())[:3],
    )


def _load_player_surface_stats() -> None:
    """Backward-compatible alias for older code paths."""
    _load_player_stats_if_needed()


def _lookup_player_stats_row(
    tour: str,
    window: str,
    surface: str,
    player_name: Optional[str],
    *,
    ta_id: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """Return the raw stats row from the CSV for a given (tour, window, surface, player)."""
    _load_player_surface_stats()
    t = (tour or "").upper().strip()
    w = (window or "").lower().strip()
    s = _surface_key(surface)
    pname = _norm_name_stats(player_name)

    # 1) ID lookup
    if ta_id is not None and _PLAYER_STATS_ID is not None:
        try:
            ta_int = int(ta_id)
        except Exception:
            ta_int = None
        if ta_int is not None:
            row = _PLAYER_STATS_ID.get((t, w, s, ta_int))
            if row:
                return row

    # 2) Name lookup
    if pname and _PLAYER_STATS is not None:
        for v in _name_variants_for_stats(pname):
            row = _PLAYER_STATS.get((t, w, s, v))
            if row:
                return row

            # try "Last First"
            parts = v.split()
            if len(parts) >= 2:
                swapped = " ".join([parts[-1]] + parts[:-1])
                row = _PLAYER_STATS.get((t, w, s, swapped))
                if row:
                    return row

    return None


def _lookup_player_stats_row_by_norm_name(
    tour: str,
    window: str,
    surface: str,
    resolved_name_norm: Optional[str],
) -> Optional[dict[str, Any]]:
    _load_player_surface_stats()
    if not resolved_name_norm or _PLAYER_STATS is None:
        return None
    t = (tour or "").upper().strip()
    w = (window or "").lower().strip()
    s = _surface_key(surface)
    return _PLAYER_STATS.get((t, w, s, resolved_name_norm))


def _pick_window_for_metric(
    row_12m: Optional[dict[str, Any]],
    row_all: Optional[dict[str, Any]],
    metric_key: str,
    n_key: str,
    *,
    min_matches: int = 5,
) -> tuple[Optional[float], Optional[str]]:
    """Pick 12-month if reliable, else all-time."""
    if row_12m and row_12m.get(metric_key) is not None and int(row_12m.get(n_key) or 0) >= min_matches:
        return float(row_12m.get(metric_key)), "12 month"
    if row_all and row_all.get(metric_key) is not None and int(row_all.get(n_key) or 0) >= 1:
        return float(row_all.get(metric_key)), "all time"
    # fall back to whichever exists even if match-count is small
    if row_12m and row_12m.get(metric_key) is not None:
        return float(row_12m.get(metric_key)), "12 month"
    if row_all and row_all.get(metric_key) is not None:
        return float(row_all.get(metric_key)), "all time"
    return None, None


def _get_player_rates(
    tour: str,
    surface: str,
    player_name: Optional[str],
    *,
    ta_id: Optional[int] = None,
) -> tuple[float, float]:
    """Return (p_service_point_won, p_return_point_won) from the consolidated CSV.

    We try "12 month" first, then "all time". If missing, we fall back to tour/surface averages,
    and finally to sane tour defaults.
    """
    _load_player_surface_stats()

    t = (tour or "").upper().strip()
    s = _surface_key(surface)

    row_12m = _lookup_player_stats_row(t, "12 month", s, player_name, ta_id=ta_id)
    row_all = _lookup_player_stats_row(t, "all time", s, player_name, ta_id=ta_id)

    p_svc, _w_svc = _pick_window_for_metric(row_12m, row_all, "svc_pts", "svc_n", min_matches=5)
    p_ret, _w_ret = _pick_window_for_metric(row_12m, row_all, "ret_pts", "ret_n", min_matches=5)

    # tour/window/surface averages (prefer 12-month averages)
    if (p_svc is None or p_ret is None) and _PLAYER_STATS_AVG is not None:
        avg12 = _PLAYER_STATS_AVG.get((t, "12 month", s), {}) if isinstance(_PLAYER_STATS_AVG, dict) else {}
        avgall = _PLAYER_STATS_AVG.get((t, "all time", s), {}) if isinstance(_PLAYER_STATS_AVG, dict) else {}

        if p_svc is None:
            p_svc = avg12.get("svc_pts") or avgall.get("svc_pts")
        if p_ret is None:
            p_ret = avg12.get("ret_pts") or avgall.get("ret_pts")

    # final fallback: sane tour defaults
    if p_svc is None or p_ret is None:
        if t == "WTA":
            p_svc = 0.60 if p_svc is None else p_svc
            p_ret = 0.40 if p_ret is None else p_ret
        else:
            p_svc = 0.62 if p_svc is None else p_svc
            p_ret = 0.38 if p_ret is None else p_ret

    return (_clamp01(float(p_svc)), _clamp01(float(p_ret)))


def _get_player_style_metrics(
    tour: str,
    player_name: Optional[str],
    *,
    ta_id: Optional[int] = None,
    overall: bool = True,
) -> dict[str, Optional[float]]:
    """Return overall (all-surfaces) style metrics from player_surface_stats."""
    _load_player_surface_stats()

    t = (tour or "").upper().strip()
    surfaces = ("hard", "clay", "grass")

    svc_metrics = {
        "svc_pts",
        "svc_aces_pg",
        "svc_dfs_pg",
        "svc_first_pct",
        "svc_first_win",
        "svc_second_win",
        "svc_bp_save_pct",
        "svc_hold_pct",
    }
    ret_metrics = {
        "ret_pts",
        "ret_first_win",
        "ret_second_win",
        "ret_bp_win_pct",
        "ret_opp_aces_pg",
        "ret_opp_hold_pct",
    }

    def _rows(window: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for s in surfaces:
            r = _lookup_player_stats_row(t, window, s, player_name, ta_id=ta_id)
            if r:
                rows.append(r)
        return rows

    rows_12m = _rows("12 month")
    rows_all = _rows("all time")

    def _weighted_avg(rows: list[dict[str, Any]], metric: str) -> Optional[float]:
        if not rows:
            return None
        total = 0.0
        wsum = 0.0
        for r in rows:
            v = r.get(metric)
            if v is None:
                continue
            n = r.get("svc_n") if metric in svc_metrics else r.get("ret_n")
            try:
                w = float(n) if n is not None else 0.0
            except Exception:
                w = 0.0
            if w <= 0:
                continue
            total += float(v) * w
            wsum += w
        if wsum > 0:
            return total / wsum
        return None

    def _avg_fallback(window: str, metric: str) -> Optional[float]:
        if _PLAYER_STATS_AVG is None or not isinstance(_PLAYER_STATS_AVG, dict):
            return None
        vals: list[float] = []
        for s in surfaces:
            v = _PLAYER_STATS_AVG.get((t, window, s), {}).get(metric)
            if v is not None:
                vals.append(float(v))
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def pick(metric: str) -> Optional[float]:
        v = _weighted_avg(rows_12m, metric)
        if v is not None:
            return v
        v = _weighted_avg(rows_all, metric)
        if v is not None:
            return v
        v = _avg_fallback("12 month", metric)
        if v is not None:
            return v
        return _avg_fallback("all time", metric)

    return {
        "svc_pts": pick("svc_pts"),
        "ret_pts": pick("ret_pts"),
        "ace_pg": pick("svc_aces_pg"),
        "df_pg": pick("svc_dfs_pg"),
        "first_in": pick("svc_first_pct"),
        "first_win": pick("svc_first_win"),
        "second_win": pick("svc_second_win"),
        "bp_save": pick("svc_bp_save_pct"),
        "hold": pick("svc_hold_pct"),
        "ret_first_win": pick("ret_first_win"),
        "ret_second_win": pick("ret_second_win"),
        "bp_win": pick("ret_bp_win_pct"),
        "ret_opp_aces": pick("ret_opp_aces_pg"),
        "ret_opp_hold": pick("ret_opp_hold_pct"),
    }


def _style_label(style: dict[str, Optional[float]]) -> Optional[str]:
    if not style:
        return None

    svc_pts = style.get("svc_pts")
    ret_pts = style.get("ret_pts")
    ace_pg = style.get("ace_pg")
    df_pg = style.get("df_pg")
    first_in = style.get("first_in")
    first_win = style.get("first_win")
    second_win = style.get("second_win")
    ret_first = style.get("ret_first_win")
    ret_second = style.get("ret_second_win")
    bp_win = style.get("bp_win")
    bp_save = style.get("bp_save")

    if all(v is None for v in (svc_pts, ret_pts, ace_pg, df_pg, first_in, first_win, second_win, ret_first, ret_second, bp_win, bp_save)):
        return None

    # Heuristic scoring (overall style, not surface-specific)
    serve_strength = (svc_pts or 0.0) + 0.25 * (first_win or 0.0) + 0.15 * (second_win or 0.0)
    return_strength = (ret_pts or 0.0) + 0.35 * (ret_first or 0.0) + 0.35 * (ret_second or 0.0) + 0.1 * (bp_win or 0.0)
    power_score = (ace_pg or 0.0) * 1.5 + (first_win or 0.0) * 0.4
    consistency_pen = (df_pg or 0.0) * 0.8

    # Big Server / Power Player
    if power_score >= 1.0 and serve_strength >= (return_strength + 0.15):
        return "big_server_power"

    # Serve-and-Volleyer (very serve-heavy, high first-strike success, weaker return)
    if power_score >= 0.85 and (first_in or 0.0) >= 0.60 and (first_win or 0.0) >= 0.70 and (ret_pts or 0.0) <= 0.38:
        return "serve_and_volley"

    # Counterpuncher (strong return, low power)
    if return_strength >= (serve_strength + 0.12) and (ace_pg or 0.0) <= 0.25:
        return "counterpuncher"

    # Aggressive Baseliner (power + higher errors)
    if power_score >= 0.65 and consistency_pen >= 0.25 and (ret_pts or 0.0) >= 0.38:
        return "aggressive_baseliner"

    # Pusher (low power, low errors, relies on consistency)
    if (ace_pg or 0.0) <= 0.20 and (df_pg or 0.0) <= 0.20 and (ret_pts or 0.0) >= 0.38:
        return "pusher"

    # All-Court (balanced, above-average both sides)
    if serve_strength >= 0.70 and return_strength >= 0.75:
        return "all_court"

    # Baseliner default
    return "baseliner"


def _style_summary(
    p1_name: Optional[str],
    p2_name: Optional[str],
    p1_style: dict[str, Optional[float]],
    p2_style: dict[str, Optional[float]],
) -> dict[str, Any]:
    return {
        "p1": {"name": p1_name, "label": _style_label(p1_style)},
        "p2": {"name": p2_name, "label": _style_label(p2_style)},
    }


def _fmt_pct(p: Optional[float]) -> Optional[str]:
    if p is None:
        return None
    try:
        return f"{float(p) * 100.0:.1f}%"
    except Exception:
        return None


def _pick_summary(
    pick_side: Optional[str],
    p1_name: Optional[str],
    p2_name: Optional[str],
    p1_prob: Optional[float],
    p2_prob: Optional[float],
    p1_nv: Optional[float],
    p2_nv: Optional[float],
    p1_odds: Optional[int],
    p2_odds: Optional[int],
    p1_wr20: Optional[float],
    p2_wr20: Optional[float],
    p1_surf_wr: Optional[float],
    p2_surf_wr: Optional[float],
    p1_svc_pts_w: Optional[float],
    p2_svc_pts_w: Optional[float],
    p1_ret_pts_w: Optional[float],
    p2_ret_pts_w: Optional[float],
    d_rest_days: Optional[float],
    d_matches_10d: Optional[float],
    h2h_p1_win_pct: Optional[float],
    h2h_total: Optional[int],
    style_summary: Optional[dict[str, Any]],
    *,
    # --- new params for richer summary ---
    surface: Optional[str] = None,
    p1_elo: Optional[float] = None,
    p2_elo: Optional[float] = None,
    p1_rank: Optional[int] = None,
    p2_rank: Optional[int] = None,
    h2h_p1_wins: int = 0,
    h2h_p2_wins: int = 0,
    h2h_surface_p1_wins: int = 0,
    h2h_surface_p2_wins: int = 0,
    h2h_surface_total: int = 0,
    individual_predictions: Optional[list] = None,
    p1_ace_pg: Optional[float] = None,
    p2_ace_pg: Optional[float] = None,
    p1_bp_save: Optional[float] = None,
    p2_bp_save: Optional[float] = None,
    p1_bp_win: Optional[float] = None,
    p2_bp_win: Optional[float] = None,
    p1_matches_played: Optional[int] = None,
    p2_matches_played: Optional[int] = None,
    elo_used_median: bool = False,
) -> Optional[str]:
    """Build a concise, readable pick summary with structured sections.

    Output format:
        {intro}  ←  who over whom, odds stance, value verdict
        Key factors: factor1; factor2; ...
        Risks: risk1; risk2; ...
    """
    if pick_side not in ("p1", "p2"):
        return None

    pick_name = p1_name if pick_side == "p1" else p2_name
    opp_name = p2_name if pick_side == "p1" else p1_name
    pick_prob = p1_prob if pick_side == "p1" else p2_prob
    opp_prob = p2_prob if pick_side == "p1" else p1_prob
    pick_nv = p1_nv if pick_side == "p1" else p2_nv
    odds = p1_odds if pick_side == "p1" else p2_odds
    opp_odds = p2_odds if pick_side == "p1" else p1_odds

    # Helper: pick-side value selector (returns pick's value first, opp second)
    def _pick(v1, v2):
        return (v1, v2) if pick_side == "p1" else (v2, v1)

    pick_elo, opp_elo = _pick(p1_elo, p2_elo)
    pick_rank, opp_rank = _pick(p1_rank, p2_rank)
    pick_wr20, opp_wr20 = _pick(p1_wr20, p2_wr20)
    pick_surf, opp_surf = _pick(p1_surf_wr, p2_surf_wr)
    pick_svc, opp_svc = _pick(p1_svc_pts_w, p2_svc_pts_w)
    pick_ret, opp_ret = _pick(p1_ret_pts_w, p2_ret_pts_w)
    pick_ace, opp_ace = _pick(p1_ace_pg, p2_ace_pg)
    pick_bps, opp_bps = _pick(p1_bp_save, p2_bp_save)
    pick_bpw, opp_bpw = _pick(p1_bp_win, p2_bp_win)
    pick_mp, opp_mp = _pick(p1_matches_played, p2_matches_played)
    pick_h2h_w, opp_h2h_w = _pick(h2h_p1_wins, h2h_p2_wins)
    pick_h2h_sw, opp_h2h_sw = _pick(h2h_surface_p1_wins, h2h_surface_p2_wins)

    surf_label = _normalize_surface_family(surface).capitalize() if surface else None

    # --- 1. Intro line ---
    label = None
    if odds is not None:
        label = "Fav" if odds < 0 else "Dog"
    elif pick_prob is not None:
        label = "Fav" if pick_prob >= 0.55 else ("Dog" if pick_prob <= 0.45 else None)

    edge_pp = None
    edge_verdict = None
    if pick_prob is not None and pick_nv is not None:
        edge_pp = (pick_prob - pick_nv) * 100.0
        if edge_pp >= 8.0:
            edge_verdict = "strong value"
        elif edge_pp >= 4.0:
            edge_verdict = "value"
        elif edge_pp >= 1.0:
            edge_verdict = "slight value"
        elif edge_pp <= -6.0:
            edge_verdict = "overpriced"
        elif edge_pp <= -3.0:
            edge_verdict = "a bit rich"
        else:
            edge_verdict = "fairly priced"

    header = str(pick_name or "?") + " over " + str(opp_name or "?")
    tags: list[str] = []
    if label:
        tags.append(label)
    if odds is not None:
        tags.append(f"{odds:+d}")
    if tags:
        header += f" ({', '.join(tags)})"

    # Ranking context
    rank_note = ""
    if pick_rank is not None and opp_rank is not None:
        rank_note = f" Rank #{pick_rank} vs #{opp_rank}."
    elif pick_rank is not None:
        rank_note = f" Rank #{pick_rank}."

    # Prob + edge line
    prob_parts: list[str] = []
    if pick_prob is not None:
        prob_parts.append(f"Model {pick_prob*100:.1f}%")
    if pick_nv is not None:
        prob_parts.append(f"Mkt {pick_nv*100:.1f}%")
    if edge_pp is not None:
        sign = "+" if edge_pp >= 0 else ""
        prob_parts.append(f"Edge {sign}{edge_pp:.1f}pp")
    prob_line = " | ".join(prob_parts)
    if edge_verdict:
        prob_line += f" → {edge_verdict}" if prob_line else edge_verdict

    sentence1 = f"{header}.{rank_note}"
    if prob_line:
        sentence1 += f" {prob_line}."

    # --- 2. Key factors (advantages) ---
    factors: list[str] = []
    risks: list[str] = []

    # ELO edge
    if pick_elo is not None and opp_elo is not None and not elo_used_median:
        elo_diff = pick_elo - opp_elo
        if elo_diff >= 150:
            factors.append(f"ELO +{int(elo_diff)} (dominant)")
        elif elo_diff >= 75:
            factors.append(f"ELO +{int(elo_diff)} (clear edge)")
        elif elo_diff >= 30:
            factors.append(f"ELO +{int(elo_diff)}")
        elif elo_diff <= -75:
            risks.append(f"ELO deficit {int(elo_diff)}")
        elif elo_diff <= -30:
            risks.append(f"ELO slightly lower ({int(elo_diff)})")

    # Surface form
    if pick_surf is not None and opp_surf is not None:
        surf_diff = pick_surf - opp_surf
        surf_ctx = f" on {surf_label.lower()}" if surf_label and surf_label != "Unknown" else ""
        if surf_diff >= 0.15:
            factors.append(f"much stronger{surf_ctx} ({pick_surf*100:.0f}% vs {opp_surf*100:.0f}%)")
        elif surf_diff >= 0.06:
            factors.append(f"better{surf_ctx} form ({pick_surf*100:.0f}% vs {opp_surf*100:.0f}%)")
        elif surf_diff <= -0.06:
            risks.append(f"opp better{surf_ctx} ({opp_surf*100:.0f}% vs {pick_surf*100:.0f}%)")

    # Serve dominance
    if pick_svc is not None and opp_svc is not None:
        svc_diff = pick_svc - opp_svc
        if svc_diff >= 0.03:
            factors.append(f"stronger serve ({pick_svc*100:.1f}% vs {opp_svc*100:.1f}% pts won)")
        elif svc_diff >= 0.01:
            factors.append("serve edge")
        elif svc_diff <= -0.03:
            risks.append("opp has serve advantage")

    # Return pressure
    if pick_ret is not None and opp_ret is not None:
        ret_diff = pick_ret - opp_ret
        if ret_diff >= 0.03:
            factors.append(f"return pressure ({pick_ret*100:.1f}% vs {opp_ret*100:.1f}% pts won)")
        elif ret_diff >= 0.01:
            factors.append("return edge")
        elif ret_diff <= -0.03:
            risks.append("opp returns better")

    # Break point resilience
    if pick_bps is not None and opp_bps is not None:
        bps_diff = pick_bps - opp_bps
        if bps_diff >= 0.05:
            factors.append(f"clutch on BP save ({pick_bps*100:.0f}% vs {opp_bps*100:.0f}%)")
        elif bps_diff <= -0.05:
            risks.append(f"BP save gap ({pick_bps*100:.0f}% vs {opp_bps*100:.0f}%)")

    # Break point conversion
    if pick_bpw is not None and opp_bpw is not None:
        bpw_diff = pick_bpw - opp_bpw
        if bpw_diff >= 0.05:
            factors.append("better BP conversion")
        elif bpw_diff <= -0.05:
            risks.append("opp converts BPs better")

    # Recent form (last 20 matches)
    if pick_wr20 is not None and opp_wr20 is not None:
        form_diff = pick_wr20 - opp_wr20
        if form_diff >= 0.15:
            factors.append(f"hot form L20 ({pick_wr20*100:.0f}% vs {opp_wr20*100:.0f}%)")
        elif form_diff >= 0.07:
            factors.append(f"better form L20 ({pick_wr20*100:.0f}% vs {opp_wr20*100:.0f}%)")
        elif form_diff <= -0.07:
            risks.append(f"form trails L20 ({pick_wr20*100:.0f}% vs {opp_wr20*100:.0f}%)")

    # Rest & fatigue
    if d_rest_days is not None:
        rest = d_rest_days if pick_side == "p1" else -d_rest_days
        if rest >= 3:
            factors.append(f"+{int(rest)}d extra rest")
        elif rest >= 1:
            factors.append("fresher (extra rest)")
        elif rest <= -3:
            risks.append(f"short rest ({int(abs(rest))}d less)")
        elif rest <= -1:
            risks.append("less rest")

    if d_matches_10d is not None:
        load = d_matches_10d if pick_side == "p1" else -d_matches_10d
        if load <= -3:
            factors.append("lighter recent schedule")
        elif load <= -2:
            factors.append("lighter load (10d)")
        elif load >= 3:
            risks.append("heavy recent schedule")
        elif load >= 2:
            risks.append("heavier load (10d)")

    # H2H with actual record
    if h2h_total and h2h_total >= 2:
        pick_h2h_pct = pick_h2h_w / h2h_total if h2h_total > 0 else 0.5
        h2h_str = f"H2H {pick_h2h_w}-{opp_h2h_w}"
        if h2h_surface_total and h2h_surface_total >= 2 and surf_label and surf_label != "Unknown":
            h2h_str += f" ({pick_h2h_sw}-{opp_h2h_sw} {surf_label.lower()})"
        if pick_h2h_pct >= 0.7:
            factors.append(f"{h2h_str} — dominant")
        elif pick_h2h_pct >= 0.55:
            factors.append(h2h_str)
        elif pick_h2h_pct <= 0.3:
            risks.append(f"{h2h_str} — trails")
        elif pick_h2h_pct <= 0.45:
            risks.append(h2h_str)

    # Model agreement/disagreement (ELO vs XGB vs Market)
    if individual_predictions:
        pred_probs: dict[str, float] = {}
        for ip in individual_predictions:
            if not isinstance(ip, dict):
                continue
            name = str(ip.get("name") or "").lower()
            p1p = ip.get("p1_prob") or ip.get("p1Prob")
            if p1p is not None:
                try:
                    val = float(p1p)
                    pick_val = val if pick_side == "p1" else (1.0 - val)
                    pred_probs[name] = pick_val
                except (TypeError, ValueError):
                    pass
        if len(pred_probs) >= 2:
            vals = list(pred_probs.values())
            spread = max(vals) - min(vals)
            if spread <= 0.05:
                factors.append("models agree")
            elif spread >= 0.15:
                high_m = max(pred_probs, key=pred_probs.get)  # type: ignore
                low_m = min(pred_probs, key=pred_probs.get)  # type: ignore
                risks.append(f"models split ({high_m} {pred_probs[high_m]*100:.0f}% vs {low_m} {pred_probs[low_m]*100:.0f}%)")

    # Style matchup
    try:
        p1_sl = style_summary.get("p1", {}).get("label") if style_summary else None
        p2_sl = style_summary.get("p2", {}).get("label") if style_summary else None
        pick_style = p1_sl if pick_side == "p1" else p2_sl
        opp_style = p2_sl if pick_side == "p1" else p1_sl
        if pick_style and opp_style and pick_style != opp_style:
            factors.append(f"{pick_style.replace('_', ' ')} vs {opp_style.replace('_', ' ')}")
    except Exception:
        pass

    # Sample size warning
    if pick_mp is not None and pick_mp < 10:
        risks.append(f"small sample ({pick_mp} matches)")
    if elo_used_median:
        risks.append("ELO is estimated (median fill)")

    # --- 3. Assemble output ---
    parts = [sentence1]
    if factors:
        parts.append("Key factors: " + "; ".join(factors) + ".")
    if risks:
        parts.append("Risks: " + "; ".join(risks) + ".")
    if not factors and not risks:
        parts.append("Limited data for deeper analysis.")

    return " ".join(parts).strip() or None


def _style_debug(
    tour: str,
    surface: str,
    p1_name: Optional[str],
    p2_name: Optional[str],
    p1_ta_id: Optional[int],
    p2_ta_id: Optional[int],
) -> dict[str, Any]:
    _load_player_surface_stats()
    t = (tour or "").upper().strip()
    s = _surface_key(surface)
    surfaces = ("hard", "clay", "grass")

    def _found_any(name: Optional[str], ta_id: Optional[int], window: str) -> bool:
        for surf in surfaces:
            row = _lookup_player_stats_row(t, window, surf, name, ta_id=ta_id)
            if row is not None:
                return True
        return False

    return {
        "stats_path": str(_PLAYER_STATS_PATH) if _PLAYER_STATS_PATH else None,
        "p1_found_12m_any_surface": _found_any(p1_name, p1_ta_id, "12 month"),
        "p1_found_all_any_surface": _found_any(p1_name, p1_ta_id, "all time"),
        "p2_found_12m_any_surface": _found_any(p2_name, p2_ta_id, "12 month"),
        "p2_found_all_any_surface": _found_any(p2_name, p2_ta_id, "all time"),
        "surface_used_for_match": s,
    }


def _has_any_player_stats(
    tour: str,
    surface: str,
    player_name: Optional[str],
    *,
    ta_id: Optional[int] = None,
) -> bool:
    """True if we can find *any* row (12m or all time) for this player."""
    return (
        _lookup_player_stats_row(tour, "12 month", surface, player_name, ta_id=ta_id) is not None
        or _lookup_player_stats_row(tour, "all time", surface, player_name, ta_id=ta_id) is not None
    )


def _safe_float_dict(d: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not isinstance(d, dict):
        return None
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                out[k] = None
            else:
                out[k] = float(v)
        else:
            out[k] = v
    return out


def _blend(a: Optional[float], b: Optional[float], w: float = 0.5) -> Optional[float]:
    if a is None and b is None:
        return None
    if a is None:
        return float(b)
    if b is None:
        return float(a)
    return float(w) * float(a) + float(1.0 - w) * float(b)


def _expected_break_points_faced(p_server_point_win: float) -> float:
    """Expected number of break points faced per service game given point-win prob p.

    Counts *break-point points played* (i.e., how many times the returner has a 1-point chance to win the game).
    Uses a small recursion + a closed-form for the deuce loop.
    """
    p = _clamp(float(p_server_point_win), 1e-6, 1.0 - 1e-6)
    q = 1.0 - p

    # closed-form expected BP points from deuce
    # E_D = q / (p^2 + q^2)
    denom = (p * p) + (q * q)
    E_D = (q / denom) if denom > 0 else 0.0
    E_AS = q * E_D          # adv-server
    E_AR = 1.0 + p * E_D    # adv-return is itself a break point

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def F(a: int, b: int) -> float:
        # terminal (server holds)
        if a >= 4 and (a - b) >= 2:
            return 0.0
        # terminal (server broken)
        if b >= 4 and (b - a) >= 2:
            return 0.0

        # deuce/adv loop states
        if a == 3 and b == 3:
            return E_D
        if a == 4 and b == 3:
            return E_AS
        if a == 3 and b == 4:
            return E_AR

        # receiver game-point states (0-40, 15-40, 30-40)
        is_bp = (b == 3 and a <= 2)
        add = 1.0 if is_bp else 0.0

        return add + p * F(a + 1, b) + q * F(a, b + 1)

    return float(F(0, 0))


def _project_match_props(
    *,
    tour: str,
    surface: str,
    p1_name: str,
    p2_name: str,
    p1_ta_id: Optional[int],
    p2_ta_id: Optional[int],
    expected_total_games: Optional[float],
    pAserve: Optional[float],
    pBserve: Optional[float],
) -> Optional[dict[str, Any]]:
    """Project aces + break-point related quantities for an upcoming match.

    Output is intended for display / lightweight betting props, not as a perfect physical simulator.
    """
    if expected_total_games is None or pAserve is None or pBserve is None:
        return None

    t = (tour or "").upper().strip()
    s = _surface_key(surface)

    # pull player rows
    p1_12 = _lookup_player_stats_row(t, "12 month", s, p1_name, ta_id=p1_ta_id)
    p1_all = _lookup_player_stats_row(t, "all time", s, p1_name, ta_id=p1_ta_id)
    p2_12 = _lookup_player_stats_row(t, "12 month", s, p2_name, ta_id=p2_ta_id)
    p2_all = _lookup_player_stats_row(t, "all time", s, p2_name, ta_id=p2_ta_id)

    # ace rates: service-side rate blended with opponent allowed-aces rate
    p1_svc_aces, p1_svc_win = _pick_window_for_metric(p1_12, p1_all, "svc_aces_pg", "svc_n", min_matches=5)
    p2_svc_aces, p2_svc_win = _pick_window_for_metric(p2_12, p2_all, "svc_aces_pg", "svc_n", min_matches=5)
    p1_opp_aces_allowed, p1_ret_win = _pick_window_for_metric(p1_12, p1_all, "ret_opp_aces_pg", "ret_n", min_matches=5)
    p2_opp_aces_allowed, p2_ret_win = _pick_window_for_metric(p2_12, p2_all, "ret_opp_aces_pg", "ret_n", min_matches=5)

    # fall back to averages if player metrics missing
    if _PLAYER_STATS_AVG is not None and isinstance(_PLAYER_STATS_AVG, dict):
        avg12 = _PLAYER_STATS_AVG.get((t, "12 month", s), {})
        avgall = _PLAYER_STATS_AVG.get((t, "all time", s), {})
        # Also try "hard" as generic fallback surface when s is "all"/unknown
        avg12_hc = _PLAYER_STATS_AVG.get((t, "12 month", "hard"), {}) if not avg12 else avg12
        avgall_hc = _PLAYER_STATS_AVG.get((t, "all time", "hard"), {}) if not avgall else avgall
        if p1_svc_aces is None:
            p1_svc_aces = avg12.get("svc_aces_pg") or avgall.get("svc_aces_pg") or avg12_hc.get("svc_aces_pg") or avgall_hc.get("svc_aces_pg")
        if p2_svc_aces is None:
            p2_svc_aces = avg12.get("svc_aces_pg") or avgall.get("svc_aces_pg") or avg12_hc.get("svc_aces_pg") or avgall_hc.get("svc_aces_pg")
        if p1_opp_aces_allowed is None:
            p1_opp_aces_allowed = avg12.get("ret_opp_aces_pg") or avgall.get("ret_opp_aces_pg") or avg12_hc.get("ret_opp_aces_pg") or avgall_hc.get("ret_opp_aces_pg")
        if p2_opp_aces_allowed is None:
            p2_opp_aces_allowed = avg12.get("ret_opp_aces_pg") or avgall.get("ret_opp_aces_pg") or avg12_hc.get("ret_opp_aces_pg") or avgall_hc.get("ret_opp_aces_pg")

    # Ultimate fallback: tour-level ace defaults (aces per service game)
    _ace_defaults = {"ATP": 0.45, "WTA": 0.22}
    _opp_ace_defaults = {"ATP": 0.40, "WTA": 0.20}
    if p1_svc_aces is None:
        p1_svc_aces = _ace_defaults.get(t, 0.35)
    if p2_svc_aces is None:
        p2_svc_aces = _ace_defaults.get(t, 0.35)
    if p1_opp_aces_allowed is None:
        p1_opp_aces_allowed = _opp_ace_defaults.get(t, 0.30)
    if p2_opp_aces_allowed is None:
        p2_opp_aces_allowed = _opp_ace_defaults.get(t, 0.30)

    p1_ace_rate_matchup = _blend(p1_svc_aces, p2_opp_aces_allowed, 0.5)
    p2_ace_rate_matchup = _blend(p2_svc_aces, p1_opp_aces_allowed, 0.5)

    # service games split ~50/50
    exp_svc_games_each = float(expected_total_games) / 2.0

    p1_aces = (float(p1_ace_rate_matchup) * exp_svc_games_each) if p1_ace_rate_matchup is not None else None
    p2_aces = (float(p2_ace_rate_matchup) * exp_svc_games_each) if p2_ace_rate_matchup is not None else None

    # break points: expected BP faced per service game from point probability
    bp_faced_p1 = _expected_break_points_faced(float(pAserve)) * exp_svc_games_each
    bp_faced_p2 = _expected_break_points_faced(float(pBserve)) * exp_svc_games_each
    bp_created_p1 = bp_faced_p2
    bp_created_p2 = bp_faced_p1

    # breaks of serve (games) from hold probability
    holdA = _hold_prob_from_point(float(pAserve))
    holdB = _hold_prob_from_point(float(pBserve))
    breaks_p1 = (1.0 - holdB) * exp_svc_games_each
    breaks_p2 = (1.0 - holdA) * exp_svc_games_each

    conv_p1 = (breaks_p1 / bp_created_p1) if bp_created_p1 > 1e-9 else None
    conv_p2 = (breaks_p2 / bp_created_p2) if bp_created_p2 > 1e-9 else None

    # optional: blend with historical BP metrics (if present) as a display hint
    p1_bp_conv_hist, _ = _pick_window_for_metric(p1_12, p1_all, "ret_bp_win_pct", "ret_n", min_matches=5)
    p2_bp_conv_hist, _ = _pick_window_for_metric(p2_12, p2_all, "ret_bp_win_pct", "ret_n", min_matches=5)
    p1_bp_save_hist, _ = _pick_window_for_metric(p1_12, p1_all, "svc_bp_save_pct", "svc_n", min_matches=5)
    p2_bp_save_hist, _ = _pick_window_for_metric(p2_12, p2_all, "svc_bp_save_pct", "svc_n", min_matches=5)

    return {
        "expected_service_games_each": exp_svc_games_each,
        "aces": {
            "p1_expected": p1_aces,
            "p2_expected": p2_aces,
            "total_expected": (p1_aces + p2_aces) if (p1_aces is not None and p2_aces is not None) else None,
            "p1_rate_per_service_game": p1_ace_rate_matchup,
            "p2_rate_per_service_game": p2_ace_rate_matchup,
            "windows_used": {
                "p1_service": p1_svc_win,
                "p1_return": p1_ret_win,
                "p2_service": p2_svc_win,
                "p2_return": p2_ret_win,
            },
        },
        "break_points": {
            "p1_created": bp_created_p1,
            "p2_created": bp_created_p2,
            "p1_faced": bp_faced_p1,
            "p2_faced": bp_faced_p2,
            "p1_breaks": breaks_p1,
            "p2_breaks": breaks_p2,
            "p1_implied_conv": conv_p1,
            "p2_implied_conv": conv_p2,
            "p1_hist_conv": p1_bp_conv_hist,
            "p2_hist_conv": p2_bp_conv_hist,
            "p1_hist_save": p1_bp_save_hist,
            "p2_hist_save": p2_bp_save_hist,
        },
        "inputs": {
            "pAserve": float(pAserve),
            "pBserve": float(pBserve),
            "holdA": float(holdA),
            "holdB": float(holdB),
            "expected_total_games": float(expected_total_games),
        },
    }
async def _predictions_today_enhanced_impl(
    days_ahead: int = Query(0, ge=0, le=14),
    include_incomplete: bool = Query(True),
    bust_cache: bool = Query(False),
    min_edge: float = Query(0.025, ge=0.0, le=1.0),
    max_odds_age_min: int = Query(180, ge=1, le=1440),
    max_overround: float = Query(0.08, ge=0.0, le=1.0),
):
    t0 = time.perf_counter()
    today = date.today()
    dates = [today + timedelta(days=i) for i in range(days_ahead + 1)]
    now_utc = dt.datetime.now(dt.timezone.utc)

    cache = _get_pred_cache(
        days_ahead,
        include_incomplete,
        min_edge=min_edge,
        max_odds_age_min=max_odds_age_min,
        max_overround=max_overround,
    )
    if not bust_cache:
        cached = await cache.get()
        if cached:
            cached["cached"] = True
            return EloPredictionsResponse(**cached)

    items: List[EloPrediction] = []
    variants_by_base: dict[str, list[str]] = {}
    rolling_by_norm: dict[str, dict] = {}
    match_data_list: List[Dict[str, Any]] = []
    row_extras: List[Dict[str, Any]] = []

    # ====== DB PHASE: fetch rows + rolling stats + H2H, then release connection ======
    async with engine.begin() as conn:
        t_sql0 = time.perf_counter()
        rows = await fetch_matches_elo_light_rows(
            conn=conn,
            sql_stmt=SOFASCORE_MATCHES_ELO_SQL_LIGHT,
            dates=dates,
            include_incomplete=include_incomplete,
        )
        t_sql1 = time.perf_counter()
        logger.info("enhanced: SQL fetch rows=%s time_ms=%.1f", len(rows), (t_sql1 - t_sql0) * 1000)
        # Batch rolling stats lookup for the full slate (avoids per-match DB calls)
        base_norms: list[str] = []
        for rr in rows:
            for nm in (rr.get("p1_name"), rr.get("p2_name")):
                bn = _norm_name(nm)
                if bn:
                    base_norms.append(bn)

        base_norms = sorted(set(base_norms))
        t_roll0 = time.perf_counter()
        variants_by_base = await _batch_alias_variants(conn, base_norms)
        rolling_by_norm = await _batch_fetch_rolling_stats(conn, variants_by_base)
        t_roll1 = time.perf_counter()
        logger.info("enhanced: rolling stats time_ms=%.1f", (t_roll1 - t_roll0) * 1000)
        h2h_cache: dict[tuple[int, int, str, date], Tuple[int, int, int, int, int, int]] = {}

        t_loop0 = time.perf_counter()
        pred_time = 0.0
        skip_model = (os.getenv("SKIP_MODEL_PREDICTIONS") or "").strip().lower() in {"1", "true", "yes", "y", "on"}

        # ====== PASS 1: Collect match data + per-row context ======
        for r in rows:
            surface = (r.get("surface") or "").lower()
            match_date = r.get("match_date")
            p1_id = r.get("p1_player_id")
            p2_id = r.get("p2_player_id")

            if match_date:
                key = (
                    int(min(p1_id or 0, p2_id or 0)),
                    int(max(p1_id or 0, p2_id or 0)),
                    surface,
                    match_date,
                )
            else:
                key = (0, 0, surface, date.today())

            row_h2h_total = r.get("h2h_total_matches")
            row_h2h_surface_total = r.get("h2h_surface_matches")

            if row_h2h_total is not None:
                h2h_p1_wins = int(r.get("h2h_p1_wins") or 0)
                h2h_p2_wins = int(r.get("h2h_p2_wins") or 0)
                h2h_total = int(row_h2h_total or 0)
                h2h_surface_p1_wins = int(r.get("h2h_surface_p1_wins") or 0)
                h2h_surface_p2_wins = int(r.get("h2h_surface_p2_wins") or 0)
                h2h_surface_total = int(row_h2h_surface_total or 0)
            elif key in h2h_cache:
                h2h_p1_wins, h2h_p2_wins, h2h_total, h2h_surface_p1_wins, h2h_surface_p2_wins, h2h_surface_total = h2h_cache[key]
            else:
                h2h_p1_wins, h2h_p2_wins, h2h_total, h2h_surface_p1_wins, h2h_surface_p2_wins, h2h_surface_total = await _get_h2h_from_db(
                    conn=conn,
                    p1_canonical_id=p1_id,
                    p2_canonical_id=p2_id,
                    p1_name=r.get("p1_name"),
                    p2_name=r.get("p2_name"),
                    surface=surface,
                    as_of=match_date or date.today(),
                )
                h2h_cache[key] = (
                    h2h_p1_wins,
                    h2h_p2_wins,
                    h2h_total,
                    h2h_surface_p1_wins,
                    h2h_surface_p2_wins,
                    h2h_surface_total,
                )

            # Rolling stats (DB-alias robust lookup)
            p1_roll = _rolling_for_player_from_batch(r.get("p1_name"), variants_by_base, rolling_by_norm)
            p2_roll = _rolling_for_player_from_batch(r.get("p2_name"), variants_by_base, rolling_by_norm)

            p1_wr20 = _safe_float(p1_roll.get("win_rate_last_20")) if p1_roll else None
            p2_wr20 = _safe_float(p2_roll.get("win_rate_last_20")) if p2_roll else None

            p1_matches = int(p1_roll["matches_played"]) if p1_roll and p1_roll.get("matches_played") is not None else None
            p2_matches = int(p2_roll["matches_played"]) if p2_roll and p2_roll.get("matches_played") is not None else None

            p1_hard = _safe_float(p1_roll.get("hard_win_rate_last_10")) if p1_roll else None
            p2_hard = _safe_float(p2_roll.get("hard_win_rate_last_10")) if p2_roll else None
            p1_clay = _safe_float(p1_roll.get("clay_win_rate_last_10")) if p1_roll else None
            p2_clay = _safe_float(p2_roll.get("clay_win_rate_last_10")) if p2_roll else None
            p1_grass = _safe_float(p1_roll.get("grass_win_rate_last_10")) if p1_roll else None
            p2_grass = _safe_float(p2_roll.get("grass_win_rate_last_10")) if p2_roll else None

            p1_style = _get_player_style_metrics(r.get("tour"), r.get("p1_name"), ta_id=r.get("p1_ta_id"), overall=True)
            p2_style = _get_player_style_metrics(r.get("tour"), r.get("p2_name"), ta_id=r.get("p2_ta_id"), overall=True)
            style_summary = _style_summary(r.get("p1_name"), r.get("p2_name"), p1_style, p2_style)
            style_debug = _style_debug(
                r.get("tour"),
                surface,
                r.get("p1_name"),
                r.get("p2_name"),
                r.get("p1_ta_id"),
                r.get("p2_ta_id"),
            )

            def surf_fallback(surf_val: Optional[float], wr20: Optional[float]) -> float:
                if surf_val is not None:
                    return float(surf_val)
                if wr20 is not None:
                    return float(wr20)
                return 0.5

            if "hard" in surface:
                p1_surf_wr = surf_fallback(p1_hard, p1_wr20)
                p2_surf_wr = surf_fallback(p2_hard, p2_wr20)
            elif "clay" in surface:
                p1_surf_wr = surf_fallback(p1_clay, p1_wr20)
                p2_surf_wr = surf_fallback(p2_clay, p2_wr20)
            elif "grass" in surface:
                p1_surf_wr = surf_fallback(p1_grass, p1_wr20)
                p2_surf_wr = surf_fallback(p2_grass, p2_wr20)
            else:
                p1_surf_wr = surf_fallback(None, p1_wr20)
                p2_surf_wr = surf_fallback(None, p2_wr20)

            # Market odds
            p1_mkt = r.get("p1_odds_american")
            p2_mkt = r.get("p2_odds_american")
            odds_fetched_at = r.get("odds_fetched_at")

            p1_imp = american_to_implied_prob_model(p1_mkt) if p1_mkt is not None else None
            p2_imp = american_to_implied_prob_model(p2_mkt) if p2_mkt is not None else None
            p1_nv, p2_nv = no_vig_two_way_model(p1_imp, p2_imp)
            odds_age_minutes = _odds_age_minutes(odds_fetched_at, now_utc=now_utc)
            market_overround_main = (float(p1_imp + p2_imp - 1.0) if (p1_imp is not None and p2_imp is not None) else None)
            has_fresh_odds = bool(odds_age_minutes is not None and odds_age_minutes <= float(max_odds_age_min))
            missing_main_odds = bool(p1_imp is None or p2_imp is None)
            high_overround = bool(market_overround_main is not None and market_overround_main > float(max_overround))
            predictor_quality_flags = []
            if missing_main_odds:
                predictor_quality_flags.append("missing_main_odds")
            if not has_fresh_odds:
                predictor_quality_flags.append("stale_odds")
            if high_overround:
                predictor_quality_flags.append("high_overround")

            med_elo = r.get("med_elo")
            p1_elo_raw = r.get("p1_elo_raw")
            p2_elo_raw = r.get("p2_elo_raw")
            p1_elo_latest = r.get("p1_elo_latest")
            p2_elo_latest = r.get("p2_elo_latest")
            p1_elo_used_median = p1_elo_raw is None and p1_elo_latest is None and med_elo is not None
            p2_elo_used_median = p2_elo_raw is None and p2_elo_latest is None and med_elo is not None
            elo_used_median = p1_elo_used_median or p2_elo_used_median

            h2h_p1_win_pct = (h2h_p1_wins / h2h_total) if h2h_total > 0 else 0.5
            h2h_surface_p1_win_pct = (h2h_surface_p1_wins / h2h_surface_total) if h2h_surface_total > 0 else 0.5

            match_data = {
                "match_id": r.get("match_id"),
                "match_key": r.get("match_key"),
                "match_date": r.get("match_date"),
                "tour": r.get("tour"),
                "tournament": r.get("tournament"),
                "round": r.get("round"),
                "surface": surface,
                "best_of": int(r.get("best_of", 3) or 3),

                "p1_name": r.get("p1_name"),
                "p2_name": r.get("p2_name"),
                "p1_ta_id": r.get("p1_ta_id"),
                "p2_ta_id": r.get("p2_ta_id"),

                # ELO (fallback to tour median if missing)
                "p1_elo": float(p1_elo_raw) if p1_elo_raw is not None else (float(p1_elo_latest) if p1_elo_latest is not None else (float(med_elo) if med_elo is not None else None)),
                "p2_elo": float(p2_elo_raw) if p2_elo_raw is not None else (float(p2_elo_latest) if p2_elo_latest is not None else (float(med_elo) if med_elo is not None else None)),
                "elo_used_median": elo_used_median,
                "p1_elo_used_median": p1_elo_used_median,
                "p2_elo_used_median": p2_elo_used_median,
                # Surface-specific ELO (for v3 model)
                "p1_helo": _safe_float(r.get("p1_helo_raw")),
                "p1_celo": _safe_float(r.get("p1_celo_raw")),
                "p1_gelo": _safe_float(r.get("p1_gelo_raw")),
                "p2_helo": _safe_float(r.get("p2_helo_raw")),
                "p2_celo": _safe_float(r.get("p2_celo_raw")),
                "p2_gelo": _safe_float(r.get("p2_gelo_raw")),
                # Ranking + age (for v3 model)
                "p1_rank": r.get("p1_rank"),
                "p2_rank": r.get("p2_rank"),
                "p1_age": _safe_float(r.get("p1_age")),
                "p2_age": _safe_float(r.get("p2_age")),

                # Rolling
                "p1_win_rate_last_20": p1_wr20,
                "p2_win_rate_last_20": p2_wr20,
                "p1_matches_played": p1_matches,
                "p2_matches_played": p2_matches,
                "p1_win_rate_surface": p1_surf_wr,
                "p2_win_rate_surface": p2_surf_wr,

                # Style proxies (from player_surface_stats)
                "p1_svc_pts_w": p1_style.get("svc_pts"),
                "p2_svc_pts_w": p2_style.get("svc_pts"),
                "p1_ret_pts_w": p1_style.get("ret_pts"),
                "p2_ret_pts_w": p2_style.get("ret_pts"),
                "p1_ace_pg": p1_style.get("ace_pg"),
                "p2_ace_pg": p2_style.get("ace_pg"),
                "p1_bp_save": p1_style.get("bp_save"),
                "p2_bp_save": p2_style.get("bp_save"),
                "p1_bp_win": p1_style.get("bp_win"),
                "p2_bp_win": p2_style.get("bp_win"),
                "style_summary": style_summary,
                "style_debug": style_debug,

                # Fatigue / recent load (diffs)
                "d_rest_days": (
                    (r.get("p1_rest_days") or 0) - (r.get("p2_rest_days") or 0)
                ),
                "d_matches_10d": (
                    (r.get("p1_matches_10d") or 0) - (r.get("p2_matches_10d") or 0)
                ),
                # matches_30d not in SQL; leave to default in predictor

                # H2H
                "h2h_p1_win_pct": h2h_p1_win_pct,
                "h2h_total_matches": h2h_total,
                "h2h_surface_p1_win_pct": h2h_surface_p1_win_pct,
                "h2h_surface_matches": h2h_surface_total,

                # Odds
                "p1_odds_american": p1_mkt,
                "p2_odds_american": p2_mkt,
                "odds_age_minutes": odds_age_minutes,
                "market_overround_main": market_overround_main,
                "has_fresh_odds": has_fresh_odds,
                "missing_main_odds": missing_main_odds,
                "high_overround": high_overround,
                "predictor_quality_flags": predictor_quality_flags,
                "max_overround": float(max_overround),
                "sofascore_total_games_line": _safe_float(r.get("sofascore_total_games_line")),
                "sofascore_total_games_over_american": r.get("sofascore_total_games_over_american"),
                "sofascore_total_games_under_american": r.get("sofascore_total_games_under_american"),
                "sofascore_spread_p1_line": _safe_float(r.get("sofascore_spread_p1_line")),
                "sofascore_spread_p2_line": _safe_float(r.get("sofascore_spread_p2_line")),
                "sofascore_spread_p1_odds_american": r.get("sofascore_spread_p1_odds_american"),
                "sofascore_spread_p2_odds_american": r.get("sofascore_spread_p2_odds_american"),

                # Tennis Insight features (if your SQL/view provides them)
                "serve_return_edge_w": r.get("serve_return_edge_w"),
                "d_srv_pts_w_w": r.get("d_srv_pts_w_w"),
                "d_ret_pts_w_w": r.get("d_ret_pts_w_w"),
                "d_hold_w": r.get("d_hold_w"),
                "missing_any": r.get("missing_any"),
                "profile_weight": r.get("profile_weight"),
            }

            match_data_list.append(match_data)
            row_extras.append({
                "r": r,
                "surface": surface,
                "p1_mkt": p1_mkt,
                "p2_mkt": p2_mkt,
                "p1_imp": p1_imp,
                "p2_imp": p2_imp,
                "p1_nv": p1_nv,
                "p2_nv": p2_nv,
                "odds_age_minutes": odds_age_minutes,
                "market_overround_main": market_overround_main,
                "has_fresh_odds": has_fresh_odds,
                "missing_main_odds": missing_main_odds,
                "high_overround": high_overround,
                "predictor_quality_flags": predictor_quality_flags,
                "odds_fetched_at": odds_fetched_at,
                "h2h_p1_wins": h2h_p1_wins,
                "h2h_p2_wins": h2h_p2_wins,
                "h2h_total": h2h_total,
                "h2h_surface_p1_wins": h2h_surface_p1_wins,
                "h2h_surface_p2_wins": h2h_surface_p2_wins,
                "h2h_surface_total": h2h_surface_total,
                "p1_wr20": p1_wr20,
                "p2_wr20": p2_wr20,
                "elo_used_median": elo_used_median,
                "p1_elo_used_median": p1_elo_used_median,
                "p2_elo_used_median": p2_elo_used_median,
                "style_summary": style_summary,
                "style_debug": style_debug,
            })

        logger.info("enhanced: pass1 (DB) time_ms=%.1f", (time.perf_counter() - t_loop0) * 1000)
    # ---- DB connection is now released ----

    # ====== PASS 1.5: Enrich match_data with TA rolling features (v3) ======
    try:
        ta_match_ids = [md["match_id"] for md in match_data_list if md.get("match_id")]
        if ta_match_ids:
            async with engine.begin() as ta_conn:
                ta_lookup = await fetch_ta_feature_diffs(conn=ta_conn, match_ids=ta_match_ids)
                for md in match_data_list:
                    ta = ta_lookup.get(str(md.get("match_id")))
                    if ta:
                        for k, v in ta.items():
                            if k != "match_id":
                                md[k] = float(v) if v is not None else 0.0
            logger.info("enhanced: enriched %d/%d matches with TA features", len(ta_lookup), len(match_data_list))
    except Exception as e:
        logger.warning("enhanced: TA feature enrichment failed (non-fatal): %s", e)

    # ====== PASS 2: Batch model inference (off the event loop) ======
    if skip_model:
        _skip_pred = {
            "p1_win_prob": None, "p2_win_prob": None,
            "predicted_winner": None, "missing_reason": "SKIP_MODEL_PREDICTIONS",
            "method": "skipped", "num_predictors": 0, "individual_predictions": [],
        }
        predictions = [_skip_pred] * len(match_data_list)
    else:
        t_pred0 = time.perf_counter()
        predictions = await predict_batch_combined_async(match_data_list)
        pred_time = time.perf_counter() - t_pred0

    # ====== PASS 3: Assemble response objects ======
    try:
        fair_step = int(os.getenv("FAIR_ODDS_ROUND_STEP", "1"))
    except Exception:
        fair_step = 1
    try:
        w_model = float(os.getenv("ODDS_BLEND_WEIGHT", "0.65"))
    except Exception:
        w_model = 0.65

    for _i, (match_data, pred, extras) in enumerate(zip(match_data_list, predictions, row_extras)):
        r = extras["r"]
        surface = extras["surface"]
        p1_mkt = extras["p1_mkt"]
        p2_mkt = extras["p2_mkt"]
        p1_imp = extras["p1_imp"]
        p2_imp = extras["p2_imp"]
        p1_nv = extras["p1_nv"]
        p2_nv = extras["p2_nv"]
        odds_age_minutes = extras["odds_age_minutes"]
        market_overround_main = extras["market_overround_main"]
        has_fresh_odds = extras["has_fresh_odds"]
        missing_main_odds = extras["missing_main_odds"]
        high_overround = extras["high_overround"]
        row_quality_flags = list(extras["predictor_quality_flags"] or [])
        odds_fetched_at = extras["odds_fetched_at"]
        h2h_p1_wins = extras["h2h_p1_wins"]
        h2h_p2_wins = extras["h2h_p2_wins"]
        h2h_total = extras["h2h_total"]
        h2h_surface_p1_wins = extras["h2h_surface_p1_wins"]
        h2h_surface_p2_wins = extras["h2h_surface_p2_wins"]
        h2h_surface_total = extras["h2h_surface_total"]
        p1_wr20 = extras["p1_wr20"]
        p2_wr20 = extras["p2_wr20"]
        elo_used_median = extras["elo_used_median"]
        p1_elo_used_median = extras["p1_elo_used_median"]
        p2_elo_used_median = extras["p2_elo_used_median"]
        style_summary = extras["style_summary"]
        style_debug = extras["style_debug"]

        p1_prob = pred.get("p1_win_prob")
        p2_prob = pred.get("p2_win_prob")

        p1_fair = _round_american(_prob_to_american(p1_prob), fair_step)
        p2_fair = _round_american(_prob_to_american(p2_prob), fair_step)

        p1_blend = blend_probs_logit(p1_prob, p1_nv, w_model=w_model)
        p2_blend = (1.0 - p1_blend) if p1_blend is not None else None
        edge_no_vig = (float(p1_prob) - float(p1_nv)) if (p1_prob is not None and p1_nv is not None) else None
        edge_blended = (float(p1_blend) - float(p1_nv)) if (p1_blend is not None and p1_nv is not None) else None

        predicted_blend = None
        if p1_blend is not None:
            predicted_blend = "p1" if p1_blend >= 0.5 else "p2"

        pred_quality_flags = list(pred.get("predictor_quality_flags") or [])
        quality_flags = sorted(set(row_quality_flags + pred_quality_flags))
        critical_quality = any(
            f in {"elo_missing", "xgb_missing", "market_missing", "missing_main_odds"} or f.endswith("_error")
            for f in quality_flags
        )
        gate_reasons: list[str] = []
        if edge_blended is None:
            gate_reasons.append("missing_edge")
        elif abs(float(edge_blended)) < float(min_edge):
            gate_reasons.append("low_edge")
        if not has_fresh_odds:
            gate_reasons.append("stale_odds")
        if high_overround:
            gate_reasons.append("high_overround")
        if p1_mkt is None or p2_mkt is None:
            gate_reasons.append("missing_market_odds")
        if critical_quality:
            gate_reasons.append("critical_quality")

        bet_eligible = len(gate_reasons) == 0
        bet_side = "none"
        if bet_eligible and edge_blended is not None:
            bet_side = "p1" if float(edge_blended) > 0 else ("p2" if float(edge_blended) < 0 else "none")
            if bet_side == "none":
                bet_eligible = False
                gate_reasons.append("zero_edge")
        kelly_fraction_capped = None
        if bet_eligible and p1_blend is not None and p2_blend is not None:
            if bet_side == "p1":
                kelly_fraction_capped = _kelly_fraction(p1_blend, p1_mkt, cap=0.02)
            elif bet_side == "p2":
                kelly_fraction_capped = _kelly_fraction(p2_blend, p2_mkt, cap=0.02)

        value_bet_summary = "No value bet (gate blocked)." if not bet_eligible else "No value bet."
        if bet_eligible and edge_blended is not None:
            side_name = r.get("p1_name") if bet_side == "p1" else r.get("p2_name")
            side_odds = p1_mkt if bet_side == "p1" else p2_mkt
            try:
                # edge_blended is p1-relative; flip sign for p2 value messaging.
                edge_side = float(edge_blended) if bet_side == "p1" else (-float(edge_blended))
                edge_pp = 100.0 * edge_side
                kelly_pct = (100.0 * float(kelly_fraction_capped)) if kelly_fraction_capped is not None else 0.0
                value_bet_summary = (
                    f"Value bet: {side_name} ({bet_side}, {side_odds:+d}) | "
                    f"edge {edge_pp:+.1f}pp | Kelly cap {kelly_pct:.1f}%."
                )
            except Exception:
                value_bet_summary = f"Value bet: {side_name} ({bet_side})."

        pick_side = predicted_blend or pred.get("predicted_winner")
        pick_summary = _pick_summary(
            pick_side=pick_side,
            p1_name=r.get("p1_name"),
            p2_name=r.get("p2_name"),
            p1_prob=p1_blend if p1_blend is not None else p1_prob,
            p2_prob=p2_blend if p2_blend is not None else p2_prob,
            p1_nv=p1_nv,
            p2_nv=p2_nv,
            p1_odds=p1_mkt,
            p2_odds=p2_mkt,
            p1_wr20=p1_wr20,
            p2_wr20=p2_wr20,
            p1_surf_wr=match_data.get("p1_win_rate_surface"),
            p2_surf_wr=match_data.get("p2_win_rate_surface"),
            p1_svc_pts_w=match_data.get("p1_svc_pts_w"),
            p2_svc_pts_w=match_data.get("p2_svc_pts_w"),
            p1_ret_pts_w=match_data.get("p1_ret_pts_w"),
            p2_ret_pts_w=match_data.get("p2_ret_pts_w"),
            d_rest_days=match_data.get("d_rest_days"),
            d_matches_10d=match_data.get("d_matches_10d"),
            h2h_p1_win_pct=match_data.get("h2h_p1_win_pct"),
            h2h_total=match_data.get("h2h_total_matches"),
            style_summary=style_summary,
            # new enriched params
            surface=surface,
            p1_elo=match_data.get("p1_elo"),
            p2_elo=match_data.get("p2_elo"),
            p1_rank=match_data.get("p1_rank"),
            p2_rank=match_data.get("p2_rank"),
            h2h_p1_wins=h2h_p1_wins,
            h2h_p2_wins=h2h_p2_wins,
            h2h_surface_p1_wins=h2h_surface_p1_wins,
            h2h_surface_p2_wins=h2h_surface_p2_wins,
            h2h_surface_total=h2h_surface_total,
            individual_predictions=pred.get("individual_predictions"),
            p1_ace_pg=match_data.get("p1_ace_pg"),
            p2_ace_pg=match_data.get("p2_ace_pg"),
            p1_bp_save=match_data.get("p1_bp_save"),
            p2_bp_save=match_data.get("p2_bp_save"),
            p1_bp_win=match_data.get("p1_bp_win"),
            p2_bp_win=match_data.get("p2_bp_win"),
            p1_matches_played=match_data.get("p1_matches_played"),
            p2_matches_played=match_data.get("p2_matches_played"),
            elo_used_median=elo_used_median,
        )

        has_elo = (match_data.get("p1_elo") is not None and match_data.get("p2_elo") is not None)
        has_rolling = (p1_wr20 is not None and p2_wr20 is not None)
        has_odds = (p1_mkt is not None and p2_mkt is not None)

        # --- Lightweight props (no DP sim — that was the 5-min bottleneck) ---
        sim_payload = None
        sim_error: Optional[str] = None
        sim_used_defaults: bool = False
        props_payload: Optional[dict[str, Any]] = None
        pAserve: Optional[float] = None
        pBserve: Optional[float] = None
        try:
            sim_tour = r.get("tour")
            sim_surface = r.get("surface")
            sim_best_of = int(r.get("best_of") or 3)

            p1_svc, p1_ret = _get_player_rates(sim_tour, sim_surface, r.get("p1_name"), ta_id=r.get("p1_ta_id"))
            p2_svc, p2_ret = _get_player_rates(sim_tour, sim_surface, r.get("p2_name"), ta_id=r.get("p2_ta_id"))

            pAserve = _combine_point_probs(p1_svc, p2_ret)
            pBserve = _combine_point_probs(p2_svc, p1_ret)

            # Regress toward tour/surface baseline
            try:
                blend_w = float(os.getenv("TOTALS_BASE_BLEND", "0.6"))
            except Exception:
                blend_w = 0.6
            blend_w = _clamp(float(blend_w), 0.0, 1.0)
            base_spw = _base_serve_point_win(sim_tour, sim_surface)
            pAserve = _clamp01(pAserve * blend_w + base_spw * (1.0 - blend_w))
            pBserve = _clamp01(pBserve * blend_w + base_spw * (1.0 - blend_w))

            holdA = _hold_prob_from_point(pAserve)
            holdB = _hold_prob_from_point(pBserve)

            # Fast expected-games estimate (no DP, just arithmetic)
            avg_hold = (holdA + holdB) / 2.0
            est_games_per_set = _clamp(9.6 + 8.0 * (avg_hold - 0.75), 8.0, 13.0)

            p_m = _clamp01(float(p1_prob)) if p1_prob is not None else 0.5
            if sim_best_of == 5:
                est_sets = 3.0 + 4.0 * p_m * (1.0 - p_m)
            else:
                est_sets = 2.0 + 2.0 * p_m * (1.0 - p_m)
            expected_total_games = est_games_per_set * est_sets

            props_payload = _project_match_props(
                tour=sim_tour,
                surface=sim_surface,
                p1_name=str(r.get("p1_name") or ""),
                p2_name=str(r.get("p2_name") or ""),
                p1_ta_id=r.get("p1_ta_id"),
                p2_ta_id=r.get("p2_ta_id"),
                expected_total_games=expected_total_games,
                pAserve=pAserve,
                pBserve=pBserve,
            )
        except Exception:
            props_payload = None

        model_spread_prob = p1_blend if p1_blend is not None else p1_prob
        model_spread_p1_line, model_spread_p2_line = _derive_model_game_spread_lines(model_spread_prob)

        pred_kwargs = dict(
            match_id=str(r.get("match_id")),
            match_key=r.get("match_key"),
            match_date=r.get("match_date"),
            match_start_utc=_dt_to_iso_z(r.get("match_start_utc")),
            tour=r.get("tour"),
            tournament=r.get("tournament"),
            round=r.get("round"),
            surface=surface,
            best_of=int(r.get("best_of", 3) or 3),

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

            p1_market_odds_american=p1_mkt,
            p2_market_odds_american=p2_mkt,
            odds_fetched_at=_dt_to_iso_z(odds_fetched_at),

            p1_market_implied_prob=p1_imp,
            p2_market_implied_prob=p2_imp,
            p1_market_no_vig_prob=p1_nv,
            p2_market_no_vig_prob=p2_nv,

            h2h_p1_wins=h2h_p1_wins,
            h2h_p2_wins=h2h_p2_wins,
            h2h_total_matches=h2h_total,
            h2h_surface_p1_wins=h2h_surface_p1_wins,
            h2h_surface_p2_wins=h2h_surface_p2_wins,
            h2h_surface_matches=h2h_surface_total,

            p1_blended_win_prob=p1_blend,
            p2_blended_win_prob=p2_blend,
            predicted_winner_blended=predicted_blend,
            edge_no_vig=edge_no_vig,
            edge_blended=edge_blended,
            bet_eligible=bet_eligible,
            bet_side=bet_side,
            kelly_fraction_capped=kelly_fraction_capped,
            value_bet_summary=value_bet_summary,

            missing_reason=pred.get("missing_reason"),
            predicted_winner=pred.get("predicted_winner"),

            inputs={
                "method": pred.get("method"),
                "num_predictors": pred.get("num_predictors"),
                "individual_predictions": pred.get("individual_predictions"),
                "effective_weights": pred.get("effective_weights", {}),
                "predictor_quality_flags": quality_flags,
                "has_elo": has_elo,
                "elo_used_median": elo_used_median,
                "p1_elo_used_median": p1_elo_used_median,
                "p2_elo_used_median": p2_elo_used_median,
                "has_rolling": has_rolling,
                "has_odds": has_odds,
                "odds_age_minutes": odds_age_minutes,
                "market_overround_main": market_overround_main,
                "has_fresh_odds": has_fresh_odds,
                "missing_main_odds": missing_main_odds,
                "high_overround": high_overround,
                "min_edge": float(min_edge),
                "max_odds_age_min": int(max_odds_age_min),
                "max_overround": float(max_overround),
                "critical_quality": critical_quality,
                "gate_reasons": gate_reasons,
                "gate_blocked": len(gate_reasons) > 0,
                "edge_abs": abs(float(edge_blended)) if edge_blended is not None else None,
                "bet_eligible": bet_eligible,
                "bet_side": bet_side,
                "kelly_fraction_capped": kelly_fraction_capped,
                "has_total_games_line": (r.get("sofascore_total_games_line") is not None),
                "has_game_spread": (r.get("sofascore_spread_p1_line") is not None or r.get("sofascore_spread_p2_line") is not None),
                "has_model_game_spread": (model_spread_p1_line is not None and model_spread_p2_line is not None),
                "model_spread_source_prob": float(model_spread_prob) if model_spread_prob is not None else None,
                "model_spread_p1_line": model_spread_p1_line,
                "model_spread_p2_line": model_spread_p2_line,
                "sofascore_total_games_line": _safe_float(r.get("sofascore_total_games_line")),
                "sofascore_total_games_over_american": r.get("sofascore_total_games_over_american"),
                "sofascore_total_games_under_american": r.get("sofascore_total_games_under_american"),
                "sofascore_spread_p1_line": _safe_float(r.get("sofascore_spread_p1_line")),
                "sofascore_spread_p2_line": _safe_float(r.get("sofascore_spread_p2_line")),
                "sofascore_spread_p1_odds_american": r.get("sofascore_spread_p1_odds_american"),
                "sofascore_spread_p2_odds_american": r.get("sofascore_spread_p2_odds_american"),
                "totals_sets_sim": sim_payload,
                "totals_sets_error": sim_error,
                "totals_sets_used_defaults": sim_used_defaults,
                "pAserve": float(pAserve) if pAserve is not None else None,
                "pBserve": float(pBserve) if pBserve is not None else None,
                "projected_props": props_payload,
                "style_summary": style_summary,
                "style_debug": style_debug,
                "pick_summary": pick_summary,
                "model_pick_summary": pick_summary,
                "model_pick_side": pick_side,
                "value_bet_summary": value_bet_summary,
                "value_bet_side": bet_side,
            },
        )

        # Optional: set top-level fields if schema supports them
        if sim_payload and hasattr(EloPrediction, "model_fields"):
            mf = EloPrediction.model_fields
            if "projected_total_games" in mf:
                pred_kwargs["projected_total_games"] = sim_payload.get("expected_total_games")
            if "projected_sets" in mf:
                pred_kwargs["projected_sets"] = (f"{float(sim_payload.get('expected_sets')):.2f}" if sim_payload.get("expected_sets") is not None else None)
        if hasattr(EloPrediction, "model_fields"):
            mf = EloPrediction.model_fields
            if "pAserve" in mf:
                pred_kwargs["pAserve"] = float(pAserve) if pAserve is not None else None
            if "pBserve" in mf:
                pred_kwargs["pBserve"] = float(pBserve) if pBserve is not None else None

        items.append(EloPrediction(**pred_kwargs))
    t_loop1 = time.perf_counter()
    logger.info(
        "enhanced: loop time_ms=%.1f pred_time_ms=%.1f",
        (t_loop1 - t_loop0) * 1000,
        pred_time * 1000,
    )

    response = EloPredictionsResponse(
        as_of=today,
        source="combined_ensemble",
        cached=False,
        count=len(items),
        items=items,
    )

    await cache.set(response.model_dump(mode="json"))
    logger.info("enhanced: total_time_ms=%.1f", (time.perf_counter() - t0) * 1000)
    return response


@router.get("/predictions/today/enhanced", response_model=EloPredictionsResponse)
async def predictions_today_enhanced(
    days_ahead: int = Query(0, ge=0, le=14),
    include_incomplete: bool = Query(True),
    bust_cache: bool = Query(False),
    min_edge: float = Query(0.025, ge=0.0, le=1.0),
    max_odds_age_min: int = Query(180, ge=1, le=1440),
    max_overround: float = Query(0.08, ge=0.0, le=1.0),
):
    return await get_predictions_today_enhanced_service(
        days_ahead=days_ahead,
        include_incomplete=include_incomplete,
        bust_cache=bust_cache,
        min_edge=min_edge,
        max_odds_age_min=max_odds_age_min,
        max_overround=max_overround,
        impl=_predictions_today_enhanced_impl,
    )


@router.get("/predictions/today", response_model=EloPredictionsResponse)
async def predictions_today(
    days_ahead: int = Query(0, ge=0, le=14),
    include_incomplete: bool = Query(True),
    bust_cache: bool = Query(False),
):
    return await get_predictions_today_service(
        days_ahead=days_ahead,
        include_incomplete=include_incomplete,
        bust_cache=bust_cache,
        impl=predictions_today_enhanced,
    )


@router.get("/predictions/today/enhanced/h2h", response_model=List[H2HMatch])
async def predictions_today_enhanced_h2h(event_id: int):
    return await get_h2h_matches_service(
        event_id=event_id,
        fetch_payload=fetch_h2h_raw_payload,
        extract_matches=_extract_h2h_matches,
        h2h_model=H2HMatch,
    )


async def _debug_stats_lookup_impl(player: str = Query(...), tour: str = Query("ATP"), surface: str = Query("hard")):
    """Temporary debug endpoint — remove after diagnosing."""
    _load_player_surface_stats()
    norm = _norm_name(player)
    s = _surface_key(surface)
    t = tour.upper().strip()
    row_12 = _lookup_player_stats_row(t, "12 month", s, player)
    row_all = _lookup_player_stats_row(t, "all time", s, player)
    # also check if the key exists directly
    direct_12 = _PLAYER_STATS.get((t, "12 month", s, norm)) if _PLAYER_STATS else None
    direct_all = _PLAYER_STATS.get((t, "all time", s, norm)) if _PLAYER_STATS else None
    sample_keys = [k for k in (_PLAYER_STATS or {}) if k[3] and "sinner" in k[3]][:5]
    return {
        "query": {"player": player, "norm": norm, "tour": t, "surface": s},
        "stats_path": str(_PLAYER_STATS_PATH),
        "total_keys": len(_PLAYER_STATS) if _PLAYER_STATS else 0,
        "row_12m": row_12,
        "row_all": row_all,
        "direct_12": direct_12 is not None,
        "direct_all": direct_all is not None,
        "sample_sinner_keys": sample_keys,
    }


@router.get("/predictions/debug/stats-lookup")
async def debug_stats_lookup(player: str = Query(...), tour: str = Query("ATP"), surface: str = Query("hard")):
    return await get_debug_stats_lookup_service(
        player=player,
        tour=tour,
        surface=surface,
        impl=_debug_stats_lookup_impl,
    )


def _diff_num(a: Any, b: Any) -> Optional[float]:
    fa = _safe_float(a)
    fb = _safe_float(b)
    if fa is None or fb is None:
        return None
    return float(fa - fb)


async def _build_player_stats_payload(
    player: str,
    tour: str = "ATP",
    surface: str = "hard",
    debug: bool = False,
) -> dict[str, Any]:
    return await build_player_stats_payload_service(
        player=player,
        tour=tour,
        surface=surface,
        debug=debug,
        engine=engine,
        load_player_surface_stats=_load_player_surface_stats,
        norm_name=_norm_name,
        norm_name_stats=_norm_name_stats,
        name_variants_for_stats=_name_variants_for_stats,
        name_tokens=_name_tokens,
        resolve_name_match=_resolve_name_match,
        player_stats_names_ts=_PLAYER_STATS_NAMES_TS or {},
        player_stats_token_index_ts=_PLAYER_STATS_TOKEN_INDEX_TS or {},
        lookup_player_stats_row_by_norm_name=_lookup_player_stats_row_by_norm_name,
        safe_float_dict=_safe_float_dict,
        build_token_index=_build_token_index,
        get_player_rates=_get_player_rates,
        get_player_style_metrics=_get_player_style_metrics,
        style_label=_style_label,
        surface_key=_surface_key,
        player_stats_path=_PLAYER_STATS_PATH,
    )


@router.get("/players/stats")
async def player_full_stats(
    player: str = Query(..., min_length=2, description="Player name"),
    tour: str = Query("ATP", description="ATP or WTA"),
    surface: str = Query("hard", description="hard, clay, grass, or unknown"),
    debug: bool = Query(False, description="Include detailed name-resolution debug payload"),
):
    return await get_player_full_stats_service(
        player=player,
        tour=tour,
        surface=surface,
        debug=debug,
        impl=_build_player_stats_payload,
    )


async def _players_compare_impl(
    player1: str = Query(..., min_length=2, description="Left player name"),
    player2: str = Query(..., min_length=2, description="Right player name"),
    tour: str = Query("ATP", description="ATP or WTA"),
    surface: str = Query("hard", description="hard, clay, grass, or unknown"),
    event_id: Optional[int] = Query(None, description="Optional SofaScore event id for direct H2H fetch"),
    debug: bool = Query(False, description="Include detailed debug payload"),
):
    return await build_players_compare_payload(
        player1=player1,
        player2=player2,
        tour=tour,
        surface=surface,
        event_id=event_id,
        debug=debug,
        validate_inputs=validate_players_compare_inputs,
        build_player_stats_payload=_build_player_stats_payload,
        build_players_compare_context=build_players_compare_context,
        build_players_compare_response=build_players_compare_response,
        norm_name_stats=_norm_name_stats,
        norm_name=_norm_name,
        engine=engine,
        resolve_canonical_player_id=_resolve_canonical_player_id,
        get_last_matchups_from_db=_get_last_matchups_from_db,
        get_player_last_matches_from_db=_get_player_last_matches_from_db,
        get_h2h_from_db=_get_h2h_from_db,
        norm_name_simple=_norm_name_simple,
        normalize_surface_family=_normalize_surface_family,
        diff_num=_diff_num,
        surface_key=_surface_key,
    )


@router.get("/players/compare")
async def players_compare(
    player1: str = Query(..., min_length=2, description="Left player name"),
    player2: str = Query(..., min_length=2, description="Right player name"),
    tour: str = Query("ATP", description="ATP or WTA"),
    surface: str = Query("hard", description="hard, clay, grass, or unknown"),
    event_id: Optional[int] = Query(None, description="Optional SofaScore event id for direct H2H fetch"),
    debug: bool = Query(False, description="Include detailed debug payload"),
):
    return await get_players_compare_service(
        player1=player1,
        player2=player2,
        tour=tour,
        surface=surface,
        event_id=event_id,
        debug=debug,
        impl=_players_compare_impl,
    )

