"""
sofascore_matchups.py

SofaScore tennis ingest (async SQLAlchemy + asyncpg) — Playwright fetch to bypass 403

- Pulls scheduled tennis events for a date window
- Keeps singles + team ties (e.g., United Cup country vs country)
- Skips doubles
- Writes into `tennis_matches` with safe UPSERT on sofascore_event_id
- ALSO writes raw events into `sofascore_events_raw` (JSONB)
- OPTIONAL: ingest odds into `tennis_matches` + `sofascore_event_odds_raw` (JSONB)

Recommended auto-fixes (optional, set via env vars):
- Auto backfill canonical ids for matches after ingest (set-based SQL)
- Auto fill TA mapping + Elo after ingest (set-based SQL)

ENV VARS
Core:
- DATABASE_URL=postgresql+asyncpg://USER:PASS@HOST:5432/DBNAME?sslmode=require
  (or DATABASE_URL_ASYNC)
- START_DATE=YYYY-MM-DD
- END_DATE=YYYY-MM-DD
- DEFAULT_DAYS_BACK=1       (used only when START_DATE/END_DATE are unset)
- DEFAULT_DAYS_AHEAD=1      (used only when START_DATE/END_DATE are unset)
- TOUR_FILTER=ATP,WTA
- INCLUDE_WTA125=1          (if TOUR_FILTER includes WTA, also keep WTA 125 events)
- RAW_FILTERED_ONLY=1       (store raw only for kept matches)

Odds:
- INGEST_ODDS=1
- ODDS_PROVIDER_ID=1
- ODDS_SLEEP_SECONDS=0.25

H2H:
- INGEST_H2H=1
- INGEST_H2H_EVENTS=1       (also fetch /event/{customId}/h2h/events when customId is available)
- H2H_SLEEP_SECONDS=0.2

Auto-fix Canonicals:
- AUTO_BACKFILL_CANONICAL=1   (default true)

Auto-fill Elo:
- AUTO_FILL_ELO=1             (default true)
- ELO_USE_FALLBACK=0          (default false)
- ELO_FALLBACK=1500           (used if fallback enabled)
  (Includes automatic fuzzy TA mapping via fix_ta_elo_mapping for accents,
   NBSP, hyphens, name reordering — no more manual fixes needed)

Auto-fill H2H:
- AUTO_BACKFILL_H2H=1         (default true)

MATCH DATE + TIME
✅ match_start_utc comes from event["startTimestamp"] converted to UTC datetime (TIMESTAMPTZ)
✅ match_date comes from event["startTimestamp"] converted to Central Time (US/Central) (date only)
   Fallback to `day` only if startTimestamp is missing.
"""

from __future__ import annotations

import os
import re
import ssl
import json
import asyncio
import logging
import datetime as dt
from typing import List, Optional, Tuple, Set, Any
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from dotenv import load_dotenv
from playwright.async_api import async_playwright

from sqlalchemy import text, bindparam, BigInteger
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncConnection
from sqlalchemy.dialects.postgresql import JSONB

from app.utils.datetime_utils import match_date_central
from app.ingest.tennis.player_aliases import resolve_player_id
from app.ingest.tennis.fix_ta_elo_mapping import run_fix as run_fuzzy_ta_fix

load_dotenv()

logger = logging.getLogger("sofascore_ingest")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

BASE_URL = "https://www.sofascore.com/api/v1"
BROWSER_HEADERS = {
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# =============================================================================
# Auto-backfill canonical ids from sofascore mapping (tennis_player_sources)
# =============================================================================

BACKFILL_CANON_P1_SQL = text(
    """
    UPDATE tennis_matches m
    SET p1_canonical_id = s.player_id
    FROM tennis_player_sources s
    WHERE s.source = 'sofascore'
      AND NULLIF(btrim(s.source_player_id), '') ~ '^[0-9]+$'
      AND NULLIF(btrim(s.source_player_id), '')::bigint = m.p1_sofascore_player_id
      AND m.match_date BETWEEN :start AND :end
      AND m.p1_sofascore_player_id IS NOT NULL
      AND m.p1_canonical_id IS NULL
    """
)

BACKFILL_CANON_P2_SQL = text(
    """
    UPDATE tennis_matches m
    SET p2_canonical_id = s.player_id
    FROM tennis_player_sources s
    WHERE s.source = 'sofascore'
      AND NULLIF(btrim(s.source_player_id), '') ~ '^[0-9]+$'
      AND NULLIF(btrim(s.source_player_id), '')::bigint = m.p2_sofascore_player_id
      AND m.match_date BETWEEN :start AND :end
      AND m.p2_sofascore_player_id IS NOT NULL
      AND m.p2_canonical_id IS NULL
    """
)

# =============================================================================
# Auto-fill TA mapping + Elo
# - tennis_player_sources: (player_id=canonical_id, source=tennisabstract_elo_atp|wta, source_player_id=ta_player_id)
# - then copy ta_player_id onto matches
# - then fill match elo from tennisabstract_elo_snapshots (latest <= match_date)
# =============================================================================

INSERT_TA_MAPS_SQL = text(
    r"""
    WITH match_players AS (
      SELECT upper(tour) AS tour, p1_canonical_id AS canonical_id, p1_name AS name
      FROM tennis_matches
      WHERE match_date BETWEEN :start AND :end
        AND p1_canonical_id IS NOT NULL

      UNION ALL

      SELECT upper(tour) AS tour, p2_canonical_id AS canonical_id, p2_name AS name
      FROM tennis_matches
      WHERE match_date BETWEEN :start AND :end
        AND p2_canonical_id IS NOT NULL
    ),
    need AS (
      SELECT DISTINCT tour, canonical_id, name
      FROM match_players mp
      WHERE mp.tour IN ('ATP','WTA')
        AND NOT EXISTS (
          SELECT 1
          FROM tennis_player_sources tps
          WHERE tps.player_id = mp.canonical_id
            AND tps.source = CASE
                WHEN mp.tour = 'WTA' THEN 'tennisabstract_elo_wta'
                ELSE 'tennisabstract_elo_atp'
            END
        )
    ),
    latest AS (
      SELECT tour, max(as_of_date) AS as_of_date
      FROM tennisabstract_elo_snapshots
      GROUP BY tour
    ),
    norm_need AS (
      SELECT
        n.tour,
        n.canonical_id,
        lower(
          regexp_replace(
            regexp_replace(
              unaccent(replace(coalesce(n.name,''), chr(160), ' ')),
              '[^[:alnum:] ]', '', 'g'),
            E'\\s+', ' ', 'g'
          )
        ) AS n_norm
      FROM need n
    ),
    norm_snap AS (
      SELECT
        s.tour,
        s.player_id AS ta_player_id,
        lower(
          regexp_replace(
            regexp_replace(
              unaccent(replace(coalesce(s.player_name,''), chr(160), ' ')),
              '[^[:alnum:] ]', '', 'g'),
            E'\\s+', ' ', 'g'
          )
        ) AS s_norm,
        s.as_of_date
      FROM tennisabstract_elo_snapshots s
      WHERE s.player_id IS NOT NULL
    ),
    cand_base AS (
      SELECT DISTINCT
        nn.tour,
        nn.canonical_id,
        ns.ta_player_id
      FROM norm_need nn
      JOIN latest l
        ON l.tour = nn.tour
      JOIN norm_snap ns
        ON ns.tour = nn.tour
       AND ns.as_of_date = l.as_of_date
       AND ns.s_norm = nn.n_norm
    ),
    hits AS (
      SELECT
        tour,
        canonical_id,
        COUNT(*) AS hit_cnt,
        MIN(ta_player_id) AS ta_player_id
      FROM cand_base
      GROUP BY tour, canonical_id
    )
    INSERT INTO tennis_player_sources (player_id, source, source_player_id)
    SELECT
      canonical_id,
      CASE WHEN tour='WTA' THEN 'tennisabstract_elo_wta' ELSE 'tennisabstract_elo_atp' END,
      ta_player_id::text
    FROM hits
    WHERE hit_cnt = 1
      AND ta_player_id IS NOT NULL
    ON CONFLICT DO NOTHING;
    """
)

BACKFILL_MATCH_TA_P1_SQL = text(
    """
    UPDATE tennis_matches m
    SET p1_ta_player_id = NULLIF(btrim(tps.source_player_id), '')::bigint
    FROM tennis_player_sources tps
    WHERE m.match_date BETWEEN :start AND :end
      AND m.p1_ta_player_id IS NULL
      AND m.p1_canonical_id IS NOT NULL
      AND tps.player_id = m.p1_canonical_id
      AND tps.source = CASE WHEN upper(m.tour)='WTA' THEN 'tennisabstract_elo_wta' ELSE 'tennisabstract_elo_atp' END
      AND NULLIF(btrim(tps.source_player_id), '') ~ '^[0-9]+$'
    """
)

BACKFILL_MATCH_TA_P2_SQL = text(
    """
    UPDATE tennis_matches m
    SET p2_ta_player_id = NULLIF(btrim(tps.source_player_id), '')::bigint
    FROM tennis_player_sources tps
    WHERE m.match_date BETWEEN :start AND :end
      AND m.p2_ta_player_id IS NULL
      AND m.p2_canonical_id IS NOT NULL
      AND tps.player_id = m.p2_canonical_id
      AND tps.source = CASE WHEN upper(m.tour)='WTA' THEN 'tennisabstract_elo_wta' ELSE 'tennisabstract_elo_atp' END
      AND NULLIF(btrim(tps.source_player_id), '') ~ '^[0-9]+$'
    """
)

# IMPORTANT FIX:
# Avoid "invalid reference to FROM-clause entry for table m" by building a ranked subquery
# that already joined tennis_matches, then updating by match_id.

FILL_MATCH_ELO_P1_SQL = text(
    """
    WITH ranked AS (
      SELECT DISTINCT ON (m.match_id)
        m.match_id,
        s.elo,
        s.as_of_date
      FROM tennis_matches m
      JOIN tennisabstract_elo_snapshots s
        ON s.tour = upper(m.tour)
       AND s.player_id = m.p1_ta_player_id
       AND s.as_of_date <= m.match_date
      WHERE m.match_date BETWEEN :start AND :end
        AND upper(m.tour) IN ('ATP','WTA')
        AND m.p1_ta_player_id IS NOT NULL
        AND m.p1_elo IS NULL
      ORDER BY m.match_id, s.as_of_date DESC
    )
    UPDATE tennis_matches m
    SET
      p1_elo = r.elo,
      p1_elo_as_of_date = r.as_of_date,
      p1_elo_source = 'tennisabstract'
    FROM ranked r
    WHERE m.match_id = r.match_id
    """
)

FILL_MATCH_ELO_P2_SQL = text(
    """
    WITH ranked AS (
      SELECT DISTINCT ON (m.match_id)
        m.match_id,
        s.elo,
        s.as_of_date
      FROM tennis_matches m
      JOIN tennisabstract_elo_snapshots s
        ON s.tour = upper(m.tour)
       AND s.player_id = m.p2_ta_player_id
       AND s.as_of_date <= m.match_date
      WHERE m.match_date BETWEEN :start AND :end
        AND upper(m.tour) IN ('ATP','WTA')
        AND m.p2_ta_player_id IS NOT NULL
        AND m.p2_elo IS NULL
      ORDER BY m.match_id, s.as_of_date DESC
    )
    UPDATE tennis_matches m
    SET
      p2_elo = r.elo,
      p2_elo_as_of_date = r.as_of_date,
      p2_elo_source = 'tennisabstract'
    FROM ranked r
    WHERE m.match_id = r.match_id
    """
)

FILL_FALLBACK_ELO_SQL = text(
    """
    UPDATE tennis_matches
    SET
      p1_elo = COALESCE(p1_elo, :fallback),
      p1_elo_source = COALESCE(p1_elo_source, 'fallback')
    WHERE match_date BETWEEN :start AND :end
      AND p1_canonical_id IS NOT NULL
      AND p1_elo IS NULL;

    UPDATE tennis_matches
    SET
      p2_elo = COALESCE(p2_elo, :fallback),
      p2_elo_source = COALESCE(p2_elo_source, 'fallback')
    WHERE match_date BETWEEN :start AND :end
      AND p2_canonical_id IS NOT NULL
      AND p2_elo IS NULL;
    """
)

# =============================================================================
# Helpers
# =============================================================================

def _normalize_unix_ts_seconds(ts: object) -> Optional[int]:
    if ts is None:
        return None
    try:
        v = int(ts)
    except Exception:
        return None
    if v <= 0:
        return None
    # If it's clearly milliseconds (13 digits), convert to seconds.
    if v > 10_000_000_000:
        v = v // 1000
    return v


def _ts_to_utc_dt(ts_seconds: Optional[int]) -> Optional[dt.datetime]:
    if not ts_seconds:
        return None
    return dt.datetime.fromtimestamp(ts_seconds, tz=dt.timezone.utc)


async def get_json(path: str, params: Optional[dict] = None, *, context=None) -> dict:
    url = f"{BASE_URL}{path}"
    if params:
        url = f"{url}?{urlencode(params)}"

    # Short-lived context path
    if context is None:
        async with async_playwright() as p:
            browser2 = await p.chromium.launch(headless=True)
            context2 = await browser2.new_context(extra_http_headers=BROWSER_HEADERS)
            page = await context2.new_page()

            await page.goto("https://www.sofascore.com/", wait_until="domcontentloaded", timeout=60000)
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=60000)

            if not resp:
                await browser2.close()
                raise RuntimeError(f"No response for {url}")

            if resp.status != 200:
                body = await resp.text()
                await browser2.close()
                raise RuntimeError(f"SofaScore API failed {resp.status} for {url}. Body (first 300): {body[:300]}")

            body = await resp.text()
            await browser2.close()
        return json.loads(body)

    # Shared context path
    page = await context.new_page()
    resp = await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    if not resp:
        await page.close()
        raise RuntimeError(f"No response for {url}")

    if resp.status != 200:
        body = await resp.text()
        await page.close()
        raise RuntimeError(f"SofaScore API failed {resp.status} for {url}. Body (first 300): {body[:300]}")

    body = await resp.text()
    await page.close()
    return json.loads(body)


async def fetch_matches_for_date(day: dt.date, *, context=None) -> List[dict]:
    payload = await get_json(f"/sport/tennis/scheduled-events/{day.isoformat()}", context=context)
    return payload.get("events") or []


def safe_get(d: dict, *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


_COUNTRY_PAT = re.compile(r"^[A-Za-z][A-Za-z .'-]{1,}$")
COUNTRY_ALIASES = {
    "usa": "United States",
    "u s a": "United States",
    "u.s.a": "United States",
    "great britain": "United Kingdom",
    "gb": "United Kingdom",
    "czech republic": "Czechia",
}


def looks_like_country(name: str) -> bool:
    if not name:
        return False
    s = name.strip()
    if len(s) < 3:
        return False
    if "/" in s or "," in s:
        return False
    return bool(_COUNTRY_PAT.match(s))


def normalize_country(name: str) -> str:
    s = (name or "").strip().lower().replace(".", "")
    s = re.sub(r"\s+", " ", s)
    return COUNTRY_ALIASES.get(s, (name or "").strip())


def get_players_from_event(event: dict) -> Optional[Tuple[dict, dict]]:
    competitors = event.get("competitors")
    if isinstance(competitors, list) and len(competitors) >= 2:
        home = away = None
        for c in competitors:
            if c.get("home") is True:
                home = c
            if c.get("away") is True:
                away = c
        if home and away:
            return home, away
        return competitors[0], competitors[1]

    home_team = event.get("homeTeam")
    away_team = event.get("awayTeam")
    if isinstance(home_team, dict) and isinstance(away_team, dict):
        return home_team, away_team

    return None


def get_competitor_name(c: Optional[dict]) -> str:
    if not c:
        return ""
    return (c.get("name") or c.get("shortName") or "").strip()


def get_competitor_id(c: Optional[dict]) -> Optional[int]:
    if not c:
        return None
    cid = c.get("id")
    if isinstance(cid, int):
        return cid
    try:
        return int(cid) if cid is not None else None
    except Exception:
        return None


def is_doubles_event(event: dict) -> bool:
    tname = ((event.get("tournament") or {}).get("name") or "").lower()
    if "doubles" in tname:
        return True

    players = get_players_from_event(event)
    if players:
        a, b = players
        if "/" in get_competitor_name(a) or "/" in get_competitor_name(b):
            return True

    return False


def extract_category_name(event: dict) -> str:
    return (
        (safe_get(event, "tournament", "category", "name", default="") or "").strip().lower()
        or (safe_get(event, "tournament", "uniqueTournament", "category", "name", default="") or "").strip().lower()
    )


def detect_tour_label(event: dict) -> str:
    tname = ((event.get("tournament") or {}).get("name") or "").strip()
    tname_l = tname.lower()

    players = get_players_from_event(event)
    a_name = b_name = ""
    if players:
        a, b = players
        a_name = get_competitor_name(a)
        b_name = get_competitor_name(b)

    def is_probably_country(s: str) -> bool:
        if looks_like_country(s):
            return " " not in s.strip()
        return False

    if "united cup" in tname_l:
        if is_probably_country(a_name) and is_probably_country(b_name):
            return "TEAM"
        return "MIXED"

    if "davis cup" in tname_l:
        if is_probably_country(a_name) and is_probably_country(b_name):
            return "TEAM"
        return "ATP"

    if any(x in tname_l for x in ["billie jean king", "fed cup"]):
        if is_probably_country(a_name) and is_probably_country(b_name):
            return "TEAM"
        return "WTA"

    if any(x in tname_l for x in ["hopman cup", "atp cup"]):
        if is_probably_country(a_name) and is_probably_country(b_name):
            return "TEAM"
        return "TEAM"

    cat = extract_category_name(event)
    if "wta" in cat:
        return "WTA"
    if "atp" in cat:
        return "ATP"
    if "challenger" in cat:
        return "CHALLENGER"
    if "itf" in cat:
        return "ITF"

    if "wta" in tname_l:
        return "WTA"
    if "atp" in tname_l:
        return "ATP"
    if "challenger" in tname_l:
        return "CHALLENGER"
    if "itf" in tname_l:
        return "ITF"

    return "UNKNOWN"


def extract_score_raw(event: dict) -> Optional[str]:
    hs = safe_get(event, "homeScore", "current", default=None)
    aws = safe_get(event, "awayScore", "current", default=None)
    if hs is not None and aws is not None:
        return f"{hs}-{aws}"

    score = event.get("score")
    if isinstance(score, dict):
        hs2 = score.get("home")
        aw2 = score.get("away")
        if hs2 is not None and aw2 is not None:
            return f"{hs2}-{aw2}"

    return None


def extract_surface(event: dict) -> Optional[str]:
    surface = safe_get(event, "tournament", "uniqueTournament", "groundType", default=None)
    if isinstance(surface, str) and surface.strip():
        return surface.strip()
    surface = safe_get(event, "tournament", "uniqueTournament", "surface", default=None)
    if isinstance(surface, str) and surface.strip():
        return surface.strip()
    return None


def _surface_family(surface: Optional[str]) -> str:
    s = (surface or "").lower()
    if "hard" in s:
        return "hard"
    if "clay" in s:
        return "clay"
    if "grass" in s:
        return "grass"
    return "unknown"


def _h2h_surface_from_event(event: dict) -> Optional[str]:
    surface = event.get("groundType")
    if isinstance(surface, str) and surface.strip():
        return surface.strip()
    surface = safe_get(event, "tournament", "uniqueTournament", "groundType", default=None)
    if isinstance(surface, str) and surface.strip():
        return surface.strip()
    surface = safe_get(event, "tournament", "uniqueTournament", "surface", default=None)
    if isinstance(surface, str) and surface.strip():
        return surface.strip()
    return None


def _compute_h2h_from_events(
    h2h_events_payload: dict,
    p1_id: Optional[int],
    p2_id: Optional[int],
    surface_family_current: str,
) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    events = safe_get(h2h_events_payload, "events", "events", default=[]) or []
    if not isinstance(events, list) or not p1_id or not p2_id:
        return None, None, None, None, None, None

    p1_wins = p2_wins = total = 0
    s_p1_wins = s_p2_wins = s_total = 0

    for ev in events:
        home_id = safe_get(ev, "homeTeam", "id", default=None)
        away_id = safe_get(ev, "awayTeam", "id", default=None)
        if home_id is None or away_id is None:
            continue
        if {home_id, away_id} != {p1_id, p2_id}:
            continue

        winner = ev.get("winnerCode")
        if winner not in (1, 2):
            continue

        total += 1
        if (winner == 1 and home_id == p1_id) or (winner == 2 and away_id == p1_id):
            p1_wins += 1
        else:
            p2_wins += 1

        if surface_family_current != "unknown":
            ev_surface = _surface_family(_h2h_surface_from_event(ev))
            if ev_surface == surface_family_current:
                s_total += 1
                if (winner == 1 and home_id == p1_id) or (winner == 2 and away_id == p1_id):
                    s_p1_wins += 1
                else:
                    s_p2_wins += 1

    return p1_wins, p2_wins, total, s_p1_wins, s_p2_wins, s_total


def extract_round(event: dict) -> Optional[str]:
    rnd = safe_get(event, "roundInfo", "name", default=None)
    if isinstance(rnd, str) and rnd.strip():
        return rnd.strip()
    return None


def extract_status(event: dict) -> Optional[str]:
    return safe_get(event, "status", "type", default=None)


def _extract_match_times(day: dt.date, event: dict) -> tuple[dt.date, Optional[dt.datetime], Optional[int]]:
    ts = _normalize_unix_ts_seconds(event.get("startTimestamp"))
    if ts:
        try:
            match_date = match_date_central(ts)
        except Exception:
            match_date = day
        start_utc = _ts_to_utc_dt(ts)
        return match_date, start_utc, ts
    return day, None, None


def extract_match_row(day: dt.date, event: dict) -> Optional[dict]:
    event_id = event.get("id")
    if not event_id:
        return None

    players = get_players_from_event(event)
    if not players:
        return None

    home, away = players
    p1_name = get_competitor_name(home)
    p2_name = get_competitor_name(away)

    # capture SofaScore player IDs
    p1_sofa_id = get_competitor_id(home)
    p2_sofa_id = get_competitor_id(away)

    tournament = ((event.get("tournament") or {}).get("name") or None)
    status = extract_status(event)
    surface = extract_surface(event)
    rnd = extract_round(event)
    score_raw = extract_score_raw(event)

    tour_label = detect_tour_label(event)

    if tour_label == "TEAM":
        # countries, not players
        p1_name = normalize_country(p1_name)
        p2_name = normalize_country(p2_name)
        p1_sofa_id = None
        p2_sofa_id = None

    match_date, match_start_utc, start_ts = _extract_match_times(day, event)

    return {
        "match_key": f"sofascore:{int(event_id)}",
        "sofascore_event_id": int(event_id),

        "match_date": match_date,
        "match_start_utc": match_start_utc,
        "start_timestamp": start_ts,

        "tour": tour_label,
        "tournament": tournament,
        "status": status,
        "round": rnd,
        "surface": surface,
        "p1_name": p1_name,
        "p2_name": p2_name,
        "score_raw": score_raw,

        "p1_sofascore_player_id": p1_sofa_id,
        "p2_sofascore_player_id": p2_sofa_id,

        "p1_canonical_id": None,
        "p2_canonical_id": None,
        "winner_canonical_id": None,

        # odds (nullable)
        "p1_odds_american": None,
        "p2_odds_american": None,
        "odds_fetched_at": None,
        "sofascore_p1_odds_american": None,
        "sofascore_p2_odds_american": None,
        "sofascore_odds_fetched_at": None,
        "flashscore_p1_odds_american": None,
        "flashscore_p2_odds_american": None,
        "flashscore_odds_fetched_at": None,
        "sofascore_total_games_line": None,
        "sofascore_total_games_over_american": None,
        "sofascore_total_games_under_american": None,
        "sofascore_spread_p1_line": None,
        "sofascore_spread_p2_line": None,
        "sofascore_spread_p1_odds_american": None,
        "sofascore_spread_p2_odds_american": None,

        # h2h (nullable)
        "h2h_p1_wins": None,
        "h2h_p2_wins": None,
        "h2h_total_matches": None,
        "h2h_surface_p1_wins": None,
        "h2h_surface_p2_wins": None,
        "h2h_surface_matches": None,
    }


def parse_bool_env(key: str, default: bool = False) -> bool:
    v = (os.getenv(key) or "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "y", "on"}


def parse_float_env(key: str, default: float) -> float:
    v = (os.getenv(key) or "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def parse_int_env(key: str, default: int) -> int:
    v = (os.getenv(key) or "").strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def get_tour_filter_set() -> Optional[Set[str]]:
    v = (os.getenv("TOUR_FILTER") or "").strip()
    if not v:
        return None
    parts = [p.strip().upper() for p in v.split(",") if p.strip()]
    return set(parts) if parts else None


def is_allowed_by_filter(event: dict, tour_filter: Optional[Set[str]], include_wta125: bool) -> bool:
    if not tour_filter:
        return True

    cat_norm = re.sub(r"\s+", " ", (extract_category_name(event) or "")).strip()
    label = detect_tour_label(event).upper()
    wanted = {w.upper() for w in tour_filter}

    def match_one(w: str) -> bool:
        w = w.upper()

        if w == "WTA":
            if cat_norm == "wta":
                return True
            if include_wta125 and ("wta" in cat_norm and "125" in cat_norm):
                return True
            return label == "WTA"

        if w == "ATP":
            if cat_norm == "atp":
                return True
            return label == "ATP"

        if w == "CHALLENGER":
            return ("challenger" in cat_norm) or (label == "CHALLENGER")

        if w == "ITF":
            return ("itf" in cat_norm) or (label == "ITF")

        return (w.lower() in cat_norm) or (w == label)

    return any(match_one(w) for w in wanted)


# =============================================================================
# Odds helpers (best-effort)
# =============================================================================

def decimal_to_american(decimal_odds: float) -> Optional[int]:
    try:
        d = float(decimal_odds)
    except Exception:
        return None
    if d <= 1.0:
        return None
    if d >= 2.0:
        return int(round((d - 1.0) * 100.0))
    return int(round(-100.0 / (d - 1.0)))


def fractional_to_decimal(frac: str) -> Optional[float]:
    if not frac or not isinstance(frac, str):
        return None
    s = frac.strip()
    if "/" not in s:
        return None
    try:
        a_str, b_str = s.split("/", 1)
        a = float(a_str.strip())
        b = float(b_str.strip())
        if b == 0:
            return None
        return 1.0 + (a / b)
    except Exception:
        return None


def fractional_to_american(frac: str) -> Optional[int]:
    d = fractional_to_decimal(frac)
    if d is None:
        return None
    return decimal_to_american(d)


def extract_best_effort_american_odds(odds_payload: dict, event_id: int = None, tour: str = None) -> tuple[Optional[int], Optional[int]]:
    if not isinstance(odds_payload, dict) or not odds_payload:
        return None, None

    markets = odds_payload.get("markets")
    if not isinstance(markets, list) or not markets:
        return None, None

    ml_market = None
    for m in markets:
        if isinstance(m, dict) and m.get("marketId") == 1:
            ml_market = m
            break
    if not ml_market:
        return None, None

    choices = ml_market.get("choices")
    if not isinstance(choices, list) or len(choices) < 2:
        return None, None

    p1 = p2 = None
    for ch in choices:
        ch_name = str(ch.get("name") or "").strip()
        frac = ch.get("fractionalValue") or ch.get("initialFractionalValue")
        if isinstance(frac, str) and frac.strip():
            am = fractional_to_american(frac)
            if ch_name == "1":
                p1 = am
            elif ch_name == "2":
                p2 = am
    return p1, p2


def _parse_total_line_candidate(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except Exception:
            return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def extract_total_games_market(odds_payload: dict) -> tuple[Optional[float], Optional[int], Optional[int]]:
    if not isinstance(odds_payload, dict) or not odds_payload:
        return None, None, None

    markets = odds_payload.get("markets")
    if not isinstance(markets, list) or not markets:
        return None, None, None

    target = None
    for m in markets:
        if not isinstance(m, dict):
            continue
        if m.get("marketId") == 12:
            target = m
            break

    if not target:
        return None, None, None

    line = None
    # SofaScore payload is inconsistent by provider; try multiple places.
    line = (
        _parse_total_line_candidate(target.get("line"))
        or _parse_total_line_candidate(target.get("handicap"))
        or _parse_total_line_candidate(target.get("choiceGroup"))
    )
    if line is None:
        mname = str(target.get("marketName") or "")
        mm = re.search(r"(\d+(?:\.\d+)?)", mname)
        if mm:
            line = _parse_total_line_candidate(mm.group(1))

    over = under = None
    for ch in (target.get("choices") or []):
        if not isinstance(ch, dict):
            continue
        cname = str(ch.get("name") or "").strip().lower()
        frac = ch.get("fractionalValue") or ch.get("initialFractionalValue")
        am = fractional_to_american(frac) if isinstance(frac, str) and frac.strip() else None
        if "over" in cname:
            over = am
        elif "under" in cname:
            under = am
        if line is None:
            line = (
                _parse_total_line_candidate(ch.get("line"))
                or _parse_total_line_candidate(ch.get("handicap"))
                or line
            )

    return line, over, under


def extract_game_spread_market(odds_payload: dict) -> tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    if not isinstance(odds_payload, dict) or not odds_payload:
        return None, None, None, None

    markets = odds_payload.get("markets")
    if not isinstance(markets, list) or not markets:
        return None, None, None, None

    target = None
    for m in markets:
        if not isinstance(m, dict):
            continue
        name = str(m.get("marketName") or "").strip().lower()
        if "handicap" in name and "set" not in name:
            target = m
            break
        if "spread" in name and "set" not in name:
            target = m
            break

    if not target:
        return None, None, None, None

    base_line = (
        _parse_total_line_candidate(target.get("choiceGroup"))
        or _parse_total_line_candidate(target.get("line"))
        or _parse_total_line_candidate(target.get("handicap"))
    )
    p1_line = base_line
    p2_line = (-base_line if base_line is not None else None)
    p1_odds = p2_odds = None

    for ch in (target.get("choices") or []):
        if not isinstance(ch, dict):
            continue
        cname = str(ch.get("name") or "").strip()
        frac = ch.get("fractionalValue") or ch.get("initialFractionalValue")
        am = fractional_to_american(frac) if isinstance(frac, str) and frac.strip() else None

        line_val = (
            _parse_total_line_candidate(ch.get("line"))
            or _parse_total_line_candidate(ch.get("handicap"))
            or _parse_total_line_candidate(ch.get("choiceGroup"))
        )
        if cname == "1":
            p1_odds = am
            if line_val is not None:
                p1_line = line_val
        elif cname == "2":
            p2_odds = am
            if line_val is not None:
                p2_line = line_val

    if p1_line is not None and p2_line is None:
        p2_line = -p1_line
    elif p2_line is not None and p1_line is None:
        p1_line = -p2_line

    return p1_line, p2_line, p1_odds, p2_odds


async def fetch_odds_for_event(event_id: int, *, context=None) -> dict:
    provider_id = parse_int_env("ODDS_PROVIDER_ID", 1)
    try:
        return await get_json(f"/event/{event_id}/odds/{provider_id}/all", context=context)
    except Exception:
        return {}


async def fetch_h2h_summary(event_id: int, *, context=None) -> dict:
    try:
        return await get_json(f"/event/{event_id}/h2h", context=context)
    except Exception:
        return {}


async def fetch_h2h_events(match_code: str, *, context=None) -> dict:
    if not match_code:
        return {}
    try:
        return await get_json(f"/event/{match_code}/h2h/events", context=context)
    except Exception:
        return {}


# =============================================================================
# DB schema / ensure tables
# =============================================================================

ENSURE_SQL_STATEMENTS = [
    "CREATE EXTENSION IF NOT EXISTS pgcrypto",
    # Minimal tennis_player_sources in case it's missing (won't touch if already exists)
    """
    CREATE TABLE IF NOT EXISTS tennis_player_sources (
      player_id INTEGER NOT NULL,
      source TEXT NOT NULL,
      source_player_id TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      PRIMARY KEY (source, source_player_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sofascore_events_raw (
        event_id   BIGINT PRIMARY KEY,
        payload    JSONB NOT NULL,
        fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sofascore_event_odds_raw (
        event_id   BIGINT PRIMARY KEY,
        payload    JSONB NOT NULL,
        fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sofascore_event_h2h_raw (
        event_id   BIGINT PRIMARY KEY,
        payload    JSONB NOT NULL,
        fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # If tennis_matches doesn't exist, create it (safe if it already exists).
    # NOTE: if your existing tennis_matches has different constraints, CREATE TABLE IF NOT EXISTS won't modify it.
    """
    CREATE TABLE IF NOT EXISTS tennis_matches (
        match_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        match_key TEXT NOT NULL UNIQUE,

        match_date DATE NOT NULL,
        match_start_utc TIMESTAMPTZ NULL,
        start_timestamp BIGINT NULL,

        tour TEXT NOT NULL,

        tournament TEXT NULL,
        status TEXT NULL,
        "round" TEXT NULL,
        surface TEXT NULL,

        p1_name TEXT NOT NULL,
        p2_name TEXT NOT NULL,

        p1_sofascore_player_id BIGINT NULL,
        p2_sofascore_player_id BIGINT NULL,

        p1_canonical_id INTEGER NULL,
        p2_canonical_id INTEGER NULL,
        winner_canonical_id INTEGER NULL,

        sofascore_event_id BIGINT UNIQUE NOT NULL,
        score_raw TEXT NULL,

        p1_odds_american INTEGER NULL,
        p2_odds_american INTEGER NULL,
        odds_fetched_at TIMESTAMPTZ NULL,
        sofascore_p1_odds_american INTEGER NULL,
        sofascore_p2_odds_american INTEGER NULL,
        sofascore_odds_fetched_at TIMESTAMPTZ NULL,
        flashscore_p1_odds_american INTEGER NULL,
        flashscore_p2_odds_american INTEGER NULL,
        flashscore_odds_fetched_at TIMESTAMPTZ NULL,
        sofascore_total_games_line DOUBLE PRECISION NULL,
        sofascore_total_games_over_american INTEGER NULL,
        sofascore_total_games_under_american INTEGER NULL,
        sofascore_spread_p1_line DOUBLE PRECISION NULL,
        sofascore_spread_p2_line DOUBLE PRECISION NULL,
        sofascore_spread_p1_odds_american INTEGER NULL,
        sofascore_spread_p2_odds_american INTEGER NULL,

        p1_ta_player_id BIGINT NULL,
        p2_ta_player_id BIGINT NULL,
        p1_elo DOUBLE PRECISION NULL,
        p2_elo DOUBLE PRECISION NULL,
        p1_elo_as_of_date DATE NULL,
        p2_elo_as_of_date DATE NULL,
        p1_elo_source TEXT NULL,
        p2_elo_source TEXT NULL,

        h2h_p1_wins INTEGER NULL,
        h2h_p2_wins INTEGER NULL,
        h2h_total_matches INTEGER NULL,
        h2h_surface_p1_wins INTEGER NULL,
        h2h_surface_p2_wins INTEGER NULL,
        h2h_surface_matches INTEGER NULL,

        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # Add missing columns safely if table already exists
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS p1_sofascore_player_id BIGINT NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS p2_sofascore_player_id BIGINT NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS match_start_utc TIMESTAMPTZ NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS start_timestamp BIGINT NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS p1_odds_american INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS p2_odds_american INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS odds_fetched_at TIMESTAMPTZ NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS sofascore_p1_odds_american INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS sofascore_p2_odds_american INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS sofascore_odds_fetched_at TIMESTAMPTZ NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS flashscore_p1_odds_american INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS flashscore_p2_odds_american INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS flashscore_odds_fetched_at TIMESTAMPTZ NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS sofascore_total_games_line DOUBLE PRECISION NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS sofascore_total_games_over_american INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS sofascore_total_games_under_american INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS sofascore_spread_p1_line DOUBLE PRECISION NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS sofascore_spread_p2_line DOUBLE PRECISION NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS sofascore_spread_p1_odds_american INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS sofascore_spread_p2_odds_american INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS p1_ta_player_id BIGINT NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS p2_ta_player_id BIGINT NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS p1_elo DOUBLE PRECISION NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS p2_elo DOUBLE PRECISION NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS p1_elo_as_of_date DATE NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS p2_elo_as_of_date DATE NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS p1_elo_source TEXT NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS p2_elo_source TEXT NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS h2h_p1_wins INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS h2h_p2_wins INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS h2h_total_matches INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS h2h_surface_p1_wins INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS h2h_surface_p2_wins INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS h2h_surface_matches INTEGER NULL",
    # Indexes
    "CREATE INDEX IF NOT EXISTS ix_tennis_matches_match_date ON tennis_matches (match_date)",
    "CREATE INDEX IF NOT EXISTS ix_tennis_matches_match_start_utc ON tennis_matches (match_start_utc)",
    "CREATE INDEX IF NOT EXISTS ix_tennis_matches_tournament ON tennis_matches (tournament)",
    "CREATE INDEX IF NOT EXISTS ix_tennis_matches_tour ON tennis_matches (tour)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_tennis_matches_sofa_event ON tennis_matches (sofascore_event_id)",
    "CREATE INDEX IF NOT EXISTS ix_tennis_matches_p1_sofa ON tennis_matches (p1_sofascore_player_id)",
    "CREATE INDEX IF NOT EXISTS ix_tennis_matches_p2_sofa ON tennis_matches (p2_sofascore_player_id)",
    "CREATE INDEX IF NOT EXISTS ix_tennis_matches_p1_ta ON tennis_matches (p1_ta_player_id)",
    "CREATE INDEX IF NOT EXISTS ix_tennis_matches_p2_ta ON tennis_matches (p2_ta_player_id)",
    "CREATE INDEX IF NOT EXISTS ix_sofascore_events_raw_fetched_at ON sofascore_events_raw (fetched_at)",
    "CREATE INDEX IF NOT EXISTS ix_sofascore_event_odds_raw_fetched_at ON sofascore_event_odds_raw (fetched_at)",
    "CREATE INDEX IF NOT EXISTS ix_sofascore_event_h2h_raw_fetched_at ON sofascore_event_h2h_raw (fetched_at)",
    # Helpful index for trim/cast joins on sofascore ids in tennis_player_sources
    """
    CREATE INDEX IF NOT EXISTS ix_tps_sofa_id_bigint
    ON tennis_player_sources ((btrim(source_player_id)::bigint))
    WHERE source='sofascore' AND source_player_id ~ '^[[:space:]]*[0-9]+[[:space:]]*$'
    """,
]


async def ensure_tables(conn: AsyncConnection) -> None:
    for stmt in ENSURE_SQL_STATEMENTS:
        await conn.execute(text(stmt))


UPSERT_EVENT_RAW_SQL = (
    text(
        """
        INSERT INTO sofascore_events_raw (event_id, payload, fetched_at)
        VALUES (:event_id, :payload, NOW())
        ON CONFLICT (event_id)
        DO UPDATE SET
          payload = EXCLUDED.payload,
          fetched_at = NOW()
        """
    ).bindparams(bindparam("payload", type_=JSONB))
)

UPSERT_ODDS_RAW_SQL = (
    text(
        """
        INSERT INTO sofascore_event_odds_raw (event_id, payload, fetched_at)
        VALUES (:event_id, :payload, NOW())
        ON CONFLICT (event_id)
        DO UPDATE SET
          payload = EXCLUDED.payload,
          fetched_at = NOW()
        """
    ).bindparams(bindparam("payload", type_=JSONB))
)

UPSERT_H2H_RAW_SQL = (
    text(
        """
        INSERT INTO sofascore_event_h2h_raw (event_id, payload, fetched_at)
        VALUES (:event_id, :payload, NOW())
        ON CONFLICT (event_id)
        DO UPDATE SET
          payload = EXCLUDED.payload,
          fetched_at = NOW()
        """
    ).bindparams(bindparam("payload", type_=JSONB))
)

# IMPORTANT: never wipe canonical ids / surface / round / score / odds if excluded values are NULL
UPSERT_MATCH_SQL = text(
    """
    INSERT INTO tennis_matches (
        match_key,
        match_date,
        match_start_utc,
        start_timestamp,
        tour,
        tournament,
        status,
        "round",
        surface,
        p1_name,
        p2_name,

        p1_sofascore_player_id,
        p2_sofascore_player_id,

        p1_canonical_id,
        p2_canonical_id,
        winner_canonical_id,
        sofascore_event_id,
        score_raw,

        p1_odds_american,
        p2_odds_american,
        odds_fetched_at,
        sofascore_p1_odds_american,
        sofascore_p2_odds_american,
        sofascore_odds_fetched_at,
        flashscore_p1_odds_american,
        flashscore_p2_odds_american,
        flashscore_odds_fetched_at,
        sofascore_total_games_line,
        sofascore_total_games_over_american,
        sofascore_total_games_under_american,
        sofascore_spread_p1_line,
        sofascore_spread_p2_line,
        sofascore_spread_p1_odds_american,
        sofascore_spread_p2_odds_american,

        h2h_p1_wins,
        h2h_p2_wins,
        h2h_total_matches,
        h2h_surface_p1_wins,
        h2h_surface_p2_wins,
        h2h_surface_matches,

        created_at,
        updated_at
    )
    VALUES (
        :match_key,
        :match_date,
        :match_start_utc,
        :start_timestamp,
        :tour,
        :tournament,
        :status,
        :round,
        :surface,
        :p1_name,
        :p2_name,

        :p1_sofascore_player_id,
        :p2_sofascore_player_id,

        :p1_canonical_id,
        :p2_canonical_id,
        :winner_canonical_id,
        :sofascore_event_id,
        :score_raw,

        :p1_odds_american,
        :p2_odds_american,
        :odds_fetched_at,
        :sofascore_p1_odds_american,
        :sofascore_p2_odds_american,
        :sofascore_odds_fetched_at,
        :flashscore_p1_odds_american,
        :flashscore_p2_odds_american,
        :flashscore_odds_fetched_at,
        :sofascore_total_games_line,
        :sofascore_total_games_over_american,
        :sofascore_total_games_under_american,
        :sofascore_spread_p1_line,
        :sofascore_spread_p2_line,
        :sofascore_spread_p1_odds_american,
        :sofascore_spread_p2_odds_american,

        :h2h_p1_wins,
        :h2h_p2_wins,
        :h2h_total_matches,
        :h2h_surface_p1_wins,
        :h2h_surface_p2_wins,
        :h2h_surface_matches,

        NOW(),
        NOW()
    )
    ON CONFLICT (sofascore_event_id)
    DO UPDATE SET
        match_key = EXCLUDED.match_key,
        match_date = EXCLUDED.match_date,

        match_start_utc = COALESCE(EXCLUDED.match_start_utc, tennis_matches.match_start_utc),
        start_timestamp = COALESCE(EXCLUDED.start_timestamp, tennis_matches.start_timestamp),

        tour = EXCLUDED.tour,
        tournament = COALESCE(EXCLUDED.tournament, tennis_matches.tournament),
        status = COALESCE(EXCLUDED.status, tennis_matches.status),
        "round" = COALESCE(EXCLUDED."round", tennis_matches."round"),
        surface = COALESCE(EXCLUDED.surface, tennis_matches.surface),
        p1_name = EXCLUDED.p1_name,
        p2_name = EXCLUDED.p2_name,

        p1_sofascore_player_id = COALESCE(EXCLUDED.p1_sofascore_player_id, tennis_matches.p1_sofascore_player_id),
        p2_sofascore_player_id = COALESCE(EXCLUDED.p2_sofascore_player_id, tennis_matches.p2_sofascore_player_id),

        p1_canonical_id = COALESCE(EXCLUDED.p1_canonical_id, tennis_matches.p1_canonical_id),
        p2_canonical_id = COALESCE(EXCLUDED.p2_canonical_id, tennis_matches.p2_canonical_id),
        winner_canonical_id = COALESCE(EXCLUDED.winner_canonical_id, tennis_matches.winner_canonical_id),
        score_raw = COALESCE(EXCLUDED.score_raw, tennis_matches.score_raw),

        p1_odds_american = COALESCE(EXCLUDED.p1_odds_american, tennis_matches.p1_odds_american),
        p2_odds_american = COALESCE(EXCLUDED.p2_odds_american, tennis_matches.p2_odds_american),
        odds_fetched_at  = COALESCE(EXCLUDED.odds_fetched_at,  tennis_matches.odds_fetched_at),
        sofascore_p1_odds_american = COALESCE(EXCLUDED.sofascore_p1_odds_american, tennis_matches.sofascore_p1_odds_american),
        sofascore_p2_odds_american = COALESCE(EXCLUDED.sofascore_p2_odds_american, tennis_matches.sofascore_p2_odds_american),
        sofascore_odds_fetched_at  = COALESCE(EXCLUDED.sofascore_odds_fetched_at,  tennis_matches.sofascore_odds_fetched_at),
        flashscore_p1_odds_american = COALESCE(EXCLUDED.flashscore_p1_odds_american, tennis_matches.flashscore_p1_odds_american),
        flashscore_p2_odds_american = COALESCE(EXCLUDED.flashscore_p2_odds_american, tennis_matches.flashscore_p2_odds_american),
        flashscore_odds_fetched_at  = COALESCE(EXCLUDED.flashscore_odds_fetched_at,  tennis_matches.flashscore_odds_fetched_at),
        sofascore_total_games_line = COALESCE(EXCLUDED.sofascore_total_games_line, tennis_matches.sofascore_total_games_line),
        sofascore_total_games_over_american = COALESCE(EXCLUDED.sofascore_total_games_over_american, tennis_matches.sofascore_total_games_over_american),
        sofascore_total_games_under_american = COALESCE(EXCLUDED.sofascore_total_games_under_american, tennis_matches.sofascore_total_games_under_american),
        sofascore_spread_p1_line = COALESCE(EXCLUDED.sofascore_spread_p1_line, tennis_matches.sofascore_spread_p1_line),
        sofascore_spread_p2_line = COALESCE(EXCLUDED.sofascore_spread_p2_line, tennis_matches.sofascore_spread_p2_line),
        sofascore_spread_p1_odds_american = COALESCE(EXCLUDED.sofascore_spread_p1_odds_american, tennis_matches.sofascore_spread_p1_odds_american),
        sofascore_spread_p2_odds_american = COALESCE(EXCLUDED.sofascore_spread_p2_odds_american, tennis_matches.sofascore_spread_p2_odds_american),

        h2h_p1_wins = COALESCE(EXCLUDED.h2h_p1_wins, tennis_matches.h2h_p1_wins),
        h2h_p2_wins = COALESCE(EXCLUDED.h2h_p2_wins, tennis_matches.h2h_p2_wins),
        h2h_total_matches = COALESCE(EXCLUDED.h2h_total_matches, tennis_matches.h2h_total_matches),
        h2h_surface_p1_wins = COALESCE(EXCLUDED.h2h_surface_p1_wins, tennis_matches.h2h_surface_p1_wins),
        h2h_surface_p2_wins = COALESCE(EXCLUDED.h2h_surface_p2_wins, tennis_matches.h2h_surface_p2_wins),
        h2h_surface_matches = COALESCE(EXCLUDED.h2h_surface_matches, tennis_matches.h2h_surface_matches),

        updated_at = NOW()
    """
)

H2H_BACKFILL_SELECT_SQL = text(
    """
    SELECT
      m.match_id,
      m.surface,
      m.p1_sofascore_player_id AS p1_id,
      m.p2_sofascore_player_id AS p2_id,
      r.payload
    FROM tennis_matches m
    JOIN sofascore_event_h2h_raw r
      ON r.event_id = m.sofascore_event_id
    WHERE m.match_date BETWEEN :start AND :end
    """
)

H2H_BACKFILL_UPDATE_SQL = text(
    """
    UPDATE tennis_matches
    SET
      h2h_p1_wins = COALESCE(:h2h_p1_wins, h2h_p1_wins),
      h2h_p2_wins = COALESCE(:h2h_p2_wins, h2h_p2_wins),
      h2h_total_matches = COALESCE(:h2h_total_matches, h2h_total_matches),
      h2h_surface_p1_wins = COALESCE(:h2h_surface_p1_wins, h2h_surface_p1_wins),
      h2h_surface_p2_wins = COALESCE(:h2h_surface_p2_wins, h2h_surface_p2_wins),
      h2h_surface_matches = COALESCE(:h2h_surface_matches, h2h_surface_matches),
      updated_at = NOW()
    WHERE match_id = :match_id
    """
)

# =============================================================================
# Canonical resolution
# =============================================================================

async def resolve_ids_for_row(conn: AsyncConnection, row: dict, event: dict) -> dict:
    async def _player_exists(pid: int) -> bool:
        res = await conn.execute(
            text("SELECT 1 FROM tennis_players WHERE id = :id LIMIT 1"),
            {"id": pid},
        )
        return res.first() is not None

    async def _player_id_from_source(external_id: Optional[str]) -> Optional[int]:
        """
        Robust sofascore mapping lookup:
        - trims whitespace
        - validates numeric
        - avoids ':sid::bigint' syntax that can break param parsing
        """
        if not external_id:
            return None
        try:
            sid = int(external_id)
        except Exception:
            return None

        stmt = text(
            """
            SELECT player_id
            FROM tennis_player_sources
            WHERE source = 'sofascore'
              AND NULLIF(btrim(source_player_id), '') ~ '^[0-9]+$'
              AND NULLIF(btrim(source_player_id), '')::bigint = :sid
            LIMIT 1
            """
        ).bindparams(bindparam("sid", type_=BigInteger))

        res = await conn.execute(stmt, {"sid": sid})
        r = res.first()
        return int(r[0]) if r else None

    async def _player_id_from_name(name: str, expected_gender: Optional[str]) -> Optional[int]:
        if not name or not expected_gender:
            return None
        res = await conn.execute(
            text(
                """
                SELECT id
                FROM tennis_players
                WHERE gender = :g
                  AND name_fold = name_fold(:n)
                LIMIT 1
                """
            ),
            {"g": expected_gender, "n": name},
        )
        r = res.first()
        return int(r[0]) if r else None

    p1_name = row.get("p1_name") or ""
    p2_name = row.get("p2_name") or ""

    # Team/country ties: no canonical ids
    if looks_like_country(p1_name) and looks_like_country(p2_name):
        row["p1_canonical_id"] = None
        row["p2_canonical_id"] = None
        return row

    expected_gender: Optional[str] = None
    if (row.get("tour") or "").upper() == "ATP":
        expected_gender = "M"
    elif (row.get("tour") or "").upper() == "WTA":
        expected_gender = "F"

    players = get_players_from_event(event)
    p1_external_id: Optional[str] = None
    p2_external_id: Optional[str] = None
    if players:
        home, away = players
        cid1 = get_competitor_id(home)
        cid2 = get_competitor_id(away)
        p1_external_id = str(cid1) if cid1 is not None else None
        p2_external_id = str(cid2) if cid2 is not None else None

    # p1
    pid = await _player_id_from_source(p1_external_id)
    if pid and await _player_exists(pid):
        row["p1_canonical_id"] = pid
    else:
        r1 = await resolve_player_id(
            conn,
            source="sofascore",
            alias_name=p1_name,
            external_id=p1_external_id,
            auto_create_pending=True,
        )
        if r1.player_id and (not r1.is_pending) and await _player_exists(int(r1.player_id)):
            row["p1_canonical_id"] = int(r1.player_id)
        else:
            pid2 = await _player_id_from_name(p1_name, expected_gender)
            row["p1_canonical_id"] = pid2 if (pid2 and await _player_exists(pid2)) else None

    # p2
    pid = await _player_id_from_source(p2_external_id)
    if pid and await _player_exists(pid):
        row["p2_canonical_id"] = pid
    else:
        r2 = await resolve_player_id(
            conn,
            source="sofascore",
            alias_name=p2_name,
            external_id=p2_external_id,
            auto_create_pending=True,
        )
        if r2.player_id and (not r2.is_pending) and await _player_exists(int(r2.player_id)):
            row["p2_canonical_id"] = int(r2.player_id)
        else:
            pid2 = await _player_id_from_name(p2_name, expected_gender)
            row["p2_canonical_id"] = pid2 if (pid2 and await _player_exists(pid2)) else None

    return row


def parse_date_env(key: str) -> Optional[dt.date]:
    v = os.getenv(key)
    if not v:
        return None
    return dt.date.fromisoformat(v.strip())


async def _auto_backfill_canonicals(conn: AsyncConnection, start: dt.date, end: dt.date) -> None:
    if not parse_bool_env("AUTO_BACKFILL_CANONICAL", default=True):
        return
    await conn.execute(BACKFILL_CANON_P1_SQL, {"start": start, "end": end})
    await conn.execute(BACKFILL_CANON_P2_SQL, {"start": start, "end": end})


async def _auto_backfill_ta_and_elo(conn: AsyncConnection, start: dt.date, end: dt.date) -> None:
    if not parse_bool_env("AUTO_FILL_ELO", default=True):
        return

    # 1a) create TA mappings where unique by normalized name+tour on latest snapshot
    await conn.execute(INSERT_TA_MAPS_SQL, {"start": start, "end": end})

    # 1b) fuzzy token-based TA matching for players the exact matcher missed
    #     (handles accents, NBSP, name reordering, hyphens, suffixes)
    try:
        await run_fuzzy_ta_fix(conn, start, end, quiet=True)
    except Exception as exc:
        logger.warning("fix_ta_elo fuzzy pass failed (non-fatal): %s", exc)

    # 2) copy TA ids onto matches
    await conn.execute(BACKFILL_MATCH_TA_P1_SQL, {"start": start, "end": end})
    await conn.execute(BACKFILL_MATCH_TA_P2_SQL, {"start": start, "end": end})

    # 3) fill Elo from snapshots (latest <= match_date)
    await conn.execute(FILL_MATCH_ELO_P1_SQL, {"start": start, "end": end})
    await conn.execute(FILL_MATCH_ELO_P2_SQL, {"start": start, "end": end})

    # 4) optional fallback Elo to eliminate blanks
    if parse_bool_env("ELO_USE_FALLBACK", default=False):
        fallback = float(os.getenv("ELO_FALLBACK", "1500"))
        await conn.execute(FILL_FALLBACK_ELO_SQL, {"start": start, "end": end, "fallback": fallback})


async def _auto_backfill_h2h_from_raw(conn: AsyncConnection, start: dt.date, end: dt.date) -> None:
    if not parse_bool_env("AUTO_BACKFILL_H2H", default=True):
        return

    res = await conn.execute(H2H_BACKFILL_SELECT_SQL, {"start": start, "end": end})
    rows = res.mappings().all()
    if not rows:
        return

    for r in rows:
        payload = r.get("payload") or {}
        if not isinstance(payload, dict):
            continue

        h2h_summary = payload.get("summary") or {}
        h2h_events_payload = payload.get("events") or {}
        surface_family = _surface_family(r.get("surface"))

        p1_id = r.get("p1_id")
        p2_id = r.get("p2_id")

        # Overall from summary if present
        team_duel = safe_get(h2h_summary, "teamDuel", default=None) or {}
        home_wins = team_duel.get("homeWins")
        away_wins = team_duel.get("awayWins")
        draws = team_duel.get("draws", 0)

        h2h_p1_wins = h2h_p2_wins = h2h_total = None
        if isinstance(home_wins, int) and isinstance(away_wins, int) and isinstance(draws, int):
            h2h_p1_wins = home_wins
            h2h_p2_wins = away_wins
            h2h_total = home_wins + away_wins + draws

        comp = _compute_h2h_from_events(h2h_events_payload, p1_id, p2_id, surface_family)
        if comp[2] is not None and h2h_total is None:
            h2h_p1_wins, h2h_p2_wins, h2h_total = comp[0], comp[1], comp[2]

        await conn.execute(
            H2H_BACKFILL_UPDATE_SQL,
            {
                "match_id": r["match_id"],
                "h2h_p1_wins": h2h_p1_wins,
                "h2h_p2_wins": h2h_p2_wins,
                "h2h_total_matches": h2h_total,
                "h2h_surface_p1_wins": comp[3] if comp[2] is not None else None,
                "h2h_surface_p2_wins": comp[4] if comp[2] is not None else None,
                "h2h_surface_matches": comp[5] if comp[2] is not None else None,
            },
        )


# =============================================================================
# Ingest loop
# =============================================================================

async def ingest_day(conn: AsyncConnection, day: dt.date, *, context=None) -> int:
    logger.info("Fetching SofaScore matches for %s", day.isoformat())

    tour_filter = get_tour_filter_set()
    include_wta125 = parse_bool_env("INCLUDE_WTA125", default=False)
    raw_filtered_only = parse_bool_env("RAW_FILTERED_ONLY", default=False)

    ingest_odds = parse_bool_env("INGEST_ODDS", default=False)
    odds_sleep = parse_float_env("ODDS_SLEEP_SECONDS", 0.25)
    ingest_h2h = parse_bool_env("INGEST_H2H", default=False)
    ingest_h2h_events = parse_bool_env("INGEST_H2H_EVENTS", default=True)
    h2h_sleep = parse_float_env("H2H_SLEEP_SECONDS", 0.2)

    events = await fetch_matches_for_date(day, context=context)
    logger.info("SofaScore returned %s total events for %s", len(events), day.isoformat())

    kept = skipped = saved = 0

    for ev in events:
        if is_doubles_event(ev):
            skipped += 1
            continue

        allowed = is_allowed_by_filter(ev, tour_filter, include_wta125)

        # Store raw (all), unless RAW_FILTERED_ONLY=1
        if not raw_filtered_only:
            ev_id = ev.get("id")
            if ev_id:
                await conn.execute(UPSERT_EVENT_RAW_SQL, {"event_id": int(ev_id), "payload": ev})

        if not allowed:
            skipped += 1
            continue

        # Store raw for kept matches if RAW_FILTERED_ONLY=1
        if raw_filtered_only:
            ev_id = ev.get("id")
            if ev_id:
                await conn.execute(UPSERT_EVENT_RAW_SQL, {"event_id": int(ev_id), "payload": ev})

        row = extract_match_row(day, ev)
        if not row:
            skipped += 1
            continue

        if not row.get("match_key") or not row.get("p1_name") or not row.get("p2_name") or not row.get("tour"):
            skipped += 1
            continue

        kept += 1

        row = await resolve_ids_for_row(conn, row, ev)

        # Odds (best-effort) — never break ingest if missing
        if ingest_odds:
            ev_id = int(row["sofascore_event_id"])
            tour_label = row.get("tour", "UNKNOWN")
            odds_payload = await fetch_odds_for_event(ev_id, context=context)

            if odds_payload:
                await conn.execute(UPSERT_ODDS_RAW_SQL, {"event_id": ev_id, "payload": odds_payload})
                o1, o2 = extract_best_effort_american_odds(odds_payload, event_id=ev_id, tour=tour_label)
                total_line, total_over, total_under = extract_total_games_market(odds_payload)
                spread_p1_line, spread_p2_line, spread_p1_odds, spread_p2_odds = extract_game_spread_market(odds_payload)
                fetched_at = dt.datetime.now(dt.timezone.utc)
                row["p1_odds_american"] = o1
                row["p2_odds_american"] = o2
                row["odds_fetched_at"] = fetched_at
                row["sofascore_p1_odds_american"] = o1
                row["sofascore_p2_odds_american"] = o2
                row["sofascore_odds_fetched_at"] = fetched_at
                row["sofascore_total_games_line"] = total_line
                row["sofascore_total_games_over_american"] = total_over
                row["sofascore_total_games_under_american"] = total_under
                row["sofascore_spread_p1_line"] = spread_p1_line
                row["sofascore_spread_p2_line"] = spread_p2_line
                row["sofascore_spread_p1_odds_american"] = spread_p1_odds
                row["sofascore_spread_p2_odds_american"] = spread_p2_odds

            if odds_sleep > 0:
                await asyncio.sleep(odds_sleep)

        # H2H (best-effort) â€” never break ingest if missing
        if ingest_h2h:
            ev_id = int(row["sofascore_event_id"])
            h2h_summary = await fetch_h2h_summary(ev_id, context=context)

            h2h_events_payload = {}
            if ingest_h2h_events:
                match_code = ev.get("customId")
                if match_code is not None:
                    h2h_events_payload = await fetch_h2h_events(str(match_code), context=context)

            if h2h_summary or h2h_events_payload:
                h2h_payload = {
                    "summary": h2h_summary or None,
                    "events": h2h_events_payload or None,
                    "custom_id": ev.get("customId"),
                }
                await conn.execute(UPSERT_H2H_RAW_SQL, {"event_id": ev_id, "payload": h2h_payload})

                p1_id = row.get("p1_sofascore_player_id")
                p2_id = row.get("p2_sofascore_player_id")
                surface_family = _surface_family(row.get("surface"))

                # Overall H2H from summary if available
                home_wins = safe_get(h2h_summary, "summary", "teamDuel", "homeWins", default=None)
                away_wins = safe_get(h2h_summary, "summary", "teamDuel", "awayWins", default=None)
                draws = safe_get(h2h_summary, "summary", "teamDuel", "draws", default=0)
                if isinstance(home_wins, int) and isinstance(away_wins, int) and isinstance(draws, int):
                    row["h2h_p1_wins"] = home_wins
                    row["h2h_p2_wins"] = away_wins
                    row["h2h_total_matches"] = home_wins + away_wins + draws

                # Compute overall + surface-specific from events list (fallback + surface)
                comp = _compute_h2h_from_events(h2h_events_payload, p1_id, p2_id, surface_family)
                if comp[2] is not None:
                    if row.get("h2h_total_matches") is None:
                        row["h2h_p1_wins"] = comp[0]
                        row["h2h_p2_wins"] = comp[1]
                        row["h2h_total_matches"] = comp[2]
                    row["h2h_surface_p1_wins"] = comp[3]
                    row["h2h_surface_p2_wins"] = comp[4]
                    row["h2h_surface_matches"] = comp[5]

            if h2h_sleep > 0:
                await asyncio.sleep(h2h_sleep)

        await conn.execute(UPSERT_MATCH_SQL, row)
        saved += 1

    logger.info("Kept %s events, skipped %s for %s", kept, skipped, day.isoformat())
    logger.info("Saved %s matches for %s", saved, day.isoformat())
    return saved


async def ingest_window(engine: AsyncEngine) -> None:
    start = parse_date_env("START_DATE")
    end = parse_date_env("END_DATE")

    today = dt.date.today()
    if not start and not end:
        # Rolling default window: yesterday, today, tomorrow.
        days_back = max(0, parse_int_env("DEFAULT_DAYS_BACK", 1))
        days_ahead = max(0, parse_int_env("DEFAULT_DAYS_AHEAD", 1))
        start = today - dt.timedelta(days=days_back)
        end = today + dt.timedelta(days=days_ahead)

    if start and not end:
        end = start
    if end and not start:
        start = end

    assert start is not None and end is not None

    if end < start:
        start, end = end, start

    days: List[dt.date] = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += dt.timedelta(days=1)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(extra_http_headers=BROWSER_HEADERS)

        # Warm-up to reduce 403 issues
        page = await context.new_page()
        await page.goto("https://www.sofascore.com/", wait_until="domcontentloaded", timeout=60000)
        await page.close()

        async with engine.begin() as conn:
            await ensure_tables(conn)

            for d in days:
                await ingest_day(conn, d, context=context)

                # After each day: fill canonicals + ta ids + elo
                await _auto_backfill_canonicals(conn, d, d)
                await _auto_backfill_ta_and_elo(conn, d, d)
                await _auto_backfill_h2h_from_raw(conn, d, d)

            # Final sweep for the full window
            await _auto_backfill_canonicals(conn, start, end)
            await _auto_backfill_ta_and_elo(conn, start, end)
            await _auto_backfill_h2h_from_raw(conn, start, end)

        await browser.close()


# =============================================================================
# Engine / SSL handling
# =============================================================================

def normalize_asyncpg_url_and_ssl(db_url: str) -> tuple[str, dict]:
    parts = urlsplit(db_url)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))

    sslmode = (q.pop("sslmode", None) or "").strip().lower()
    ssl_param = (q.pop("ssl", None) or "").strip().lower()

    cleaned_query = urlencode(q, doseq=True)
    cleaned_url = urlunsplit((parts.scheme, parts.netloc, parts.path, cleaned_query, parts.fragment))

    connect_args: dict = {}

    if sslmode:
        if sslmode == "disable":
            connect_args = {"ssl": False}
        elif sslmode in ("allow", "prefer", "require", "verify-ca", "verify-full"):
            ctx = ssl.create_default_context()
            # "require" in Postgres typically means encryption; verification depends on server certs.
            # If your provider uses standard CA chain, this is fine.
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
            connect_args = {"ssl": ctx}

    if not connect_args and ssl_param:
        if ssl_param in {"0", "false", "off"}:
            connect_args = {"ssl": False}
        else:
            ctx = ssl.create_default_context()
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
            connect_args = {"ssl": ctx}

    return cleaned_url, connect_args


def get_engine() -> AsyncEngine:
    db_url = os.getenv("DATABASE_URL") or os.getenv("DATABASE_URL_ASYNC")
    if not db_url:
        raise RuntimeError("Missing DATABASE_URL (expected postgresql+asyncpg://...)")

    normalized_url, connect_args = normalize_asyncpg_url_and_ssl(db_url) 
    return create_async_engine(
        normalized_url,
        future=True,
        echo=False,
        connect_args=connect_args,
        pool_pre_ping=True,
    )


if __name__ == "__main__":
    engine = get_engine()
    asyncio.run(ingest_window(engine))
