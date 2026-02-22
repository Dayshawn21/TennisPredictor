from __future__ import annotations

"""
Backfill SofaScore tennis event statistics into normalized per-player rows.

What this script does:
- Reads candidate finished ATP/WTA events from tennis_matches.
- Fetches SofaScore event stats JSON using existing Playwright helper.
- Stores raw payload in sofascore_event_stats_raw (JSONB).
- Parses core serve stats for both players and upserts into tennis_match_player_stats.

Usage:
  py -3 api/app/ingest/tennis/sofascore_event_stats_backfill.py --start-date 2026-01-01
  py -3 api/app/ingest/tennis/sofascore_event_stats_backfill.py --start-date 2026-01-01 --end-date 2026-02-21 --limit 500
  py -3 api/app/ingest/tennis/sofascore_event_stats_backfill.py --start-date 2026-01-01 --only-missing
"""

import argparse
import asyncio
import datetime as dt
import logging
import re
from typing import Any, Dict, Iterable, Optional, Tuple

from playwright.async_api import async_playwright
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import bindparam

from app.db_session import engine
from app.ingest.tennis.sofascore_matchups import BROWSER_HEADERS, get_json

logger = logging.getLogger("sofascore_event_stats_backfill")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

ENSURE_SQL = [
    """
    CREATE TABLE IF NOT EXISTS sofascore_event_stats_raw (
      event_id BIGINT PRIMARY KEY,
      payload JSONB NOT NULL,
      fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_sofascore_event_stats_raw_fetched_at ON sofascore_event_stats_raw (fetched_at)",
    """
    CREATE TABLE IF NOT EXISTS tennis_match_player_stats (
      sofascore_event_id BIGINT NOT NULL,
      player_id BIGINT NOT NULL,
      opp_player_id BIGINT NOT NULL,
      ace INTEGER NULL,
      df INTEGER NULL,
      svpt INTEGER NULL,
      first_in INTEGER NULL,
      first_won INTEGER NULL,
      second_won INTEGER NULL,
      sv_gms INTEGER NULL,
      bp_saved INTEGER NULL,
      bp_faced INTEGER NULL,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      PRIMARY KEY (sofascore_event_id, player_id)
    )
    """,
    "ALTER TABLE tennis_match_player_stats ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
    "CREATE INDEX IF NOT EXISTS ix_tmps_sofa_event ON tennis_match_player_stats (sofascore_event_id)",
]

UPSERT_RAW_SQL = (
    text(
        """
        INSERT INTO sofascore_event_stats_raw (event_id, payload, fetched_at)
        VALUES (:event_id, :payload, NOW())
        ON CONFLICT (event_id)
        DO UPDATE SET payload = EXCLUDED.payload, fetched_at = NOW()
        """
    ).bindparams(bindparam("payload", type_=JSONB))
)

UPSERT_STATS_SQL = text(
    """
    INSERT INTO tennis_match_player_stats (
      sofascore_event_id, player_id, opp_player_id,
      ace, df, svpt, first_in, first_won, second_won, sv_gms, bp_saved, bp_faced,
      updated_at
    )
    VALUES (
      :sofascore_event_id, :player_id, :opp_player_id,
      :ace, :df, :svpt, :first_in, :first_won, :second_won, :sv_gms, :bp_saved, :bp_faced,
      NOW()
    )
    ON CONFLICT (sofascore_event_id, player_id)
    DO UPDATE SET
      opp_player_id = EXCLUDED.opp_player_id,
      ace = COALESCE(EXCLUDED.ace, tennis_match_player_stats.ace),
      df = COALESCE(EXCLUDED.df, tennis_match_player_stats.df),
      svpt = COALESCE(EXCLUDED.svpt, tennis_match_player_stats.svpt),
      first_in = COALESCE(EXCLUDED.first_in, tennis_match_player_stats.first_in),
      first_won = COALESCE(EXCLUDED.first_won, tennis_match_player_stats.first_won),
      second_won = COALESCE(EXCLUDED.second_won, tennis_match_player_stats.second_won),
      sv_gms = COALESCE(EXCLUDED.sv_gms, tennis_match_player_stats.sv_gms),
      bp_saved = COALESCE(EXCLUDED.bp_saved, tennis_match_player_stats.bp_saved),
      bp_faced = COALESCE(EXCLUDED.bp_faced, tennis_match_player_stats.bp_faced),
      updated_at = NOW()
    """
)

FETCH_CANDIDATES_SQL = text(
    """
    SELECT
      m.sofascore_event_id,
      m.p1_sofascore_player_id,
      m.p2_sofascore_player_id,
      m.match_date,
      upper(m.tour) AS tour,
      lower(coalesce(m.status, '')) AS status
    FROM tennis_matches m
    WHERE m.sofascore_event_id IS NOT NULL
      AND m.match_date BETWEEN :start_date AND :end_date
      AND upper(m.tour) IN ('ATP', 'WTA')
      AND lower(coalesce(m.status, '')) IN ('finished','completed','ended','final')
      AND m.p1_sofascore_player_id IS NOT NULL
      AND m.p2_sofascore_player_id IS NOT NULL
      AND (
        :only_missing = false OR
        NOT EXISTS (
          SELECT 1
          FROM tennis_match_player_stats s
          WHERE s.sofascore_event_id = m.sofascore_event_id
        )
      )
    ORDER BY m.match_date, m.sofascore_event_id
    LIMIT :limit_rows
    """
)

STAT_ALIASES = {
    "ace": {"aces", "ace"},
    "df": {"double faults", "double fault", "dfs"},
    "svpt": {
        "service points played",
        "service points",
        "total service points",
        "serve points played",
        "total service points won",
    },
    "first_in": {"first serve in", "1st serve in", "1st serve", "first serve"},
    "first_won": {
        "first serve points won",
        "1st serve points won",
        "1st serve won",
        "first serve won",
        "first serve points",
    },
    "second_won": {
        "second serve points won",
        "2nd serve points won",
        "2nd serve won",
        "second serve won",
        "second serve points",
    },
    "sv_gms": {"service games played", "service games", "service games won"},
    "bp_saved": {"break points saved", "bp saved"},
    "bp_faced": {"break points faced", "bp faced"},
}


def _norm_name(v: str) -> str:
    s = (v or "").strip().lower()
    s = s.replace("%", "")
    s = re.sub(r"\s+", " ", s)
    return s


def _to_number(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s or s == "-":
        return None
    s = s.replace(",", "")
    if s.endswith("%"):
        s = s[:-1]
    if "/" in s:
        left, _, _ = s.partition("/")
        s = left.strip()
    try:
        return float(s)
    except Exception:
        return None


def _maybe_int(v: Optional[float]) -> Optional[int]:
    if v is None:
        return None
    return int(round(v))


def _walk_objects(obj: Any) -> Iterable[dict]:
    if isinstance(obj, dict):
        yield obj
        for vv in obj.values():
            yield from _walk_objects(vv)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_objects(item)


def _extract_from_all_period(payload: dict) -> Dict[str, Dict[str, Optional[int]]]:
    """Extract core tennis stats from the ALL period using stable SofaScore keys."""
    out: Dict[str, Dict[str, Optional[int]]] = {}
    periods = payload.get("statistics")
    if not isinstance(periods, list):
        return out

    all_block = None
    for block in periods:
        if isinstance(block, dict) and str(block.get("period", "")).upper() == "ALL":
            all_block = block
            break
    if not isinstance(all_block, dict):
        return out

    groups = all_block.get("groups")
    if not isinstance(groups, list):
        return out

    for g in groups:
        if not isinstance(g, dict):
            continue
        items = g.get("statisticsItems")
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            key = _norm_name(str(it.get("key") or it.get("name") or ""))
            if not key:
                continue
            out[key] = {
                "home_num": _maybe_int(_to_number(it.get("homeValue") if it.get("homeValue") is not None else it.get("home"))),
                "away_num": _maybe_int(_to_number(it.get("awayValue") if it.get("awayValue") is not None else it.get("away"))),
                "home_den": _maybe_int(_to_number(it.get("homeTotal"))),
                "away_den": _maybe_int(_to_number(it.get("awayTotal"))),
            }

    return out


def _first_scalar_from_dict(d: dict) -> Optional[Any]:
    for k in ("value", "displayValue", "raw", "stat", "count", "numerator", "wins", "current"):
        if k in d and d.get(k) is not None:
            return d.get(k)
    return None


def _extract_side_raw(d: dict, side: str) -> Optional[Any]:
    keys = (
        (f"{side}", f"{side}Value", f"{side}Team", f"value{side.title()}", f"{side}TeamValue")
        if side in {"home", "away"}
        else ()
    )
    for k in keys:
        if k in d:
            v = d.get(k)
            if isinstance(v, dict):
                vv = _first_scalar_from_dict(v)
                if vv is not None:
                    return vv
            elif v is not None:
                return v
    return None


def _split_num_den(v: Any) -> Tuple[Optional[float], Optional[float]]:
    if v is None:
        return None, None
    if isinstance(v, dict):
        num = _to_number(v.get("numerator"))
        den = _to_number(v.get("denominator") or v.get("total"))
        if num is not None or den is not None:
            return num, den
        v = _first_scalar_from_dict(v)
        if v is None:
            return None, None

    s = str(v).strip()
    if not s or s == "-":
        return None, None

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except Exception:
            return None, None

    num = _to_number(s)
    return num, None


def _extract_named_home_away(payload: dict) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}

    for d in _walk_objects(payload):
        raw_name = d.get("name") or d.get("title") or d.get("key") or d.get("statName")
        if not raw_name:
            t = d.get("statisticsType")
            if isinstance(t, dict):
                raw_name = t.get("name") or t.get("slug")
            elif isinstance(t, str):
                raw_name = t
        if not raw_name:
            continue

        n = _norm_name(str(raw_name))

        home_raw = _extract_side_raw(d, "home")
        away_raw = _extract_side_raw(d, "away")
        home_num, home_den = _split_num_den(home_raw)
        away_num, away_den = _split_num_den(away_raw)

        if home_num is None and away_num is None and home_den is None and away_den is None:
            continue

        out[n] = {
            "home_num": home_num,
            "away_num": away_num,
            "home_den": home_den,
            "away_den": away_den,
        }

    return out


def _pick_metric(stats: Dict[str, Dict[str, Optional[float]]], key: str, part: str = "num") -> Tuple[Optional[int], Optional[int]]:
    aliases = STAT_ALIASES[key]
    for name, vals in stats.items():
        if name in aliases:
            if part == "den":
                return _maybe_int(vals.get("home_den")), _maybe_int(vals.get("away_den"))
            return _maybe_int(vals.get("home_num")), _maybe_int(vals.get("away_num"))
    return None, None


async def _fetch_event_stats(event_id: int, *, context) -> dict:
    paths = [
        f"/event/{event_id}/statistics",
        f"/event/{event_id}/statistics/0",
    ]
    last_err: Optional[Exception] = None
    for p in paths:
        try:
            return await get_json(p, context=context)
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    return {}


async def ensure_tables() -> None:
    async with engine.begin() as conn:
        for stmt in ENSURE_SQL:
            await conn.execute(text(stmt))


async def load_candidates(start_date: dt.date, end_date: dt.date, only_missing: bool, limit_rows: int) -> list[dict]:
    async with engine.begin() as conn:
        res = await conn.execute(
            FETCH_CANDIDATES_SQL,
            {
                "start_date": start_date,
                "end_date": end_date,
                "only_missing": bool(only_missing),
                "limit_rows": int(limit_rows),
            },
        )
        return [dict(r._mapping) for r in res.fetchall()]


async def backfill(start_date: dt.date, end_date: dt.date, only_missing: bool, limit_rows: int, dry_run: bool) -> dict:
    await ensure_tables()
    candidates = await load_candidates(start_date, end_date, only_missing, limit_rows)
    logger.info("candidates=%d start=%s end=%s only_missing=%s", len(candidates), start_date, end_date, only_missing)

    counts = {
        "candidates": len(candidates),
        "raw_saved": 0,
        "parsed_events": 0,
        "upserts": 0,
        "parse_empty": 0,
        "fetch_errors": 0,
    }

    if not candidates:
        return counts

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(extra_http_headers=BROWSER_HEADERS)
        page = await context.new_page()
        await page.goto("https://www.sofascore.com/", wait_until="domcontentloaded", timeout=60000)
        await page.close()

        async with engine.begin() as conn:
            for row in candidates:
                ev_id = int(row["sofascore_event_id"])
                p1 = int(row["p1_sofascore_player_id"])
                p2 = int(row["p2_sofascore_player_id"])

                try:
                    payload = await _fetch_event_stats(ev_id, context=context)
                except Exception as e:
                    counts["fetch_errors"] += 1
                    logger.warning("event=%s fetch_failed=%s", ev_id, e)
                    continue

                if not dry_run:
                    await conn.execute(UPSERT_RAW_SQL, {"event_id": ev_id, "payload": payload})
                counts["raw_saved"] += 1

                keyed = _extract_from_all_period(payload)
                if keyed:
                    ace_h = keyed.get("aces", {}).get("home_num")
                    ace_a = keyed.get("aces", {}).get("away_num")
                    df_h = keyed.get("doublefaults", {}).get("home_num")
                    df_a = keyed.get("doublefaults", {}).get("away_num")

                    first_serve = keyed.get("firstserveaccuracy", {})
                    first_in_h = first_serve.get("home_num")
                    first_in_a = first_serve.get("away_num")
                    svpt_h = first_serve.get("home_den")
                    svpt_a = first_serve.get("away_den")

                    first_pts = keyed.get("firstservepointsaccuracy", {})
                    first_won_h = first_pts.get("home_num")
                    first_won_a = first_pts.get("away_num")

                    second_pts = keyed.get("secondservepointsaccuracy", {})
                    second_won_h = second_pts.get("home_num")
                    second_won_a = second_pts.get("away_num")

                    sv_games = keyed.get("servicegamestotal", {})
                    sv_gms_h = sv_games.get("home_num")
                    sv_gms_a = sv_games.get("away_num")

                    bp_saved = keyed.get("breakpointssaved", {})
                    bp_saved_h = bp_saved.get("home_num")
                    bp_saved_a = bp_saved.get("away_num")
                    bp_faced_h = bp_saved.get("home_den")
                    bp_faced_a = bp_saved.get("away_den")
                else:
                    named = _extract_named_home_away(payload)
                    if not named:
                        counts["parse_empty"] += 1
                        continue

                    ace_h, ace_a = _pick_metric(named, "ace")
                    df_h, df_a = _pick_metric(named, "df")
                    svpt_h, svpt_a = _pick_metric(named, "svpt")
                    first_in_h, first_in_a = _pick_metric(named, "first_in")
                    first_in_den_h, first_in_den_a = _pick_metric(named, "first_in", part="den")
                    first_won_h, first_won_a = _pick_metric(named, "first_won")
                    second_won_h, second_won_a = _pick_metric(named, "second_won")
                    sv_gms_h, sv_gms_a = _pick_metric(named, "sv_gms")
                    bp_saved_h, bp_saved_a = _pick_metric(named, "bp_saved")
                    bp_faced_h, bp_faced_a = _pick_metric(named, "bp_faced")
                    bp_saved_den_h, bp_saved_den_a = _pick_metric(named, "bp_saved", part="den")

                    # Derived fallbacks from fraction-style values.
                    if svpt_h is None:
                        svpt_h = first_in_den_h
                    if svpt_a is None:
                        svpt_a = first_in_den_a
                    if bp_faced_h is None:
                        bp_faced_h = bp_saved_den_h
                    if bp_faced_a is None:
                        bp_faced_a = bp_saved_den_a

                rows = [
                    {
                        "sofascore_event_id": ev_id,
                        "player_id": p1,
                        "opp_player_id": p2,
                        "ace": ace_h,
                        "df": df_h,
                        "svpt": svpt_h,
                        "first_in": first_in_h,
                        "first_won": first_won_h,
                        "second_won": second_won_h,
                        "sv_gms": sv_gms_h,
                        "bp_saved": bp_saved_h,
                        "bp_faced": bp_faced_h,
                    },
                    {
                        "sofascore_event_id": ev_id,
                        "player_id": p2,
                        "opp_player_id": p1,
                        "ace": ace_a,
                        "df": df_a,
                        "svpt": svpt_a,
                        "first_in": first_in_a,
                        "first_won": first_won_a,
                        "second_won": second_won_a,
                        "sv_gms": sv_gms_a,
                        "bp_saved": bp_saved_a,
                        "bp_faced": bp_faced_a,
                    },
                ]

                parsed_any = any(
                    any(v is not None for k, v in r.items() if k not in {"sofascore_event_id", "player_id", "opp_player_id"})
                    for r in rows
                )
                if not parsed_any:
                    counts["parse_empty"] += 1
                    continue

                if not dry_run:
                    for rr in rows:
                        await conn.execute(UPSERT_STATS_SQL, rr)
                counts["parsed_events"] += 1
                counts["upserts"] += 2

        await browser.close()

    return counts


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s.strip())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date", default=dt.date.today().isoformat(), help="YYYY-MM-DD")
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--only-missing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)

    counts = asyncio.run(
        backfill(
            start_date=start_date,
            end_date=end_date,
            only_missing=bool(args.only_missing),
            limit_rows=int(args.limit),
            dry_run=bool(args.dry_run),
        )
    )
    logger.info("done %s", counts)


if __name__ == "__main__":
    main()

