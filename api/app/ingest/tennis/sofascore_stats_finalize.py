from __future__ import annotations

import os
import json
import asyncio
import logging
import ssl
from typing import Any, Dict, Optional, Tuple, List
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from dotenv import load_dotenv
from playwright.async_api import async_playwright, BrowserContext

from sqlalchemy import text, bindparam
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncConnection
from sqlalchemy.dialects.postgresql import JSONB


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("sofascore_stats")

SOFASCORE_API = "https://www.sofascore.com/api/v1"

REQUEST_DELAY_SECONDS = float(os.getenv("STATS_DELAY", "0.9"))
MAX_MATCHES = int(os.getenv("MAX_MATCHES", "200"))

# Tours: "ATP,WTA" by default
ALLOWED_TOURS = [t.strip().upper() for t in (os.getenv("STATS_TOURS", "ATP,WTA") or "").split(",") if t.strip()]

# Optional date filters
START_DATE = (os.getenv("START_DATE") or "").strip() or None  # YYYY-MM-DD or None
END_DATE = (os.getenv("END_DATE") or "").strip() or None      # YYYY-MM-DD or None

BROWSER_HEADERS = {
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# ----------------------------
# DB URL normalization (sslmode -> connect_args)
# ----------------------------
def normalize_asyncpg_url_and_ssl(db_url: str) -> tuple[str, dict]:
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

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
        else:
            ctx = ssl.create_default_context()
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
    db_url = os.getenv("DATABASE_URL_ASYNC") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Missing DATABASE_URL")

    normalized_url, connect_args = normalize_asyncpg_url_and_ssl(db_url)
    return create_async_engine(
        normalized_url,
        future=True,
        echo=False,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

# ----------------------------
# SQL
# ----------------------------
# IMPORTANT: use = ANY(:tours) with an "expanding" bindparam so Postgres gets the right array
FETCH_MATCHES_SQL = text("""
SELECT
  m.id AS match_id,
  msm.source_match_id AS event_id,
  m.match_date
FROM matches m
JOIN match_source_map msm
  ON msm.canonical_match_id = m.id
WHERE msm.source = 'sofascore'
  AND m.tour IN :tours
  AND m.status = 'finished'
  AND (
    CAST(:start_date AS date) IS NULL OR m.match_date >= CAST(:start_date AS date)
  )
  AND (
    CAST(:end_date AS date) IS NULL OR m.match_date <= CAST(:end_date AS date)
  )
  AND NOT EXISTS (
    SELECT 1
    FROM match_stats s
    WHERE s.match_id = m.id
      AND s.stats_complete = TRUE
  )
ORDER BY m.match_date DESC
LIMIT :limit;
""").bindparams(
    bindparam("tours", expanding=True),
)

UPSERT_STATS_SQL = (
    text(
        """
        INSERT INTO match_stats (
          match_id,
          stats_source,
          source_event_id,
          stats_complete,
          pulled_at,
          raw_json,

          w_ace, w_df, w_svpt, w_1stin, w_1stwon, w_2ndwon,
          w_svgms, w_bpsaved, w_bpfaced,

          l_ace, l_df, l_svpt, l_1stin, l_1stwon, l_2ndwon,
          l_svgms, l_bpsaved, l_bpfaced
        )
        VALUES (
          :match_id,
          'sofascore',
          :event_id,
          :stats_complete,
          NOW(),
          :raw_json,

          :w_ace, :w_df, :w_svpt, :w_1stin, :w_1stwon, :w_2ndwon,
          :w_svgms, :w_bpsaved, :w_bpfaced,

          :l_ace, :l_df, :l_svpt, :l_1stin, :l_1stwon, :l_2ndwon,
          :l_svgms, :l_bpsaved, :l_bpfaced
        )
        ON CONFLICT (match_id) DO UPDATE
        SET
          pulled_at = NOW(),
          raw_json = EXCLUDED.raw_json,
          stats_source = CASE WHEN match_stats.stats_complete THEN match_stats.stats_source ELSE EXCLUDED.stats_source END,
          source_event_id = CASE WHEN match_stats.stats_complete THEN match_stats.source_event_id ELSE EXCLUDED.source_event_id END,
          stats_complete = CASE WHEN match_stats.stats_complete THEN TRUE ELSE EXCLUDED.stats_complete END,

          w_ace     = CASE WHEN match_stats.stats_complete THEN match_stats.w_ace     ELSE EXCLUDED.w_ace     END,
          w_df      = CASE WHEN match_stats.stats_complete THEN match_stats.w_df      ELSE EXCLUDED.w_df      END,
          w_svpt    = CASE WHEN match_stats.stats_complete THEN match_stats.w_svpt    ELSE EXCLUDED.w_svpt    END,
          w_1stin   = CASE WHEN match_stats.stats_complete THEN match_stats.w_1stin   ELSE EXCLUDED.w_1stin   END,
          w_1stwon  = CASE WHEN match_stats.stats_complete THEN match_stats.w_1stwon  ELSE EXCLUDED.w_1stwon  END,
          w_2ndwon  = CASE WHEN match_stats.stats_complete THEN match_stats.w_2ndwon  ELSE EXCLUDED.w_2ndwon  END,
          w_svgms   = CASE WHEN match_stats.stats_complete THEN match_stats.w_svgms   ELSE EXCLUDED.w_svgms   END,
          w_bpsaved = CASE WHEN match_stats.stats_complete THEN match_stats.w_bpsaved ELSE EXCLUDED.w_bpsaved END,
          w_bpfaced = CASE WHEN match_stats.stats_complete THEN match_stats.w_bpfaced ELSE EXCLUDED.w_bpfaced END,

          l_ace     = CASE WHEN match_stats.stats_complete THEN match_stats.l_ace     ELSE EXCLUDED.l_ace     END,
          l_df      = CASE WHEN match_stats.stats_complete THEN match_stats.l_df      ELSE EXCLUDED.l_df      END,
          l_svpt    = CASE WHEN match_stats.stats_complete THEN match_stats.l_svpt    ELSE EXCLUDED.l_svpt    END,
          l_1stin   = CASE WHEN match_stats.stats_complete THEN match_stats.l_1stin   ELSE EXCLUDED.l_1stin   END,
          l_1stwon  = CASE WHEN match_stats.stats_complete THEN match_stats.l_1stwon  ELSE EXCLUDED.l_1stwon  END,
          l_2ndwon  = CASE WHEN match_stats.stats_complete THEN match_stats.l_2ndwon  ELSE EXCLUDED.l_2ndwon  END,
          l_svgms   = CASE WHEN match_stats.stats_complete THEN match_stats.l_svgms   ELSE EXCLUDED.l_svgms   END,
          l_bpsaved = CASE WHEN match_stats.stats_complete THEN match_stats.l_bpsaved ELSE EXCLUDED.l_bpsaved END,
          l_bpfaced = CASE WHEN match_stats.stats_complete THEN match_stats.l_bpfaced ELSE EXCLUDED.l_bpfaced END
        """
    ).bindparams(bindparam("raw_json", type_=JSONB))
)

# ----------------------------
# SofaScore fetch via Playwright
# ----------------------------
async def api_get_json(context: BrowserContext, path: str) -> Tuple[Optional[dict], Optional[int]]:
    url = f"{SOFASCORE_API}{path}"
    resp = await context.request.get(url)
    if not resp:
        return None, None
    status = resp.status
    if status != 200:
        return None, status
    try:
        return await resp.json(), status
    except Exception:
        txt = await resp.text()
        logger.warning("Non-JSON response for %s (first 200): %s", url, txt[:200])
        return None, status


async def fetch_event_and_stats(context: BrowserContext, event_id: int) -> Tuple[Optional[dict], Optional[dict], Optional[int], Optional[int]]:
    event, es = await api_get_json(context, f"/event/{event_id}")
    stats, ss = await api_get_json(context, f"/event/{event_id}/statistics")
    return event, stats, es, ss

# ----------------------------
# Stats parsing
# ----------------------------
def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def extract_stat_map(stats_payload: dict) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    out: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    groups = (stats_payload or {}).get("statistics", {}).get("groups") or []
    for g in groups:
        for item in g.get("statisticsItems") or []:
            name = (item.get("name") or "").strip().lower()
            if not name:
                continue
            out[name] = (_to_int(item.get("home")), _to_int(item.get("away")))
    return out


def pick_stat(m: Dict[str, Tuple[Optional[int], Optional[int]]], *names: str) -> Optional[Tuple[Optional[int], Optional[int]]]:
    for n in names:
        key = n.strip().lower()
        if key in m:
            return m[key]
    return None


def parse_wl_stats(event_payload: dict, stats_payload: dict) -> Optional[dict]:
    if not event_payload or not stats_payload:
        return None

    winner_code = (event_payload.get("event") or {}).get("winnerCode")
    if winner_code not in (1, 2):
        return None

    smap = extract_stat_map(stats_payload)
    if not smap:
        return None

    def get_pair(*possible_names: str) -> Optional[Tuple[Optional[int], Optional[int]]]:
        return pick_stat(smap, *possible_names)

    aces = get_pair("aces")
    dfs = get_pair("double faults", "double fault")
    svpt = get_pair("service points")
    first_in = get_pair("1st serve in", "first serve in")
    first_won = get_pair("1st serve won", "first serve won")
    second_won = get_pair("2nd serve won", "second serve won")
    svgms = get_pair("service games")
    bpsaved = get_pair("break points saved")
    bpfaced = get_pair("break points faced")

    required = [aces, dfs, first_in, first_won, second_won, bpsaved, bpfaced]
    if any(p is None for p in required):
        # store raw_json but do not mark complete yet
        return {
            "stats_complete": False,
            "w_ace": None, "w_df": None, "w_svpt": None, "w_1stin": None, "w_1stwon": None, "w_2ndwon": None,
            "w_svgms": None, "w_bpsaved": None, "w_bpfaced": None,
            "l_ace": None, "l_df": None, "l_svpt": None, "l_1stin": None, "l_1stwon": None, "l_2ndwon": None,
            "l_svgms": None, "l_bpsaved": None, "l_bpfaced": None,
        }

    def winner_loser(pair: Tuple[Optional[int], Optional[int]]) -> Tuple[Optional[int], Optional[int]]:
        home, away = pair
        return (home, away) if winner_code == 1 else (away, home)

    w_ace, l_ace = winner_loser(aces)
    w_df, l_df = winner_loser(dfs)
    w_svpt, l_svpt = winner_loser(svpt) if svpt else (None, None)
    w_1stin, l_1stin = winner_loser(first_in)
    w_1stwon, l_1stwon = winner_loser(first_won)
    w_2ndwon, l_2ndwon = winner_loser(second_won)
    w_svgms, l_svgms = winner_loser(svgms) if svgms else (None, None)
    w_bpsaved, l_bpsaved = winner_loser(bpsaved)
    w_bpfaced, l_bpfaced = winner_loser(bpfaced)

    return {
        "stats_complete": True,
        "w_ace": w_ace, "w_df": w_df, "w_svpt": w_svpt, "w_1stin": w_1stin, "w_1stwon": w_1stwon, "w_2ndwon": w_2ndwon,
        "w_svgms": w_svgms, "w_bpsaved": w_bpsaved, "w_bpfaced": w_bpfaced,
        "l_ace": l_ace, "l_df": l_df, "l_svpt": l_svpt, "l_1stin": l_1stin, "l_1stwon": l_1stwon, "l_2ndwon": l_2ndwon,
        "l_svgms": l_svgms, "l_bpsaved": l_bpsaved, "l_bpfaced": l_bpfaced,
    }

# ----------------------------
# Runner
# ----------------------------
async def fetch_targets(conn: AsyncConnection) -> List[tuple]:
    rows = await conn.execute(
        FETCH_MATCHES_SQL,
        {
            "tours": ALLOWED_TOURS,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "limit": MAX_MATCHES,
        },
    )
    return rows.fetchall()


async def main() -> None:
    engine = get_engine()

    async with engine.begin() as conn:
        targets = await fetch_targets(conn)

    logger.info("Found %s finished matches needing stats (tours=%s start=%s end=%s)",
                len(targets), ALLOWED_TOURS, START_DATE, END_DATE)

    if not targets:
        await engine.dispose()
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(extra_http_headers=BROWSER_HEADERS)

        # Warm cookies
        page = await context.new_page()
        await page.goto("https://www.sofascore.com/", wait_until="domcontentloaded", timeout=60000)
        await page.close()

        async with engine.begin() as conn:
            for (match_id, event_id, match_date) in targets:
                event_id = int(event_id)

                event_payload, stats_payload, es, ss = await fetch_event_and_stats(context, event_id)

                if ss == 403 or es == 403:
                    logger.warning("403 challenge for event %s (event=%s stats=%s)", event_id, es, ss)
                    await asyncio.sleep(2.0)
                    continue

                if not stats_payload:
                    logger.warning("Stats not ready for event %s (event=%s stats=%s)", event_id, es, ss)
                    await asyncio.sleep(REQUEST_DELAY_SECONDS)
                    continue

                parsed = parse_wl_stats(event_payload, stats_payload)
                if not parsed:
                    logger.warning("Could not parse stats for event %s", event_id)
                    await asyncio.sleep(REQUEST_DELAY_SECONDS)
                    continue

                raw_json = {"event": event_payload, "statistics": stats_payload}

                await conn.execute(
                    UPSERT_STATS_SQL,
                    {
                        "match_id": match_id,
                        "event_id": event_id,
                        "raw_json": raw_json,
                        **parsed,
                    },
                )

                logger.info("Saved stats (complete=%s) match_id=%s event_id=%s date=%s",
                            parsed["stats_complete"], match_id, event_id, match_date)

                await asyncio.sleep(REQUEST_DELAY_SECONDS)

        await browser.close()

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
