from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
from typing import Any, Dict, List

import asyncpg
from dotenv import load_dotenv
from playwright.async_api import async_playwright

from app.ingest.tennis.sofascore_matchups import (
    BROWSER_HEADERS,
    extract_best_effort_american_odds,
    extract_game_spread_market,
    extract_total_games_market,
    fetch_odds_for_event,
)

load_dotenv()

DONE_STATUSES = ("finished", "completed", "ended", "cancelled", "canceled", "postponed", "walkover")


async def _load_candidates(conn: asyncpg.Connection, hours_ahead: int, limit: int) -> List[dict]:
    return [
        dict(r)
        for r in await conn.fetch(
            """
            SELECT
                match_id::text AS match_id,
                sofascore_event_id,
                match_start_utc,
                tour,
                p1_name,
                p2_name
            FROM tennis_matches
            WHERE sofascore_event_id IS NOT NULL
              AND match_start_utc IS NOT NULL
              AND match_start_utc >= NOW()
              AND match_start_utc <= (NOW() + (($1::text || ' hours')::interval))
              AND COALESCE(lower(status), '') <> ALL($2::text[])
            ORDER BY match_start_utc ASC
            LIMIT $3
            """,
            str(int(hours_ahead)),
            list(DONE_STATUSES),
            int(limit),
        )
    ]


async def _upsert_odds_raw(conn: asyncpg.Connection, event_id: int, payload: dict):
    await conn.execute(
        """
        INSERT INTO sofascore_event_odds_raw (event_id, payload, fetched_at)
        VALUES ($1, $2::jsonb, NOW())
        ON CONFLICT (event_id)
        DO UPDATE SET payload = EXCLUDED.payload, fetched_at = EXCLUDED.fetched_at
        """,
        int(event_id),
        payload,
    )


async def _update_match_odds(
    conn: asyncpg.Connection,
    *,
    event_id: int,
    p1_odds: int | None,
    p2_odds: int | None,
    total_line: float | None,
    total_over: int | None,
    total_under: int | None,
    spread_p1_line: float | None,
    spread_p2_line: float | None,
    spread_p1_odds: int | None,
    spread_p2_odds: int | None,
):
    await conn.execute(
        """
        UPDATE tennis_matches
        SET
            p1_odds_american = COALESCE($2, p1_odds_american),
            p2_odds_american = COALESCE($3, p2_odds_american),
            odds_fetched_at = NOW(),
            sofascore_p1_odds_american = COALESCE($2, sofascore_p1_odds_american),
            sofascore_p2_odds_american = COALESCE($3, sofascore_p2_odds_american),
            sofascore_odds_fetched_at = NOW(),
            sofascore_total_games_line = COALESCE($4, sofascore_total_games_line),
            sofascore_total_games_over_american = COALESCE($5, sofascore_total_games_over_american),
            sofascore_total_games_under_american = COALESCE($6, sofascore_total_games_under_american),
            sofascore_spread_p1_line = COALESCE($7, sofascore_spread_p1_line),
            sofascore_spread_p2_line = COALESCE($8, sofascore_spread_p2_line),
            sofascore_spread_p1_odds_american = COALESCE($9, sofascore_spread_p1_odds_american),
            sofascore_spread_p2_odds_american = COALESCE($10, sofascore_spread_p2_odds_american)
        WHERE sofascore_event_id = $1
        """,
        int(event_id),
        p1_odds,
        p2_odds,
        total_line,
        total_over,
        total_under,
        spread_p1_line,
        spread_p2_line,
        spread_p1_odds,
        spread_p2_odds,
    )


async def main(*, hours_ahead: int, sleep_sec: float, limit: int):
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    if database_url.startswith("postgresql+asyncpg://"):
        database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

    conn = await asyncpg.connect(database_url)
    try:
        candidates = await _load_candidates(conn, hours_ahead=hours_ahead, limit=limit)
        print(f"Candidates in next {hours_ahead}h: {len(candidates)}")
        if not candidates:
            return

        fetched = 0
        updated = 0
        failed = 0

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(extra_http_headers=BROWSER_HEADERS)
            page = await context.new_page()
            await page.goto("https://www.sofascore.com/", wait_until="domcontentloaded", timeout=60000)
            await page.close()

            for row in candidates:
                event_id = int(row["sofascore_event_id"])
                try:
                    payload = await fetch_odds_for_event(event_id, context=context)
                    fetched += 1
                    if not payload:
                        if sleep_sec > 0:
                            await asyncio.sleep(sleep_sec)
                        continue

                    await _upsert_odds_raw(conn, event_id, payload)

                    p1_odds, p2_odds = extract_best_effort_american_odds(payload, event_id=event_id, tour=row.get("tour"))
                    total_line, total_over, total_under = extract_total_games_market(payload)
                    spread_p1_line, spread_p2_line, spread_p1_odds, spread_p2_odds = extract_game_spread_market(payload)

                    await _update_match_odds(
                        conn,
                        event_id=event_id,
                        p1_odds=p1_odds,
                        p2_odds=p2_odds,
                        total_line=total_line,
                        total_over=total_over,
                        total_under=total_under,
                        spread_p1_line=spread_p1_line,
                        spread_p2_line=spread_p2_line,
                        spread_p1_odds=spread_p1_odds,
                        spread_p2_odds=spread_p2_odds,
                    )
                    updated += 1
                except Exception as e:
                    failed += 1
                    print(f"Failed event_id={event_id}: {e}")
                if sleep_sec > 0:
                    await asyncio.sleep(sleep_sec)

            await context.close()
            await browser.close()

        print(f"Done. fetched={fetched} updated={updated} failed={failed}")
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh SofaScore odds only for near-start tennis matches.")
    parser.add_argument("--hours-ahead", type=int, default=4, help="Refresh matches starting in next N hours.")
    parser.add_argument("--sleep-sec", type=float, default=0.25, help="Delay between event odds fetches.")
    parser.add_argument("--limit", type=int, default=300, help="Maximum matches to refresh in one run.")
    args = parser.parse_args()

    asyncio.run(
        main(
            hours_ahead=max(1, int(args.hours_ahead)),
            sleep_sec=max(0.0, float(args.sleep_sec)),
            limit=max(1, int(args.limit)),
        )
    )
