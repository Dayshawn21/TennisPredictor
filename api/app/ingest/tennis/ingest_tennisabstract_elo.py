#!/usr/bin/env python3
"""
Weekly Tennis Abstract Elo snapshot downloader (ATP + WTA).

What it does
- Downloads the Tennis Abstract Elo pages:
    * https://tennisabstract.com/reports/atp_elo_ratings.html
    * https://tennisabstract.com/reports/wta_elo_ratings.html
- Parses the ratings table into a normalized DataFrame.
- Adds `tour` and `as_of_date` (from the page's "Last update:" line when present).

Outputs (choose one):
- CSV mode (default): writes ATP + WTA CSV files into --out folder
- DB mode (--to-db): upserts ATP + WTA into `tennisabstract_elo_snapshots` via SQLAlchemy async engine

Run examples (from api/):
  CSV:
    python -m app.ingest.tennis.ingest_tennisabstract_elo --out data\\elo

  DB:
    python -m app.ingest.tennis.ingest_tennisabstract_elo --to-db

Optional
- Use --as-of YYYY-MM-DD to override as_of_date.
- Use --no-atp / --no-wta to skip a tour.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import re
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

load_dotenv()

ATP_URL = "https://tennisabstract.com/reports/atp_elo_ratings.html"
WTA_URL = "https://tennisabstract.com/reports/wta_elo_ratings.html"


def _extract_as_of_date(html: str) -> dt.date | None:
    """Find 'Last update: YYYY-MM-DD' in page HTML."""
    m = re.search(r"Last\s+update:\s*(\d{4}-\d{2}-\d{2})", html, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return dt.date.fromisoformat(m.group(1))
    except ValueError:
        return None


def _snake(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def _pick_leaderboard_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Tennis Abstract pages often contain small 'tables' for headers/notes.
    Pick the one that looks like the Elo leaderboard by checking for
    'player' and 'elo' in the column names. Fallback: widest table.
    """
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if any("player" in c for c in cols) and any("elo" in c for c in cols):
            return t.copy()

    # Fallback: widest table is usually the leaderboard
    return max(tables, key=lambda x: x.shape[1]).copy()


def fetch_table(url: str, tour: str, as_of_override: dt.date | None = None) -> tuple[pd.DataFrame, dt.date]:
    """Download URL, parse tables via pandas.read_html, normalize columns, and add metadata."""
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    html = r.text

    as_of = as_of_override or _extract_as_of_date(html) or dt.date.today()

    tables = pd.read_html(StringIO(html))
    if not tables:
        raise RuntimeError(f"No tables found at {url}")

    df = _pick_leaderboard_table(tables)

    # Normalize columns to snake_case and disambiguate duplicates.
    new_cols: list[str] = []
    seen: dict[str, int] = {}
    for c in df.columns:
        c2 = _snake(c)
        if c2 in seen:
            seen[c2] += 1
            c2 = f"{c2}_{seen[c2]}"
        else:
            seen[c2] = 0
        new_cols.append(c2)
    df.columns = new_cols

    # Add tour + as_of_date
    df.insert(0, "tour", tour.upper())
    df.insert(1, "as_of_date", as_of.isoformat())

    return df, as_of


def _get_col(df: pd.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_int(v: Any) -> int | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    # Sometimes ranks are floats like "1.0"
    try:
        return int(float(s))
    except ValueError:
        return None


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


async def upsert_snapshot(engine, df: pd.DataFrame) -> int:
    """
    Upsert rows into tennisabstract_elo_snapshots.

    Expected table:
      tennisabstract_elo_snapshots(
        tour text, as_of_date date,
        elo_rank int, player_name text, age numeric,
        elo numeric, helo numeric, celo numeric, gelo numeric,
        peak_elo numeric, peak_month text,
        official_rank int, log_diff numeric,
        created_at timestamptz default now(),
        PRIMARY KEY (tour, as_of_date, player_name)
      )
    """
    col_player = _get_col(df, "player", "player_name", "name")
    col_rank = _get_col(df, "rank", "elo_rank", "rk")
    col_age = _get_col(df, "age")
    col_elo = _get_col(df, "elo")
    col_helo = _get_col(df, "helo")
    col_celo = _get_col(df, "celo")
    col_gelo = _get_col(df, "gelo")
    col_peak_elo = _get_col(df, "peak_elo", "peak")
    col_peak_month = _get_col(df, "peak_month")
    col_official_rank = _get_col(df, "official_rank", "atp_rank", "wta_rank", "rank_1")
    col_log_diff = _get_col(df, "log_diff", "logdiff")

    if not col_player or not col_elo:
        raise RuntimeError(f"Could not find required columns (player/elo). Columns: {list(df.columns)}")

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        player_name = str(r[col_player]).strip()
        if not player_name or player_name.lower() == "nan":
            continue

        rows.append(
            {
                "tour": str(r["tour"]).strip().upper(),
                "as_of_date": dt.date.fromisoformat(str(r["as_of_date"])),
                "elo_rank": _to_int(r[col_rank]) if col_rank else None,
                "player_name": player_name,
                "age": _to_float(r[col_age]) if col_age else None,
                "elo": _to_float(r[col_elo]),
                "helo": _to_float(r[col_helo]) if col_helo else None,
                "celo": _to_float(r[col_celo]) if col_celo else None,
                "gelo": _to_float(r[col_gelo]) if col_gelo else None,
                "peak_elo": _to_float(r[col_peak_elo]) if col_peak_elo else None,
                "peak_month": (str(r[col_peak_month]).strip() if col_peak_month else None),
                "official_rank": _to_int(r[col_official_rank]) if col_official_rank else None,
                "log_diff": _to_float(r[col_log_diff]) if col_log_diff else None,
            }
        )

    sql = text(
        """
        INSERT INTO tennisabstract_elo_snapshots (
          tour, as_of_date, elo_rank, player_name, age,
          elo, helo, celo, gelo,
          peak_elo, peak_month, official_rank, log_diff
        )
        VALUES (
          :tour, :as_of_date, :elo_rank, :player_name, :age,
          :elo, :helo, :celo, :gelo,
          :peak_elo, :peak_month, :official_rank, :log_diff
        )
        ON CONFLICT (tour, as_of_date, player_name)
        DO UPDATE SET
          elo_rank      = EXCLUDED.elo_rank,
          age           = EXCLUDED.age,
          elo           = EXCLUDED.elo,
          helo          = EXCLUDED.helo,
          celo          = EXCLUDED.celo,
          gelo          = EXCLUDED.gelo,
          peak_elo      = EXCLUDED.peak_elo,
          peak_month    = EXCLUDED.peak_month,
          official_rank = EXCLUDED.official_rank,
          log_diff      = EXCLUDED.log_diff,
          created_at    = now();
        """
    )

    async with engine.begin() as conn:
        await conn.execute(sql, rows)

    return len(rows)


async def post_fix(engine, backfill_days: int) -> None:
    """
    Post-ingest maintenance:
      1) Fill player_name_fold in snapshots (best-effort)
      1b) Propagate player_id from older snapshots to newer rows
      2) Refresh tennisabstract_elo_latest
      3) Rebuild TA mappings (name_fold + gender)
      4) Backfill recent match Elo (latest)
    """
    async with engine.begin() as conn:
        # 1) Ensure unaccent is available (best-effort)
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent;"))
        except Exception:
            pass

        # 1) Fill / refresh name_fold for snapshots
        #    Replace non-breaking space (chr 160) with regular space BEFORE
        #    running unaccent + whitespace normalisation so that names
        #    like 'Francisco\xa0Comesana' fold to 'francisco comesana'.
        await conn.execute(text(
            r"""
            UPDATE tennisabstract_elo_snapshots
            SET player_name_fold = lower(trim(regexp_replace(
                unaccent(replace(player_name, chr(160), ' ')),
                '\s+', ' ', 'g')))
            WHERE player_name_fold IS NULL
               OR player_name_fold = ''
               OR player_name_fold <> lower(trim(regexp_replace(
                    unaccent(replace(player_name, chr(160), ' ')),
                    '\s+', ' ', 'g')));
            """
        ))

        # 1b) Propagate player_id from older snapshots to newer rows that
        #     have the same (tour, player_name_fold) but player_id IS NULL.
        #     This ensures newly-ingested snapshots inherit the TA numeric ID
        #     so the snap CTE and latest-table rebuild include them.
        res = await conn.execute(text(
            """
            UPDATE tennisabstract_elo_snapshots s
            SET player_id = known.player_id
            FROM (
              SELECT DISTINCT ON (tour, player_name_fold)
                tour, player_name_fold, player_id
              FROM tennisabstract_elo_snapshots
              WHERE player_id IS NOT NULL
                AND player_name_fold IS NOT NULL
                AND player_name_fold <> ''
              ORDER BY tour, player_name_fold, as_of_date DESC
            ) known
            WHERE s.player_id IS NULL
              AND s.player_name_fold IS NOT NULL
              AND s.player_name_fold <> ''
              AND s.player_name_fold = known.player_name_fold
              AND s.tour = known.tour;
            """
        ))
        print(f"  ➜ Propagated player_id to {res.rowcount} snapshot rows")

        # 2) Refresh latest table (replace contents)
        await conn.execute(text(
            """
            DO $$
            BEGIN
              IF EXISTS (
                SELECT 1
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'tennisabstract_elo_latest'
                  AND c.relkind = 'v'
                  AND n.nspname = 'public'
              ) THEN
                EXECUTE 'ALTER VIEW public.tennisabstract_elo_latest RENAME TO tennisabstract_elo_latest_view';
              END IF;
            END $$;
            """
        ))
        await conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS tennisabstract_elo_latest (
              tour text,
              as_of_date date,
              elo_rank integer,
              player_name text,
              age numeric,
              elo numeric,
              helo numeric,
              celo numeric,
              gelo numeric,
              peak_elo numeric,
              peak_month text,
              official_rank integer,
              log_diff numeric,
              created_at timestamptz,
              player_id bigint
            );
            """
        ))
        await conn.execute(text("DELETE FROM tennisabstract_elo_latest;"))
        await conn.execute(text(
            """
            INSERT INTO tennisabstract_elo_latest (
              tour, as_of_date, elo_rank, player_name, age,
              elo, helo, celo, gelo,
              peak_elo, peak_month, official_rank, log_diff,
              created_at, player_id
            )
            SELECT DISTINCT ON (tour, player_id)
              tour, as_of_date, elo_rank, player_name, age,
              elo, helo, celo, gelo,
              peak_elo, peak_month, official_rank, log_diff,
              created_at, player_id
            FROM tennisabstract_elo_snapshots
            WHERE player_id IS NOT NULL
            ORDER BY tour, player_id, as_of_date DESC;
            """
        ))

        # 2b) Refresh Elo medians table (tour x as_of_date)
        await conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS tennisabstract_elo_medians (
              tour text NOT NULL,
              as_of_date date NOT NULL,
              med_elo float8,
              med_helo float8,
              med_celo float8,
              med_gelo float8,
              updated_at timestamptz NOT NULL DEFAULT now(),
              PRIMARY KEY (tour, as_of_date)
            );
            """
        ))
        await conn.execute(text("TRUNCATE TABLE tennisabstract_elo_medians;"))
        await conn.execute(text(
            """
            INSERT INTO tennisabstract_elo_medians (tour, as_of_date, med_elo, med_helo, med_celo, med_gelo)
            SELECT
              tour,
              as_of_date,
              (percentile_cont(0.5) WITHIN GROUP (ORDER BY elo::double precision))::float8  AS med_elo,
              (percentile_cont(0.5) WITHIN GROUP (ORDER BY helo::double precision))::float8 AS med_helo,
              (percentile_cont(0.5) WITHIN GROUP (ORDER BY celo::double precision))::float8 AS med_celo,
              (percentile_cont(0.5) WITHIN GROUP (ORDER BY gelo::double precision))::float8 AS med_gelo
            FROM tennisabstract_elo_snapshots
            WHERE elo IS NOT NULL
            GROUP BY 1,2;
            """
        ))

        # 3) Rebuild TA mappings from snapshots (name_fold + gender)
        # Step A: insert new mappings, avoid duplicate (source, source_player_id)
        await conn.execute(text(
            """
            WITH ta AS (
              SELECT DISTINCT ON (tour, player_id)
                player_id,
                tour,
                player_name_fold
              FROM tennisabstract_elo_snapshots
              WHERE player_id IS NOT NULL AND player_name_fold IS NOT NULL AND player_name_fold <> ''
              ORDER BY tour, player_id, as_of_date DESC
            ),
            tp AS (
              SELECT id AS player_id, name_fold, gender
              FROM tennis_players
              WHERE name_fold IS NOT NULL AND name_fold <> ''
            ),
            src AS (
              SELECT
                tp.player_id,
                CASE WHEN ta.tour = 'ATP' THEN 'tennisabstract_elo_atp' ELSE 'tennisabstract_elo_wta' END AS source,
                ta.player_id::text AS source_player_id
              FROM tp
              JOIN ta ON ta.player_name_fold = tp.name_fold
              WHERE (ta.tour = 'ATP' AND tp.gender = 'M')
                 OR (ta.tour = 'WTA' AND tp.gender = 'F')
            ),
            filtered AS (
              SELECT s.*
              FROM src s
              WHERE NOT EXISTS (
                SELECT 1
                FROM tennis_player_sources tps
                WHERE tps.source = s.source
                  AND tps.source_player_id = s.source_player_id
                  AND tps.player_id <> s.player_id
              )
            )
            INSERT INTO tennis_player_sources (player_id, source, source_player_id)
            SELECT player_id, source, source_player_id
            FROM filtered
            ON CONFLICT (player_id, source) DO NOTHING;
            """
        ))
        # Step B: update existing mappings for player/source
        await conn.execute(text(
            """
            WITH ta AS (
              SELECT DISTINCT ON (tour, player_id)
                player_id,
                tour,
                player_name_fold
              FROM tennisabstract_elo_snapshots
              WHERE player_id IS NOT NULL AND player_name_fold IS NOT NULL AND player_name_fold <> ''
              ORDER BY tour, player_id, as_of_date DESC
            ),
            tp AS (
              SELECT id AS player_id, name_fold, gender
              FROM tennis_players
              WHERE name_fold IS NOT NULL AND name_fold <> ''
            ),
            src AS (
              SELECT
                tp.player_id,
                CASE WHEN ta.tour = 'ATP' THEN 'tennisabstract_elo_atp' ELSE 'tennisabstract_elo_wta' END AS source,
                ta.player_id::text AS source_player_id
              FROM tp
              JOIN ta ON ta.player_name_fold = tp.name_fold
              WHERE (ta.tour = 'ATP' AND tp.gender = 'M')
                 OR (ta.tour = 'WTA' AND tp.gender = 'F')
            )
            UPDATE tennis_player_sources tps
            SET source_player_id = src.source_player_id
            FROM src
            WHERE tps.player_id = src.player_id
              AND tps.source = src.source;
            """
        ))

        # Step C: Fix broken TA source mappings.
        #   Some canonical IDs (from sofascore) point to the wrong
        #   tennis_players row, so Step B's name_fold join misses them.
        #   Detect sources whose ta_id doesn't exist in elo_latest and
        #   repair them by matching the sofascore source_name against
        #   elo_latest player_name.
        #
        #   Two-phase: first delete conflicting claims by other canonicals,
        #   then update the broken sources with the correct ta_id.

        # Phase 1: Delete TA source rows that block the fix.
        #   If canonical X has a broken ta_id, and canonical Y owns the
        #   correct ta_id (matched by name), delete Y's claim so X can
        #   take it.  Y is typically a stale/duplicate player row.
        del_res = await conn.execute(text(
            r"""
            WITH broken AS (
              SELECT tps.player_id, tps.source, tps.source_player_id
              FROM tennis_player_sources tps
              WHERE tps.source IN ('tennisabstract_elo_atp', 'tennisabstract_elo_wta')
                AND trim(tps.source_player_id) ~ '^[0-9]+$'
                AND NOT EXISTS (
                  SELECT 1 FROM tennisabstract_elo_latest el
                  WHERE el.player_id = tps.source_player_id::bigint
                    AND upper(el.tour) = CASE WHEN tps.source = 'tennisabstract_elo_atp'
                                              THEN 'ATP' ELSE 'WTA' END
                )
            ),
            sofa_names AS (
              SELECT b.player_id, b.source,
                     sf.source_name AS sofa_name
              FROM broken b
              JOIN tennis_player_sources sf
                ON sf.player_id = b.player_id AND sf.source = 'sofascore'
              WHERE sf.source_name IS NOT NULL AND sf.source_name <> ''
            ),
            target_ta AS (
              SELECT DISTINCT ON (sn.player_id, sn.source)
                sn.player_id, sn.source,
                el.player_id::text AS new_ta_id
              FROM sofa_names sn
              JOIN tennisabstract_elo_latest el
                ON lower(trim(regexp_replace(
                     unaccent(replace(el.player_name, chr(160), ' ')),
                     '\s+', ' ', 'g')))
                 = lower(trim(regexp_replace(
                     unaccent(sn.sofa_name),
                     '\s+', ' ', 'g')))
                AND upper(el.tour) = CASE WHEN sn.source = 'tennisabstract_elo_atp'
                                          THEN 'ATP' ELSE 'WTA' END
              ORDER BY sn.player_id, sn.source
            )
            DELETE FROM tennis_player_sources d
            USING target_ta t
            WHERE d.source = t.source
              AND d.source_player_id = t.new_ta_id
              AND d.player_id <> t.player_id;
            """
        ))
        if del_res.rowcount:
            print(f"  ➜ Removed {del_res.rowcount} conflicting TA source claims")

        # Phase 2: Now update the broken sources with the correct ta_id.
        fix_res = await conn.execute(text(
            r"""
            WITH broken AS (
              SELECT tps.player_id, tps.source, tps.source_player_id
              FROM tennis_player_sources tps
              WHERE tps.source IN ('tennisabstract_elo_atp', 'tennisabstract_elo_wta')
                AND trim(tps.source_player_id) ~ '^[0-9]+$'
                AND NOT EXISTS (
                  SELECT 1 FROM tennisabstract_elo_latest el
                  WHERE el.player_id = tps.source_player_id::bigint
                    AND upper(el.tour) = CASE WHEN tps.source = 'tennisabstract_elo_atp'
                                              THEN 'ATP' ELSE 'WTA' END
                )
            ),
            sofa_names AS (
              SELECT b.player_id, b.source,
                     sf.source_name AS sofa_name
              FROM broken b
              JOIN tennis_player_sources sf
                ON sf.player_id = b.player_id AND sf.source = 'sofascore'
              WHERE sf.source_name IS NOT NULL AND sf.source_name <> ''
            ),
            fixed AS (
              SELECT DISTINCT ON (sn.player_id, sn.source)
                sn.player_id, sn.source,
                el.player_id::text AS new_ta_id
              FROM sofa_names sn
              JOIN tennisabstract_elo_latest el
                ON lower(trim(regexp_replace(
                     unaccent(replace(el.player_name, chr(160), ' ')),
                     '\s+', ' ', 'g')))
                 = lower(trim(regexp_replace(
                     unaccent(sn.sofa_name),
                     '\s+', ' ', 'g')))
                AND upper(el.tour) = CASE WHEN sn.source = 'tennisabstract_elo_atp'
                                          THEN 'ATP' ELSE 'WTA' END
              ORDER BY sn.player_id, sn.source
            )
            UPDATE tennis_player_sources tps
            SET source_player_id = f.new_ta_id
            FROM fixed f
            WHERE tps.player_id = f.player_id
              AND tps.source = f.source;
            """
        ))
        print(f"  ➜ Fixed {fix_res.rowcount} broken TA source mappings via sofascore name")

        # 4) Backfill recent matches with latest Elo
        await conn.execute(text(
            """
            UPDATE tennis_matches m
            SET
              p1_elo = l1.elo,
              p2_elo = l2.elo,
              p1_elo_as_of_date = l1.as_of_date,
              p2_elo_as_of_date = l2.as_of_date,
              p1_elo_source = 'ta_latest',
              p2_elo_source = 'ta_latest'
            FROM tennis_player_sources s1
            JOIN tennisabstract_elo_latest l1
              ON l1.player_id = s1.source_player_id::bigint
            JOIN tennis_player_sources s2
              ON s2.source IN ('tennisabstract_elo_atp', 'tennisabstract_elo_wta')
            JOIN tennisabstract_elo_latest l2
              ON l2.player_id = s2.source_player_id::bigint
            WHERE m.match_date >= (CURRENT_DATE - (CAST(:days AS int) * INTERVAL '1 day'))
              AND s1.player_id = m.p1_canonical_id
              AND s2.player_id = m.p2_canonical_id
              AND s1.source IN ('tennisabstract_elo_atp', 'tennisabstract_elo_wta')
              AND trim(s1.source_player_id) ~ '^[0-9]+$'
              AND trim(s2.source_player_id) ~ '^[0-9]+$';
            """
        ), {"days": int(backfill_days)})


async def main_async() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/elo", help="Output folder for CSV snapshots (CSV mode)")
    ap.add_argument("--as-of", default=None, help="Override as_of_date (YYYY-MM-DD)")
    ap.add_argument("--no-atp", action="store_true", help="Skip ATP")
    ap.add_argument("--no-wta", action="store_true", help="Skip WTA")
    ap.add_argument("--to-db", action="store_true", help="Upsert results into Postgres instead of CSV")
    ap.add_argument("--post-fix", action="store_true", help="Run post-ingest maintenance steps")
    ap.add_argument("--backfill-days", type=int, default=30, help="Days of matches to backfill Elo")
    args = ap.parse_args()

    as_of_override = dt.date.fromisoformat(args.as_of) if args.as_of else None

    engine = None
    if args.to_db:
        db_url = os.getenv("DATABASE_URL_ASYNC") or os.getenv("DATABASE_URL")
        if not db_url:
            raise RuntimeError("Set DATABASE_URL_ASYNC (preferred) or DATABASE_URL in your .env")
        engine = create_async_engine(db_url, echo=False, future=True,  connect_args={"ssl": "require"},)

    if not args.no_atp:
        atp_df, atp_as_of = fetch_table(ATP_URL, "ATP", as_of_override)
        if args.to_db:
            n = await upsert_snapshot(engine, atp_df)
            print(f"Upserted {n} ATP rows (as_of={atp_as_of})")
        else:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            atp_file = out_dir / f"tennisabstract_atp_elo_{atp_as_of.isoformat()}.csv"
            atp_df.to_csv(atp_file, index=False)
            print(f"Wrote {atp_file} ({len(atp_df)} rows)")

    if not args.no_wta:
        wta_df, wta_as_of = fetch_table(WTA_URL, "WTA", as_of_override)
        if args.to_db:
            n = await upsert_snapshot(engine, wta_df)
            print(f"Upserted {n} WTA rows (as_of={wta_as_of})")
        else:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            wta_file = out_dir / f"tennisabstract_wta_elo_{wta_as_of.isoformat()}.csv"
            wta_df.to_csv(wta_file, index=False)
            print(f"Wrote {wta_file} ({len(wta_df)} rows)")

    if engine:
        if args.post_fix:
            await post_fix(engine, args.backfill_days)
        await engine.dispose()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
