"""
One-time backfill to map tennis_players to TennisAbstract Elo using fuzzy name matching.

Run:
  python -m app.ingest.tennis.backfill_ta_elo_mapping

Notes:
- Enforces 1-to-1 mapping (no two tennis_players can claim the same TA player_id).
- Uses gendered sources: tennisabstract_elo_m / tennisabstract_elo_f
- Skips any rows that would conflict with UNIQUE(source, source_player_id)
"""

from __future__ import annotations

import os
import ssl
import asyncio
from typing import Any, Dict, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


def normalize_asyncpg_url_and_ssl(db_url: str) -> Tuple[str, Dict[str, Any]]:
    u = urlparse(db_url)
    q = parse_qs(u.query)
    sslmode = (q.get("sslmode", [""])[0] or "").lower()

    connect_args: Dict[str, Any] = {}
    if sslmode in {"require", "verify-ca", "verify-full"}:
        connect_args["ssl"] = ssl.create_default_context()

    q.pop("sslmode", None)
    new_query = urlencode({k: v[0] for k, v in q.items()})
    new_url = urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))
    return new_url, connect_args


SQL_DELETE_OLD = text("""
delete from tennis_player_sources
where source in ('tennisabstract_elo', 'tennisabstract_elo_m', 'tennisabstract_elo_f');
""")


SQL_BACKFILL = text(r"""
with ta_players as (
  select distinct
    tour,
    player_id,
    player_name_fold,
    player_name
  from tennisabstract_elo_snapshots
  where player_id is not null
    and player_name_fold is not null
    and player_name_fold <> ''
),
candidates as (
  select
    tp.id as tennis_player_id,
    ta.player_id as ta_player_id,
    ta.player_name as ta_name,
    case when ta.tour = 'ATP' then 'tennisabstract_elo_m' else 'tennisabstract_elo_f' end as source,
    similarity(tp.name_fold, ta.player_name_fold) as sim
  from tennis_players tp
  join ta_players ta
    on tp.gender = case when ta.tour = 'ATP' then 'M' else 'F' end
   and tp.name_fold is not null
   and tp.name_fold <> ''
   and similarity(tp.name_fold, ta.player_name_fold) >= :min_sim
),
ranked as (
  select
    *,
    row_number() over (
      partition by tennis_player_id
      order by sim desc, ta_player_id
    ) as rn_for_player,
    row_number() over (
      partition by source, ta_player_id
      order by sim desc, tennis_player_id
    ) as rn_for_ta
  from candidates
),
best as (
  -- "Mutual best": best TA for this player AND best player for this TA.
  select
    tennis_player_id,
    source,
    ta_player_id::text as source_player_id,
    ta_name as source_name
  from ranked
  where rn_for_player = 1
    and rn_for_ta = 1
)
insert into tennis_player_sources (player_id, source, source_player_id, source_name)
select tennis_player_id, source, source_player_id, source_name
from best
on conflict do nothing
returning 1;
""")


SQL_COVERAGE_TODAY = text("""
select elo_status, count(*) as cnt
from v_api_fixtures_elo_unified
where match_date = current_date
group by elo_status
order by cnt desc;
""")


async def main() -> None:
    db_url = os.getenv("DATABASE_URL_ASYNC") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Set DATABASE_URL_ASYNC or DATABASE_URL (Neon: add ?sslmode=require)")

    normalized_url, connect_args = normalize_asyncpg_url_and_ssl(db_url)

    # Force SSL for Neon even if sslmode wasn't included
    if "ssl" not in connect_args:
        connect_args["ssl"] = ssl.create_default_context()

    min_sim = float(os.getenv("TA_SIM_THRESHOLD", "0.75"))  # bump to 0.80 if you want stricter

    engine = create_async_engine(
        normalized_url,
        echo=False,  # set True only if you want spammy SQL logs
        future=True,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    async with engine.begin() as conn:
        await conn.execute(text("create extension if not exists pg_trgm;"))

        res = await conn.execute(SQL_DELETE_OLD)
        print(f"Deleted {res.rowcount} old TA mappings")

        res = await conn.execute(SQL_BACKFILL, {"min_sim": min_sim})
        inserted = res.rowcount or 0
        print(f"Inserted {inserted} TA mappings (min_sim={min_sim})")

        print("\nElo coverage for today:")
        cov = await conn.execute(SQL_COVERAGE_TODAY)
        for row in cov.all():
            print(f"  {row.elo_status}: {row.cnt}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
