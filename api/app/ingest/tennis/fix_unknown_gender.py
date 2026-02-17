# app/ingest/tennis/fix_unknown_gender.py
from __future__ import annotations

import os
import ssl
import asyncio
from typing import Any, Dict, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


def normalize_asyncpg_url_and_ssl(db_url: str) -> Tuple[str, Dict[str, Any]]:
    """
    Neon requires SSL. If URL has ?sslmode=require we create an SSL context.
    We also strip sslmode from the URL query after converting it into connect_args.
    """
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


SQL_UPDATE_U = text(r"""
with u as (
  select
    tp.id,
    tp.name,
    tp.name_fold,
    upper(left(btrim(tp.name), 1)) as init,
    lower(regexp_replace(btrim(tp.name), '^.*\s', '')) as last_token
  from tennis_players tp
  where tp.gender = 'U'
),
cand as (
  select
    u.id as u_id,
    p.gender as inferred_gender
  from u
  join tennis_players p
    on p.gender in ('M','F')
   and (
        (u.name_fold is not null and u.name_fold <> '' and p.name_fold = u.name_fold)
     or (
          upper(left(btrim(p.name), 1)) = u.init
      and lower(regexp_replace(btrim(p.name), '^.*\s', '')) = u.last_token
     )
   )
),
dec as (
  select
    u_id,
    case when count(distinct inferred_gender) = 1 then min(inferred_gender) end as target_gender
  from cand
  group by u_id
),
del as (
  -- Avoid uq_tennis_players_name_gender (name, gender)
  delete from tennis_players urow
  using dec d
  where urow.id = d.u_id
    and d.target_gender is not null
    and exists (
      select 1
      from tennis_players ex
      where ex.name = urow.name
        and ex.gender = d.target_gender
    )
  returning 1
),
upd as (
  -- IMPORTANT: do NOT set ta_player_id here (avoids uq_tennis_players_gender_ta_player_id)
  update tennis_players urow
  set gender = d.target_gender
  from dec d
  where urow.id = d.u_id
    and d.target_gender is not null
    and not exists (
      select 1
      from tennis_players ex
      where ex.name = urow.name
        and ex.gender = d.target_gender
    )
  returning 1
)
select
  (select count(*) from dec where target_gender is not null) as resolved,
  (select count(*) from upd) as updated,
  (select count(*) from del) as deleted_conflicts;
""")


SQL_REASON_BREAKDOWN = text(r"""
with u as (
  select
    id,
    name,
    name_fold,
    upper(left(btrim(name), 1)) as init,
    lower(regexp_replace(btrim(name), '^.*\s', '')) as last_token
  from tennis_players
  where gender='U'
),
cand as (
  select
    u.id as u_id,
    count(distinct p.gender) as gender_variants,
    count(*) as matches
  from u
  join tennis_players p
    on p.gender in ('M','F')
   and (
        (u.name_fold is not null and u.name_fold <> '' and p.name_fold = u.name_fold)
     or (upper(left(btrim(p.name), 1)) = u.init
         and lower(regexp_replace(btrim(p.name), '^.*\s', '')) = u.last_token)
   )
  group by u.id
)
select
  case
    when cand.u_id is null then 'NO_MATCH'
    when cand.gender_variants > 1 then 'AMBIGUOUS_M_AND_F'
    else 'SHOULD_HAVE_RESOLVED'
  end as reason,
  count(*) as cnt
from u
left join cand on cand.u_id = u.id
group by 1
order by 2 desc;
""")


async def main() -> None:
    db_url = os.getenv("DATABASE_URL_ASYNC") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            "Missing DATABASE_URL_ASYNC or DATABASE_URL. "
            "Example: postgresql+asyncpg://user:pass@host:5432/db?sslmode=require"
        )

    normalized_url, connect_args = normalize_asyncpg_url_and_ssl(db_url)
    if "ssl" not in connect_args:
        connect_args["ssl"] = ssl.create_default_context()

    engine = create_async_engine(
        normalized_url,
        echo=False,
        future=True,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    async with engine.begin() as conn:
        before_u = (await conn.execute(text("select count(*) from tennis_players where gender='U';"))).scalar_one()

        res = await conn.execute(SQL_UPDATE_U)
        row = res.one()

        after_u = (await conn.execute(text("select count(*) from tennis_players where gender='U';"))).scalar_one()

        print(
            f"before_U={before_u} | resolved={row.resolved}, updated={row.updated}, "
            f"deleted_conflicts={row.deleted_conflicts} | after_U={after_u}"
        )

        # Show why remaining U rows weren't resolved
        print("\nRemaining U breakdown:")
        breakdown = await conn.execute(SQL_REASON_BREAKDOWN)
        for reason, cnt in breakdown.all():
            print(f"  {reason}: {cnt}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
