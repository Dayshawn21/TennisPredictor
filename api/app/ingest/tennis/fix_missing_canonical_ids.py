from __future__ import annotations

import argparse
import asyncio
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text

from app.db_session import engine


MATCHES_SQL = text(r"""
SELECT
  m.match_id,
  m.match_key,
  m.match_date,
  upper(m.tour) AS tour,

  m.p1_name,
  m.p1_canonical_id,
  m.p1_sofascore_player_id,

  m.p2_name,
  m.p2_canonical_id,
  m.p2_sofascore_player_id,

  -- expected canonical from sofascore mapping
  s1.player_id AS expected_p1_canonical_id,
  s2.player_id AS expected_p2_canonical_id

FROM tennis_matches m

LEFT JOIN tennis_player_sources s1
  ON s1.source = 'sofascore'
 AND NULLIF(trim(s1.source_player_id), '')::bigint = m.p1_sofascore_player_id

LEFT JOIN tennis_player_sources s2
  ON s2.source = 'sofascore'
 AND NULLIF(trim(s2.source_player_id), '')::bigint = m.p2_sofascore_player_id

WHERE m.match_date BETWEEN :start AND :end
  AND (
       m.p1_sofascore_player_id IS NOT NULL
    OR m.p2_sofascore_player_id IS NOT NULL
  )
ORDER BY m.match_date, m.match_key;
""")


UPDATE_P1_SQL = text("UPDATE tennis_matches SET p1_canonical_id = :cid WHERE match_id = :mid")
UPDATE_P2_SQL = text("UPDATE tennis_matches SET p2_canonical_id = :cid WHERE match_id = :mid")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start).date()
    end = datetime.fromisoformat(args.end).date()

    fixed = 0
    skipped_no_map = 0
    already_ok = 0

    async with engine.begin() as conn:
        row = (await conn.execute(text(
            "SELECT current_database(), inet_server_addr(), inet_server_port(), current_user"
        ))).one()
        print("SCRIPT DB:", row[0], row[1], row[2], row[3])

        rows = (await conn.execute(MATCHES_SQL, {"start": start, "end": end})).mappings().all()

        for r in rows:
            mid = r["match_id"]
            key = r.get("match_key")

            # ---- P1 ----
            exp1 = r.get("expected_p1_canonical_id")
            cur1 = r.get("p1_canonical_id")

            if exp1 is None and r.get("p1_sofascore_player_id") is not None:
                skipped_no_map += 1
                print(f"⚠️  NO MAP p1 {key}: sofascore_player_id={r.get('p1_sofascore_player_id')} name='{r.get('p1_name')}'")
            elif exp1 is not None:
                if cur1 == exp1:
                    already_ok += 1
                else:
                    fixed += 1
                    print(f"✅ FIX p1 {key}: '{r.get('p1_name')}' canonical {cur1} -> {exp1} (sofa_id={r.get('p1_sofascore_player_id')})")
                    if not args.dry_run:
                        await conn.execute(UPDATE_P1_SQL, {"cid": exp1, "mid": mid})

            # ---- P2 ----
            exp2 = r.get("expected_p2_canonical_id")
            cur2 = r.get("p2_canonical_id")

            if exp2 is None and r.get("p2_sofascore_player_id") is not None:
                skipped_no_map += 1
                print(f"⚠️  NO MAP p2 {key}: sofascore_player_id={r.get('p2_sofascore_player_id')} name='{r.get('p2_name')}'")
            elif exp2 is not None:
                if cur2 == exp2:
                    already_ok += 1
                else:
                    fixed += 1
                    print(f"✅ FIX p2 {key}: '{r.get('p2_name')}' canonical {cur2} -> {exp2} (sofa_id={r.get('p2_sofascore_player_id')})")
                    if not args.dry_run:
                        await conn.execute(UPDATE_P2_SQL, {"cid": exp2, "mid": mid})

    print(f"\nDone. fixed={fixed} already_ok={already_ok} skipped_no_map={skipped_no_map} dry_run={args.dry_run}")


if __name__ == "__main__":
    asyncio.run(main())
