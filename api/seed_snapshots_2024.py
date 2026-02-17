import asyncio
from sqlalchemy import text
from app.db import get_db
from datetime import date

AS_OF = date(2024, 12, 31)
SURFACE = "hard"


# Placeholder snapshot values (you will replace with real 2024 stats later)
SNAPS = [
    ("Jaume Munar",
     74.0, 23.0, 75.0, 24.0, 76.0, 25.0, 0.30, 0.32, 0.05, 0.05, 18.0, 52.0, 17.0, 50.0),
    ("Sebastian Baez",
     77.0, 26.0, 78.0, 27.0, 79.0, 28.0, 0.18, 0.20, 0.04, 0.04, 20.0, 54.0, 21.0, 55.0),
    ("Jessica Bouzas Maneiro",
     71.0, 21.0, 72.0, 22.0, 73.0, 23.0, 0.12, 0.13, 0.03, 0.03, 15.0, 49.0, 14.0, 48.0),
    ("Solana Sierra",
     69.0, 20.0, 70.0, 21.0, 71.0, 22.0, 0.10, 0.11, 0.04, 0.04, 14.0, 47.0, 13.0, 46.0),
]

async def main():
    async for db in get_db():
        for (name, last5_hold, last5_break, last10_hold, last10_break,
             surf_last10_hold, surf_last10_break, last10_aces_pg, surf_last10_aces_pg,
             last10_df_pg, surf_last10_df_pg, last10_tb_match_rate, last10_tb_win_pct,
             surf_last10_tb_match_rate, surf_last10_tb_win_pct) in SNAPS:

            pid = await db.execute(text("SELECT id FROM tennis_players WHERE name=:n"), {"n": name})
            row = pid.first()
            if not row:
                raise RuntimeError(f"Player missing: {name}")
            player_id = int(row[0])

            await db.execute(
                text("""
                    INSERT INTO tennis_player_snapshots (
                      player_id, as_of_date, surface,
                      last5_hold, last5_break,
                      last10_hold, last10_break,
                      surf_last10_hold, surf_last10_break,
                      last10_aces_pg, surf_last10_aces_pg,
                      last10_df_pg, surf_last10_df_pg,
                      last10_tb_match_rate, last10_tb_win_pct,
                      surf_last10_tb_match_rate, surf_last10_tb_win_pct
                    )
                    VALUES (
                      :player_id, :as_of_date, :surface,
                      :last5_hold, :last5_break,
                      :last10_hold, :last10_break,
                      :surf_last10_hold, :surf_last10_break,
                      :last10_aces_pg, :surf_last10_aces_pg,
                      :last10_df_pg, :surf_last10_df_pg,
                      :last10_tb_match_rate, :last10_tb_win_pct,
                      :surf_last10_tb_match_rate, :surf_last10_tb_win_pct
                    )
                    ON CONFLICT (player_id, surface, as_of_date) DO UPDATE SET
                      last5_hold = EXCLUDED.last5_hold,
                      last5_break = EXCLUDED.last5_break,
                      last10_hold = EXCLUDED.last10_hold,
                      last10_break = EXCLUDED.last10_break,
                      surf_last10_hold = EXCLUDED.surf_last10_hold,
                      surf_last10_break = EXCLUDED.surf_last10_break,
                      last10_aces_pg = EXCLUDED.last10_aces_pg,
                      surf_last10_aces_pg = EXCLUDED.surf_last10_aces_pg,
                      last10_df_pg = EXCLUDED.last10_df_pg,
                      surf_last10_df_pg = EXCLUDED.surf_last10_df_pg,
                      last10_tb_match_rate = EXCLUDED.last10_tb_match_rate,
                      last10_tb_win_pct = EXCLUDED.last10_tb_win_pct,
                      surf_last10_tb_match_rate = EXCLUDED.surf_last10_tb_match_rate,
                      surf_last10_tb_win_pct = EXCLUDED.surf_last10_tb_win_pct
                """),
                {
                    "player_id": player_id,
                    "as_of_date": AS_OF,
                    "surface": SURFACE,
                    "last5_hold": last5_hold,
                    "last5_break": last5_break,
                    "last10_hold": last10_hold,
                    "last10_break": last10_break,
                    "surf_last10_hold": surf_last10_hold,
                    "surf_last10_break": surf_last10_break,
                    "last10_aces_pg": last10_aces_pg,
                    "surf_last10_aces_pg": surf_last10_aces_pg,
                    "last10_df_pg": last10_df_pg,
                    "surf_last10_df_pg": surf_last10_df_pg,
                    "last10_tb_match_rate": last10_tb_match_rate,
                    "last10_tb_win_pct": last10_tb_win_pct,
                    "surf_last10_tb_match_rate": surf_last10_tb_match_rate,
                    "surf_last10_tb_win_pct": surf_last10_tb_win_pct,
                },
            )

        await db.commit()
        print("Seeded 2024 snapshots (hard).")
        break

asyncio.run(main())
