import os
import re
import unicodedata
from datetime import date

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

CSV_PATH = r"C:\Users\dayshawn\Desktop\SportApp\api\cvs\player_stats_ATP_WTA_shrunk_k20.csv"
CONN_STR = os.environ["DATABASE_URL_PSQL"]  # your Neon connection string env var

def norm_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

out = pd.DataFrame({
    "tour": df["Tour"].astype(str).str.upper().str.strip(),
    "player": df["Player"].astype(str),
    "player_key": df["Player"].map(norm_name),
    "match_stat_matches": pd.to_numeric(df["Match Stat Matches"], errors="coerce").fillna(0).astype(int),

    # shrunk stats -> clean table columns
    "aces_per_game": pd.to_numeric(df["Aces per Game (shrunk_k=20)"], errors="coerce"),
    "dfs_per_game": pd.to_numeric(df["DFs per Game (shrunk_k=20)"], errors="coerce"),
    "first_serve_pct": pd.to_numeric(df["1st Serve % (shrunk_k=20)"], errors="coerce"),
    "first_serve_won_pct": pd.to_numeric(df["1st Serve W % (shrunk_k=20)"], errors="coerce"),
    "second_serve_won_pct": pd.to_numeric(df["2nd Serve W % (shrunk_k=20)"], errors="coerce"),
    "service_pts_won_pct": pd.to_numeric(df["Service Pts W % (shrunk_k=20)"], errors="coerce"),
    "bp_save_pct": pd.to_numeric(df["BP save % (shrunk_k=20)"], errors="coerce"),
    "service_hold_pct": pd.to_numeric(df["Service hold % (shrunk_k=20)"], errors="coerce"),

    "first_return_won_pct": pd.to_numeric(df["1st Return W % (shrunk_k=20)"], errors="coerce"),
    "second_return_won_pct": pd.to_numeric(df["2nd Return W % (shrunk_k=20)"], errors="coerce"),
    "return_pts_won_pct": pd.to_numeric(df["Return Pts W % (shrunk_k=20)"], errors="coerce"),
    "bp_won_pct": pd.to_numeric(df["BP W % (shrunk_k=20)"], errors="coerce"),
    "opponent_hold_pct": pd.to_numeric(df["Opponent Hold % (shrunk_k=20)"], errors="coerce"),

    "snapshot_date": date.today(),
})

cols = list(out.columns)
rows = [tuple(r) for r in out.itertuples(index=False, name=None)]

insert_sql = f"""
insert into public.tennis_insight_player_stats ({",".join(cols)})
values %s
on conflict (tour, player_key, snapshot_date) do update set
  player = excluded.player,
  match_stat_matches = excluded.match_stat_matches,
  aces_per_game = excluded.aces_per_game,
  dfs_per_game = excluded.dfs_per_game,
  first_serve_pct = excluded.first_serve_pct,
  first_serve_won_pct = excluded.first_serve_won_pct,
  second_serve_won_pct = excluded.second_serve_won_pct,
  service_pts_won_pct = excluded.service_pts_won_pct,
  bp_save_pct = excluded.bp_save_pct,
  service_hold_pct = excluded.service_hold_pct,
  first_return_won_pct = excluded.first_return_won_pct,
  second_return_won_pct = excluded.second_return_won_pct,
  return_pts_won_pct = excluded.return_pts_won_pct,
  bp_won_pct = excluded.bp_won_pct,
  opponent_hold_pct = excluded.opponent_hold_pct;
"""

with psycopg2.connect(CONN_STR) as conn:
    with conn.cursor() as cur:
        # optional: wipe today's snapshot so reruns are clean
        cur.execute("delete from public.tennis_insight_player_stats where snapshot_date = current_date;")
        execute_values(cur, insert_sql, rows, page_size=1000)
    conn.commit()

print(f"Loaded {len(rows)} rows into public.tennis_insight_player_stats for snapshot_date={date.today()}.")
