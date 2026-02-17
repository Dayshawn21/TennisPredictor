# predictions_history_service.py
from sqlalchemy import text
from app.db_session import engine

UPSERT_SQL = text("""
INSERT INTO predictions_history (
  match_id, match_date, tour, surface, tournament_name, round,
  p1_id, p2_id,
  p1_elo_overall, p2_elo_overall,
  p1_elo_surface, p2_elo_surface,
  d_elo_overall, d_elo_surface,
  p1_prob, p1_fair_american,
  generated_at, updated_at
)
VALUES (
  :match_id, :match_date, :tour, :surface, :tournament_name, :round,
  :p1_id, :p2_id,
  :p1_elo_overall, :p2_elo_overall,
  :p1_elo_surface, :p2_elo_surface,
  :d_elo_overall, :d_elo_surface,
  :p1_prob, :p1_fair_american,
  now(), now()
)
ON CONFLICT (match_id) DO UPDATE SET
  p1_prob = EXCLUDED.p1_prob,
  p1_fair_american = EXCLUDED.p1_fair_american,
  updated_at = now();
""")

def upsert_predictions_history(rows: list[dict]) -> int:
    with engine.begin() as conn:
        conn.execute(UPSERT_SQL, rows)
    return len(rows)
