import os
import hashlib
import datetime as dt
from typing import List, Dict
import psycopg

def _hash_key(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def build_match_key(match_date: dt.date, tour: str, tournament: str | None, p1: str, p2: str) -> str:
    base = f"{match_date.isoformat()}|{tour}|{tournament or ''}|{p1}|{p2}"
    return _hash_key(base)

UPSERT_SQL = """
INSERT INTO upcoming_matches
(match_key, match_date, match_time, source_url, tour, tournament, p1_name, p2_name, status, updated_at)
VALUES
(%(match_key)s, %(match_date)s, %(match_time)s, %(source_url)s, %(tour)s, %(tournament)s, %(p1_name)s, %(p2_name)s, %(status)s, now())
ON CONFLICT (match_key) DO UPDATE SET
  match_date   = EXCLUDED.match_date,
  match_time   = EXCLUDED.match_time,
  source_url   = EXCLUDED.source_url,
  tour         = EXCLUDED.tour,
  tournament   = EXCLUDED.tournament,
  p1_name      = EXCLUDED.p1_name,
  p2_name      = EXCLUDED.p2_name,
  status       = EXCLUDED.status,
  updated_at   = now();
"""

def upsert_upcoming_matches(matches: List[Dict], match_date: dt.date) -> List[Dict]:
    """
    Writes matches to DB, returns the same list but with match_key added.
    Requires env DATABASE_URL.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")

    enriched = []
    for m in matches:
        tour = m["tour"]
        tournament = m.get("tournament")
        p1 = m["p1"]
        p2 = m["p2"]
        match_key = build_match_key(match_date, tour, tournament, p1, p2)

        enriched.append({
            "match_key": match_key,
            "match_date": match_date,
            "match_time": m.get("time"),
            "source_url": m.get("source_url"),
            "tour": tour,
            "tournament": tournament,
            "p1_name": p1,
            "p2_name": p2,
            "status": m.get("status"),
        })

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(UPSERT_SQL, enriched)
        conn.commit()

    return enriched
