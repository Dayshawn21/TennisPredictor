from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

# ✅ FAST version:
# - No correlated subqueries per match
# - Computes "as_of" snapshot once per tour for the requested date
# - Computes medians once per tour (overall + hard/clay/grass)
# - Joins by canonical -> tennis_player_sources(source='tennisabstract_elo') -> TA player_id
#
# Returned columns:
# - p1_elo/p2_elo + p1_helo/p1_celo/p1_gelo (+ p2_*)
# - surface-based p1_elo_used / p2_elo_used
# - missing_reason when mapping is missing

GET_FEATURES_BY_DATE = text("""
WITH matches AS (
  SELECT
    tm.match_id,
    tm.match_key,
    tm.match_date,
    upper(tm.tour) AS tour,
    tm.tournament,
    tm.round,
    tm.surface,
    tm.p1_name,
    tm.p2_name,
    tm.p1_canonical_id AS p1_player_id,
    tm.p2_canonical_id AS p2_player_id,
    CASE
      WHEN tm.surface ILIKE '%hard%'  THEN 'Hard'
      WHEN tm.surface ILIKE '%clay%'  THEN 'Clay'
      WHEN tm.surface ILIKE '%grass%' THEN 'Grass'
      ELSE NULL
    END AS surface_key
  FROM tennis_matches tm
  WHERE tm.match_date = :match_date
    AND upper(tm.tour) IN ('ATP','WTA')
),

mapped AS (
  SELECT
    m.*,
    ta1.source_player_id::bigint AS p1_ta_id,
    ta2.source_player_id::bigint AS p2_ta_id
  FROM matches m
  LEFT JOIN tennis_player_sources ta1
    ON ta1.player_id = m.p1_player_id AND ta1.source = 'tennisabstract_elo'
  LEFT JOIN tennis_player_sources ta2
    ON ta2.player_id = m.p2_player_id AND ta2.source = 'tennisabstract_elo'
),

asof AS (
  SELECT
    upper(s.tour) AS tour,
    max(s.as_of_date) AS as_of_date
  FROM tennisabstract_elo_snapshots s
  WHERE s.as_of_date <= :match_date
    AND upper(s.tour) IN ('ATP','WTA')
  GROUP BY 1
),

snap AS (
  SELECT s.*
  FROM tennisabstract_elo_snapshots s
  JOIN asof a
    ON upper(s.tour) = a.tour
   AND s.as_of_date = a.as_of_date
),

med AS (
  SELECT
    upper(tour) AS tour,
    (percentile_cont(0.5) WITHIN GROUP (ORDER BY elo::double precision))::numeric  AS med_elo,
    (percentile_cont(0.5) WITHIN GROUP (ORDER BY helo::double precision))::numeric AS med_helo,
    (percentile_cont(0.5) WITHIN GROUP (ORDER BY celo::double precision))::numeric AS med_celo,
    (percentile_cont(0.5) WITHIN GROUP (ORDER BY gelo::double precision))::numeric AS med_gelo
  FROM snap
  GROUP BY 1
)

SELECT
  m.match_id,
  m.match_key,
  m.match_date,
  m.tour,
  m.tournament,
  m.round,
  m.surface,
  m.p1_name,
  m.p2_name,

  m.p1_ta_id,
  m.p2_ta_id,

  -- Elo (overall + surfaces) with median fallback per tour
  COALESCE(s1.elo,  med.med_elo)  AS p1_elo,
  COALESCE(s1.helo, med.med_helo) AS p1_helo,
  COALESCE(s1.celo, med.med_celo) AS p1_celo,
  COALESCE(s1.gelo, med.med_gelo) AS p1_gelo,

  COALESCE(s2.elo,  med.med_elo)  AS p2_elo,
  COALESCE(s2.helo, med.med_helo) AS p2_helo,
  COALESCE(s2.celo, med.med_celo) AS p2_celo,
  COALESCE(s2.gelo, med.med_gelo) AS p2_gelo,

  -- Surface-specific "used" rating
  CASE
    WHEN m.surface_key = 'Hard'  THEN COALESCE(s1.helo, s1.elo, med.med_helo, med.med_elo)
    WHEN m.surface_key = 'Clay'  THEN COALESCE(s1.celo, s1.elo, med.med_celo, med.med_elo)
    WHEN m.surface_key = 'Grass' THEN COALESCE(s1.gelo, s1.elo, med.med_gelo, med.med_elo)
    ELSE COALESCE(s1.elo, med.med_elo)
  END AS p1_elo_used,

  CASE
    WHEN m.surface_key = 'Hard'  THEN COALESCE(s2.helo, s2.elo, med.med_helo, med.med_elo)
    WHEN m.surface_key = 'Clay'  THEN COALESCE(s2.celo, s2.elo, med.med_celo, med.med_elo)
    WHEN m.surface_key = 'Grass' THEN COALESCE(s2.gelo, s2.elo, med.med_gelo, med.med_elo)
    ELSE COALESCE(s2.elo, med.med_elo)
  END AS p2_elo_used,

  -- Explain why true Elo wasn't used
  CASE
    WHEN m.p1_player_id IS NULL OR m.p2_player_id IS NULL
      THEN 'Missing canonical player_id(s) on tennis_matches.'
    WHEN m.p1_ta_id IS NULL AND m.p2_ta_id IS NULL
      THEN 'both players missing tennisabstract_elo mapping'
    WHEN m.p1_ta_id IS NULL
      THEN 'p1 missing tennisabstract_elo mapping'
    WHEN m.p2_ta_id IS NULL
      THEN 'p2 missing tennisabstract_elo mapping'
    WHEN s1.player_id IS NULL AND s2.player_id IS NULL
      THEN 'TA ids exist but not found in TA snapshot for as_of_date'
    WHEN s1.player_id IS NULL
      THEN 'p1 TA id not found in TA snapshot for as_of_date'
    WHEN s2.player_id IS NULL
      THEN 'p2 TA id not found in TA snapshot for as_of_date'
    ELSE NULL
  END AS missing_reason

FROM mapped m
LEFT JOIN med
  ON med.tour = m.tour
LEFT JOIN snap s1
  ON upper(s1.tour) = m.tour AND s1.player_id = m.p1_ta_id
LEFT JOIN snap s2
  ON upper(s2.tour) = m.tour AND s2.player_id = m.p2_ta_id
ORDER BY m.tour, m.tournament, m.p1_name, m.p2_name;
""")


async def fetch_match_features_by_date(
    conn: AsyncConnection,
    match_date: date,
) -> List[Dict[str, Any]]:
    """
    Returns ALL ATP/WTA matches for the date from tennis_matches,
    with Elo fields (overall + hard/clay/grass) + missing_reason.
    Optimized to avoid per-row correlated subqueries.
    """
    # Ensure unaccent exists (safe to run; if already created, no-op)
    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent;"))

    res = await conn.execute(GET_FEATURES_BY_DATE, {"match_date": match_date})
    return [dict(r) for r in res.mappings().all()]
