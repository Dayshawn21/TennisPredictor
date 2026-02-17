-- ============================================================
--  ELO MISSING DIAGNOSTIC
--  Run this in your Postgres client (pgAdmin, DBeaver, psql)
--  Change the date on line 12 to the match day you want to check
-- ============================================================

-- ① Set your target date here
WITH params AS (
  SELECT
    CURRENT_DATE        AS match_date,   -- ← change me if needed
    ARRAY['ATP','WTA']  AS tours
),

-- ② Collect every player side from today's matches
players AS (
  SELECT DISTINCT ON (cid, name, tour)
    cid, name, tour, side, match_id, opponent
  FROM (
    SELECT
      m.p1_canonical_id AS cid,
      m.p1_name         AS name,
      upper(m.tour)     AS tour,
      'P1'              AS side,
      m.match_id,
      m.p2_name         AS opponent
    FROM tennis_matches m, params p
    WHERE m.match_date = p.match_date
      AND upper(m.tour) = ANY(p.tours)
    UNION ALL
    SELECT
      m.p2_canonical_id,
      m.p2_name,
      upper(m.tour),
      'P2',
      m.match_id,
      m.p1_name
    FROM tennis_matches m, params p
    WHERE m.match_date = p.match_date
      AND upper(m.tour) = ANY(p.tours)
  ) raw
  ORDER BY cid, name, tour, side
),

-- ③ Look up each player's TA source mapping + elo_latest
chain AS (
  SELECT
    pl.name,
    pl.tour,
    pl.cid                       AS canonical_id,
    tp.name                      AS tp_name,
    tps.source_player_id         AS ta_id,
    el.elo                       AS elo_latest,
    el.player_name               AS ta_name,
    -- Try to find the REAL TA player by matching sofascore name
    sf.source_name               AS sofascore_name,
    fix_el.player_id             AS fixable_ta_id,
    fix_el.elo                   AS fixable_elo,
    fix_el.player_name           AS fixable_ta_name
  FROM players pl

  -- canonical player record
  LEFT JOIN tennis_players tp
    ON tp.id = pl.cid

  -- TA source mapping
  LEFT JOIN tennis_player_sources tps
    ON tps.player_id = pl.cid
    AND tps.source = CASE WHEN pl.tour = 'ATP'
                          THEN 'tennisabstract_elo_atp'
                          ELSE 'tennisabstract_elo_wta' END

  -- does the TA id actually resolve?
  LEFT JOIN tennisabstract_elo_latest el
    ON trim(tps.source_player_id) ~ '^[0-9]+$'
    AND el.player_id = tps.source_player_id::bigint
    AND upper(el.tour) = pl.tour

  -- sofascore name (for repair hints)
  LEFT JOIN tennis_player_sources sf
    ON sf.player_id = pl.cid
    AND sf.source = 'sofascore'

  -- can we find the real TA entry by name?
  LEFT JOIN tennisabstract_elo_latest fix_el
    ON el.player_id IS NULL              -- only when broken
    AND sf.source_name IS NOT NULL
    AND lower(trim(regexp_replace(
          unaccent(replace(fix_el.player_name, chr(160), ' ')),
          '\s+', ' ', 'g')))
      = lower(trim(regexp_replace(
          unaccent(sf.source_name),
          '\s+', ' ', 'g')))
    AND upper(fix_el.tour) = pl.tour
),

-- ④ Classify each player
classified AS (
  SELECT
    *,
    CASE
      WHEN canonical_id IS NULL          THEN '1_NO_CANONICAL'
      WHEN ta_id IS NULL                 THEN '2_NO_TA_SOURCE'
      WHEN elo_latest IS NULL
       AND fixable_ta_id IS NOT NULL     THEN '3_WRONG_TA_ID (auto-fixable)'
      WHEN elo_latest IS NULL            THEN '4_TA_ID_NOT_IN_LATEST'
      ELSE                                    '5_OK'
    END AS status
  FROM chain
)

-- ⑤ Output — broken players first, then OK
SELECT
  status,
  name                                   AS match_name,
  tour,
  canonical_id                           AS cid,
  tp_name                                AS tennis_players_name,
  ta_id,
  round(elo_latest::numeric, 1)          AS elo,
  sofascore_name                         AS sofa_name,
  fixable_ta_id                          AS fix_ta_id,
  round(fixable_elo::numeric, 1)         AS fix_elo,
  fixable_ta_name                        AS fix_ta_name
FROM classified
ORDER BY
  status,                -- broken first
  tour,
  name;
