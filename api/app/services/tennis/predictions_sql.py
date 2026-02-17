from __future__ import annotations

from sqlalchemy import text

SOFASCORE_MATCHES_ELO_SQL = text(
    r"""
    WITH matches AS (
      SELECT
        m.match_id,
        m.match_key,
        m.match_date,
        m.match_start_utc,
        upper(m.tour) AS tour,
        m.tournament AS tournament,
        m."round" AS round,
        COALESCE(NULLIF(m.surface,''), 'unknown') AS surface,
        m.status,
        m.score_raw,
        m.p1_name AS p1_name,
        m.p2_name AS p2_name,
        m.p1_canonical_id AS p1_player_id,
        m.p2_canonical_id AS p2_player_id,
        m.p1_odds_american,
        m.p2_odds_american,
        m.odds_fetched_at,
        m.sofascore_total_games_line,
        m.sofascore_total_games_over_american,
        m.sofascore_total_games_under_american,
        m.sofascore_spread_p1_line,
        m.sofascore_spread_p2_line,
        m.sofascore_spread_p1_odds_american,
        m.sofascore_spread_p2_odds_american,
        CASE
          WHEN upper(m.tour) = 'ATP'
           AND (
                m.tournament ILIKE '%Australian Open%'
             OR m.tournament ILIKE '%Roland Garros%'
             OR m.tournament ILIKE '%French Open%'
             OR m.tournament ILIKE '%Wimbledon%'
             OR m.tournament ILIKE '%US Open%'
           )
           AND COALESCE(m."round",'') !~* 'qual'
          THEN 5
          ELSE 3
        END AS best_of
      FROM tennis_matches m
      WHERE m.match_date = ANY(:dates)
        AND upper(m.tour) IN ('ATP','WTA')
        AND (
          :include_incomplete = true
          OR COALESCE(lower(m.status),'') IN ('finished','completed','ended')
        )
    ),
    mapped AS (
      SELECT
        x.*,
        NULLIF(trim(ta1.source_player_id), '')::bigint AS p1_ta_id,
        NULLIF(trim(ta2.source_player_id), '')::bigint AS p2_ta_id
      FROM matches x
      LEFT JOIN tennis_player_sources ta1
        ON ta1.player_id = x.p1_player_id
       AND ta1.source = CASE
         WHEN x.tour = 'ATP' THEN 'tennisabstract_elo_atp'
         ELSE 'tennisabstract_elo_wta'
       END
      LEFT JOIN tennis_player_sources ta2
        ON ta2.player_id = x.p2_player_id
       AND ta2.source = CASE
         WHEN x.tour = 'ATP' THEN 'tennisabstract_elo_atp'
         ELSE 'tennisabstract_elo_wta'
       END
    ),
    fatigue AS (
      SELECT
        m.*,

        p1_last.last_match_date AS p1_last_match_date,
        p1_last.rest_days       AS p1_rest_days,
        p1_last.went_distance   AS p1_last_went_distance,

        p1_wl.matches_10d       AS p1_matches_10d,
        p1_wl.sets_10d          AS p1_sets_10d,

        p2_last.last_match_date AS p2_last_match_date,
        p2_last.rest_days       AS p2_rest_days,
        p2_last.went_distance   AS p2_last_went_distance,

        p2_wl.matches_10d       AS p2_matches_10d,
        p2_wl.sets_10d          AS p2_sets_10d

      FROM mapped m

      LEFT JOIN LATERAL (
        SELECT
          pm.match_date AS last_match_date,
          (m.match_date - pm.match_date)::int AS rest_days,
          (sp.sets_played = bo.best_of_pm) AS went_distance
        FROM tennis_matches pm
        CROSS JOIN LATERAL (
          SELECT
            CASE
              WHEN COALESCE(pm.score_raw,'') ~ '^\s*\d+\s*-\s*\d+\s*$' THEN
                split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 1)::int +
                split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 2)::int
              ELSE
                (SELECT count(*)
                 FROM unnest(regexp_split_to_array(COALESCE(pm.score_raw,''), '\s+')) t(tok)
                 WHERE regexp_replace(tok, '\(.*\)', '', 'g') ~ '^\d+\-\d+$'
                )
            END AS sets_played
        ) sp
        CROSS JOIN LATERAL (
          SELECT
            CASE
              WHEN upper(pm.tour) = 'ATP'
               AND (
                    pm.tournament ILIKE '%Australian Open%'
                 OR pm.tournament ILIKE '%Roland Garros%'
                 OR pm.tournament ILIKE '%French Open%'
                 OR pm.tournament ILIKE '%Wimbledon%'
                 OR pm.tournament ILIKE '%US Open%'
               )
               AND COALESCE(pm."round",'') !~* 'qual'
              THEN 5 ELSE 3
            END AS best_of_pm
        ) bo
        WHERE m.p1_player_id IS NOT NULL
          AND pm.match_id <> m.match_id
          AND (pm.p1_canonical_id = m.p1_player_id OR pm.p2_canonical_id = m.p1_player_id)
          AND pm.match_date <= m.match_date
          AND COALESCE(lower(pm.status),'') IN ('finished','completed','ended')
          AND COALESCE(pm.score_raw,'') <> ''
        ORDER BY pm.match_date DESC
        LIMIT 1
      ) p1_last ON TRUE

      LEFT JOIN LATERAL (
        SELECT
          count(*)::int AS matches_10d,
          COALESCE(sum(sp.sets_played), 0)::int AS sets_10d
        FROM tennis_matches pm
        CROSS JOIN LATERAL (
          SELECT
            CASE
              WHEN COALESCE(pm.score_raw,'') ~ '^\s*\d+\s*-\s*\d+\s*$' THEN
                split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 1)::int +
                split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 2)::int
              ELSE
                (SELECT count(*)
                 FROM unnest(regexp_split_to_array(COALESCE(pm.score_raw,''), '\s+')) t(tok)
                 WHERE regexp_replace(tok, '\(.*\)', '', 'g') ~ '^\d+\-\d+$'
                )
            END AS sets_played
        ) sp
        WHERE m.p1_player_id IS NOT NULL
          AND pm.match_id <> m.match_id
          AND (pm.p1_canonical_id = m.p1_player_id OR pm.p2_canonical_id = m.p1_player_id)
          AND pm.match_date >= (m.match_date - interval '10 days')
          AND pm.match_date <= m.match_date
          AND COALESCE(lower(pm.status),'') IN ('finished','completed','ended')
          AND COALESCE(pm.score_raw,'') <> ''
      ) p1_wl ON TRUE

      LEFT JOIN LATERAL (
        SELECT
          pm.match_date AS last_match_date,
          (m.match_date - pm.match_date)::int AS rest_days,
          (sp.sets_played = bo.best_of_pm) AS went_distance
        FROM tennis_matches pm
        CROSS JOIN LATERAL (
          SELECT
            CASE
              WHEN COALESCE(pm.score_raw,'') ~ '^\s*\d+\s*-\s*\d+\s*$' THEN
                split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 1)::int +
                split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 2)::int
              ELSE
                (SELECT count(*)
                 FROM unnest(regexp_split_to_array(COALESCE(pm.score_raw,''), '\s+')) t(tok)
                 WHERE regexp_replace(tok, '\(.*\)', '', 'g') ~ '^\d+\-\d+$'
                )
            END AS sets_played
        ) sp
        CROSS JOIN LATERAL (
          SELECT
            CASE
              WHEN upper(pm.tour) = 'ATP'
               AND (
                    pm.tournament ILIKE '%Australian Open%'
                 OR pm.tournament ILIKE '%Roland Garros%'
                 OR pm.tournament ILIKE '%French Open%'
                 OR pm.tournament ILIKE '%Wimbledon%'
                 OR pm.tournament ILIKE '%US Open%'
               )
               AND COALESCE(pm."round",'') !~* 'qual'
              THEN 5 ELSE 3
            END AS best_of_pm
        ) bo
        WHERE m.p2_player_id IS NOT NULL
          AND pm.match_id <> m.match_id
          AND (pm.p1_canonical_id = m.p2_player_id OR pm.p2_canonical_id = m.p2_player_id)
          AND pm.match_date <= m.match_date
          AND COALESCE(lower(pm.status),'') IN ('finished','completed','ended')
          AND COALESCE(pm.score_raw,'') <> ''
        ORDER BY pm.match_date DESC
        LIMIT 1
      ) p2_last ON TRUE

      LEFT JOIN LATERAL (
        SELECT
          count(*)::int AS matches_10d,
          COALESCE(sum(sp.sets_played), 0)::int AS sets_10d
        FROM tennis_matches pm
        CROSS JOIN LATERAL (
          SELECT
            CASE
              WHEN COALESCE(pm.score_raw,'') ~ '^\s*\d+\s*-\s*\d+\s*$' THEN
                split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 1)::int +
                split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 2)::int
              ELSE
                (SELECT count(*)
                 FROM unnest(regexp_split_to_array(COALESCE(pm.score_raw,''), '\s+')) t(tok)
                 WHERE regexp_replace(tok, '\(.*\)', '', 'g') ~ '^\d+\-\d+$'
                )
            END AS sets_played
        ) sp
        WHERE m.p2_player_id IS NOT NULL
          AND pm.match_id <> m.match_id
          AND (pm.p1_canonical_id = m.p2_player_id OR pm.p2_canonical_id = m.p2_player_id)
          AND pm.match_date >= (m.match_date - interval '10 days')
          AND pm.match_date <= m.match_date
          AND COALESCE(lower(pm.status),'') IN ('finished','completed','ended')
          AND COALESCE(pm.score_raw,'') <> ''
      ) p2_wl ON TRUE
    ),
    asof AS (
      SELECT
        d.match_date,
        d.tour,
        max(s.as_of_date) AS as_of_date
      FROM (SELECT DISTINCT match_date, tour FROM fatigue) d
      JOIN tennisabstract_elo_snapshots s
        ON upper(s.tour) = d.tour
       AND s.as_of_date <= d.match_date
      GROUP BY d.match_date, d.tour
    ),
    snap AS (
      SELECT DISTINCT ON (upper(s.tour), s.as_of_date, s.player_id)
        upper(s.tour) AS tour,
        s.as_of_date,
        s.player_id,
        s.elo,
        s.helo,
        s.celo,
        s.gelo,
        s.created_at,
        s.player_name
      FROM tennisabstract_elo_snapshots s
      JOIN asof a
        ON upper(s.tour) = a.tour
       AND s.as_of_date = a.as_of_date
      WHERE s.player_id IS NOT NULL
      ORDER BY
        upper(s.tour),
        s.as_of_date,
        s.player_id,
        (s.elo  IS NOT NULL) DESC,
        (s.helo IS NOT NULL) DESC,
        (s.celo IS NOT NULL) DESC,
        (s.gelo IS NOT NULL) DESC,
        s.created_at DESC,
        s.player_name ASC
    ),
    med AS (
      SELECT
        tour,
        as_of_date,
        (percentile_cont(0.5) WITHIN GROUP (ORDER BY elo::double precision))::float8  AS med_elo,
        (percentile_cont(0.5) WITHIN GROUP (ORDER BY helo::double precision))::float8 AS med_helo,
        (percentile_cont(0.5) WITHIN GROUP (ORDER BY celo::double precision))::float8 AS med_celo,
        (percentile_cont(0.5) WITHIN GROUP (ORDER BY gelo::double precision))::float8 AS med_gelo
      FROM snap
      GROUP BY 1,2
    ),
   final_rows AS (
  SELECT
    m.*,

    -- ✅ ONLY columns that actually exist in tennis_insight_match_features_final
    ti.serve_return_edge_w::float8    AS serve_return_edge_w,
    ti.d_srv_pts_w_w::float8          AS d_srv_pts_w_w,
    ti.d_ret_pts_w_w::float8          AS d_ret_pts_w_w,
    ti.d_hold_w::float8               AS d_hold_w,
    ti.missing_any                    AS missing_any,
    ti.profile_weight::float8         AS profile_weight,

    a.as_of_date,

    s1.elo::float8  AS p1_elo_raw,
    s1.helo::float8 AS p1_helo_raw,
    s1.celo::float8 AS p1_celo_raw,
    s1.gelo::float8 AS p1_gelo_raw,

    s2.elo::float8  AS p2_elo_raw,
    s2.helo::float8 AS p2_helo_raw,
    s2.celo::float8 AS p2_celo_raw,
    s2.gelo::float8 AS p2_gelo_raw,

    md.med_elo,
    md.med_helo,
    md.med_celo,
    md.med_gelo

  FROM fatigue m
  LEFT JOIN public.tennis_insight_match_features_final ti
    ON ti.match_id = m.match_id
  LEFT JOIN asof a
    ON a.match_date = m.match_date AND a.tour = m.tour
  LEFT JOIN med md
    ON md.tour = m.tour AND md.as_of_date = a.as_of_date
  LEFT JOIN snap s1
    ON s1.tour = m.tour
   AND s1.as_of_date = a.as_of_date
   AND s1.player_id = m.p1_ta_id
  LEFT JOIN snap s2
    ON s2.tour = m.tour
   AND s2.as_of_date = a.as_of_date
   AND s2.player_id = m.p2_ta_id
),

    dedup AS (
      SELECT DISTINCT ON (match_id) *
      FROM final_rows
      ORDER BY
        match_id,
        (p1_player_id IS NOT NULL AND p2_player_id IS NOT NULL) DESC,
        (as_of_date IS NOT NULL) DESC,
        (p1_ta_id IS NOT NULL AND p2_ta_id IS NOT NULL) DESC,
        (p1_elo_raw IS NOT NULL AND p2_elo_raw IS NOT NULL) DESC
    )
    SELECT *
    FROM dedup
    ORDER BY match_date, tour, tournament, p1_name;
    """
)
