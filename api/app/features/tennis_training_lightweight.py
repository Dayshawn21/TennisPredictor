# FILE: api/app/features/tennis_training_lightweight.py
"""
Extract training features for lightweight XGBoost model.
Uses existing tennisabstract_elo_snapshots table (no backfill needed).
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import logging
import csv
from datetime import date, timedelta
from typing import List
import asyncpg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable not set")

# Fix asyncpg scheme requirement
if DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

OUTPUT_CSV = "tennis_training_lightweight.csv"


async def fetch_training_data(conn) -> List[dict]:
    """
    Fetch finished matches and join with closest ELO snapshot.
    We'll use a window to find the most recent ELO snapshot before each match.
    """
    query = """
    WITH matches AS (
      SELECT
        m.match_id,
        m.match_date,
        m.tour,
        m.surface,
        m.tournament,
        m.round,
        m.p1_name,
        m.p2_name,
        m.p1_canonical_id,
        m.p2_canonical_id,
        m.p1_sofascore_player_id,
        m.p2_sofascore_player_id,
        -- Determine winner from score
        CASE
          WHEN m.score ~ '^[0-9]+-[0-9]+$' THEN
            CASE
              WHEN CAST(split_part(m.score, '-', 1) AS int) >
                   CAST(split_part(m.score, '-', 2) AS int)
              THEN true
              ELSE false
            END
          -- For set scores like "6-4 7-5", count sets won
          WHEN m.score_raw IS NOT NULL THEN
            (SELECT COUNT(*) FILTER (
              WHERE regexp_replace(tok, '\\(.*\\)', '', 'g') ~ '^\\d+\\-\\d+$'
                AND split_part(regexp_replace(tok, '\\(.*\\)', '', 'g'), '-', 1)::int >
                    split_part(regexp_replace(tok, '\\(.*\\)', '', 'g'), '-', 2)::int
            ) > COUNT(*) / 2
            FROM unnest(regexp_split_to_array(m.score_raw, '\\s+')) t(tok))
          ELSE NULL
        END AS p1_won
      FROM tennis_matches m
      WHERE m.status IN ('finished', 'completed', 'ended')
        AND m.match_date >= '2020-01-01'
        AND m.match_date < CURRENT_DATE
        AND m.tour IN ('ATP', 'WTA')
        AND (m.score IS NOT NULL OR m.score_raw IS NOT NULL)
    ),
    matches_with_elo AS (
      SELECT
        m.*,
        
        -- P1 ELO (most recent snapshot before match date)
        (SELECT s.elo
         FROM tennisabstract_elo_snapshots s
         JOIN tennis_player_sources tps 
           ON tps.source_player_id = s.player_name
           AND tps.source = CASE WHEN m.tour = 'ATP' 
                                THEN 'tennisabstract_elo_atp'
                                ELSE 'tennisabstract_elo_wta' END
         WHERE s.tour = m.tour
           AND s.as_of_date <= m.match_date
           AND tps.player_id = m.p1_canonical_id
         ORDER BY s.as_of_date DESC
         LIMIT 1
        ) AS p1_elo,
        
        (SELECT s.helo
         FROM tennisabstract_elo_snapshots s
         JOIN tennis_player_sources tps 
           ON tps.source_player_id = s.player_name
           AND tps.source = CASE WHEN m.tour = 'ATP' 
                                THEN 'tennisabstract_elo_atp'
                                ELSE 'tennisabstract_elo_wta' END
         WHERE s.tour = m.tour
           AND s.as_of_date <= m.match_date
           AND tps.player_id = m.p1_canonical_id
         ORDER BY s.as_of_date DESC
         LIMIT 1
        ) AS p1_helo,
        
        (SELECT s.celo
         FROM tennisabstract_elo_snapshots s
         JOIN tennis_player_sources tps 
           ON tps.source_player_id = s.player_name
           AND tps.source = CASE WHEN m.tour = 'ATP' 
                                THEN 'tennisabstract_elo_atp'
                                ELSE 'tennisabstract_elo_wta' END
         WHERE s.tour = m.tour
           AND s.as_of_date <= m.match_date
           AND tps.player_id = m.p1_canonical_id
         ORDER BY s.as_of_date DESC
         LIMIT 1
        ) AS p1_celo,
        
        (SELECT s.gelo
         FROM tennisabstract_elo_snapshots s
         JOIN tennis_player_sources tps 
           ON tps.source_player_id = s.player_name
           AND tps.source = CASE WHEN m.tour = 'ATP' 
                                THEN 'tennisabstract_elo_atp'
                                ELSE 'tennisabstract_elo_wta' END
         WHERE s.tour = m.tour
           AND s.as_of_date <= m.match_date
           AND tps.player_id = m.p1_canonical_id
         ORDER BY s.as_of_date DESC
         LIMIT 1
        ) AS p1_gelo,
        
        (SELECT s.elo_rank
         FROM tennisabstract_elo_snapshots s
         JOIN tennis_player_sources tps 
           ON tps.source_player_id = s.player_name
           AND tps.source = CASE WHEN m.tour = 'ATP' 
                                THEN 'tennisabstract_elo_atp'
                                ELSE 'tennisabstract_elo_wta' END
         WHERE s.tour = m.tour
           AND s.as_of_date <= m.match_date
           AND tps.player_id = m.p1_canonical_id
         ORDER BY s.as_of_date DESC
         LIMIT 1
        ) AS p1_elo_rank,
        
        (SELECT s.age
         FROM tennisabstract_elo_snapshots s
         JOIN tennis_player_sources tps 
           ON tps.source_player_id = s.player_name
           AND tps.source = CASE WHEN m.tour = 'ATP' 
                                THEN 'tennisabstract_elo_atp'
                                ELSE 'tennisabstract_elo_wta' END
         WHERE s.tour = m.tour
           AND s.as_of_date <= m.match_date
           AND tps.player_id = m.p1_canonical_id
         ORDER BY s.as_of_date DESC
         LIMIT 1
        ) AS p1_age,
        
        -- P2 ELO (same pattern)
        (SELECT s.elo FROM tennisabstract_elo_snapshots s
         JOIN tennis_player_sources tps ON tps.source_player_id = s.player_name
           AND tps.source = CASE WHEN m.tour = 'ATP' THEN 'tennisabstract_elo_atp' ELSE 'tennisabstract_elo_wta' END
         WHERE s.tour = m.tour AND s.as_of_date <= m.match_date AND tps.player_id = m.p2_canonical_id
         ORDER BY s.as_of_date DESC LIMIT 1) AS p2_elo,
        
        (SELECT s.helo FROM tennisabstract_elo_snapshots s
         JOIN tennis_player_sources tps ON tps.source_player_id = s.player_name
           AND tps.source = CASE WHEN m.tour = 'ATP' THEN 'tennisabstract_elo_atp' ELSE 'tennisabstract_elo_wta' END
         WHERE s.tour = m.tour AND s.as_of_date <= m.match_date AND tps.player_id = m.p2_canonical_id
         ORDER BY s.as_of_date DESC LIMIT 1) AS p2_helo,
        
        (SELECT s.celo FROM tennisabstract_elo_snapshots s
         JOIN tennis_player_sources tps ON tps.source_player_id = s.player_name
           AND tps.source = CASE WHEN m.tour = 'ATP' THEN 'tennisabstract_elo_atp' ELSE 'tennisabstract_elo_wta' END
         WHERE s.tour = m.tour AND s.as_of_date <= m.match_date AND tps.player_id = m.p2_canonical_id
         ORDER BY s.as_of_date DESC LIMIT 1) AS p2_celo,
        
        (SELECT s.gelo FROM tennisabstract_elo_snapshots s
         JOIN tennis_player_sources tps ON tps.source_player_id = s.player_name
           AND tps.source = CASE WHEN m.tour = 'ATP' THEN 'tennisabstract_elo_atp' ELSE 'tennisabstract_elo_wta' END
         WHERE s.tour = m.tour AND s.as_of_date <= m.match_date AND tps.player_id = m.p2_canonical_id
         ORDER BY s.as_of_date DESC LIMIT 1) AS p2_gelo,
        
        (SELECT s.elo_rank FROM tennisabstract_elo_snapshots s
         JOIN tennis_player_sources tps ON tps.source_player_id = s.player_name
           AND tps.source = CASE WHEN m.tour = 'ATP' THEN 'tennisabstract_elo_atp' ELSE 'tennisabstract_elo_wta' END
         WHERE s.tour = m.tour AND s.as_of_date <= m.match_date AND tps.player_id = m.p2_canonical_id
         ORDER BY s.as_of_date DESC LIMIT 1) AS p2_elo_rank,
        
        (SELECT s.age FROM tennisabstract_elo_snapshots s
         JOIN tennis_player_sources tps ON tps.source_player_id = s.player_name
           AND tps.source = CASE WHEN m.tour = 'ATP' THEN 'tennisabstract_elo_atp' ELSE 'tennisabstract_elo_wta' END
         WHERE s.tour = m.tour AND s.as_of_date <= m.match_date AND tps.player_id = m.p2_canonical_id
         ORDER BY s.as_of_date DESC LIMIT 1) AS p2_age
      
      FROM matches m
      WHERE m.p1_won IS NOT NULL
        AND m.p1_canonical_id IS NOT NULL
        AND m.p2_canonical_id IS NOT NULL
    )
    SELECT * FROM matches_with_elo
    WHERE p1_elo IS NOT NULL AND p2_elo IS NOT NULL
    ORDER BY match_date
    LIMIT 100000;
    """
    
    rows = await conn.fetch(query)
    logger.info("Fetched %d matches with ELO data", len(rows))
    
    return [dict(r) for r in rows]


def engineer_features(row: dict) -> dict:
    """Build feature dict for XGBoost training."""
    
    # Determine best_of
    tour = (row["tour"] or "").upper()
    tournament = (row["tournament"] or "").lower()
    is_slam = any(k in tournament for k in [
        "australian open", "french open", "roland garros",
        "wimbledon", "us open"
    ])
    best_of = 5 if (is_slam and tour == "ATP") else 3
    
    # Surface encoding
    surface = (row["surface"] or "").lower()
    is_hard = 1 if "hard" in surface else 0
    is_clay = 1 if "clay" in surface else 0
    is_grass = 1 if "grass" in surface else 0
    
    # Tournament tier
    is_grand_slam = 1 if is_slam else 0
    is_masters = 1 if "masters" in tournament else 0
    
    # ELO features
    p1_elo = float(row["p1_elo"]) if row["p1_elo"] else 1500.0
    p2_elo = float(row["p2_elo"]) if row["p2_elo"] else 1500.0
    d_elo = p1_elo - p2_elo
    
    # Surface-specific ELO
    if is_hard:
        p1_surf_elo = float(row["p1_helo"]) if row["p1_helo"] else p1_elo
        p2_surf_elo = float(row["p2_helo"]) if row["p2_helo"] else p2_elo
    elif is_clay:
        p1_surf_elo = float(row["p1_celo"]) if row["p1_celo"] else p1_elo
        p2_surf_elo = float(row["p2_celo"]) if row["p2_celo"] else p2_elo
    elif is_grass:
        p1_surf_elo = float(row["p1_gelo"]) if row["p1_gelo"] else p1_elo
        p2_surf_elo = float(row["p2_gelo"]) if row["p2_gelo"] else p2_elo
    else:
        p1_surf_elo = p1_elo
        p2_surf_elo = p2_elo
    
    d_surface_elo = p1_surf_elo - p2_surf_elo
    
    # Surface specialization (how much better on this surface vs overall)
    p1_surf_spec = p1_surf_elo - p1_elo
    p2_surf_spec = p2_surf_elo - p2_elo
    d_surf_spec = p1_surf_spec - p2_surf_spec
    
    # Rankings
    p1_elo_rank = int(row["p1_elo_rank"]) if row["p1_elo_rank"] else 100
    p2_elo_rank = int(row["p2_elo_rank"]) if row["p2_elo_rank"] else 100
    d_elo_rank = p2_elo_rank - p1_elo_rank  # Lower rank = better
    
    # Age
    p1_age = float(row["p1_age"]) if row["p1_age"] else 27.0
    p2_age = float(row["p2_age"]) if row["p2_age"] else 27.0
    d_age = p1_age - p2_age
    
    return {
        "match_id": str(row["match_id"]),
        "match_date": str(row["match_date"]),
        "tour": row["tour"],
        "surface": surface,
        "p1_name": row["p1_name"],
        "p2_name": row["p2_name"],
        
        # Label
        "y": 1 if row["p1_won"] else 0,
        
        # Features
        "d_elo": d_elo,
        "d_surface_elo": d_surface_elo,
        "d_surf_spec": d_surf_spec,
        "d_elo_rank": d_elo_rank,
        "d_age": d_age,
        "is_hard": is_hard,
        "is_clay": is_clay,
        "is_grass": is_grass,
        "is_grand_slam": is_grand_slam,
        "is_masters": is_masters,
        "best_of": best_of,
    }


async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    
    try:
        logger.info("Fetching training data from database...")
        raw_matches = await fetch_training_data(conn)
        
        logger.info("Engineering features...")
        rows = []
        for m in raw_matches:
            try:
                row = engineer_features(m)
                rows.append(row)
            except Exception as e:
                logger.warning("Failed to engineer features for match %s: %s", 
                              m.get("match_id"), e)
                continue
        
        logger.info("Engineered %d feature rows", len(rows))
        
        # Write to CSV
        if rows:
            fieldnames = list(rows[0].keys())
            with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info("✅ Wrote %d rows to %s", len(rows), OUTPUT_CSV)
            
            # Print stats
            df_y = pd.DataFrame(rows)
            logger.info("Label distribution: %s", df_y["y"].value_counts().to_dict())
            logger.info("Date range: %s to %s", df_y["match_date"].min(), df_y["match_date"].max())
            logger.info("Tours: %s", df_y["tour"].value_counts().to_dict())
        else:
            logger.warning("No rows to write")
    
    except Exception as e:
        logger.error("Error in main: %s", e, exc_info=True)
        raise
    finally:
        await conn.close()


if __name__ == "__main__":
    import pandas as pd
    asyncio.run(main())