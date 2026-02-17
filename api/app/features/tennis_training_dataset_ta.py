from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import logging
import csv
from typing import List

import asyncpg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_3bxYFijyoeD4@ep-dawn-wave-a8928yr9-pooler.eastus2.azure.neon.tech/neondb?sslmode=require",
)

OUTPUT_CSV = "tennis_training_ta.csv"


async def fetch_feature_rows(conn) -> List[asyncpg.Record]:
    """
    Grab feature rows where we have core last-10 hold/break features.
    Aces / DFs / tiebreak fields can be NULL; we’ll keep them as-is.
    """
    rows = await conn.fetch(
        """
        SELECT
            match_id,
            match_date,
            tour,
            surface,
            p1_name,
            p2_name,
            p1_won,

            -- Core hold/break
            p1_last5_hold_pct,
            p1_last5_break_pct,
            p1_last10_hold_pct,
            p1_last10_break_pct,

            p2_last5_hold_pct,
            p2_last5_break_pct,
            p2_last10_hold_pct,
            p2_last10_break_pct,

            p1_surface_last10_hold_pct,
            p1_surface_last10_break_pct,
            p2_surface_last10_hold_pct,
            p2_surface_last10_break_pct,

            -- Aces per service game
            p1_last10_aces_per_svc_game,
            p2_last10_aces_per_svc_game,
            p1_surface_last10_aces_per_svc_game,
            p2_surface_last10_aces_per_svc_game,

            -- Double faults per service game
            p1_last10_df_per_svc_game,
            p2_last10_df_per_svc_game,
            p1_surface_last10_df_per_svc_game,
            p2_surface_last10_df_per_svc_game,

            -- Tiebreak features (overall)
            p1_last10_tb_match_rate,
            p1_last10_tb_win_pct,
            p2_last10_tb_match_rate,
            p2_last10_tb_win_pct,

            -- Tiebreak features (surface-specific)
            p1_surface_last10_tb_match_rate,
            p1_surface_last10_tb_win_pct,
            p2_surface_last10_tb_match_rate,
            p2_surface_last10_tb_win_pct

        FROM tennis_features_ta
        WHERE
            p1_last10_hold_pct IS NOT NULL
            AND p1_last10_break_pct IS NOT NULL
            AND p2_last10_hold_pct IS NOT NULL
            AND p2_last10_break_pct IS NOT NULL
        """
    )
    logger.info("Fetched %d feature rows for training", len(rows))
    return rows


def build_feature_row(r: asyncpg.Record) -> dict:
    """
    Build ML-friendly feature dict:
      - label: y (1 if p1 wins – p1 is always winner in our pipeline)
      - features: mostly (p1 - p2) differences
    """
    y = 1 if r["p1_won"] else 0

    def diff(a, b):
        if a is None or b is None:
            return None
        return float(a) - float(b)

    row = {
        # --- Meta (good for debugging / grouping) ---
        "match_id": str(r["match_id"]),
        "match_date": str(r["match_date"]),
        "tour": r["tour"],
        "surface": r["surface"] or "",
        "p1_name": r["p1_name"],
        "p2_name": r["p2_name"],

        # --- Label ---
        "y": y,

        # --- Core difference features (hold/break) ---
        "d_last5_hold": diff(r["p1_last5_hold_pct"], r["p2_last5_hold_pct"]),
        "d_last5_break": diff(r["p1_last5_break_pct"], r["p2_last5_break_pct"]),
        "d_last10_hold": diff(r["p1_last10_hold_pct"], r["p2_last10_hold_pct"]),
        "d_last10_break": diff(r["p1_last10_break_pct"], r["p2_last10_break_pct"]),

        "d_surf_last10_hold": diff(
            r["p1_surface_last10_hold_pct"], r["p2_surface_last10_hold_pct"]
        ),
        "d_surf_last10_break": diff(
            r["p1_surface_last10_break_pct"], r["p2_surface_last10_break_pct"]
        ),

        # --- Aces per service game (overall + surface, diffs) ---
        "d_last10_aces_pg": diff(
            r["p1_last10_aces_per_svc_game"], r["p2_last10_aces_per_svc_game"]
        ),
        "d_surf_last10_aces_pg": diff(
            r["p1_surface_last10_aces_per_svc_game"],
            r["p2_surface_last10_aces_per_svc_game"],
        ),

        # --- Double faults per service game (overall + surface, diffs) ---
        "d_last10_df_pg": diff(
            r["p1_last10_df_per_svc_game"], r["p2_last10_df_per_svc_game"]
        ),
        "d_surf_last10_df_pg": diff(
            r["p1_surface_last10_df_per_svc_game"],
            r["p2_surface_last10_df_per_svc_game"],
        ),

        # --- Tiebreak features (overall diffs) ---
        "d_last10_tb_match_rate": diff(
            r["p1_last10_tb_match_rate"], r["p2_last10_tb_match_rate"]
        ),
        "d_last10_tb_win_pct": diff(
            r["p1_last10_tb_win_pct"], r["p2_last10_tb_win_pct"]
        ),

        # --- Tiebreak features (surface-specific diffs) ---
        "d_surf_last10_tb_match_rate": diff(
            r["p1_surface_last10_tb_match_rate"],
            r["p2_surface_last10_tb_match_rate"],
        ),
        "d_surf_last10_tb_win_pct": diff(
            r["p1_surface_last10_tb_win_pct"],
            r["p2_surface_last10_tb_win_pct"],
        ),

        # --- Optional raw features (for debugging / extra models) ---

        # Hold/break raw
        "p1_last10_hold": r["p1_last10_hold_pct"],
        "p1_last10_break": r["p1_last10_break_pct"],
        "p2_last10_hold": r["p2_last10_hold_pct"],
        "p2_last10_break": r["p2_last10_break_pct"],

        # Aces raw
        "p1_last10_aces_pg": r["p1_last10_aces_per_svc_game"],
        "p2_last10_aces_pg": r["p2_last10_aces_per_svc_game"],
        "p1_surf_last10_aces_pg": r["p1_surface_last10_aces_per_svc_game"],
        "p2_surf_last10_aces_pg": r["p2_surface_last10_aces_per_svc_game"],

        # Double faults raw
        "p1_last10_df_pg": r["p1_last10_df_per_svc_game"],
        "p2_last10_df_pg": r["p2_last10_df_per_svc_game"],
        "p1_surf_last10_df_pg": r["p1_surface_last10_df_per_svc_game"],
        "p2_surf_last10_df_pg": r["p2_surface_last10_df_per_svc_game"],

        # Tiebreak raw
        "p1_last10_tb_match_rate_raw": r["p1_last10_tb_match_rate"],
        "p1_last10_tb_win_raw": r["p1_last10_tb_win_pct"],
        "p2_last10_tb_match_rate_raw": r["p2_last10_tb_match_rate"],
        "p2_last10_tb_win_raw": r["p2_last10_tb_win_pct"],

        "p1_surf_last10_tb_match_rate_raw": r["p1_surface_last10_tb_match_rate"],
        "p1_surf_last10_tb_win_surf_raw": r["p1_surface_last10_tb_win_pct"],
        "p2_surf_last10_tb_match_rate_raw": r["p2_surface_last10_tb_match_rate"],
        "p2_surf_last10_tb_win_surf_raw": r["p2_surface_last10_tb_win_pct"],
    }

    return row


async def build_training_csv() -> None:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")

    conn = await asyncpg.connect(DATABASE_URL)

    try:
        rows = await fetch_feature_rows(conn)
        if not rows:
            logger.warning("No feature rows found – did you run tennis_feature_builder_ta?")
            return

        feature_rows = [build_feature_row(r) for r in rows]

        fieldnames = list(feature_rows[0].keys())

        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(feature_rows)

        logger.info("Wrote %d rows to %s", len(feature_rows), OUTPUT_CSV)

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(build_training_csv())
