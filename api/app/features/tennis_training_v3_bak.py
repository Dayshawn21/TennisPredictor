# FILE: api/app/features/tennis_training_v3.py
"""
Build the V3 training CSV — the most comprehensive feature set.

Merges:
  - Simple rolling features  (from tennis_training_simple.csv)
  - TA rolling features       (from tennis_training_ta.csv)
  - ELO / ranking / age       (from tennisabstract_elo_snapshots via DB)
  - Round encoding             (from tennis_matches via DB)

Output columns  (40 ML features + 6 meta):
  match_id, match_date, tour, p1_name, p2_name, y,
  --- ELO block (NEW) ---
  d_elo, d_surface_elo, d_rank_log, d_age,
  --- rolling form ---
  d_win_rate_last_20, d_win_rate_surface, d_experience,
  d_rest_days, d_matches_10d, d_matches_30d,
  --- context ---
  is_hard, is_clay, is_grass, is_grand_slam, is_masters, best_of,
  round_ordinal,
  --- H2H ---
  h2h_p1_win_pct, h2h_total_matches, h2h_surface_p1_win_pct, h2h_surface_matches,
  --- style proxies (restored) ---
  d_svc_pts_w, d_ret_pts_w, d_ace_pg, d_bp_save, d_bp_win,
  --- TA rolling ---
  d_last5_hold, d_last5_break, d_last10_hold, d_last10_break,
  d_surf_last10_hold, d_surf_last10_break,
  d_last10_aces_pg, d_surf_last10_aces_pg,
  d_last10_df_pg, d_surf_last10_df_pg,
  d_last10_tb_match_rate, d_last10_tb_win_pct,
  d_surf_last10_tb_match_rate, d_surf_last10_tb_win_pct

Usage:
  python -m app.features.tennis_training_v3
  python -m app.features.tennis_training_v3 --simple tennis_training_simple.csv --ta tennis_training_ta.csv --out tennis_training_v3.csv
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import pathlib
import re
import unicodedata
from typing import Dict, Optional, Tuple

import asyncpg
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_SIMPLE = PROJECT_ROOT / "tennis_training_simple.csv"
DEFAULT_TA = PROJECT_ROOT / "tennis_training_ta.csv"
DEFAULT_OUT = PROJECT_ROOT / "tennis_training_v3.csv"

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_3bxYFijyoeD4@ep-dawn-wave-a8928yr9-pooler.eastus2.azure.neon.tech/neondb?sslmode=require",
)
if DATABASE_URL and DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")


# ── Round ordinal encoding ──────────────────────────────────────────────────

ROUND_ORDER: Dict[str, int] = {
    "qualification": 1,
    "q1": 1, "q2": 2, "q3": 3,
    "round of 128": 4, "r128": 4,
    "round of 64": 5, "r64": 5,
    "round of 32": 6, "r32": 6,
    "round of 16": 7, "r16": 7,
    "quarterfinal": 8, "quarterfinals": 8, "qf": 8,
    "semifinal": 9, "semifinals": 9, "sf": 9,
    "final": 10, "f": 10,
    # Davis Cup / team events
    "round robin": 5,
    "group": 5,
}


def _round_ordinal(round_str: Optional[str]) -> int:
    """Encode round as ordinal integer (0 = unknown)."""
    if not round_str or not isinstance(round_str, str):
        return 0
    s = round_str.strip().lower()
    # Check direct match first
    if s in ROUND_ORDER:
        return ROUND_ORDER[s]
    # Fuzzy match
    for key, val in ROUND_ORDER.items():
        if key in s:
            return val
    return 0


def _norm_name(name: Optional[str]) -> str:
    if not name or not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _surface_key(surface: Optional[str]) -> str:
    s = (str(surface) if surface and isinstance(surface, str) else "").lower()
    if "clay" in s:
        return "clay"
    if "grass" in s:
        return "grass"
    return "hard"


# ── TA features we expect from the TA CSV ───────────────────────────────────

TA_FEATURES = [
    "d_last5_hold", "d_last5_break",
    "d_last10_hold", "d_last10_break",
    "d_surf_last10_hold", "d_surf_last10_break",
    "d_last10_aces_pg", "d_surf_last10_aces_pg",
    "d_last10_df_pg", "d_surf_last10_df_pg",
    "d_last10_tb_match_rate", "d_last10_tb_win_pct",
    "d_surf_last10_tb_match_rate", "d_surf_last10_tb_win_pct",
]

# Style proxy features from simple CSV
STYLE_FEATURES = [
    "d_svc_pts_w", "d_ret_pts_w", "d_ace_pg", "d_bp_save", "d_bp_win",
]


# ── DB: Fetch ELO / ranking / age / round for each match ────────────────────

async def fetch_elo_round_data(conn: asyncpg.Connection) -> pd.DataFrame:
    """
    For every finished match, return:
      match_id, round, p1_elo, p2_elo, p1_surface_elo, p2_surface_elo,
      p1_rank, p2_rank, p1_age, p2_age
    """
    rows = await conn.fetch(
        """
        WITH match_base AS (
            SELECT
                m.match_id::text AS match_id,
                m.match_date,
                m.tour,
                m.surface,
                m.round,
                m.p1_name,
                m.p2_name,
                -- Cached ELO on the match row itself
                m.p1_elo,
                m.p2_elo,
                -- TA player IDs for snapshot lookup
                m.p1_ta_player_id,
                m.p2_ta_player_id
            FROM tennis_matches m
            WHERE m.status IN ('finished', 'completed', 'ended')
              AND m.match_date >= '2020-01-01'
              AND m.tour IN ('ATP', 'WTA')
        )
        SELECT
            mb.match_id,
            mb.match_date,
            mb.surface,
            mb.round,
            -- Overall ELO: prefer cached, then snapshot
            COALESCE(mb.p1_elo, e1.elo)::float AS p1_elo,
            COALESCE(mb.p2_elo, e2.elo)::float AS p2_elo,
            -- Surface-specific ELO
            CASE
                WHEN mb.surface ILIKE '%%hard%%'  THEN e1.helo
                WHEN mb.surface ILIKE '%%clay%%'  THEN e1.celo
                WHEN mb.surface ILIKE '%%grass%%' THEN e1.gelo
                ELSE e1.elo
            END::float AS p1_surface_elo,
            CASE
                WHEN mb.surface ILIKE '%%hard%%'  THEN e2.helo
                WHEN mb.surface ILIKE '%%clay%%'  THEN e2.celo
                WHEN mb.surface ILIKE '%%grass%%' THEN e2.gelo
                ELSE e2.elo
            END::float AS p2_surface_elo,
            -- Rank and age
            e1.official_rank::int AS p1_rank,
            e2.official_rank::int AS p2_rank,
            e1.age::float AS p1_age,
            e2.age::float AS p2_age
        FROM match_base mb
        LEFT JOIN LATERAL (
            SELECT elo, helo, celo, gelo, official_rank, age
            FROM tennisabstract_elo_snapshots s
            WHERE s.player_id = mb.p1_ta_player_id
              AND s.as_of_date <= mb.match_date
            ORDER BY s.as_of_date DESC
            LIMIT 1
        ) e1 ON TRUE
        LEFT JOIN LATERAL (
            SELECT elo, helo, celo, gelo, official_rank, age
            FROM tennisabstract_elo_snapshots s
            WHERE s.player_id = mb.p2_ta_player_id
              AND s.as_of_date <= mb.match_date
            ORDER BY s.as_of_date DESC
            LIMIT 1
        ) e2 ON TRUE
        """
    )
    logger.info("Fetched ELO/rank/age/round for %d matches", len(rows))
    return pd.DataFrame([dict(r) for r in rows])


# ── Merge everything ─────────────────────────────────────────────────────────

async def build_v3_csv(
    simple_path: pathlib.Path,
    ta_path: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    # ── 1. Load simple CSV ───────────────────────────────────────────────────
    if not simple_path.exists():
        raise FileNotFoundError(f"Missing simple CSV: {simple_path}")
    simple = pd.read_csv(simple_path)
    logger.info("Simple CSV: %d rows, cols=%s", len(simple), list(simple.columns)[:10])

    # ── 2. Load TA CSV ───────────────────────────────────────────────────────
    if not ta_path.exists():
        raise FileNotFoundError(f"Missing TA CSV: {ta_path}")
    ta = pd.read_csv(ta_path)
    ta_keep = ["match_id"] + [c for c in TA_FEATURES if c in ta.columns]
    ta = ta[ta_keep].copy()
    logger.info("TA CSV: %d rows", len(ta))

    # ── 3. Merge simple + TA (left join to keep matches without TA data) ────
    merged = simple.merge(ta, on="match_id", how="left")
    logger.info("After simple+TA merge: %d rows (%d with TA data)",
                len(merged), merged[TA_FEATURES[0]].notna().sum() if TA_FEATURES else 0)

    # ── 4. Fetch ELO / ranking / age / round from DB ────────────────────────
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        elo_df = await fetch_elo_round_data(conn)
    finally:
        await conn.close()

    # ── 5. Merge ELO data on match_id ────────────────────────────────────────
    elo_df = elo_df[
        ["match_id", "round", "p1_elo", "p2_elo",
         "p1_surface_elo", "p2_surface_elo",
         "p1_rank", "p2_rank", "p1_age", "p2_age"]
    ].copy()
    # Drop duplicates (in case of multiple ELO joins)
    elo_df = elo_df.drop_duplicates(subset=["match_id"])
    merged = merged.merge(elo_df, on="match_id", how="left")
    logger.info("After ELO merge: %d rows, p1_elo non-null: %d",
                len(merged), merged["p1_elo"].notna().sum())

    # ── 6. Compute new ELO-based differential features ───────────────────────
    #
    # IMPORTANT: The simple CSV already applied a deterministic swap
    # (MD5 hash → p1/p2 assignment), so p1/p2 in the CSV may NOT be the
    # same as p1/p2 in the DB. We need to use the ELO values as-is from
    # the match row (which are for DB's p1/p2), then check if a swap
    # happened and invert if needed.
    #
    # The simple CSV stores the names in p1_name / p2_name AFTER any swap.
    # The elo_df has p1_elo / p2_elo from the DB (BEFORE swap).
    # We need to detect the swap by comparing names.

    # We already have p1_name/p2_name from the simple CSV.
    # Fetch the original p1_name/p2_name from the DB to detect swap.
    conn2 = await asyncpg.connect(DATABASE_URL)
    try:
        orig_names = await conn2.fetch(
            """
            SELECT match_id::text AS match_id, p1_name AS db_p1_name, p2_name AS db_p2_name
            FROM tennis_matches
            WHERE status IN ('finished', 'completed', 'ended')
              AND match_date >= '2020-01-01'
              AND tour IN ('ATP', 'WTA')
            """
        )
    finally:
        await conn2.close()

    orig_df = pd.DataFrame([dict(r) for r in orig_names])
    merged = merged.merge(orig_df, on="match_id", how="left")

    # Detect if the simple CSV swapped p1/p2
    merged["_swapped"] = (
        merged["p1_name"].fillna("").str.strip().str.lower()
        != merged["db_p1_name"].fillna("").str.strip().str.lower()
    )
    swap_count = merged["_swapped"].sum()
    logger.info("Swapped rows detected: %d / %d", swap_count, len(merged))

    # Build ELO diffs accounting for swap
    def _elo_diff(row, col_p1, col_p2) -> float:
        e1 = row.get(col_p1)
        e2 = row.get(col_p2)
        if pd.isna(e1) or pd.isna(e2):
            return 0.0
        diff = float(e1) - float(e2)
        return -diff if row.get("_swapped", False) else diff

    merged["d_elo"] = merged.apply(lambda r: _elo_diff(r, "p1_elo", "p2_elo"), axis=1)
    merged["d_surface_elo"] = merged.apply(
        lambda r: _elo_diff(r, "p1_surface_elo", "p2_surface_elo"), axis=1
    )

    # Ranking diff (log-space, lower is better → negate)
    def _rank_diff(row) -> float:
        r1 = row.get("p1_rank")
        r2 = row.get("p2_rank")
        if pd.isna(r1) or pd.isna(r2) or r1 is None or r2 is None:
            return 0.0
        r1, r2 = float(r1), float(r2)
        if r1 <= 0 or r2 <= 0:
            return 0.0
        # Negative = p1 ranked higher (lower number), which is GOOD for p1
        diff = math.log(r2) - math.log(r1)
        return -diff if row.get("_swapped", False) else diff

    merged["d_rank_log"] = merged.apply(_rank_diff, axis=1)

    # Age diff
    def _age_diff(row) -> float:
        a1 = row.get("p1_age")
        a2 = row.get("p2_age")
        if pd.isna(a1) or pd.isna(a2):
            return 0.0
        diff = float(a1) - float(a2)
        return -diff if row.get("_swapped", False) else diff

    merged["d_age"] = merged.apply(_age_diff, axis=1)

    # Round ordinal
    merged["round_ordinal"] = merged["round"].apply(_round_ordinal)

    # ── 7. Select final columns ──────────────────────────────────────────────
    meta_cols = ["match_id", "match_date", "tour", "p1_name", "p2_name", "y"]

    elo_cols = ["d_elo", "d_surface_elo", "d_rank_log", "d_age"]

    rolling_cols = [
        "d_win_rate_last_20", "d_win_rate_surface", "d_experience",
        "d_rest_days", "d_matches_10d", "d_matches_30d",
    ]

    context_cols = [
        "is_hard", "is_clay", "is_grass", "is_grand_slam", "is_masters",
        "best_of", "round_ordinal",
    ]

    h2h_cols = [
        "h2h_p1_win_pct", "h2h_total_matches",
        "h2h_surface_p1_win_pct", "h2h_surface_matches",
    ]

    # Style proxies (from simple CSV — may not all exist)
    style_cols = [c for c in STYLE_FEATURES if c in merged.columns]
    if len(style_cols) < len(STYLE_FEATURES):
        missing_style = set(STYLE_FEATURES) - set(style_cols)
        logger.warning("Missing style proxy columns (will fill with 0): %s", missing_style)
        for c in missing_style:
            merged[c] = 0.0
        style_cols = STYLE_FEATURES

    ta_cols = [c for c in TA_FEATURES if c in merged.columns]
    if len(ta_cols) < len(TA_FEATURES):
        missing_ta = set(TA_FEATURES) - set(ta_cols)
        logger.warning("Missing TA columns (will fill with 0): %s", missing_ta)
        for c in missing_ta:
            merged[c] = 0.0
        ta_cols = TA_FEATURES

    all_cols = meta_cols + elo_cols + rolling_cols + context_cols + h2h_cols + style_cols + ta_cols

    # Verify all columns exist
    for c in all_cols:
        if c not in merged.columns:
            logger.error("Column %s not found in merged DataFrame!", c)
            merged[c] = 0.0

    out = merged[all_cols].copy()

    # ── 8. Summary stats ─────────────────────────────────────────────────────
    feature_cols = elo_cols + rolling_cols + context_cols + h2h_cols + style_cols + ta_cols
    logger.info("Total feature columns: %d", len(feature_cols))

    # Report ELO coverage
    elo_present = (out["d_elo"] != 0.0).sum()
    logger.info("ELO coverage: %d / %d (%.1f%%)", elo_present, len(out), 100 * elo_present / len(out))

    rank_present = (out["d_rank_log"] != 0.0).sum()
    logger.info("Rank coverage: %d / %d (%.1f%%)", rank_present, len(out), 100 * rank_present / len(out))

    age_present = (out["d_age"] != 0.0).sum()
    logger.info("Age coverage: %d / %d (%.1f%%)", age_present, len(out), 100 * age_present / len(out))

    round_present = (out["round_ordinal"] != 0).sum()
    logger.info("Round coverage: %d / %d (%.1f%%)", round_present, len(out), 100 * round_present / len(out))

    y_dist = out["y"].value_counts()
    logger.info("Label distribution:\n%s", y_dist)

    out.to_csv(out_path, index=False)
    logger.info("✅ Wrote %d rows × %d cols to %s", len(out), len(out.columns), out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build V3 training CSV (ELO + TA + style + round)")
    ap.add_argument("--simple", type=str, default=str(DEFAULT_SIMPLE))
    ap.add_argument("--ta", type=str, default=str(DEFAULT_TA))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    asyncio.run(build_v3_csv(
        pathlib.Path(args.simple),
        pathlib.Path(args.ta),
        pathlib.Path(args.out),
    ))


if __name__ == "__main__":
    main()
