# FILE: api/app/features/tennis_training_v3.py
"""
Build the V3 training CSV — the most comprehensive feature set.

Merges:
  - Simple rolling features  (from tennis_training_simple.csv)
  - TA rolling features       (from tennis_training_ta.csv)
  - Insight player profiles   (from tennis_insight_match_features in DB — fills 2025 gaps)
  - Per-surface CSVs          (12-month surface service/return stats — fills surface gaps)
  - ELO / ranking / age       (from tennisabstract_elo_snapshots via DB)
  - Round encoding             (from tennis_matches via DB)

Output columns  (40 ML features + 6 meta):
  match_id, match_date, tour, p1_name, p2_name, y,
  --- ELO block ---
  d_elo, d_surface_elo, d_rank_log, d_age,
  --- rolling form ---
  d_win_rate_last_20, d_win_rate_surface, d_experience,
  d_rest_days, d_matches_10d, d_matches_30d,
  --- context ---
  is_hard, is_clay, is_grass, is_grand_slam, is_masters, best_of,
  round_ordinal,
  --- H2H ---
  h2h_p1_win_pct, h2h_total_matches, h2h_surface_p1_win_pct, h2h_surface_matches,
  --- style proxies ---
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
CVS_DIR = PROJECT_ROOT / "cvs"

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
    if s in ROUND_ORDER:
        return ROUND_ORDER[s]
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


# ── Per-surface CSV loader ──────────────────────────────────────────────────

def _load_surface_csvs() -> Dict[str, pd.DataFrame]:
    """
    Load per-surface (12-month) service+return CSVs and merge them into
    per-player-surface DataFrames keyed by (tour, surface).
    Returns dict: {("ATP","hard"): df, ("ATP","clay"): df, ...}
    """
    SURFACE_FILES = {
        ("ATP", "hard"): (
            "ATP Match Stats - 12 month hard service.csv",
            "ATP Match Stats - 12 month hard return.csv",
        ),
        ("ATP", "clay"): (
            "ATP Match Stats - 12 month clay service.csv",
            # No 12-month clay return for ATP — fall back handled below
            "ATP Match Stats - 12 month clay return.csv",
        ),
        ("ATP", "grass"): (
            "ATP Match Stats - 12 month grass service.csv",
            "ATP Match Stats - 12 month grass return.csv",
        ),
        ("WTA", "hard"): (
            "WTA Match Stats - 12 month hard service.csv",
            "WTA Match Stats - 12 month hard return.csv",
        ),
        ("WTA", "clay"): (
            "WTA Match Stats - 12 month clay service.csv",
            "WTA Match Stats - 12 month clay return.csv",
        ),
        ("WTA", "grass"): (
            "WTA Match Stats - 12 month grass service.csv",
            "WTA Match Stats - 12 month grass return.csv",
        ),
    }

    # Fallback files: use All Time if 12-month is missing
    FALLBACK_FILES = {
        ("ATP", "clay"): (
            "ATP Match Stats -  All Time Clay Service.csv",
            "ATP Match Stats -  All Time Clay Return.csv",
        ),
        ("ATP", "hard"): (
            "ATP Match Stats - All Time Hard Court Service.csv",
            "ATP Match Stats -  All Time Hard Court return.csv",
        ),
        ("ATP", "grass"): (
            "ATP Match Stats -  All Time Grass Service.csv",
            "ATP Match Stats -  All Time Gras Return.csv",
        ),
    }

    result = {}
    for (tour, surf), (svc_file, ret_file) in SURFACE_FILES.items():
        svc_path = CVS_DIR / svc_file
        ret_path = CVS_DIR / ret_file
        # Try fallback if primary files missing
        if not svc_path.exists() or not ret_path.exists():
            fb = FALLBACK_FILES.get((tour, surf))
            if fb:
                svc_path = CVS_DIR / fb[0]
                ret_path = CVS_DIR / fb[1]
                logger.info("Using All-Time fallback for %s/%s", tour, surf)
            if not svc_path.exists() or not ret_path.exists():
                logger.warning("Missing surface CSV: %s or %s", svc_file, ret_file)
                continue
        svc = pd.read_csv(svc_path)
        ret = pd.read_csv(ret_path)

        # Normalize player names for joining
        svc["player_key"] = svc["Player"].apply(_norm_name)
        ret["player_key"] = ret["Player"].apply(_norm_name)

        # Rename to canonical column names
        svc = svc.rename(columns={
            "Service hold %": "surf_hold_pct",
            "Service Pts W %": "surf_svc_pts_w",
            "Aces per Game": "surf_aces_pg",
            "DFs per Game": "surf_dfs_pg",
            "BP save %": "surf_bp_save",
            "1st Serve W %": "surf_1st_svc_w",
            "2nd Serve W %": "surf_2nd_svc_w",
        })
        ret = ret.rename(columns={
            "Opponent Hold %": "surf_opp_hold_pct",
            "Return Pts W %": "surf_ret_pts_w",
            "BP W %": "surf_bp_w",
            "1st Return W %": "surf_1st_ret_w",
            "2nd Return W %": "surf_2nd_ret_w",
        })

        merged = svc[["player_key", "surf_hold_pct", "surf_svc_pts_w",
                       "surf_aces_pg", "surf_dfs_pg", "surf_bp_save"]].merge(
            ret[["player_key", "surf_opp_hold_pct", "surf_ret_pts_w", "surf_bp_w"]],
            on="player_key", how="outer"
        )
        result[(tour, surf)] = merged
        logger.info("Surface CSV %s/%s: %d players", tour, surf, len(merged))

    return result


# ── DB: Fetch insight match features ────────────────────────────────────────

async def fetch_insight_features(conn: asyncpg.Connection) -> pd.DataFrame:
    """
    Get per-match player profiles from tennis_insight_match_features.
    These are overall (not rolling) stats but cover 2020–2026.
    """
    rows = await conn.fetch(
        """
        SELECT
            match_id::text AS match_id,
            tour,
            match_date,
            surface,
            p1_name AS imf_p1_name,
            p2_name AS imf_p2_name,
            p1_service_hold_pct,
            p1_opponent_hold_pct,
            p1_service_pts_won_pct,
            p1_return_pts_won_pct,
            p1_aces_per_game,
            p1_dfs_per_game,
            p1_bp_save_pct,
            p1_bp_won_pct,
            p1_matches,
            p2_service_hold_pct,
            p2_opponent_hold_pct,
            p2_service_pts_won_pct,
            p2_return_pts_won_pct,
            p2_aces_per_game,
            p2_dfs_per_game,
            p2_bp_save_pct,
            p2_bp_won_pct,
            p2_matches,
            p1_missing,
            p2_missing
        FROM tennis_insight_match_features
        """
    )
    logger.info("Fetched %d insight match features", len(rows))
    return pd.DataFrame([dict(r) for r in rows])


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
                m.p1_elo,
                m.p2_elo,
                m.p1_odds_american,
                m.p2_odds_american,
                m.sofascore_total_games_line,
                m.sofascore_total_games_over_american,
                m.sofascore_total_games_under_american,
                m.sofascore_spread_p1_line,
                m.sofascore_spread_p2_line,
                m.sofascore_spread_p1_odds_american,
                m.sofascore_spread_p2_odds_american,
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
            COALESCE(mb.p1_elo, e1.elo)::float AS p1_elo,
            COALESCE(mb.p2_elo, e2.elo)::float AS p2_elo,
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
            e1.official_rank::int AS p1_rank,
            e2.official_rank::int AS p2_rank,
            e1.age::float AS p1_age,
            e2.age::float AS p2_age,
            mb.p1_odds_american,
            mb.p2_odds_american,
            mb.sofascore_total_games_line,
            mb.sofascore_total_games_over_american,
            mb.sofascore_total_games_under_american,
            mb.sofascore_spread_p1_line,
            mb.sofascore_spread_p2_line,
            mb.sofascore_spread_p1_odds_american,
            mb.sofascore_spread_p2_odds_american
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


def _american_to_implied_prob(odds: Optional[float]) -> Optional[float]:
    if odds is None or pd.isna(odds):
        return None
    try:
        o = int(float(odds))
    except Exception:
        return None
    if o > 0:
        return 100.0 / (float(o) + 100.0)
    if o < 0:
        v = float(-o)
        return v / (v + 100.0)
    return None


def _no_vig_two_way(p1: Optional[float], p2: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    if p1 is None or p2 is None:
        return (None, None)
    s = float(p1) + float(p2)
    if s <= 0:
        return (None, None)
    return (float(p1) / s, float(p2) / s)


# ── Backfill: map insight features → TA / style features ────────────────────

def _backfill_from_insight(
    merged: pd.DataFrame,
    imf: pd.DataFrame,
    surface_dfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    For rows where TA/style features are NaN, backfill from insight match features.
    Also uses per-surface CSVs for surface-specific features.

    The insight features are overall player profiles (not rolling-window),
    but they're the best proxy we have for 2025 matches.
    """
    if imf.empty:
        logger.warning("No insight features loaded — skipping backfill")
        return merged

    # Keep only what we need from insight
    imf_cols = ["match_id", "imf_p1_name", "imf_p2_name", "surface",
                "tour",
                "p1_service_hold_pct", "p1_opponent_hold_pct",
                "p1_service_pts_won_pct", "p1_return_pts_won_pct",
                "p1_aces_per_game", "p1_dfs_per_game",
                "p1_bp_save_pct", "p1_bp_won_pct",
                "p2_service_hold_pct", "p2_opponent_hold_pct",
                "p2_service_pts_won_pct", "p2_return_pts_won_pct",
                "p2_aces_per_game", "p2_dfs_per_game",
                "p2_bp_save_pct", "p2_bp_won_pct",
                "p1_missing", "p2_missing"]
    imf = imf[imf_cols].copy()

    # Join on match_id
    merged = merged.merge(imf, on="match_id", how="left", suffixes=("", "_imf"))

    # Detect swap: the training CSV's p1 may differ from insight's p1
    merged["_imf_swapped"] = (
        merged["p1_name"].fillna("").str.strip().str.lower()
        != merged["imf_p1_name"].fillna("").str.strip().str.lower()
    )

    imf_present = merged["imf_p1_name"].notna().sum()
    logger.info("Insight match features joined: %d / %d rows", imf_present, len(merged))
    logger.info("Insight-swapped vs training: %d", merged["_imf_swapped"].sum())

    # ── Mapping: feature → (insight_col_a, insight_col_b) where diff = a - b
    STYLE_MAP = {
        "d_svc_pts_w": ("p1_service_pts_won_pct", "p2_service_pts_won_pct"),
        "d_ret_pts_w": ("p1_return_pts_won_pct", "p2_return_pts_won_pct"),
        "d_ace_pg":    ("p1_aces_per_game", "p2_aces_per_game"),
        "d_bp_save":   ("p1_bp_save_pct", "p2_bp_save_pct"),
        "d_bp_win":    ("p1_bp_won_pct", "p2_bp_won_pct"),
    }

    TA_MAP = {
        "d_last5_hold":     ("p1_service_hold_pct", "p2_service_hold_pct"),
        "d_last10_hold":    ("p1_service_hold_pct", "p2_service_hold_pct"),
        "d_last5_break":    ("p2_opponent_hold_pct", "p1_opponent_hold_pct"),  # break% = 100 - opp_hold%
        "d_last10_break":   ("p2_opponent_hold_pct", "p1_opponent_hold_pct"),
        "d_last10_aces_pg": ("p1_aces_per_game", "p2_aces_per_game"),
        "d_last10_df_pg":   ("p1_dfs_per_game", "p2_dfs_per_game"),
    }

    ALL_BACKFILL = {**STYLE_MAP, **TA_MAP}

    # Vectorized backfill
    for feat, (col_a, col_b) in ALL_BACKFILL.items():
        if feat not in merged.columns:
            merged[feat] = np.nan

        # Compute insight-based diff
        a_vals = pd.to_numeric(merged[col_a], errors="coerce")
        b_vals = pd.to_numeric(merged[col_b], errors="coerce")
        insight_diff = a_vals - b_vals
        # Flip sign for rows where training CSV swapped p1/p2 relative to insight
        insight_diff = insight_diff.where(~merged["_imf_swapped"], -insight_diff)

        # Only fill where current value is NaN or exactly 0
        mask = merged[feat].isna() | (merged[feat] == 0.0)
        filled_count = (mask & insight_diff.notna()).sum()
        merged.loc[mask, feat] = insight_diff.where(mask)
        if filled_count > 0:
            logger.info("  Backfilled %s: %d rows from insight", feat, filled_count)

    # ── Backfill surface-specific TA features from per-surface CSVs ──────────
    if surface_dfs:
        _backfill_surface_features(merged, surface_dfs)

    # ── Backfill tiebreak features (set to 0 if not available) ───────────────
    for tb_feat in ["d_last10_tb_match_rate", "d_last10_tb_win_pct",
                    "d_surf_last10_tb_match_rate", "d_surf_last10_tb_win_pct"]:
        if tb_feat not in merged.columns:
            merged[tb_feat] = 0.0
        merged[tb_feat] = merged[tb_feat].fillna(0.0)

    # Clean up temp columns from insight join
    drop_cols = [c for c in merged.columns if c.startswith("imf_") or c.startswith("_imf")
                 or c.startswith("p1_service") or c.startswith("p2_service")
                 or c.startswith("p1_return") or c.startswith("p2_return")
                 or c.startswith("p1_aces") or c.startswith("p2_aces")
                 or c.startswith("p1_dfs") or c.startswith("p2_dfs")
                 or c.startswith("p1_bp_") or c.startswith("p2_bp_")
                 or c.startswith("p1_opponent") or c.startswith("p2_opponent")
                 or c.startswith("p1_missing") or c.startswith("p2_missing")
                 or c.startswith("p1_matches") or c.startswith("p2_matches")]
    drop_cols = [c for c in drop_cols if c in merged.columns]
    merged = merged.drop(columns=drop_cols, errors="ignore")

    return merged


def _backfill_surface_features(
    merged: pd.DataFrame,
    surface_dfs: Dict[str, pd.DataFrame],
) -> None:
    """
    Fill d_surf_last10_hold, d_surf_last10_break, d_surf_last10_aces_pg,
    d_surf_last10_df_pg from per-surface CSVs where they're NaN.
    """
    # Determine which column has the surface info
    if "surface_imf" in merged.columns:
        surf_col = "surface_imf"
    elif "surface" in merged.columns:
        surf_col = "surface"
    else:
        logger.warning("No surface column available for surface backfill")
        return

    SURF_FEATURE_MAP = {
        "d_surf_last10_hold":    "surf_hold_pct",
        "d_surf_last10_break":   "surf_opp_hold_pct",     # needs inversion
        "d_surf_last10_aces_pg": "surf_aces_pg",
        "d_surf_last10_df_pg":   "surf_dfs_pg",
    }

    # Build fast lookup: (tour, surface, player_key) → row
    surf_lookup: Dict[tuple, dict] = {}
    for (tour, surf), df in surface_dfs.items():
        for _, row in df.iterrows():
            pk = row.get("player_key", "")
            if pk:
                surf_lookup[(tour, surf, pk)] = row.to_dict()

    logger.info("Surface lookup built: %d entries", len(surf_lookup))

    # Ensure target columns exist
    for feat in SURF_FEATURE_MAP:
        if feat not in merged.columns:
            merged[feat] = np.nan

    filled_counts = {k: 0 for k in SURF_FEATURE_MAP}

    for idx in merged.index:
        tour = str(merged.at[idx, "tour"]).upper() if pd.notna(merged.at[idx, "tour"]) else ""
        surf = _surface_key(merged.at[idx, surf_col] if pd.notna(merged.at[idx, surf_col]) else "")
        p1_key = _norm_name(merged.at[idx, "p1_name"])
        p2_key = _norm_name(merged.at[idx, "p2_name"])

        p1_row = surf_lookup.get((tour, surf, p1_key))
        p2_row = surf_lookup.get((tour, surf, p2_key))

        for feat, col_name in SURF_FEATURE_MAP.items():
            current = merged.at[idx, feat]
            if pd.notna(current) and current != 0.0:
                continue  # already has data

            v1 = float(p1_row[col_name]) if p1_row is not None and pd.notna(p1_row.get(col_name)) else np.nan
            v2 = float(p2_row[col_name]) if p2_row is not None and pd.notna(p2_row.get(col_name)) else np.nan

            if pd.notna(v1) and pd.notna(v2):
                if feat == "d_surf_last10_break":
                    diff = v2 - v1  # break% = 100 - opp_hold%
                else:
                    diff = v1 - v2
                merged.at[idx, feat] = diff
                filled_counts[feat] += 1

    for feat, cnt in filled_counts.items():
        if cnt > 0:
            logger.info("  Backfilled %s: %d rows from surface CSVs", feat, cnt)


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

    # ── 3. Merge simple + TA (left join to keep 2025 matches) ────────────────
    merged = simple.merge(ta, on="match_id", how="left")
    ta_coverage = merged[TA_FEATURES[0]].notna().sum() if TA_FEATURES else 0
    logger.info("After simple+TA merge: %d rows (%d with TA data)", len(merged), ta_coverage)

    # ── 4. Load per-surface CSVs ─────────────────────────────────────────────
    surface_dfs = _load_surface_csvs()

    # ── 5. Fetch insight match features + ELO data from DB ───────────────────
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        imf_df = await fetch_insight_features(conn)
        elo_df = await fetch_elo_round_data(conn)
    finally:
        await conn.close()

    # ── 6. Backfill style + TA features from insight + surface CSVs ──────────
    logger.info("--- BACKFILL PHASE ---")
    merged = _backfill_from_insight(merged, imf_df, surface_dfs)

    # Report TA coverage after backfill
    for feat in TA_FEATURES + STYLE_FEATURES:
        if feat in merged.columns:
            present = (merged[feat].notna() & (merged[feat] != 0.0)).sum()
            logger.info("  %s coverage after backfill: %d / %d (%.1f%%)",
                        feat, present, len(merged), 100 * present / max(len(merged), 1))

    # ── 7. Merge ELO data on match_id ────────────────────────────────────────
    elo_df = elo_df[
        ["match_id", "round", "p1_elo", "p2_elo",
         "p1_surface_elo", "p2_surface_elo",
         "p1_rank", "p2_rank", "p1_age", "p2_age",
         "p1_odds_american", "p2_odds_american",
         "sofascore_total_games_line",
         "sofascore_total_games_over_american",
         "sofascore_total_games_under_american",
         "sofascore_spread_p1_line",
         "sofascore_spread_p2_line",
         "sofascore_spread_p1_odds_american",
         "sofascore_spread_p2_odds_american"]
    ].copy()
    elo_df = elo_df.drop_duplicates(subset=["match_id"])
    merged = merged.merge(elo_df, on="match_id", how="left")
    logger.info("After ELO merge: %d rows, p1_elo non-null: %d",
                len(merged), merged["p1_elo"].notna().sum())

    # ── 8. Fetch original names for swap detection (ELO is from DB perspective)
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

    # Detect if the simple CSV swapped p1/p2 relative to the DB
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

    # Ranking diff (log-space)
    def _rank_diff(row) -> float:
        r1 = row.get("p1_rank")
        r2 = row.get("p2_rank")
        if pd.isna(r1) or pd.isna(r2) or r1 is None or r2 is None:
            return 0.0
        r1, r2 = float(r1), float(r2)
        if r1 <= 0 or r2 <= 0:
            return 0.0
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

    # Market totals/spread features
    merged["market_total_over_implied"] = merged["sofascore_total_games_over_american"].apply(_american_to_implied_prob)
    merged["market_total_under_implied"] = merged["sofascore_total_games_under_american"].apply(_american_to_implied_prob)
    nv_total = merged.apply(
        lambda r: _no_vig_two_way(r.get("market_total_over_implied"), r.get("market_total_under_implied")),
        axis=1,
    )
    merged["market_total_over_no_vig"] = nv_total.apply(lambda x: x[0])
    merged["market_total_under_no_vig"] = nv_total.apply(lambda x: x[1])
    merged["market_total_overround"] = (
        merged["market_total_over_implied"] + merged["market_total_under_implied"] - 1.0
    )
    merged["has_total_line"] = merged["sofascore_total_games_line"].notna().astype(int)
    merged["market_total_line"] = merged["sofascore_total_games_line"].astype(float)
    merged["market_total_line_centered"] = merged.apply(
        lambda r: (
            float(r["market_total_line"]) - (22.0 if str(r.get("tour", "")).upper() == "ATP" else 21.5)
        ) if pd.notna(r.get("market_total_line")) else 0.0,
        axis=1,
    )

    merged["market_spread_p1_implied"] = merged["sofascore_spread_p1_odds_american"].apply(_american_to_implied_prob)
    merged["market_spread_p2_implied"] = merged["sofascore_spread_p2_odds_american"].apply(_american_to_implied_prob)
    nv_spread = merged.apply(
        lambda r: _no_vig_two_way(r.get("market_spread_p1_implied"), r.get("market_spread_p2_implied")),
        axis=1,
    )
    merged["market_spread_p1_no_vig"] = nv_spread.apply(lambda x: x[0])
    merged["market_spread_p2_no_vig"] = nv_spread.apply(lambda x: x[1])
    merged["market_spread_p1_line"] = merged["sofascore_spread_p1_line"].astype(float)
    merged["market_spread_p2_line"] = merged["sofascore_spread_p2_line"].astype(float)
    merged["has_game_spread"] = (
        merged["sofascore_spread_p1_line"].notna() | merged["sofascore_spread_p2_line"].notna()
    ).astype(int)

    # ── 9. Select final columns ──────────────────────────────────────────────
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

    market_cols = [
        "has_total_line",
        "market_total_line",
        "market_total_line_centered",
        "market_total_over_implied",
        "market_total_under_implied",
        "market_total_over_no_vig",
        "market_total_under_no_vig",
        "market_total_overround",
        "has_game_spread",
        "market_spread_p1_line",
        "market_spread_p2_line",
        "market_spread_p1_implied",
        "market_spread_p2_implied",
        "market_spread_p1_no_vig",
        "market_spread_p2_no_vig",
    ]

    style_cols = [c for c in STYLE_FEATURES if c in merged.columns]
    if len(style_cols) < len(STYLE_FEATURES):
        for c in set(STYLE_FEATURES) - set(style_cols):
            merged[c] = 0.0
        style_cols = STYLE_FEATURES

    ta_cols = [c for c in TA_FEATURES if c in merged.columns]
    if len(ta_cols) < len(TA_FEATURES):
        for c in set(TA_FEATURES) - set(ta_cols):
            merged[c] = 0.0
        ta_cols = TA_FEATURES

    all_cols = meta_cols + elo_cols + rolling_cols + context_cols + h2h_cols + market_cols + style_cols + ta_cols

    for c in all_cols:
        if c not in merged.columns:
            logger.error("Column %s not found — filling with 0", c)
            merged[c] = 0.0

    out = merged[all_cols].copy()

    # ── 10. Summary stats ────────────────────────────────────────────────────
    feature_cols = elo_cols + rolling_cols + context_cols + h2h_cols + market_cols + style_cols + ta_cols
    logger.info("Total feature columns: %d", len(feature_cols))

    for feat_name, feat_list in [
        ("ELO", ["d_elo"]), ("Surface ELO", ["d_surface_elo"]),
        ("Rank", ["d_rank_log"]), ("Age", ["d_age"]),
        ("Round", ["round_ordinal"]),
        ("Market", market_cols), ("Style", style_cols), ("TA", ta_cols),
    ]:
        present = sum((out[c] != 0.0).sum() for c in feat_list)
        total = len(out) * len(feat_list)
        logger.info("  %s coverage: %d / %d (%.1f%%)", feat_name, present, total,
                     100 * present / max(total, 1))

    y_dist = out["y"].value_counts()
    logger.info("Label distribution:\n%s", y_dist)

    out["match_date"] = pd.to_datetime(out["match_date"])
    logger.info("Date range: %s to %s", out["match_date"].min(), out["match_date"].max())
    for yr in sorted(out["match_date"].dt.year.unique()):
        cnt = (out["match_date"].dt.year == yr).sum()
        logger.info("  %d: %d matches", yr, cnt)

    out.to_csv(out_path, index=False)
    logger.info("✅ Wrote %d rows × %d cols to %s", len(out), len(out.columns), out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build V3 training CSV (ELO + TA + insight + surface + round)")
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
