# FILE: api/app/features/tennis_training_simple.py
"""
Extract training features WITHOUT requiring historical ELO snapshots.
Uses only data available in tennis_matches table.
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import logging
import csv
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Dict, Any
from bisect import bisect_left
from datetime import timedelta
import hashlib
import asyncpg
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

OUTPUT_CSV = "tennis_training_simple.csv"

_STYLE_STATS: Optional[Dict[tuple[str, str, str, str], Dict[str, Any]]] = None
_STYLE_AVG: Optional[Dict[tuple[str, str, str], Dict[str, float]]] = None


def _norm_name(name: Optional[str]) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _surface_key(surface: Optional[str]) -> str:
    s = (surface or "").lower()
    if "clay" in s:
        return "clay"
    if "grass" in s:
        return "grass"
    if "hard" in s:
        return "hard"
    return "unknown"


def _find_player_stats_path() -> Optional[Path]:
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "player_surface_stats.csv",
        Path.cwd() / "app" / "data" / "player_surface_stats.csv",
        here.parents[1] / "data" / "player_surface_stats.csv",
        here.parents[2] / "data" / "player_surface_stats.csv",
        Path.cwd() / "player_surface_stats_final.csv",
        Path.cwd() / "app" / "data" / "player_surface_stats_final.csv",
        here.parents[1] / "data" / "player_surface_stats_final.csv",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def _load_player_style_stats() -> None:
    global _STYLE_STATS, _STYLE_AVG
    if _STYLE_STATS is not None:
        return

    path = _find_player_stats_path()
    if path is None:
        _STYLE_STATS = {}
        _STYLE_AVG = {}
        return

    stats: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    sums: Dict[tuple[str, str, str], Dict[str, float]] = {}
    cnts: Dict[tuple[str, str, str], Dict[str, int]] = {}

    def to_float(v) -> Optional[float]:
        try:
            if v is None:
                return None
            sv = str(v).strip()
            if sv == "" or sv.lower() == "nan":
                return None
            return float(sv)
        except Exception:
            return None

    def to_int(v) -> int:
        try:
            if v is None:
                return 0
            sv = str(v).strip()
            if sv == "" or sv.lower() == "nan":
                return 0
            return int(float(sv))
        except Exception:
            return 0

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            tour = str(r.get("tour") or "").upper().strip()
            window = str(r.get("window") or "").lower().strip()
            surface = str(r.get("surface") or "").lower().strip()
            player = _norm_name(r.get("player") or r.get("player_name") or r.get("name")) or ""
            if not tour or not window or not surface or not player:
                continue

            svc_pts = to_float(r.get("svc_service_pts_win_pct"))
            ret_pts = to_float(r.get("ret_return_pts_win_pct"))
            svc_n = to_int(r.get("svc_matches"))
            ret_n = to_int(r.get("ret_matches"))
            svc_aces_pg = to_float(r.get("svc_aces_per_game"))
            svc_bp_save_pct = to_float(r.get("svc_bp_save_pct"))
            ret_bp_win_pct = to_float(r.get("ret_bp_win_pct"))

            row: Dict[str, Any] = {
                "svc_pts": svc_pts,
                "ret_pts": ret_pts,
                "svc_n": svc_n,
                "ret_n": ret_n,
                "svc_aces_pg": svc_aces_pg,
                "svc_bp_save_pct": svc_bp_save_pct,
                "ret_bp_win_pct": ret_bp_win_pct,
            }

            stats[(tour, window, surface, player)] = row

            key = (tour, window, surface)
            if key not in sums:
                sums[key] = {}
                cnts[key] = {}
            for k in ("svc_pts", "ret_pts", "svc_aces_pg", "svc_bp_save_pct", "ret_bp_win_pct"):
                v = row.get(k)
                if v is None:
                    continue
                sums[key][k] = sums[key].get(k, 0.0) + float(v)
                cnts[key][k] = cnts[key].get(k, 0) + 1

    avgs: Dict[tuple[str, str, str], Dict[str, float]] = {}
    for k in sums.keys():
        avgs[k] = {}
        for metric, total in sums[k].items():
            c = cnts[k].get(metric, 0)
            if c > 0:
                avgs[k][metric] = float(total) / float(c)

    _STYLE_STATS = stats
    _STYLE_AVG = avgs


def _pick_metric(row_12m: Optional[Dict[str, Any]], row_all: Optional[Dict[str, Any]], metric: str, n_key: str, min_matches: int = 5) -> Optional[float]:
    if row_12m and row_12m.get(metric) is not None and int(row_12m.get(n_key) or 0) >= min_matches:
        return float(row_12m.get(metric))
    if row_all and row_all.get(metric) is not None and int(row_all.get(n_key) or 0) >= min_matches:
        return float(row_all.get(metric))
    return None


def _lookup_style_metrics(tour: str, surface: str, player_name: str) -> Dict[str, Optional[float]]:
    _load_player_style_stats()
    if _STYLE_STATS is None:
        return {"svc_pts": None, "ret_pts": None, "ace_pg": None, "bp_save": None, "bp_win": None}

    t = (tour or "").upper().strip()
    s = _surface_key(surface)
    p = _norm_name(player_name)

    row_12m = _STYLE_STATS.get((t, "12 month", s, p))
    row_all = _STYLE_STATS.get((t, "all time", s, p))

    def pick(metric: str, n_key: str) -> Optional[float]:
        val = _pick_metric(row_12m, row_all, metric, n_key, min_matches=5)
        if val is not None:
            return val
        if _STYLE_AVG is not None:
            avg12 = _STYLE_AVG.get((t, "12 month", s), {})
            avgall = _STYLE_AVG.get((t, "all time", s), {})
            return avg12.get(metric) or avgall.get(metric)
        return None

    return {
        "svc_pts": pick("svc_pts", "svc_n"),
        "ret_pts": pick("ret_pts", "ret_n"),
        "ace_pg": pick("svc_aces_pg", "svc_n"),
        "bp_save": pick("svc_bp_save_pct", "svc_n"),
        "bp_win": pick("ret_bp_win_pct", "ret_n"),
    }


def parse_winner_from_score(score: str) -> bool | None:
    """
    Parse score to determine if P1 won.
    Score format: "6-4 6-3" (space-separated sets)
    P1 wins if they won more sets.
    """
    if not score:
        return None
    
    try:
        sets = score.strip().split()
        p1_sets_won = 0
        p2_sets_won = 0
        
        for set_score in sets:
            # Skip invalid formats like "W/O", "RET"
            if '-' not in set_score:
                continue
            
            # Remove tiebreak notation like "(7)"
            set_score = set_score.split('(')[0]
            
            parts = set_score.split('-')
            if len(parts) != 2:
                continue
            
            p1_games = int(parts[0])
            p2_games = int(parts[1])
            
            if p1_games > p2_games:
                p1_sets_won += 1
            elif p2_games > p1_games:
                p2_sets_won += 1
        
        if p1_sets_won > p2_sets_won:
            return True
        elif p2_sets_won > p1_sets_won:
            return False
        else:
            return None
    except (ValueError, IndexError):
        return None


async def fetch_matches(conn) -> List[dict]:
    """
    Fetch finished matches with outcomes.
    We'll build rolling features from match history.
    """
    query = """
    SELECT
        m.match_id,
        m.match_date,
        m.tour,
        m.surface,
        m.tournament,
        m.round,
        m.p1_name,
        m.p2_name,
        m.score
    FROM tennis_matches m
    WHERE m.status IN ('finished', 'completed', 'ended')
      AND m.match_date >= '2020-01-01'
      AND m.match_date < '2026-01-01'
      AND m.tour IN ('ATP', 'WTA')
      AND m.score IS NOT NULL
      AND m.score != ''
    ORDER BY m.match_date, m.match_id
    """
    
    rows = await conn.fetch(query)
    logger.info("Fetched %d matches with scores", len(rows))
    return [dict(r) for r in rows]


def calculate_rolling_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling win rates and form for each player.
    """
    logger.info("Calculating rolling features...")

    matches_df = matches_df.copy()
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"], errors="coerce")
    
    # Create player-level history
    all_results = []
    
    for _, match in matches_df.iterrows():
        if pd.isna(match['p1_won']):
            continue
        
        # P1 result
        all_results.append({
            'player': match['p1_name'],
            'date': match['match_date'],
            'tour': match['tour'],
            'surface': match['surface'],
            'won': 1 if match['p1_won'] else 0,
        })
        # P2 result
        all_results.append({
            'player': match['p2_name'],
            'date': match['match_date'],
            'tour': match['tour'],
            'surface': match['surface'],
            'won': 0 if match['p1_won'] else 1,
        })
    
    results_df = pd.DataFrame(all_results).sort_values('date')
    
    # Calculate rolling stats per player
    player_stats = {}
    
    for idx, row in results_df.iterrows():
        player = row['player']
        date = row['date']
        surface = (row['surface'] or '').capitalize()
        if pd.isna(date):
            continue
        
        if player not in player_stats:
            player_stats[player] = {
                'matches': [],
                'surface_matches': {},
                'match_dates': [],
            }
        
        # Before this match, what were their stats?
        recent_matches = player_stats[player]['matches'][-20:]
        surface_matches = player_stats[player]['surface_matches'].get(surface, [])[-10:]
        match_dates = player_stats[player]['match_dates']

        # Rest days + matches in recent windows (safe fatigue/context)
        if match_dates:
            last_date = match_dates[-1]
            rest_days = (date - last_date).days
            if rest_days < 0:
                rest_days = 0
        else:
            rest_days = None

        start_10 = date - timedelta(days=10)
        start_30 = date - timedelta(days=30)
        recent_10 = len(match_dates) - bisect_left(match_dates, start_10)
        recent_30 = len(match_dates) - bisect_left(match_dates, start_30)
        
        # Store stats for this match
        results_df.loc[idx, 'win_rate_last_20'] = \
            sum(recent_matches) / len(recent_matches) if recent_matches else 0.5
        
        results_df.loc[idx, 'win_rate_surface_last_10'] = \
            sum(surface_matches) / len(surface_matches) if surface_matches else 0.5
        
        results_df.loc[idx, 'matches_played'] = len(player_stats[player]['matches'])
        results_df.loc[idx, 'rest_days'] = rest_days if rest_days is not None else 30
        results_df.loc[idx, 'matches_10d'] = recent_10
        results_df.loc[idx, 'matches_30d'] = recent_30
        
        # Update player history
        player_stats[player]['matches'].append(row['won'])
        if surface:
            if surface not in player_stats[player]['surface_matches']:
                player_stats[player]['surface_matches'][surface] = []
            player_stats[player]['surface_matches'][surface].append(row['won'])
        player_stats[player]['match_dates'].append(date)
    
    return results_df


def build_features(matches_df: pd.DataFrame, results_df: pd.DataFrame) -> List[dict]:
    """
    Build feature rows for training.
    """
    logger.info("Building feature rows...")
    
    feature_rows = []
    h2h_history: dict[tuple[str, str], dict] = {}

    def _pair_key(a: str, b: str) -> tuple[str, str]:
        return tuple(sorted([a, b]))

    def _get_h2h_stats(p1_name: str, p2_name: str, surface: str) -> tuple[int, int, int, int, int, int]:
        key = _pair_key(p1_name, p2_name)
        hist = h2h_history.get(key)
        if not hist:
            return 0, 0, 0, 0, 0, 0

        p1_wins = hist["wins"].get(p1_name, 0)
        p2_wins = hist["wins"].get(p2_name, 0)
        total = hist["total"]

        surf = surface or "unknown"
        surf_hist = hist["surface"].get(surf)
        if surf_hist:
            surf_total = surf_hist["total"]
            p1_surf_wins = surf_hist["wins"].get(p1_name, 0)
            p2_surf_wins = surf_hist["wins"].get(p2_name, 0)
        else:
            surf_total = 0
            p1_surf_wins = 0
            p2_surf_wins = 0

        return p1_wins, p2_wins, total, p1_surf_wins, p2_surf_wins, surf_total

    def _update_h2h(p1_name: str, p2_name: str, surface: str, p1_won: bool) -> None:
        key = _pair_key(p1_name, p2_name)
        if key not in h2h_history:
            h2h_history[key] = {
                "wins": {p1_name: 0, p2_name: 0},
                "total": 0,
                "surface": {},
            }

        hist = h2h_history[key]
        hist["wins"].setdefault(p1_name, 0)
        hist["wins"].setdefault(p2_name, 0)

        winner = p1_name if p1_won else p2_name
        hist["wins"][winner] += 1
        hist["total"] += 1

        surf = surface or "unknown"
        if surf not in hist["surface"]:
            hist["surface"][surf] = {"wins": {p1_name: 0, p2_name: 0}, "total": 0}
        surf_hist = hist["surface"][surf]
        surf_hist["wins"].setdefault(p1_name, 0)
        surf_hist["wins"].setdefault(p2_name, 0)
        surf_hist["wins"][winner] += 1
        surf_hist["total"] += 1

    matches_df = matches_df.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    
    for _, match in matches_df.iterrows():
        if pd.isna(match['p1_won']):
            continue
        
        # Get player stats at match time
        match_date = pd.to_datetime(match['match_date'], errors="coerce")
        if pd.isna(match_date):
            continue
        p1_stats = results_df[
            (results_df['player'] == match['p1_name']) & 
            (results_df['date'] == match_date)
        ]
        p2_stats = results_df[
            (results_df['player'] == match['p2_name']) & 
            (results_df['date'] == match_date)
        ]
        
        if p1_stats.empty or p2_stats.empty:
            continue
        
        p1 = p1_stats.iloc[0]
        p2 = p2_stats.iloc[0]

        # Deterministic swap to balance labels (avoid P1 always being winner)
        key = f"{match.get('match_id')}|{match.get('match_date')}|{match.get('p1_name')}|{match.get('p2_name')}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        do_swap = int(digest[:2], 16) % 2 == 1
        if do_swap:
            p1, p2 = p2, p1
        
        # Surface encoding
        surface = (match['surface'] or '').lower()
        is_hard = 1 if 'hard' in surface else 0
        is_clay = 1 if 'clay' in surface else 0
        is_grass = 1 if 'grass' in surface else 0
        
        # Tournament tier
        tournament = (match['tournament'] or '').lower()
        is_slam = any(k in tournament for k in [
            'australian open', 'french open', 'roland garros',
            'wimbledon', 'us open'
        ])
        is_masters = 'masters' in tournament
        
        best_of = 5 if (is_slam and match['tour'] == 'ATP') else 3
        
        y = 1 if match['p1_won'] else 0
        if do_swap:
            y = 1 - y
            p1_name = match['p2_name']
            p2_name = match['p1_name']
        else:
            p1_name = match['p1_name']
            p2_name = match['p2_name']

        p1_style = _lookup_style_metrics(match['tour'], surface, p1_name)
        p2_style = _lookup_style_metrics(match['tour'], surface, p2_name)

        def _diff(a: Optional[float], b: Optional[float], default: float = 0.0) -> float:
            if a is None or b is None:
                return default
            return float(a) - float(b)

        h2h_p1_wins, h2h_p2_wins, h2h_total, h2h_surface_p1_wins, _h2h_surface_p2_wins, h2h_surface_total = _get_h2h_stats(
            p1_name, p2_name, surface
        )
        h2h_p1_win_pct = (h2h_p1_wins / h2h_total) if h2h_total > 0 else 0.5
        h2h_surface_p1_win_pct = (h2h_surface_p1_wins / h2h_surface_total) if h2h_surface_total > 0 else 0.5

        feature_rows.append({
            'match_id': match['match_id'],
            'match_date': str(match['match_date']),
            'tour': match['tour'],
            'p1_name': p1_name,
            'p2_name': p2_name,
            'y': y,
            
            # Differential features
            'd_win_rate_last_20': float(p1.get('win_rate_last_20', 0.5) - p2.get('win_rate_last_20', 0.5)),
            'd_win_rate_surface': float(p1.get('win_rate_surface_last_10', 0.5) - p2.get('win_rate_surface_last_10', 0.5)),
            'd_experience': float(min(p1.get('matches_played', 0), 500) - min(p2.get('matches_played', 0), 500)),

            # Fatigue / recent load
            'd_rest_days': float((p1.get('rest_days', 30) or 30) - (p2.get('rest_days', 30) or 30)),
            'd_matches_10d': float((p1.get('matches_10d', 0) or 0) - (p2.get('matches_10d', 0) or 0)),
            'd_matches_30d': float((p1.get('matches_30d', 0) or 0) - (p2.get('matches_30d', 0) or 0)),
            
            # Surface
            'is_hard': is_hard,
            'is_clay': is_clay,
            'is_grass': is_grass,
            
            # Tournament
            'is_grand_slam': 1 if is_slam else 0,
            'is_masters': 1 if is_masters else 0,
            'best_of': best_of,

            # H2H (prior to match)
            'h2h_p1_win_pct': float(h2h_p1_win_pct),
            'h2h_total_matches': int(h2h_total),
            'h2h_surface_p1_win_pct': float(h2h_surface_p1_win_pct),
            'h2h_surface_matches': int(h2h_surface_total),

            # Style proxies (player_surface_stats)
            'd_svc_pts_w': _diff(p1_style.get("svc_pts"), p2_style.get("svc_pts")),
            'd_ret_pts_w': _diff(p1_style.get("ret_pts"), p2_style.get("ret_pts")),
            'd_ace_pg': _diff(p1_style.get("ace_pg"), p2_style.get("ace_pg")),
            'd_bp_save': _diff(p1_style.get("bp_save"), p2_style.get("bp_save")),
            'd_bp_win': _diff(p1_style.get("bp_win"), p2_style.get("bp_win")),
        })

        _update_h2h(match['p1_name'], match['p2_name'], surface, bool(match['p1_won']))
    
    return feature_rows


async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    
    try:
        # Fetch matches
        matches = await fetch_matches(conn)
        if not matches:
            logger.error("No matches found!")
            return
        
        matches_df = pd.DataFrame(matches)
        
        # Parse winners from scores
        logger.info("Parsing winners from scores...")
        matches_df['p1_won'] = matches_df['score'].apply(parse_winner_from_score)
        
        # Filter to matches with valid outcomes
        matches_df = matches_df[matches_df['p1_won'].notna()].copy()
        matches_df["match_date"] = pd.to_datetime(matches_df["match_date"], errors="coerce")
        matches_df = matches_df[matches_df["match_date"].notna()].copy()
        
        logger.info(f"Processing {len(matches_df)} matches with valid outcomes")
        logger.info(f"Date range: {matches_df['match_date'].min()} to {matches_df['match_date'].max()}")
        p1_wins = int((matches_df['p1_won'] == True).sum())
        p2_wins = int((matches_df['p1_won'] == False).sum())
        logger.info(f"P1 wins: {p1_wins}, P2 wins: {p2_wins}")
        
        # Calculate rolling features
        results_df = calculate_rolling_features(matches_df)
        
        # Build training features
        feature_rows = build_features(matches_df, results_df)
        
        logger.info(f"Generated {len(feature_rows)} feature rows")
        
        # Write to CSV
        if feature_rows:
            fieldnames = list(feature_rows[0].keys())
            with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(feature_rows)
            
            logger.info(f"✅ Wrote {len(feature_rows)} rows to {OUTPUT_CSV}")
            
            # Stats
            y_dist = pd.Series([r['y'] for r in feature_rows])
            logger.info(f"Label distribution: {y_dist.value_counts().to_dict()}")
        else:
            logger.warning("No feature rows generated")
    
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
