"""
Step 3: Build match-level features from Tennis Insight player stats (ATP/WTA)

Inputs:
  1) player_stats: CSV you already created in Step 2 (shrunk)
     Example: player_stats_ATP_WTA_shrunk_k20.csv
  2) matches: CSV with at least these columns:
       - tour      (ATP or WTA)
       - p1_name   (player 1 name)
       - p2_name   (player 2 name)

Output:
  - matches_with_features.csv (same rows as matches, plus engineered features)
  - missing_players.csv (any p1/p2 names that didn't match to the stats table)

Usage:
  python step3_build_match_features.py \
      --matches today_matches.csv \
      --player_stats player_stats_ATP_WTA_shrunk_k20.csv \
      --out matches_with_features.csv
"""

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


def normalize_name(name: str) -> str:
    """Lowercase, strip accents/punct, collapse spaces."""
    if pd.isna(name):
        return ""
    s = str(name).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_features(matches: pd.DataFrame, stats: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    # Ensure TOUR is present and standardized
    matches = matches.copy()
    matches["tour"] = matches["tour"].astype(str).str.upper().str.strip()

    # Build keys for joining
    matches["p1_key"] = matches["p1_name"].map(normalize_name)
    matches["p2_key"] = matches["p2_name"].map(normalize_name)

    stats = stats.copy()
    stats["Tour"] = stats["Tour"].astype(str).str.upper().str.strip()
    stats["player_key"] = stats["Player"].map(normalize_name)

    # Prefer shrunk columns from Step 2
    def sc(col: str) -> str:
        return f"{col} (shrunk_k={k})"

    base_cols = [
        "Match Stat Matches",
        sc("Service hold %"),
        sc("Opponent Hold %"),
        sc("Service Pts W %"),
        sc("Return Pts W %"),
        sc("Aces per Game"),
        sc("DFs per Game"),
        sc("BP save %"),
        sc("BP W %"),
    ]

    # Some users may have different k; fall back to raw if needed
    available = set(stats.columns)
    chosen = []
    for c in base_cols:
        if c in available:
            chosen.append(c)
        else:
            raw = c.replace(f" (shrunk_k={k})", "")
            if raw in available:
                chosen.append(raw)

    stats_small = stats[["Tour", "player_key"] + chosen].copy()

    # Merge p1 stats
    p1 = stats_small.add_prefix("p1_")
    p1 = p1.rename(columns={"p1_Tour": "tour", "p1_player_key": "p1_key"})
    out = matches.merge(p1, on=["tour", "p1_key"], how="left")

    # Merge p2 stats
    p2 = stats_small.add_prefix("p2_")
    p2 = p2.rename(columns={"p2_Tour": "tour", "p2_player_key": "p2_key"})
    out = out.merge(p2, on=["tour", "p2_key"], how="left")

    # Fill missing stats with TOUR means (safe default)
    num_cols = [c for c in out.columns if c.startswith(("p1_", "p2_"))]
    # compute means by tour from stats_small
    tour_means = stats_small.groupby("Tour").mean(numeric_only=True)

    def fill_row(row):
        t = row["tour"]
        if t in tour_means.index:
            means = tour_means.loc[t]
            for col in num_cols:
                if pd.isna(row[col]):
                    # map back to underlying stat column name (remove p1_/p2_)
                    base = col.split("_", 1)[1]
                    if base in means.index:
                        row[col] = means[base]
        return row

    out = out.apply(fill_row, axis=1)

    # Helper getter for columns (works for shrunk or raw depending on what got picked)
    def pick(side: str, stat: str) -> str:
        # side is "p1" or "p2"
        preferred = f"{side}_{sc(stat)}"
        raw = f"{side}_{stat}"
        return preferred if preferred in out.columns else raw

    # Core stats
    p1_hold = pick("p1", "Service hold %")
    p2_hold = pick("p2", "Service hold %")
    p1_opp_hold = pick("p1", "Opponent Hold %")
    p2_opp_hold = pick("p2", "Opponent Hold %")
    p1_srv_pts = pick("p1", "Service Pts W %")
    p2_srv_pts = pick("p2", "Service Pts W %")
    p1_ret_pts = pick("p1", "Return Pts W %")
    p2_ret_pts = pick("p2", "Return Pts W %")
    p1_aces = pick("p1", "Aces per Game")
    p2_aces = pick("p2", "Aces per Game")
    p1_dfs = pick("p1", "DFs per Game")
    p2_dfs = pick("p2", "DFs per Game")
    p1_bp_save = pick("p1", "BP save %")
    p2_bp_save = pick("p2", "BP save %")
    p1_bp_w = pick("p1", "BP W %")
    p2_bp_w = pick("p2", "BP W %")

    # Match-level engineered features (diffs)
    out["d_hold"] = out[p1_hold] - out[p2_hold]
    out["d_srv_pts_w"] = out[p1_srv_pts] - out[p2_srv_pts]
    out["d_ret_pts_w"] = out[p1_ret_pts] - out[p2_ret_pts]
    out["d_aces_pg"] = out[p1_aces] - out[p2_aces]
    out["d_dfs_pg"] = out[p1_dfs] - out[p2_dfs]
    out["d_bp_save"] = out[p1_bp_save] - out[p2_bp_save]
    out["d_bp_w"] = out[p1_bp_w] - out[p2_bp_w]

    # Matchup interaction features (very useful)
    out["p1_srv_vs_p2_ret_hold_edge"] = out[p1_hold] - out[p2_opp_hold]
    out["p2_srv_vs_p1_ret_hold_edge"] = out[p2_hold] - out[p1_opp_hold]
    out["serve_return_edge"] = out["p1_srv_vs_p2_ret_hold_edge"] - out["p2_srv_vs_p1_ret_hold_edge"]

    # Sample-size signals
    m1 = "p1_Match Stat Matches"
    m2 = "p2_Match Stat Matches"
    if m1 in out.columns and m2 in out.columns:
        out["min_matches_profiled"] = out[[m1, m2]].min(axis=1)
        out["d_matches_profiled"] = out[m1] - out[m2]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", required=True, help="CSV with tour, p1_name, p2_name")
    ap.add_argument("--player_stats", required=True, help="Shrunk player stats CSV from Step 2")
    ap.add_argument("--out", default="matches_with_features.csv", help="Output CSV path")
    ap.add_argument("--k", type=int, default=20, help="Shrinkage k used in Step 2 (column suffix)")
    args = ap.parse_args()

    matches = pd.read_csv(args.matches)
    stats = pd.read_csv(args.player_stats)

    needed = {"tour", "p1_name", "p2_name"}
    missing = needed - set(matches.columns)
    if missing:
        raise SystemExit(f"Matches file is missing required columns: {sorted(missing)}")

    out = build_features(matches, stats, k=args.k)

    # Report missing joins (after normalization)
    missing_rows = out[
        out["p1_key"].eq("") | out["p2_key"].eq("")
    ].copy()

    # More useful: names that didn't match stats (pre-fill stage won't detect this directly).
    # We'll detect by checking if original join was null before filling; we can approximate by
    # checking whether the raw merged fields existed as NaN before filling, but we already filled.
    # So instead, generate a list of names whose normalized key isn't in the stats keys.
    stats_keys = set((stats["Tour"].astype(str).str.upper().str.strip() + "||" + stats["Player"].map(normalize_name)).tolist())
    p1_miss = out.loc[~(out["tour"] + "||" + out["p1_key"]).isin(stats_keys), ["tour","p1_name","p1_key"]].drop_duplicates()
    p2_miss = out.loc[~(out["tour"] + "||" + out["p2_key"]).isin(stats_keys), ["tour","p2_name","p2_key"]].drop_duplicates()
    miss = pd.concat([
        p1_miss.rename(columns={"p1_name":"name","p1_key":"key"}).assign(side="p1"),
        p2_miss.rename(columns={"p2_name":"name","p2_key":"key"}).assign(side="p2"),
    ], ignore_index=True).drop_duplicates()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    miss_out = str(Path(args.out).with_suffix("").as_posix() + "_missing_players.csv")
    miss.to_csv(miss_out, index=False)

    print(f"Saved: {args.out}")
    print(f"Saved: {miss_out}")
    print(f"Rows: {len(out)} | Missing player names: {len(miss)}")


if __name__ == "__main__":
    main()
