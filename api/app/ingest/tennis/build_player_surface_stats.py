#!/usr/bin/env python3
"""
Build a single player surface stats table from many 'Match Stats' CSV exports.

Expected filenames (flexible):
  "<TOUR> Match Stats - <WINDOW> <SURFACE> service.csv"
  "<TOUR> Match Stats - <WINDOW> <SURFACE> return.csv"

Examples:
  ATP Match Stats - 12 month hard service.csv
  WTA Match Stats - all time clay return.csv

Notes:
- Handles typos like "returrn" in filenames.
- If multiple files exist for the same (tour, window, surface, kind),
  keeps the one with the most rows.
- Outputs a single CSV with standardized column names and percents converted to 0..1.

Usage:
  python build_player_surface_stats.py --input_dir ./exports --output ./player_surface_stats.csv
"""

from __future__ import annotations
import argparse, os, re
import pandas as pd
from sqlalchemy import text  # optional if you later want to load to DB

SURFACES = {"hard","clay","grass","indoor","carpet","all"}  # allow "all" surface exports

def _percent_to_prob(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce") / 100.0

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Player":"player",
        "Match Stat Matches":"matches",
        "Aces per Game":"aces_per_game",
        "DFs per Game":"dfs_per_game",
        "Ace to DF ratio":"ace_to_df_ratio",
        "1st Serve %":"first_serve_pct",
        "1st Serve W %":"first_serve_win_pct",
        "2nd Serve W %":"second_serve_win_pct",
        "Service Pts W %":"service_pts_win_pct",
        "BP save %":"bp_save_pct",
        "Service hold %":"hold_pct",
        "Opponent Aces per Game":"opp_aces_per_game",
        "Opponent DFs per Game":"opp_dfs_per_game",
        "Opponent 1st Serve %":"opp_first_serve_pct",
        "1st Return W %":"first_return_win_pct",
        "2nd Return W %":"second_return_win_pct",
        "Return Pts W %":"return_pts_win_pct",
        "BP W %":"bp_win_pct",
        "Opponent Hold %":"opp_hold_pct",
    }
    return df.rename(columns={c: mapping.get(c, c) for c in df.columns})

def _prep_service(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_cols(df).copy()
    df["player"] = df["player"].astype(str).str.strip()
    df = df.rename(columns={"matches":"svc_matches"})
    for c in ["first_serve_pct","first_serve_win_pct","second_serve_win_pct",
              "service_pts_win_pct","bp_save_pct","hold_pct"]:
        if c in df.columns:
            df[c] = _percent_to_prob(df[c])
    keep = ["player","svc_matches"]
    metric = [c for c in df.columns if c not in keep]
    df = df[keep + metric]
    df = df.rename(columns={c: f"svc_{c}" for c in metric})
    return df

def _prep_return(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_cols(df).copy()
    df["player"] = df["player"].astype(str).str.strip()
    df = df.rename(columns={"matches":"ret_matches"})
    for c in ["opp_first_serve_pct","first_return_win_pct","second_return_win_pct",
              "return_pts_win_pct","bp_win_pct","opp_hold_pct"]:
        if c in df.columns:
            df[c] = _percent_to_prob(df[c])
    keep = ["player","ret_matches"]
    metric = [c for c in df.columns if c not in keep]
    df = df[keep + metric]
    df = df.rename(columns={c: f"ret_{c}" for c in metric})
    return df

def _parse_filename(fn: str):
    # tour: ATP/WTA
    m = re.search(r"\b(ATP|WTA)\b", fn, re.IGNORECASE)
    tour = m.group(1).upper() if m else None

    # kind: service/return (accept returrn typos)
    kind = None
    if re.search(r"\bservice\b", fn, re.IGNORECASE):
        kind = "service"
    elif re.search(r"\bret(?:u)?rr?n\b|\breturn\b", fn, re.IGNORECASE):
        kind = "return"

    # surface
    surf = None
    for s in ["indoor hard","indoor","hard","clay","grass","carpet","all surfaces","all"]:
        if re.search(re.escape(s), fn, re.IGNORECASE):
            surf = "all" if "all" in s else ("indoor" if "indoor" in s else s.split()[0])
            break

    # window (best-effort): "12 month", "all time", "3 month", etc.
    window = None
    m = re.search(r"Match Stats\s*-\s*([^-]+?)\s+(hard|clay|grass|indoor|carpet|all)", fn, re.IGNORECASE)
    if m:
        window = m.group(1).strip().lower()
        window = window.replace("months","month")
    else:
        # fallback
        if re.search(r"all\s*time", fn, re.IGNORECASE):
            window = "all time"
        else:
            m2 = re.search(r"(\d+)\s*month", fn, re.IGNORECASE)
            if m2:
                window = f"{m2.group(1)} month"

    return tour, window, surf, kind

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".csv") and "match stats" in f.lower()]
    buckets = {}
    for f in files:
        tour, window, surf, kind = _parse_filename(f)
        if not all([tour, window, surf, kind]):
            continue
        key = (tour, window, surf, kind)
        path = os.path.join(args.input_dir, f)
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, encoding_errors="ignore")
        # keep the largest file for this key (helps with duplicates/typos)
        if key not in buckets or len(df) > len(buckets[key][1]):
            buckets[key] = (path, df)

    # merge service+return per (tour, window, surf)
    out_rows = []
    keys3 = sorted({(t,w,s) for (t,w,s,k) in buckets.keys()})
    for (tour, window, surf) in keys3:
        svc = buckets.get((tour, window, surf, "service"))
        ret = buckets.get((tour, window, surf, "return"))
        svc_df = _prep_service(svc[1]) if svc else None
        ret_df = _prep_return(ret[1]) if ret else None

        if svc_df is None and ret_df is None:
            continue
        if svc_df is None:
            merged = ret_df.copy()
        elif ret_df is None:
            merged = svc_df.copy()
        else:
            merged = svc_df.merge(ret_df, on="player", how="outer")

        merged["tour"] = tour
        merged["window"] = window
        merged["surface"] = surf
        out_rows.append(merged)

    if not out_rows:
        raise SystemExit("No matching CSVs found. Check filenames and --input_dir.")

    out = pd.concat(out_rows, ignore_index=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} rows -> {args.output}")

if __name__ == "__main__":
    main()
