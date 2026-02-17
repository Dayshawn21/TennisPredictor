import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import pandas as pd

def _norm_name(s: str) -> str:
    return " ".join(str(s or "").strip().lower().split())

def _clamp(x: float, lo=1e-6, hi=1 - 1e-6) -> float:
    return max(lo, min(hi, x))

@dataclass(frozen=True)
class PlayerStats:
    svc_pts_win: Optional[float]
    svc_matches: int
    ret_pts_win: Optional[float]
    ret_matches: int

class PlayerSurfaceStatsProvider:
    """
    In-memory lookup over player_surface_stats.csv.
    Keys: (tour, window, surface, normalized_player_name)
    """
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)

        # Normalize keys
        df["player_key"] = df["player"].apply(_norm_name)
        df["tour_key"] = df["tour"].astype(str).str.upper().str.strip()
        df["window_key"] = df["window"].astype(str).str.lower().str.strip()
        df["surface_key"] = df["surface"].astype(str).str.lower().str.strip()

        # Ensure numeric + NaNs become None later
        num_cols = [
            "svc_matches", "ret_matches",
            "svc_service_pts_win_pct", "ret_return_pts_win_pct",
        ]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        self._idx: Dict[Tuple[str, str, str, str], dict] = {}
        for r in df.to_dict(orient="records"):
            key = (r["tour_key"], r["window_key"], r["surface_key"], r["player_key"])
            self._idx[key] = r

        # Precompute tour/surface averages as a last-resort fallback
        self._avg: Dict[Tuple[str, str], dict] = {}
        for (tour, surface), g in df.groupby(["tour_key", "surface_key"]):
            self._avg[(tour, surface)] = {
                "svc_service_pts_win_pct": float(g["svc_service_pts_win_pct"].dropna().mean()) if g["svc_service_pts_win_pct"].notna().any() else None,
                "ret_return_pts_win_pct": float(g["ret_return_pts_win_pct"].dropna().mean()) if g["ret_return_pts_win_pct"].notna().any() else None,
            }

    def _get_row(self, tour: str, window: str, surface: str, player: str) -> Optional[dict]:
        key = (tour.upper().strip(), window.lower().strip(), surface.lower().strip(), _norm_name(player))
        return self._idx.get(key)

    def _blend(self, recent_val, recent_n, fallback_val, k: int = 20) -> Optional[float]:
        if recent_val is None and fallback_val is None:
            return None
        if recent_val is None:
            return fallback_val
        if fallback_val is None:
            return recent_val
        n = max(0, int(recent_n or 0))
        return (n * float(recent_val) + k * float(fallback_val)) / (n + k)

    def get_player_stats(self, tour: str, surface: str, player: str) -> PlayerStats:
        """
        Returns service pts win% + return pts win% with fallback:
          surface 12m -> surface all_time -> all 12m -> all all_time -> tour/surface avg
        And shrinkage when matches are small.
        """
        surf = surface.lower().strip()
        # windows as they appear in your CSV: "12 month" and "all time"
        w12 = "12 month"
        wat = "all time"

        # Service
        r12 = self._get_row(tour, w12, surf, player)
        rat = self._get_row(tour, wat, surf, player)
        r12_all = self._get_row(tour, w12, "all", player)
        rat_all = self._get_row(tour, wat, "all", player)

        svc12 = float(r12["svc_service_pts_win_pct"]) if r12 and pd.notna(r12.get("svc_service_pts_win_pct")) else None
        svc12n = int(r12.get("svc_matches") or 0) if r12 else 0
        svcat = float(rat["svc_service_pts_win_pct"]) if rat and pd.notna(rat.get("svc_service_pts_win_pct")) else None
        svcatn = int(rat.get("svc_matches") or 0) if rat else 0

        # fallbacks if surface missing
        if svc12 is None and r12_all:
            svc12 = float(r12_all["svc_service_pts_win_pct"]) if pd.notna(r12_all.get("svc_service_pts_win_pct")) else None
            svc12n = int(r12_all.get("svc_matches") or 0)
        if svcat is None and rat_all:
            svcat = float(rat_all["svc_service_pts_win_pct"]) if pd.notna(rat_all.get("svc_service_pts_win_pct")) else None
            svcatn = int(rat_all.get("svc_matches") or 0)

        svc_pts = self._blend(svc12, svc12n, svcat, k=20)

        # Return
        ret12 = float(r12["ret_return_pts_win_pct"]) if r12 and pd.notna(r12.get("ret_return_pts_win_pct")) else None
        ret12n = int(r12.get("ret_matches") or 0) if r12 else 0
        retat = float(rat["ret_return_pts_win_pct"]) if rat and pd.notna(rat.get("ret_return_pts_win_pct")) else None
        retatn = int(rat.get("ret_matches") or 0) if rat else 0

        if ret12 is None and r12_all:
            ret12 = float(r12_all["ret_return_pts_win_pct"]) if pd.notna(r12_all.get("ret_return_pts_win_pct")) else None
            ret12n = int(r12_all.get("ret_matches") or 0)
        if retat is None and rat_all:
            retat = float(rat_all["ret_return_pts_win_pct"]) if pd.notna(rat_all.get("ret_return_pts_win_pct")) else None
            retatn = int(rat_all.get("ret_matches") or 0)

        ret_pts = self._blend(ret12, ret12n, retat, k=20)

        # Last resort: tour/surface averages
        if svc_pts is None:
            svc_pts = self._avg.get((tour.upper().strip(), surf), {}).get("svc_service_pts_win_pct")
        if ret_pts is None:
            ret_pts = self._avg.get((tour.upper().strip(), surf), {}).get("ret_return_pts_win_pct")

        svc_pts = float(_clamp(svc_pts)) if svc_pts is not None else None
        ret_pts = float(_clamp(ret_pts)) if ret_pts is not None else None

        return PlayerStats(
            svc_pts_win=svc_pts,
            svc_matches=svc12n or svcatn,
            ret_pts_win=ret_pts,
            ret_matches=ret12n or retatn,
        )
