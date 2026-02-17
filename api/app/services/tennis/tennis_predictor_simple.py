"""
Simple XGBoost tennis predictor.

Option B1:
- Prefer your 9-feature model: xgb_tennis_simple.json
- Prefer your 9-line txt feature list: xgb_tennis_lightweight_features.txt (or xgb_tennis_simple_features.txt)

Env overrides:
  - XGB_TENNIS_MODEL_PATH
  - XGB_TENNIS_FEATURES_PATH
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
from xgboost import XGBClassifier
from joblib import load as joblib_load

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")


# -------------------------
# File + feature loading
# -------------------------

def _find_file(candidates: List[str]) -> Path:
    tried: List[str] = []
    for c in candidates:
        if not c:
            continue
        tried.append(c)
        p = Path(c)
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find file. Tried: {tried}")


def _read_feature_cols(path: Path) -> List[str]:
    """
    Supports:
      - .txt: one feature per line (ignores blank lines and lines starting with #)
      - .json: JSON list of feature names
    """
    text = path.read_text(encoding="utf-8").strip()

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Feature JSON must be a list. Got: {type(data).__name__}")
        cols = [str(x).strip() for x in data if str(x).strip()]
        if not cols:
            raise ValueError(f"Feature JSON list is empty: {path}")
        return cols

    # txt fallback
    cols: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        cols.append(s)

    if not cols:
        raise ValueError(f"Feature TXT list is empty: {path}")
    return cols


def _features_for_model(model_path: Path, app_dir: Path, api_dir: Path, cwd: Path, feats_env: str) -> Path:
    if feats_env:
        return _find_file([feats_env])

    name = model_path.name
    if "_v3" in name:
        candidates = [
            str(app_dir / "models" / "xgb_tennis_v3_features.txt"),
            str(api_dir / "models" / "xgb_tennis_v3_features.txt"),
            str(cwd / "models" / "xgb_tennis_v3_features.txt"),
        ]
    elif "combined_v2" in name:
        candidates = [
            str(app_dir / "models" / "xgb_tennis_combined_v2_features.txt"),
            str(api_dir / "models" / "xgb_tennis_combined_v2_features.txt"),
            str(cwd / "models" / "xgb_tennis_combined_v2_features.txt"),
        ]
    elif "simple_v2" in name:
        candidates = [
            str(app_dir / "models" / "xgb_tennis_simple_v2_features.txt"),
            str(api_dir / "models" / "xgb_tennis_simple_v2_features.txt"),
            str(cwd / "models" / "xgb_tennis_simple_v2_features.txt"),
        ]
    elif "simple" in name:
        candidates = [
            str(app_dir / "models" / "xgb_tennis_simple_features.txt"),
            str(api_dir / "models" / "xgb_tennis_simple_features.txt"),
            str(cwd / "models" / "xgb_tennis_simple_features.txt"),
        ]
    else:
        candidates = [
            str(app_dir / "models" / "xgb_tennis_lightweight_features.txt"),
            str(api_dir / "models" / "xgb_tennis_lightweight_features.txt"),
            str(cwd / "models" / "xgb_tennis_lightweight_features.txt"),
        ]

    return _find_file(candidates)


def _default_paths() -> Tuple[Path, Path]:
    """
    OPTION B1: Prefer your 9-feature model + txt list.
    """
    here = Path(__file__).resolve()
    app_dir = here.parents[2]  # .../api/app
    api_dir = here.parents[3]  # .../api
    cwd = Path.cwd()

    model_env = os.getenv("XGB_TENNIS_MODEL_PATH", "").strip()
    feats_env = os.getenv("XGB_TENNIS_FEATURES_PATH", "").strip()

    # IMPORTANT: prefer v3, then combined v2, then simple v2, then legacy.
    model_candidates = [
        model_env,
        str(app_dir / "models" / "xgb_tennis_v3.json"),
        str(api_dir / "models" / "xgb_tennis_v3.json"),
        str(cwd / "models" / "xgb_tennis_v3.json"),
        str(app_dir / "models" / "xgb_tennis_combined_v2.json"),
        str(api_dir / "models" / "xgb_tennis_combined_v2.json"),
        str(cwd / "models" / "xgb_tennis_combined_v2.json"),
        str(app_dir / "models" / "xgb_tennis_simple_v2.json"),
        str(api_dir / "models" / "xgb_tennis_simple_v2.json"),
        str(cwd / "models" / "xgb_tennis_simple_v2.json"),
        str(app_dir / "models" / "xgb_tennis_simple.json"),
        str(api_dir / "models" / "xgb_tennis_simple.json"),
        str(cwd / "models" / "xgb_tennis_simple.json"),
        # fallbacks (14-feature model) LAST
        str(app_dir / "models" / "xgb_tennis_lightweight.json"),
        str(api_dir / "models" / "xgb_tennis_lightweight.json"),
        str(cwd / "models" / "xgb_tennis_lightweight.json"),
    ]

    # Your 9-line TXT is named like lightweight_features.txt in your messages — that's fine.
    model_path = _find_file(model_candidates)
    feats_path = _features_for_model(model_path, app_dir, api_dir, cwd, feats_env)
    return model_path, feats_path


# Backwards-compatible name (in case other code still references it)
def _default_model_paths() -> Tuple[Path, Path]:
    return _default_paths()


# -------------------------
# Predictor
# -------------------------

class SimpleTennisPredictor:
    def __init__(self) -> None:
        model_path, feats_path = _default_paths()

        self.model_path = model_path
        self.feats_path = feats_path
        self.feature_cols = _read_feature_cols(feats_path)
        self.calibration = None

        # We only use XGBClassifier as a loader/predictor shell.
        self.model = XGBClassifier()
        self.model.load_model(str(model_path))

        # Optional calibration model (if available)
        calib_env = os.getenv("XGB_TENNIS_CALIBRATION_PATH", "").strip()
        calib_candidates = [
            calib_env,
            str(model_path).replace(".json", "_calibration.joblib"),
            str(Path(str(model_path)).with_name("xgb_tennis_v3_calibration.joblib")),
            str(Path(str(model_path)).with_name("xgb_tennis_combined_v2_calibration.joblib")),
            str(Path(str(model_path)).with_name("xgb_tennis_simple_v2_calibration.joblib")),
        ]
        for c in calib_candidates:
            if not c:
                continue
            p = Path(c)
            if p.exists():
                try:
                    self.calibration = joblib_load(p)
                    logger.info("Loaded calibration model: %s", p)
                except Exception as e:
                    logger.warning("Failed to load calibration model %s: %s", p, e)
                break

        # Guardrail: fail fast if model expects a different number of features.
        # (This is exactly what caused your "expected 14, got 9".)
        try:
            expected = int(self.model.get_booster().num_features())
        except Exception:
            expected = 0

        got = len(self.feature_cols)
        if expected and expected != got:
            raise ValueError(
                f"Feature shape mismatch, expected: {expected}, got {got}. "
                f"Model={model_path} FeaturesFile={feats_path}"
            )

    def predict_proba(self, features: Dict[str, Any]) -> float:
        row = [float(features.get(c, 0.0) or 0.0) for c in self.feature_cols]
        X = np.array(row, dtype=np.float32).reshape(1, -1)
        if self.calibration is not None:
            try:
                p = float(self.calibration.predict_proba(X)[0, 1])
            except Exception:
                p = float(self.model.predict_proba(X)[0, 1])
        else:
            p = float(self.model.predict_proba(X)[0, 1])
        # clamp
        if p < 0.001:
            p = 0.001
        if p > 0.999:
            p = 0.999
        return p

    def predict_proba_batch(self, features_list: List[Dict[str, Any]]) -> List[float]:
        """Predict probabilities for multiple matches in one vectorized call."""
        if not features_list:
            return []
        rows = []
        for features in features_list:
            rows.append([float(features.get(c, 0.0) or 0.0) for c in self.feature_cols])
        X = np.array(rows, dtype=np.float32)  # shape (N, num_features)
        if self.calibration is not None:
            try:
                probs = self.calibration.predict_proba(X)[:, 1].tolist()
            except Exception:
                probs = self.model.predict_proba(X)[:, 1].tolist()
        else:
            probs = self.model.predict_proba(X)[:, 1].tolist()
        return [max(0.001, min(0.999, float(p))) for p in probs]


_PRED: Optional[SimpleTennisPredictor] = None


def get_simple_predictor() -> SimpleTennisPredictor:
    global _PRED
    if _PRED is None:
        _PRED = SimpleTennisPredictor()
    return _PRED


# -------------------------
# Feature engineering (9 features)
# -------------------------

_INSIGHT_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}
_SURFACE_CACHE: Optional[Dict[Tuple[str, str, str, str], Dict[str, float]]] = None
_SURFACE_CACHE_LOADED: bool = False


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


def _fetch_insight_by_match_id(match_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not match_id:
        return None
    key = str(match_id)
    if key in _INSIGHT_CACHE:
        return _INSIGHT_CACHE[key]
    if not DATABASE_URL:
        _INSIGHT_CACHE[key] = None
        return None
    try:
        import psycopg2  # type: ignore
    except Exception:
        logger.warning("psycopg2 not available; skipping insight backfill")
        _INSIGHT_CACHE[key] = None
        return None

    row: Optional[Dict[str, Any]] = None
    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        match_id::text AS match_id,
                        tour,
                        match_date,
                        surface,
                        p1_name,
                        p2_name,
                        p1_service_hold_pct,
                        p1_opponent_hold_pct,
                        p1_service_pts_won_pct,
                        p1_return_pts_won_pct,
                        p1_aces_per_game,
                        p1_dfs_per_game,
                        p1_bp_save_pct,
                        p1_bp_won_pct,
                        p2_service_hold_pct,
                        p2_opponent_hold_pct,
                        p2_service_pts_won_pct,
                        p2_return_pts_won_pct,
                        p2_aces_per_game,
                        p2_dfs_per_game,
                        p2_bp_save_pct,
                        p2_bp_won_pct
                    FROM tennis_insight_match_features
                    WHERE match_id::text = %s
                    LIMIT 1
                    """,
                    (key,),
                )
                data = cur.fetchone()
                if data:
                    cols = [desc[0] for desc in cur.description]
                    row = dict(zip(cols, data))
    except Exception as e:
        logger.warning("Insight DB fetch failed for match_id=%s: %s", key, e)
        row = None

    _INSIGHT_CACHE[key] = row
    return row


def _load_surface_cache() -> Optional[Dict[Tuple[str, str, str, str], Dict[str, float]]]:
    global _SURFACE_CACHE, _SURFACE_CACHE_LOADED
    if _SURFACE_CACHE_LOADED:
        return _SURFACE_CACHE

    _SURFACE_CACHE_LOADED = True
    _SURFACE_CACHE = {}

    here = Path(__file__).resolve()
    app_dir = here.parents[2]  # .../api/app
    surface_path = app_dir / "data" / "player_surface_stats.csv"
    if not surface_path.exists():
        logger.warning("Surface stats CSV missing: %s", surface_path)
        return _SURFACE_CACHE

    try:
        with surface_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                player_key = _norm_name(row.get("player"))
                tour = (row.get("tour") or "").upper().strip()
                surface = (row.get("surface") or "").lower().strip()
                window = (row.get("window") or "").lower().strip()
                if not player_key or not tour or not surface or not window:
                    continue
                key = (tour, surface, window, player_key)
                _SURFACE_CACHE[key] = {
                    "svc_hold_pct": _safe_float(row.get("svc_hold_pct")),
                    "ret_opp_hold_pct": _safe_float(row.get("ret_opp_hold_pct")),
                    "svc_aces_per_game": _safe_float(row.get("svc_aces_per_game")),
                    "svc_dfs_per_game": _safe_float(row.get("svc_dfs_per_game")),
                }
        logger.info("Loaded surface stats: %d entries", len(_SURFACE_CACHE))
    except Exception as e:
        logger.warning("Failed to load surface stats CSV: %s", e)
    return _SURFACE_CACHE


def _maybe_backfill_from_insight(match_data: Dict[str, Any]) -> None:
    # Only try if any TA/style features are missing/zero
    ta_keys = [
        "d_last5_hold", "d_last5_break",
        "d_last10_hold", "d_last10_break",
        "d_surf_last10_hold", "d_surf_last10_break",
        "d_last10_aces_pg", "d_surf_last10_aces_pg",
        "d_last10_df_pg", "d_surf_last10_df_pg",
        "d_last10_tb_match_rate", "d_last10_tb_win_pct",
        "d_surf_last10_tb_match_rate", "d_surf_last10_tb_win_pct",
    ]
    style_keys = ["d_svc_pts_w", "d_ret_pts_w", "d_ace_pg", "d_bp_save", "d_bp_win"]
    if not any(match_data.get(k) in (None, 0, 0.0) for k in ta_keys + style_keys):
        return

    imf = _fetch_insight_by_match_id(match_data.get("match_id"))
    if not imf:
        return

    p1_name = _norm_name(match_data.get("p1_name"))
    imf_p1 = _norm_name(imf.get("p1_name"))
    swapped = bool(p1_name and imf_p1 and p1_name != imf_p1)

    def _imf_diff(col_a: str, col_b: str) -> Optional[float]:
        a = _safe_float(imf.get(col_a))
        b = _safe_float(imf.get(col_b))
        if a is None or b is None:
            return None
        diff = float(a) - float(b)
        return -diff if swapped else diff

    # Style proxies
    style_map = {
        "d_svc_pts_w": ("p1_service_pts_won_pct", "p2_service_pts_won_pct"),
        "d_ret_pts_w": ("p1_return_pts_won_pct", "p2_return_pts_won_pct"),
        "d_ace_pg": ("p1_aces_per_game", "p2_aces_per_game"),
        "d_bp_save": ("p1_bp_save_pct", "p2_bp_save_pct"),
        "d_bp_win": ("p1_bp_won_pct", "p2_bp_won_pct"),
    }
    for feat, (a, b) in style_map.items():
        if match_data.get(feat) in (None, 0, 0.0):
            v = _imf_diff(a, b)
            if v is not None:
                match_data[feat] = v

    # TA rolling proxies
    ta_map = {
        "d_last5_hold": ("p1_service_hold_pct", "p2_service_hold_pct"),
        "d_last10_hold": ("p1_service_hold_pct", "p2_service_hold_pct"),
        "d_last5_break": ("p2_opponent_hold_pct", "p1_opponent_hold_pct"),
        "d_last10_break": ("p2_opponent_hold_pct", "p1_opponent_hold_pct"),
        "d_last10_aces_pg": ("p1_aces_per_game", "p2_aces_per_game"),
        "d_last10_df_pg": ("p1_dfs_per_game", "p2_dfs_per_game"),
    }
    for feat, (a, b) in ta_map.items():
        if match_data.get(feat) in (None, 0, 0.0):
            v = _imf_diff(a, b)
            if v is not None:
                match_data[feat] = v


def _maybe_backfill_surface_stats(match_data: Dict[str, Any]) -> None:
    surface_cache = _load_surface_cache()
    if not surface_cache:
        return

    surf = _surface_key(match_data.get("surface"))
    tour = (match_data.get("tour") or "").upper()
    if not surf or not tour:
        return

    p1_key = _norm_name(match_data.get("p1_name"))
    p2_key = _norm_name(match_data.get("p2_name"))
    if not p1_key or not p2_key:
        return

    windows = ["12 month", "all time"]
    def _lookup(player_key: str) -> Optional[Dict[str, float]]:
        for w in windows:
            row = surface_cache.get((tour, surf, w, player_key))
            if row:
                return row
        return None

    p1_row = _lookup(p1_key)
    p2_row = _lookup(p2_key)
    if not p1_row or not p2_row:
        return

    def _set_if_missing(key: str, val: Optional[float]) -> None:
        if val is None:
            return
        if match_data.get(key) in (None, 0, 0.0):
            match_data[key] = val

    v1 = _safe_float(p1_row.get("svc_hold_pct"))
    v2 = _safe_float(p2_row.get("svc_hold_pct"))
    if v1 is not None and v2 is not None:
        _set_if_missing("d_surf_last10_hold", float(v1) - float(v2))

    v1 = _safe_float(p1_row.get("svc_aces_per_game"))
    v2 = _safe_float(p2_row.get("svc_aces_per_game"))
    if v1 is not None and v2 is not None:
        _set_if_missing("d_surf_last10_aces_pg", float(v1) - float(v2))

    v1 = _safe_float(p1_row.get("svc_dfs_per_game"))
    v2 = _safe_float(p2_row.get("svc_dfs_per_game"))
    if v1 is not None and v2 is not None:
        _set_if_missing("d_surf_last10_df_pg", float(v1) - float(v2))

    # break% proxy: opponent hold (lower is better return); diff is p2 - p1
    p1_opp_hold = _safe_float(p1_row.get("ret_opp_hold_pct"))
    p2_opp_hold = _safe_float(p2_row.get("ret_opp_hold_pct"))
    if p1_opp_hold is not None and p2_opp_hold is not None:
        _set_if_missing("d_surf_last10_break", float(p2_opp_hold) - float(p1_opp_hold))

def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _american_to_implied_prob(odds: Optional[int]) -> Optional[float]:
    o = _safe_int(odds)
    if o is None:
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


def _surface_flags(surface: Optional[str]) -> tuple[int, int, int]:
    s = (surface or "").lower()
    return (int("hard" in s), int("clay" in s), int("grass" in s))


def _is_grand_slam(tournament: Optional[str]) -> int:
    t = (tournament or "").lower()
    return int(any(x in t for x in ["australian open", "roland garros", "french open", "wimbledon", "us open"]))


def _is_masters(tournament: Optional[str]) -> int:
    t = (tournament or "").lower()
    return int(("masters" in t) or ("1000" in t))


# ── Round ordinal encoding (used by v3 model) ─────────────────────────────
_ROUND_ORDER: Dict[str, int] = {
    "qualification": 1, "q1": 1, "q2": 2, "q3": 3,
    "round of 128": 4, "r128": 4,
    "round of 64": 5, "r64": 5,
    "round of 32": 6, "r32": 6,
    "round of 16": 7, "r16": 7,
    "quarterfinal": 8, "quarterfinals": 8, "qf": 8,
    "semifinal": 9, "semifinals": 9, "sf": 9,
    "final": 10, "f": 10,
    "round robin": 5, "group": 5,
}


def _round_ordinal(round_str: Optional[str]) -> int:
    if not round_str:
        return 0
    s = round_str.strip().lower()
    if s in _ROUND_ORDER:
        return _ROUND_ORDER[s]
    for key, val in _ROUND_ORDER.items():
        if key in s:
            return val
    return 0


def _surface_elo(match_data: Dict[str, Any], player: str) -> Optional[float]:
    """Pick the right surface-specific ELO for a player."""
    surface = (match_data.get("surface") or "").lower()
    if "hard" in surface:
        return _safe_float(match_data.get(f"{player}_helo"))
    if "clay" in surface:
        return _safe_float(match_data.get(f"{player}_celo"))
    if "grass" in surface:
        return _safe_float(match_data.get(f"{player}_gelo"))
    return _safe_float(match_data.get(f"{player}_elo"))


def _build_features(match_data: Dict[str, Any]) -> Dict[str, Any]:
    md = dict(match_data)
    _maybe_backfill_from_insight(md)
    _maybe_backfill_surface_stats(md)
    match_data = md
    # ── ELO features (v3 — the biggest accuracy gain) ────────────────────
    p1_elo = _safe_float(match_data.get("p1_elo"), 0.0)
    p2_elo = _safe_float(match_data.get("p2_elo"), 0.0)
    d_elo = (float(p1_elo) - float(p2_elo)) if p1_elo and p2_elo else 0.0

    p1_selo = _surface_elo(match_data, "p1")
    p2_selo = _surface_elo(match_data, "p2")
    d_surface_elo = 0.0
    if p1_selo is not None and p2_selo is not None:
        d_surface_elo = float(p1_selo) - float(p2_selo)
    elif d_elo != 0.0:
        d_surface_elo = d_elo  # fallback to overall ELO diff

    # Ranking diff (log-space)
    p1_rank = _safe_float(match_data.get("p1_rank") or match_data.get("p1_official_rank"))
    p2_rank = _safe_float(match_data.get("p2_rank") or match_data.get("p2_official_rank"))
    d_rank_log = 0.0
    if p1_rank and p2_rank and p1_rank > 0 and p2_rank > 0:
        d_rank_log = math.log(p2_rank) - math.log(p1_rank)

    # Age diff
    p1_age = _safe_float(match_data.get("p1_age"))
    p2_age = _safe_float(match_data.get("p2_age"))
    d_age = 0.0
    if p1_age is not None and p2_age is not None:
        d_age = float(p1_age) - float(p2_age)

    # ── Rolling form ─────────────────────────────────────────────────────
    p1_wr20 = _safe_float(match_data.get("p1_win_rate_last_20"), 0.5) or 0.5
    p2_wr20 = _safe_float(match_data.get("p2_win_rate_last_20"), 0.5) or 0.5

    p1_surf = _safe_float(match_data.get("p1_win_rate_surface"), p1_wr20) or p1_wr20
    p2_surf = _safe_float(match_data.get("p2_win_rate_surface"), p2_wr20) or p2_wr20

    p1_mp = _safe_int(match_data.get("p1_matches_played"), 0) or 0
    p2_mp = _safe_int(match_data.get("p2_matches_played"), 0) or 0

    d_wr20 = float(p1_wr20) - float(p2_wr20)
    d_wsurf = float(p1_surf) - float(p2_surf)
    d_exp = math.log1p(p1_mp) - math.log1p(p2_mp)

    d_rest = _safe_float(match_data.get("d_rest_days"), 0.0) or 0.0
    d_m10 = _safe_float(match_data.get("d_matches_10d"), 0.0) or 0.0
    d_m30 = _safe_float(match_data.get("d_matches_30d"), 0.0) or 0.0

    # ── Context ───────────────────────────────────────────────────────────
    is_hard, is_clay, is_grass = _surface_flags(match_data.get("surface"))
    is_gs = _is_grand_slam(match_data.get("tournament"))
    is_m = _is_masters(match_data.get("tournament"))
    best_of = _safe_int(match_data.get("best_of"), 3) or 3
    round_ord = _round_ordinal(match_data.get("round"))

    # ── H2H ───────────────────────────────────────────────────────────────
    h2h_p1_win_pct = _safe_float(match_data.get("h2h_p1_win_pct"), 0.5) or 0.5
    h2h_total_matches = _safe_int(match_data.get("h2h_total_matches"), 0) or 0
    h2h_surface_p1_win_pct = _safe_float(match_data.get("h2h_surface_p1_win_pct"), 0.5) or 0.5
    h2h_surface_matches = _safe_int(match_data.get("h2h_surface_matches"), 0) or 0

    # ── Style proxies ─────────────────────────────────────────────────────
    def _diff(a: Optional[float], b: Optional[float], default: float = 0.0) -> float:
        if a is None or b is None:
            return default
        return float(a) - float(b)

    p1_svc_pts = _safe_float(match_data.get("p1_svc_pts_w"))
    p2_svc_pts = _safe_float(match_data.get("p2_svc_pts_w"))
    p1_ret_pts = _safe_float(match_data.get("p1_ret_pts_w"))
    p2_ret_pts = _safe_float(match_data.get("p2_ret_pts_w"))
    p1_ace_pg = _safe_float(match_data.get("p1_ace_pg"))
    p2_ace_pg = _safe_float(match_data.get("p2_ace_pg"))
    p1_bp_save = _safe_float(match_data.get("p1_bp_save"))
    p2_bp_save = _safe_float(match_data.get("p2_bp_save"))
    p1_bp_win = _safe_float(match_data.get("p1_bp_win"))
    p2_bp_win = _safe_float(match_data.get("p2_bp_win"))

    # ── TA rolling features (v3 — populate from match_data) ──────────────
    def _ta(key: str) -> float:
        """Grab a TA feature from match_data, default 0."""
        return _safe_float(match_data.get(key), 0.0) or 0.0

    d_svc_pts_w = _safe_float(match_data.get("d_svc_pts_w"))
    if d_svc_pts_w in (None, 0, 0.0):
        d_svc_pts_w = _diff(p1_svc_pts, p2_svc_pts)
    d_ret_pts_w = _safe_float(match_data.get("d_ret_pts_w"))
    if d_ret_pts_w in (None, 0, 0.0):
        d_ret_pts_w = _diff(p1_ret_pts, p2_ret_pts)
    d_ace_pg = _safe_float(match_data.get("d_ace_pg"))
    if d_ace_pg in (None, 0, 0.0):
        d_ace_pg = _diff(p1_ace_pg, p2_ace_pg)
    d_bp_save = _safe_float(match_data.get("d_bp_save"))
    if d_bp_save in (None, 0, 0.0):
        d_bp_save = _diff(p1_bp_save, p2_bp_save)
    d_bp_win = _safe_float(match_data.get("d_bp_win"))
    if d_bp_win in (None, 0, 0.0):
        d_bp_win = _diff(p1_bp_win, p2_bp_win)

    # Market totals / spreads (available for qualifying + WTA125 when posted)
    total_line = _safe_float(match_data.get("sofascore_total_games_line"))
    total_over_odds = _safe_int(match_data.get("sofascore_total_games_over_american"))
    total_under_odds = _safe_int(match_data.get("sofascore_total_games_under_american"))
    total_over_imp = _american_to_implied_prob(total_over_odds)
    total_under_imp = _american_to_implied_prob(total_under_odds)
    total_over_nv, total_under_nv = _no_vig_two_way(total_over_imp, total_under_imp)
    total_overround = None
    if total_over_imp is not None and total_under_imp is not None:
        total_overround = float(total_over_imp + total_under_imp - 1.0)

    p1_spread_line = _safe_float(match_data.get("sofascore_spread_p1_line"))
    p2_spread_line = _safe_float(match_data.get("sofascore_spread_p2_line"))
    p1_spread_odds = _safe_int(match_data.get("sofascore_spread_p1_odds_american"))
    p2_spread_odds = _safe_int(match_data.get("sofascore_spread_p2_odds_american"))
    p1_spread_imp = _american_to_implied_prob(p1_spread_odds)
    p2_spread_imp = _american_to_implied_prob(p2_spread_odds)
    p1_spread_nv, p2_spread_nv = _no_vig_two_way(p1_spread_imp, p2_spread_imp)

    tour = (match_data.get("tour") or "").upper()
    # Lightweight centering so totals are comparable across tours.
    total_baseline = 22.0 if tour == "ATP" else 21.5
    total_line_centered = (float(total_line) - total_baseline) if total_line is not None else 0.0
    has_total_line = 1 if total_line is not None else 0
    has_game_spread = 1 if (p1_spread_line is not None or p2_spread_line is not None) else 0

    return {
        # ELO block (NEW)
        "d_elo": d_elo,
        "d_surface_elo": d_surface_elo,
        "d_rank_log": d_rank_log,
        "d_age": d_age,
        # Rolling form
        "d_win_rate_last_20": d_wr20,
        "d_win_rate_surface": d_wsurf,
        "d_experience": d_exp,
        "d_rest_days": d_rest,
        "d_matches_10d": d_m10,
        "d_matches_30d": d_m30,
        # Context
        "is_hard": is_hard,
        "is_clay": is_clay,
        "is_grass": is_grass,
        "is_grand_slam": is_gs,
        "is_masters": is_m,
        "best_of": best_of,
        "round_ordinal": round_ord,
        # H2H
        "h2h_p1_win_pct": h2h_p1_win_pct,
        "h2h_total_matches": h2h_total_matches,
        "h2h_surface_p1_win_pct": h2h_surface_p1_win_pct,
        "h2h_surface_matches": h2h_surface_matches,
        # Style proxies
        "d_svc_pts_w": d_svc_pts_w,
        "d_ret_pts_w": d_ret_pts_w,
        "d_ace_pg": d_ace_pg,
        "d_bp_save": d_bp_save,
        "d_bp_win": d_bp_win,
        # Totals/spread market features
        "has_total_line": has_total_line,
        "market_total_line": float(total_line) if total_line is not None else 0.0,
        "market_total_line_centered": total_line_centered,
        "market_total_over_implied": float(total_over_imp) if total_over_imp is not None else 0.0,
        "market_total_under_implied": float(total_under_imp) if total_under_imp is not None else 0.0,
        "market_total_over_no_vig": float(total_over_nv) if total_over_nv is not None else 0.5,
        "market_total_under_no_vig": float(total_under_nv) if total_under_nv is not None else 0.5,
        "market_total_overround": float(total_overround) if total_overround is not None else 0.0,
        "has_game_spread": has_game_spread,
        "market_spread_p1_line": float(p1_spread_line) if p1_spread_line is not None else 0.0,
        "market_spread_p2_line": float(p2_spread_line) if p2_spread_line is not None else 0.0,
        "market_spread_p1_implied": float(p1_spread_imp) if p1_spread_imp is not None else 0.0,
        "market_spread_p2_implied": float(p2_spread_imp) if p2_spread_imp is not None else 0.0,
        "market_spread_p1_no_vig": float(p1_spread_nv) if p1_spread_nv is not None else 0.5,
        "market_spread_p2_no_vig": float(p2_spread_nv) if p2_spread_nv is not None else 0.5,
        # TA rolling
        "d_last5_hold": _ta("d_last5_hold"),
        "d_last5_break": _ta("d_last5_break"),
        "d_last10_hold": _ta("d_last10_hold"),
        "d_last10_break": _ta("d_last10_break"),
        "d_surf_last10_hold": _ta("d_surf_last10_hold"),
        "d_surf_last10_break": _ta("d_surf_last10_break"),
        "d_last10_aces_pg": _ta("d_last10_aces_pg"),
        "d_surf_last10_aces_pg": _ta("d_surf_last10_aces_pg"),
        "d_last10_df_pg": _ta("d_last10_df_pg"),
        "d_surf_last10_df_pg": _ta("d_surf_last10_df_pg"),
        "d_last10_tb_match_rate": _ta("d_last10_tb_match_rate"),
        "d_last10_tb_win_pct": _ta("d_last10_tb_win_pct"),
        "d_surf_last10_tb_match_rate": _ta("d_surf_last10_tb_match_rate"),
        "d_surf_last10_tb_win_pct": _ta("d_surf_last10_tb_win_pct"),
    }


def predict_match_xgb(match_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
      {
        p1_win_prob, p2_win_prob,
        method,
        features
      }
    """
    try:
        predictor = get_simple_predictor()
        feats = _build_features(match_data)
        p1 = predictor.predict_proba(feats)
        return {
            "p1_win_prob": p1,
            "p2_win_prob": 1.0 - p1,
            "method": "xgb_simple_9f",
            "features": feats,
        }
    except Exception:
        # If you add logger.exception("XGB predictor failed") elsewhere, it will print to your Uvicorn terminal.
        logger.exception("XGB predictor failed")
        raise


def predict_match_xgb_batch(match_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batch version — one predict_proba call for all matches."""
    predictor = get_simple_predictor()
    feats_list = [_build_features(md) for md in match_data_list]
    probs = predictor.predict_proba_batch(feats_list)
    results = []
    for feats, p1 in zip(feats_list, probs):
        results.append({
            "p1_win_prob": p1,
            "p2_win_prob": 1.0 - p1,
            "method": "xgb_simple_9f",
            "features": feats,
        })
    return results
