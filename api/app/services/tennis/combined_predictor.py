# FILE: api/app/services/tennis/combined_predictor.py
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Dict, Optional, List, Tuple

from app.services.tennis.elo_predictor_enhanced import predict_match_elo

_XGB_IMPORT_ERROR: Optional[str] = None
try:
    from app.services.tennis.tennis_predictor_simple import predict_match_xgb, predict_match_xgb_batch
except Exception as e:  # pragma: no cover - defensive runtime fallback
    predict_match_xgb = None  # type: ignore[assignment]
    predict_match_xgb_batch = None  # type: ignore[assignment]
    _XGB_IMPORT_ERROR = f"xgb_import_error:{type(e).__name__}:{e}"


def _american_to_implied_prob(odds: Optional[int]) -> Optional[float]:
    if odds is None:
        return None
    try:
        odds = int(odds)
    except Exception:
        return None

    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return None


def _no_vig_two_way(p1: Optional[float], p2: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    if p1 is None or p2 is None:
        return (None, None)
    s = p1 + p2
    if s <= 0:
        return (None, None)
    return (p1 / s, p2 / s)


def _clamp_prob(p: float) -> float:
    if p < 0.001:
        return 0.001
    if p > 0.999:
        return 0.999
    return p


def _market_quality_flags(match_data: Dict[str, Any]) -> tuple[list[str], dict[str, Any], float]:
    flags: list[str] = []

    p1_imp = _american_to_implied_prob(match_data.get("p1_odds_american"))
    p2_imp = _american_to_implied_prob(match_data.get("p2_odds_american"))
    if p1_imp is None or p2_imp is None:
        flags.append("missing_main_odds")
        overround = None
    else:
        overround = float(p1_imp + p2_imp - 1.0)

    has_fresh_odds = match_data.get("has_fresh_odds")
    if has_fresh_odds is False:
        flags.append("stale_odds")

    high_overround = match_data.get("high_overround")
    if high_overround is None and overround is not None:
        try:
            max_overround = float(match_data.get("max_overround", 0.08))
        except Exception:
            max_overround = 0.08
        high_overround = overround > max_overround
    if bool(high_overround):
        flags.append("high_overround")

    market_multiplier = 1.0
    if p1_imp is None or p2_imp is None:
        market_multiplier = 0.0
    elif "stale_odds" in flags:
        market_multiplier = 0.5
    elif "high_overround" in flags:
        market_multiplier = 0.75

    diagnostics = {
        "odds_age_minutes": match_data.get("odds_age_minutes"),
        "market_overround_main": overround if overround is not None else match_data.get("market_overround_main"),
        "has_fresh_odds": has_fresh_odds,
    }

    return flags, diagnostics, market_multiplier


def _xgb_has_key_feature_groups(match_data: Dict[str, Any]) -> bool:
    ta_keys = (
        "d_last10_hold",
        "d_last10_break",
        "d_surf_last10_hold",
        "d_surf_last10_break",
        "d_last10_tb_match_rate",
    )
    style_keys = (
        "p1_svc_pts_w",
        "p2_svc_pts_w",
        "p1_ret_pts_w",
        "p2_ret_pts_w",
    )
    rolling_keys = (
        "p1_win_rate_last_20",
        "p2_win_rate_last_20",
        "p1_win_rate_surface",
        "p2_win_rate_surface",
    )

    has_ta = any(k in match_data for k in ta_keys)
    # Upcoming matches may not exist in tennis_features_ta yet.
    # Accept Tennis Insight proxy diffs as equivalent strength-layer coverage.
    proxy_keys = (
        "serve_return_edge_w",
        "d_srv_pts_w_w",
        "d_ret_pts_w_w",
        "d_hold_w",
    )
    has_proxy = any(match_data.get(k) is not None for k in proxy_keys)
    has_style = all(match_data.get(k) is not None for k in style_keys)
    has_rolling = all(match_data.get(k) is not None for k in rolling_keys)
    return (has_ta or has_proxy or has_style) and has_rolling


def _elo_multiplier_and_flags(match_data: Dict[str, Any]) -> tuple[float, list[str]]:
    flags: list[str] = []
    heavy_fallback = bool(
        match_data.get("elo_used_median")
        or match_data.get("p1_elo_used_median")
        or match_data.get("p2_elo_used_median")
    )
    if heavy_fallback:
        flags.append("elo_fallback_used")
    return (0.7 if heavy_fallback else 1.0), flags


def _normalize_weights(
    components: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, float], float]:
    wsum = sum(float(x["weight"]) for x in components)
    effective: Dict[str, float] = {}
    if wsum > 0:
        for c in components:
            ew = float(c["weight"]) / wsum
            c["effective_weight"] = ew
            effective[c["name"]] = ew
    return components, effective, wsum


def _build_combined_result(
    match_data: Dict[str, Any],
    *,
    xgb_prob: Optional[float] = None,
    xgb_error_msg: Optional[str] = None,
) -> Dict[str, Any]:
    individual: List[Dict[str, Any]] = []
    missing: List[str] = []
    quality_flags: List[str] = []

    # Quality-aware multipliers
    elo_mult, elo_flags = _elo_multiplier_and_flags(match_data)
    quality_flags.extend(elo_flags)

    has_xgb_feature_groups = _xgb_has_key_feature_groups(match_data)
    xgb_mult = 1.0 if has_xgb_feature_groups else 0.8
    if not has_xgb_feature_groups:
        quality_flags.append("xgb_feature_groups_partial")

    market_flags, market_diag, market_mult = _market_quality_flags(match_data)
    quality_flags.extend(market_flags)

    # ELO predictor (only if ELO present)
    if match_data.get("p1_elo") is not None and match_data.get("p2_elo") is not None:
        try:
            elo_out = predict_match_elo(match_data)
            p_elo = elo_out.get("p1_win_prob")
            if p_elo is not None:
                individual.append({"name": "elo", "p1_prob": float(p_elo), "weight": 0.40 * elo_mult})
        except Exception as e:
            missing.append(f"elo_error:{type(e).__name__}:{str(e)}")
    else:
        missing.append("elo_missing")
        quality_flags.append("elo_missing")

    # XGB predictor
    if _XGB_IMPORT_ERROR:
        missing.append(_XGB_IMPORT_ERROR)
        quality_flags.append("xgb_missing")
    elif xgb_error_msg:
        missing.append(xgb_error_msg)
        quality_flags.append("xgb_missing")
    else:
        if xgb_prob is None:
            try:
                xgb_out = predict_match_xgb(match_data)
                xgb_prob = xgb_out.get("p1_win_prob")
            except Exception as e:
                missing.append(f"xgb_error:{type(e).__name__}:{e}")
                quality_flags.append("xgb_missing")
        if xgb_prob is not None:
            individual.append({"name": "xgb", "p1_prob": float(xgb_prob), "weight": 0.40 * xgb_mult})

    # Market (no-vig)
    p1_imp = _american_to_implied_prob(match_data.get("p1_odds_american"))
    p2_imp = _american_to_implied_prob(match_data.get("p2_odds_american"))
    p1_nv, _ = _no_vig_two_way(p1_imp, p2_imp)
    if p1_nv is not None and market_mult > 0.0:
        individual.append({"name": "market_no_vig", "p1_prob": float(p1_nv), "weight": 0.20 * market_mult})
    else:
        missing.append("market_missing")
        quality_flags.append("market_missing")

    if not individual:
        return {
            "p1_win_prob": None,
            "p2_win_prob": None,
            "predicted_winner": None,
            "method": "combined_ensemble",
            "num_predictors": 0,
            "individual_predictions": [],
            "missing_reason": ",".join(missing) if missing else "no_predictors",
            "effective_weights": {},
            "predictor_quality_flags": sorted(set(quality_flags)),
            "odds_age_minutes": market_diag.get("odds_age_minutes"),
            "market_overround_main": market_diag.get("market_overround_main"),
        }

    individual, effective_weights, wsum = _normalize_weights(individual)
    if wsum <= 0:
        return {
            "p1_win_prob": None,
            "p2_win_prob": None,
            "predicted_winner": None,
            "method": "combined_ensemble",
            "num_predictors": 0,
            "individual_predictions": individual,
            "missing_reason": "weights_sum_zero",
            "effective_weights": effective_weights,
            "predictor_quality_flags": sorted(set(quality_flags)),
            "odds_age_minutes": market_diag.get("odds_age_minutes"),
            "market_overround_main": market_diag.get("market_overround_main"),
        }

    p1 = sum(float(x["p1_prob"]) * float(x["effective_weight"]) for x in individual)
    p1 = _clamp_prob(float(p1))
    p2 = 1.0 - p1
    predicted = "p1" if p1 >= 0.5 else "p2"

    return {
        "p1_win_prob": p1,
        "p2_win_prob": p2,
        "predicted_winner": predicted,
        "method": "combined_ensemble",
        "num_predictors": len(individual),
        "individual_predictions": individual,
        "missing_reason": ",".join(missing) if missing else None,
        "effective_weights": effective_weights,
        "predictor_quality_flags": sorted(set(quality_flags)),
        "odds_age_minutes": market_diag.get("odds_age_minutes"),
        "market_overround_main": market_diag.get("market_overround_main"),
    }


def predict_match_combined(match_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
      {
        p1_win_prob, p2_win_prob,
        predicted_winner,
        method,
        num_predictors,
        individual_predictions,
        missing_reason
      }
    """
    return _build_combined_result(match_data)


# ---------------------------------------------------------------------------
# Batch prediction (runs CPU work off the async event loop)
# ---------------------------------------------------------------------------
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def predict_batch_combined(match_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Batch version of predict_match_combined.
    Runs all XGBoost predictions in one vectorized call, then combines with ELO + market.
    """
    # Step 1: Batch XGBoost inference (single predict_proba call)
    xgb_results: List[Optional[Dict[str, Any]]] = [None] * len(match_data_list)
    xgb_error_msg: Optional[str] = None
    if _XGB_IMPORT_ERROR:
        xgb_error_msg = _XGB_IMPORT_ERROR
    else:
        try:
            xgb_batch = predict_match_xgb_batch(match_data_list)
            for i, r in enumerate(xgb_batch):
                xgb_results[i] = r
        except Exception as e:
            xgb_error_msg = f"xgb_error:{type(e).__name__}:{e}"

    # Step 2: Per-match ELO + market + dynamic blend
    results: List[Dict[str, Any]] = []
    for i, match_data in enumerate(match_data_list):
        xgb_prob = None
        if xgb_results[i] is not None:
            xgb_prob = xgb_results[i].get("p1_win_prob")
        results.append(
            _build_combined_result(
                match_data,
                xgb_prob=float(xgb_prob) if xgb_prob is not None else None,
                xgb_error_msg=xgb_error_msg,
            )
        )

    return results


async def predict_batch_combined_async(match_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run batch prediction off the event loop so we don't block async I/O."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, predict_batch_combined, match_data_list)
