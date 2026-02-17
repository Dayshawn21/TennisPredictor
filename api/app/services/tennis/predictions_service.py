from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict, List, Optional


async def get_predictions_today_enhanced(
    *,
    days_ahead: int,
    include_incomplete: bool,
    bust_cache: bool,
    min_edge: Optional[float] = None,
    max_odds_age_min: Optional[int] = None,
    max_overround: Optional[float] = None,
    impl: Callable[..., Awaitable[Any]],
) -> Any:
    try:
        return await impl(
            days_ahead,
            include_incomplete,
            bust_cache,
            min_edge=min_edge,
            max_odds_age_min=max_odds_age_min,
            max_overround=max_overround,
        )
    except TypeError:
        # Backward compatibility for callsites still using the old impl signature.
        return await impl(days_ahead, include_incomplete, bust_cache)


async def get_predictions_today(
    *,
    days_ahead: int,
    include_incomplete: bool,
    bust_cache: bool,
    impl: Callable[[int, bool, bool], Awaitable[Any]],
) -> Any:
    return await impl(days_ahead, include_incomplete, bust_cache)


async def get_suggested_parlay(
    *,
    legs: int,
    top_n: int,
    days_ahead: int,
    include_incomplete: bool,
    min_parlay_payout: int,
    min_edge: Optional[float],
    min_ev: Optional[float],
    min_leg_odds: Optional[int],
    candidate_pool: int,
    max_overlap: int,
    objective: str,
    impl: Callable[..., Awaitable[Any]],
) -> Any:
    return await impl(
        legs=legs,
        top_n=top_n,
        days_ahead=days_ahead,
        include_incomplete=include_incomplete,
        min_parlay_payout=min_parlay_payout,
        min_edge=min_edge,
        min_ev=min_ev,
        min_leg_odds=min_leg_odds,
        candidate_pool=candidate_pool,
        max_overlap=max_overlap,
        objective=objective,
    )


async def get_h2h_matches(
    *,
    event_id: int,
    fetch_payload: Callable[[int], Awaitable[Any]],
    extract_matches: Callable[[Dict[str, Any], int], List[Dict[str, Any]]],
    h2h_model: Callable[..., Any],
) -> List[Any]:
    payload = await fetch_payload(event_id)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = None
    if isinstance(payload, dict):
        return [h2h_model(**m) for m in extract_matches(payload, 10)]
    return []


async def get_debug_stats_lookup(
    *,
    player: str,
    tour: str,
    surface: str,
    impl: Callable[[str, str, str], Awaitable[Dict[str, Any]]],
) -> Dict[str, Any]:
    return await impl(player, tour, surface)
