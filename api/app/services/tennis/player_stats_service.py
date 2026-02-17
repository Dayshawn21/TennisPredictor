from __future__ import annotations

from datetime import date
from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi import HTTPException
from sqlalchemy import text


async def get_player_full_stats(
    *,
    player: str,
    tour: str,
    surface: str,
    debug: bool,
    impl: Callable[[str, str, str, bool], Awaitable[Dict[str, Any]]],
) -> Dict[str, Any]:
    return await impl(player, tour, surface, debug)


async def get_players_compare(
    *,
    player1: str,
    player2: str,
    tour: str,
    surface: str,
    event_id: int | None,
    debug: bool,
    impl: Callable[[str, str, str, str, int | None, bool], Awaitable[Dict[str, Any]]],
) -> Dict[str, Any]:
    return await impl(player1, player2, tour, surface, event_id, debug)


def validate_players_compare_inputs(
    *,
    player1: str,
    player2: str,
    norm_name_stats: Callable[[str], str | None],
    norm_name: Callable[[str], str | None],
) -> tuple[str, str]:
    p1 = (player1 or "").strip()
    p2 = (player2 or "").strip()
    if not p1 or not p2:
        raise HTTPException(status_code=400, detail="player1 and player2 are required")
    if (norm_name_stats(p1) or norm_name(p1) or p1.lower()) == (norm_name_stats(p2) or norm_name(p2) or p2.lower()):
        raise HTTPException(status_code=400, detail="player1 and player2 must be different")
    return p1, p2


def build_players_compare_response(
    *,
    player1: str,
    player2: str,
    tour: str,
    surface: str,
    event_id: int | None,
    debug: bool,
    left: dict[str, Any],
    right: dict[str, Any],
    last_10_matchups: list[dict[str, Any]],
    left_last_10: list[dict[str, Any]],
    right_last_10: list[dict[str, Any]],
    left_last_10_resolution: dict[str, Any],
    right_last_10_resolution: dict[str, Any],
    h2h_overall: dict[str, Any],
    h2h_surface: dict[str, Any],
    norm_name_simple: Callable[[str | None], str],
    normalize_surface_family: Callable[[str | None], str],
    diff_num: Callable[[Any, Any], float | None],
    surface_key: Callable[[str], str],
) -> dict[str, Any]:
    if (h2h_overall.get("p1_wins", 0) + h2h_overall.get("p2_wins", 0) == 0) and last_10_matchups:
        p1n = norm_name_simple(player1)
        p2n = norm_name_simple(player2)
        req_surface = normalize_surface_family(surface)
        p1_w = 0
        p2_w = 0
        p1_sw = 0
        p2_sw = 0
        stot = 0
        for m in last_10_matchups:
            wn = norm_name_simple(m.get("winner_name"))
            if wn == p1n:
                p1_w += 1
            elif wn == p2n:
                p2_w += 1
            ms = normalize_surface_family(m.get("surface"))
            if ms == req_surface:
                stot += 1
                if wn == p1n:
                    p1_sw += 1
                elif wn == p2n:
                    p2_sw += 1
        h2h_overall = {"p1_wins": p1_w, "p2_wins": p2_w, "total": int(len(last_10_matchups))}
        h2h_surface = {"p1_wins": p1_sw, "p2_wins": p2_sw, "total": stot}

    h2h_record = {
        "overall": f"{h2h_overall.get('p1_wins', 0)}-{h2h_overall.get('p2_wins', 0)}",
        "surface": f"{h2h_surface.get('p1_wins', 0)}-{h2h_surface.get('p2_wins', 0)}",
    }

    def _any_resolved(payload: dict[str, Any]) -> bool:
        ms = payload.get("match_status") or {}
        return bool(ms.get("csv_rows") or ms.get("rolling") or ms.get("elo_latest"))

    def _has_any_missing(payload: dict[str, Any]) -> bool:
        ms = payload.get("match_status") or {}
        return not (bool(ms.get("csv_rows")) and bool(ms.get("rolling")) and bool(ms.get("elo_latest")))

    def _has_ambig(payload: dict[str, Any]) -> bool:
        qf = payload.get("quality_flags") or []
        return any("ambig" in str(x).lower() for x in qf)

    compare: dict[str, Any] = {
        "elo": {
            "elo": diff_num((left.get("elo_latest") or {}).get("elo"), (right.get("elo_latest") or {}).get("elo")),
            "helo": diff_num((left.get("elo_latest") or {}).get("helo"), (right.get("elo_latest") or {}).get("helo")),
            "celo": diff_num((left.get("elo_latest") or {}).get("celo"), (right.get("elo_latest") or {}).get("celo")),
            "gelo": diff_num((left.get("elo_latest") or {}).get("gelo"), (right.get("elo_latest") or {}).get("gelo")),
        },
        "rolling": {
            "win_rate_last_20": diff_num((left.get("rolling_stats") or {}).get("win_rate_last_20"), (right.get("rolling_stats") or {}).get("win_rate_last_20")),
            "hard_win_rate_last_10": diff_num((left.get("rolling_stats") or {}).get("hard_win_rate_last_10"), (right.get("rolling_stats") or {}).get("hard_win_rate_last_10")),
            "clay_win_rate_last_10": diff_num((left.get("rolling_stats") or {}).get("clay_win_rate_last_10"), (right.get("rolling_stats") or {}).get("clay_win_rate_last_10")),
            "grass_win_rate_last_10": diff_num((left.get("rolling_stats") or {}).get("grass_win_rate_last_10"), (right.get("rolling_stats") or {}).get("grass_win_rate_last_10")),
        },
        "current_surface_rates": {
            "service_points_won": diff_num((left.get("current_surface_rates") or {}).get("service_points_won"), (right.get("current_surface_rates") or {}).get("service_points_won")),
            "return_points_won": diff_num((left.get("current_surface_rates") or {}).get("return_points_won"), (right.get("current_surface_rates") or {}).get("return_points_won")),
        },
        "surface_rows": {},
    }

    for surf in ("hard", "clay", "grass"):
        l_s = ((left.get("surface_rows") or {}).get(surf) or {})
        r_s = ((right.get("surface_rows") or {}).get(surf) or {})
        l12 = l_s.get("row_12m") or {}
        r12 = r_s.get("row_12m") or {}
        la = l_s.get("row_all_time") or {}
        ra = r_s.get("row_all_time") or {}
        compare["surface_rows"][surf] = {
            "row_12m": {
                "svc_pts": diff_num(l12.get("svc_pts"), r12.get("svc_pts")),
                "ret_pts": diff_num(l12.get("ret_pts"), r12.get("ret_pts")),
                "svc_hold_pct": diff_num(l12.get("svc_hold_pct"), r12.get("svc_hold_pct")),
                "ret_bp_win_pct": diff_num(l12.get("ret_bp_win_pct"), r12.get("ret_bp_win_pct")),
            },
            "row_all_time": {
                "svc_pts": diff_num(la.get("svc_pts"), ra.get("svc_pts")),
                "ret_pts": diff_num(la.get("ret_pts"), ra.get("ret_pts")),
                "svc_hold_pct": diff_num(la.get("svc_hold_pct"), ra.get("svc_hold_pct")),
                "ret_bp_win_pct": diff_num(la.get("ret_bp_win_pct"), ra.get("ret_bp_win_pct")),
            },
        }

    quality = {
        "both_resolved": _any_resolved(left) and _any_resolved(right),
        "partial_data": _has_any_missing(left) or _has_any_missing(right),
        "any_ambiguous": _has_ambig(left) or _has_ambig(right),
    }

    debug_payload = None
    if debug:
        debug_payload = {
            "left": left.get("debug"),
            "right": right.get("debug"),
            "last_10": {
                "left": left_last_10_resolution,
                "right": right_last_10_resolution,
            },
        }

    return {
        "meta": {
            "tour": (tour or "").upper().strip(),
            "surface_requested": surface_key(surface),
            "as_of": date.today().isoformat(),
            "input_names": {"player1": player1, "player2": player2},
        },
        "players": {"left": left, "right": right},
        "compare": compare,
        "h2h": {
            "overall": h2h_overall,
            "surface": h2h_surface,
            "record": h2h_record,
            "last_10_matchups": last_10_matchups,
            "event_id": event_id,
        },
        "last_10_matches": {"left": left_last_10, "right": right_last_10},
        "last_10_resolution": {"left": left_last_10_resolution, "right": right_last_10_resolution},
        "last_10_matchups": last_10_matchups,
        "quality": quality,
        "debug": debug_payload,
    }


async def build_player_stats_payload(
    *,
    player: str,
    tour: str = "ATP",
    surface: str = "hard",
    debug: bool = False,
    engine: Any,
    load_player_surface_stats: Callable[[], None],
    norm_name: Callable[[Optional[str]], Optional[str]],
    norm_name_stats: Callable[[Optional[str]], Optional[str]],
    name_variants_for_stats: Callable[[str], list[str]],
    name_tokens: Callable[[str | None], list[str]],
    resolve_name_match: Callable[[str, set[str], dict[str, set[str]]], dict[str, Any]],
    player_stats_names_ts: dict[tuple[str, str], set[str]],
    player_stats_token_index_ts: dict[tuple[str, str], dict[str, set[str]]],
    lookup_player_stats_row_by_norm_name: Callable[[str, str, str, Optional[str]], Optional[dict[str, Any]]],
    safe_float_dict: Callable[[Optional[dict[str, Any]]], Optional[dict[str, Any]]],
    build_token_index: Callable[[set[str]], dict[str, set[str]]],
    get_player_rates: Callable[[str, str, str, Optional[int]], tuple[Optional[float], Optional[float]]],
    get_player_style_metrics: Callable[[str, str, Optional[int], bool], dict[str, Optional[float]]],
    style_label: Callable[[dict[str, Optional[float]]], Optional[str]],
    surface_key: Callable[[str], str],
    player_stats_path: Any,
) -> dict[str, Any]:
    pname = (player or "").strip()
    if not pname:
        raise HTTPException(status_code=400, detail="player is required")
    t = (tour or "").upper().strip()
    if t not in {"ATP", "WTA"}:
        raise HTTPException(status_code=400, detail="tour must be ATP or WTA")

    s = surface_key(surface)
    load_player_surface_stats()

    base_norm = norm_name(pname)
    stats_norm = norm_name_stats(pname)
    stats_name_variants = name_variants_for_stats(stats_norm) if stats_norm else []
    quality_flags: set[str] = set()
    if stats_norm and any(len(tok) == 1 for tok in name_tokens(stats_norm)):
        quality_flags.add("name_abbrev_detected")

    rolling: Optional[Dict[str, Any]] = None
    ta_id: Optional[int] = None
    elo_latest: Optional[Dict[str, Any]] = None

    surfaces = ("hard", "clay", "grass")
    surface_rows: dict[str, Any] = {}
    csv_resolution_per_surface: dict[str, Any] = {}
    csv_methods_used: list[str] = []
    canonical_stats_name: Optional[str] = None
    csv_any_found = False

    for surf in surfaces:
        pool = (player_stats_names_ts or {}).get((t, surf), set())
        tok_idx = (player_stats_token_index_ts or {}).get((t, surf), {})
        resolved = resolve_name_match(pname, pool, tok_idx)
        for flag in resolved.get("quality_flags") or []:
            quality_flags.add(str(flag))
        csv_resolution_per_surface[surf] = resolved

        chosen = resolved.get("chosen_name")
        row_12 = lookup_player_stats_row_by_norm_name(t, "12 month", surf, chosen)
        row_all = lookup_player_stats_row_by_norm_name(t, "all time", surf, chosen)
        has_any = bool(row_12 or row_all)
        if has_any:
            csv_any_found = True
            if canonical_stats_name is None:
                canonical_stats_name = str(chosen)
            csv_methods_used.append(str(resolved.get("method") or "not_found"))

        surface_rows[surf] = {
            "row_12m": safe_float_dict(row_12),
            "row_all_time": safe_float_dict(row_all),
        }

    if canonical_stats_name is None:
        canonical_stats_name = pname

    rolling_resolution: dict[str, Any] = {
        "method": "not_found",
        "chosen_name": None,
        "score": None,
        "reason": "not_checked",
        "top_candidates": [],
    }
    elo_resolution: dict[str, Any] = {
        "method": "not_found",
        "chosen_name": None,
        "score": None,
        "reason": "not_checked",
        "top_candidates": [],
    }

    async with engine.begin() as conn:
        q_roll = text(
            r"""
            SELECT
              lower(regexp_replace(regexp_replace(trim(player_name), '[^a-zA-Z0-9\s]', ' ', 'g'), '\s+', ' ', 'g')) AS nm,
              win_rate_last_20,
              matches_played,
              hard_win_rate_last_10,
              clay_win_rate_last_10,
              grass_win_rate_last_10
            FROM tennis_player_rolling_stats
            """
        )
        roll_rows = [dict(r) for r in (await conn.execute(q_roll)).mappings().all()]
        roll_by_nm: dict[str, dict[str, Any]] = {}
        for rr in roll_rows:
            nm = rr.get("nm")
            if nm and nm not in roll_by_nm:
                roll_by_nm[str(nm)] = rr
        roll_pool = set(roll_by_nm.keys())
        rolling_resolution = resolve_name_match(pname, roll_pool, build_token_index(roll_pool))
        for flag in rolling_resolution.get("quality_flags") or []:
            quality_flags.add(str(flag))
        rname = rolling_resolution.get("chosen_name")
        if rname and rname in roll_by_nm:
            rolling = dict(roll_by_nm[rname])
            rolling.pop("nm", None)

        q_elo = text(
            r"""
            SELECT
              player_id,
              as_of_date,
              elo,
              helo,
              celo,
              gelo,
              official_rank,
              age,
              player_name,
              lower(regexp_replace(regexp_replace(trim(player_name), '[^a-zA-Z0-9\s]', ' ', 'g'), '\s+', ' ', 'g')) AS nm
            FROM tennisabstract_elo_latest
            WHERE upper(tour) = :tour
            """
        )
        elo_rows = [dict(r) for r in (await conn.execute(q_elo, {"tour": t})).mappings().all()]
        elo_by_nm: dict[str, dict[str, Any]] = {}
        for er in elo_rows:
            nm = er.get("nm")
            if nm and nm not in elo_by_nm:
                elo_by_nm[str(nm)] = er
        elo_pool = set(elo_by_nm.keys())
        elo_resolution = resolve_name_match(pname, elo_pool, build_token_index(elo_pool))
        for flag in elo_resolution.get("quality_flags") or []:
            quality_flags.add(str(flag))
        ename = elo_resolution.get("chosen_name")
        if ename and ename in elo_by_nm:
            e = dict(elo_by_nm[ename])
            e.pop("nm", None)
            try:
                ta_id = int(e.get("player_id")) if e.get("player_id") is not None else None
            except Exception:
                ta_id = None
            elo_latest = e

    svc_pts, ret_pts = get_player_rates(t, s, canonical_stats_name, ta_id=None)
    style = get_player_style_metrics(t, canonical_stats_name, ta_id=None, overall=True)
    style_lbl = style_label(style)

    csv_method = "not_found"
    if csv_methods_used:
        uniq = sorted(set(csv_methods_used))
        csv_method = uniq[0] if len(uniq) == 1 else "mixed"

    match_status = {
        "csv_rows": bool(csv_any_found),
        "rolling": bool(rolling is not None),
        "elo_latest": bool(elo_latest is not None),
    }
    resolution = {
        "csv_rows": csv_method,
        "rolling": str(rolling_resolution.get("method") or "not_found"),
        "elo_latest": str(elo_resolution.get("method") or "not_found"),
    }
    match_status_detail = {
        "csv_rows": "ok" if match_status["csv_rows"] else "no_surface_rows_found",
        "rolling": "ok" if match_status["rolling"] else str(rolling_resolution.get("reason") or "not_found"),
        "elo_latest": "ok" if match_status["elo_latest"] else str(elo_resolution.get("reason") or "not_found"),
    }

    debug_payload = None
    if debug:
        debug_payload = {
            "normalization": {
                "input": pname,
                "base_norm": base_norm,
                "stats_norm": stats_norm,
                "stats_name_variants": stats_name_variants,
            },
            "sources": {
                "csv_rows": {
                    "requested_surface": s,
                    "per_surface": csv_resolution_per_surface,
                },
                "rolling": rolling_resolution,
                "elo_latest": elo_resolution,
            },
            "stats_path": str(player_stats_path) if player_stats_path else None,
        }

    return {
        "player": pname,
        "tour": t,
        "surface_requested": s,
        "normalized": {
            "base": base_norm,
            "stats_key": stats_norm,
            "stats_name_variants": stats_name_variants,
            "db_alias_variants": [],
        },
        "ids": {
            "ta_id": ta_id,
        },
        "match_status": match_status,
        "match_status_detail": match_status_detail,
        "resolution": resolution,
        "quality_flags": sorted(quality_flags),
        "rolling_stats": safe_float_dict(rolling),
        "elo_latest": safe_float_dict(elo_latest),
        "current_surface_rates": {
            "service_points_won": svc_pts,
            "return_points_won": ret_pts,
        },
        "style": {
            "label": style_lbl,
            "metrics": safe_float_dict(style),
        },
        "surface_rows": surface_rows,
        "stats_path": str(player_stats_path) if player_stats_path else None,
        "debug": debug_payload,
    }


async def build_players_compare_context(
    *,
    p1: str,
    p2: str,
    tour: str,
    surface: str,
    engine: Any,
    resolve_canonical_player_id: Callable[..., Awaitable[tuple[Optional[int], dict[str, Any]]]],
    get_last_matchups_from_db: Callable[..., Awaitable[list[dict[str, Any]]]],
    get_player_last_matches_from_db: Callable[..., Awaitable[tuple[list[dict[str, Any]], dict[str, Any]]]],
    get_h2h_from_db: Callable[..., Awaitable[tuple[int, int, int, int, int, int]]],
) -> dict[str, Any]:
    last_10_matchups: list[dict[str, Any]] = []
    left_last_10: list[dict[str, Any]] = []
    right_last_10: list[dict[str, Any]] = []
    left_last_10_resolution: dict[str, Any] = {"method": "not_found", "resolved_player_id": None, "resolved_name": None, "reason": "not_checked"}
    right_last_10_resolution: dict[str, Any] = {"method": "not_found", "resolved_player_id": None, "resolved_name": None, "reason": "not_checked"}
    h2h_overall = {"p1_wins": 0, "p2_wins": 0, "total": 0}
    h2h_surface = {"p1_wins": 0, "p2_wins": 0, "total": 0}

    async with engine.begin() as conn:
        left_cid, left_cid_meta = await resolve_canonical_player_id(conn=conn, player_name=p1, tour=tour)
        right_cid, right_cid_meta = await resolve_canonical_player_id(conn=conn, player_name=p2, tour=tour)
        last_10_matchups = await get_last_matchups_from_db(
            conn=conn,
            p1_canonical_id=left_cid,
            p2_canonical_id=right_cid,
            p1_name=p1,
            p2_name=p2,
            limit=10,
        )
        left_last_10, left_last_10_resolution = await get_player_last_matches_from_db(
            conn=conn,
            player_name=p1,
            canonical_player_id=left_cid,
            name_candidates=left_cid_meta.get("candidates"),
            limit=10,
        )
        right_last_10, right_last_10_resolution = await get_player_last_matches_from_db(
            conn=conn,
            player_name=p2,
            canonical_player_id=right_cid,
            name_candidates=right_cid_meta.get("candidates"),
            limit=10,
        )
        p1w, p2w, tot, p1sw, p2sw, stot = await get_h2h_from_db(
            conn=conn,
            p1_canonical_id=left_cid,
            p2_canonical_id=right_cid,
            p1_name=p1,
            p2_name=p2,
            surface=surface,
            as_of=date.today(),
            lookback_years=10,
        )
        h2h_overall = {"p1_wins": int(p1w), "p2_wins": int(p2w), "total": int(tot)}
        h2h_surface = {"p1_wins": int(p1sw), "p2_wins": int(p2sw), "total": int(stot)}

    return {
        "last_10_matchups": last_10_matchups,
        "left_last_10": left_last_10,
        "right_last_10": right_last_10,
        "left_last_10_resolution": left_last_10_resolution,
        "right_last_10_resolution": right_last_10_resolution,
        "h2h_overall": h2h_overall,
        "h2h_surface": h2h_surface,
    }


async def build_players_compare_payload(
    *,
    player1: str,
    player2: str,
    tour: str,
    surface: str,
    event_id: int | None,
    debug: bool,
    validate_inputs: Callable[..., tuple[str, str]],
    build_player_stats_payload: Callable[..., Awaitable[dict[str, Any]]],
    build_players_compare_context: Callable[..., Awaitable[dict[str, Any]]],
    build_players_compare_response: Callable[..., dict[str, Any]],
    norm_name_stats: Callable[[str], str | None],
    norm_name: Callable[[str], str | None],
    engine: Any,
    resolve_canonical_player_id: Callable[..., Awaitable[tuple[Optional[int], dict[str, Any]]]],
    get_last_matchups_from_db: Callable[..., Awaitable[list[dict[str, Any]]]],
    get_player_last_matches_from_db: Callable[..., Awaitable[tuple[list[dict[str, Any]], dict[str, Any]]]],
    get_h2h_from_db: Callable[..., Awaitable[tuple[int, int, int, int, int, int]]],
    norm_name_simple: Callable[[str | None], str],
    normalize_surface_family: Callable[[str | None], str],
    diff_num: Callable[[Any, Any], float | None],
    surface_key: Callable[[str], str],
) -> dict[str, Any]:
    p1, p2 = validate_inputs(
        player1=player1,
        player2=player2,
        norm_name_stats=norm_name_stats,
        norm_name=norm_name,
    )
    left = await build_player_stats_payload(player=p1, tour=tour, surface=surface, debug=debug)
    right = await build_player_stats_payload(player=p2, tour=tour, surface=surface, debug=debug)
    context = await build_players_compare_context(
        p1=p1,
        p2=p2,
        tour=tour,
        surface=surface,
        engine=engine,
        resolve_canonical_player_id=resolve_canonical_player_id,
        get_last_matchups_from_db=get_last_matchups_from_db,
        get_player_last_matches_from_db=get_player_last_matches_from_db,
        get_h2h_from_db=get_h2h_from_db,
    )
    return build_players_compare_response(
        player1=p1,
        player2=p2,
        tour=tour,
        surface=surface,
        event_id=event_id,
        debug=debug,
        left=left,
        right=right,
        last_10_matchups=context["last_10_matchups"],
        left_last_10=context["left_last_10"],
        right_last_10=context["right_last_10"],
        left_last_10_resolution=context["left_last_10_resolution"],
        right_last_10_resolution=context["right_last_10_resolution"],
        h2h_overall=context["h2h_overall"],
        h2h_surface=context["h2h_surface"],
        norm_name_simple=norm_name_simple,
        normalize_surface_family=normalize_surface_family,
        diff_num=diff_num,
        surface_key=surface_key,
    )
