from __future__ import annotations

from datetime import date
from typing import Any, Callable, Iterable, Optional, Tuple

from sqlalchemy import bindparam, text
from sqlalchemy.exc import DBAPIError

from app.db_session import engine

H2H_RAW_SQL = text(
    """
    SELECT payload
    FROM sofascore_event_h2h_raw
    WHERE event_id = :event_id
    LIMIT 1
    """
)


async def fetch_h2h_raw_payload(event_id: int) -> Any:
    async with engine.begin() as conn:
        res = await conn.execute(H2H_RAW_SQL, {"event_id": event_id})
        row = res.first()
        return row[0] if row else None


async def fetch_matches_elo_light_rows(
    *,
    conn,
    sql_stmt,
    dates: Iterable[date],
    include_incomplete: bool,
) -> list[dict[str, Any]]:
    res = await conn.execute(
        sql_stmt,
        {"dates": list(dates), "include_incomplete": include_incomplete},
    )
    return [dict(r) for r in res.mappings().all()]


async def batch_alias_variants(
    *,
    conn,
    base_norms: list[str],
) -> dict[str, list[str]]:
    base_norms = [b for b in base_norms if b]
    if not base_norms:
        return {}

    q_canon = text(
        """
        SELECT alias_name_norm, canonical_player_id
        FROM public.tennis_player_aliases
        WHERE alias_name_norm IN :alias_norms
          AND is_pending = false
          AND canonical_player_id IS NOT NULL
        """
    ).bindparams(bindparam("alias_norms", expanding=True))

    res = await conn.execute(q_canon, {"alias_norms": base_norms})
    canon_map: dict[str, int] = {}
    for row in res.fetchall():
        if row and row[0] and row[1] is not None:
            canon_map[str(row[0])] = int(row[1])

    cids = sorted(set(canon_map.values()))
    cid_to_aliases: dict[int, list[str]] = {}
    if cids:
        q_aliases = text(
            """
            SELECT canonical_player_id, alias_name_norm
            FROM public.tennis_player_aliases
            WHERE canonical_player_id IN :cids
              AND is_pending = false
            """
        ).bindparams(bindparam("cids", expanding=True))
        res2 = await conn.execute(q_aliases, {"cids": cids})
        for row in res2.fetchall():
            if not row:
                continue
            cid = row[0]
            nm = row[1]
            if cid is None or not nm:
                continue
            cid_to_aliases.setdefault(int(cid), []).append(str(nm))

    out: dict[str, list[str]] = {}
    for b in base_norms:
        variants = [b]
        cid = canon_map.get(b)
        if cid is not None:
            for v in cid_to_aliases.get(cid, []):
                if v not in variants:
                    variants.append(v)
        out[b] = variants
    return out


async def batch_fetch_rolling_stats(
    *,
    conn,
    variants_by_base: dict[str, list[str]],
) -> dict[str, dict]:
    all_norms: set[str] = set()
    for vs in variants_by_base.values():
        for v in vs:
            if v:
                all_norms.add(v)
    if not all_norms:
        return {}

    q = text(
        r"""
        SELECT
            lower(regexp_replace(trim(player_name), '\s+', ' ', 'g')) AS player_name_norm,
            win_rate_last_20,
            matches_played,
            hard_win_rate_last_10,
            clay_win_rate_last_10,
            grass_win_rate_last_10
        FROM tennis_player_rolling_stats
        WHERE lower(regexp_replace(trim(player_name), '\s+', ' ', 'g')) IN :nms
        """
    ).bindparams(bindparam("nms", expanding=True))

    res = await conn.execute(q, {"nms": list(all_norms)})
    out: dict[str, dict] = {}
    for row in res.mappings().all():
        nm = row.get("player_name_norm")
        if nm and nm not in out:
            out[str(nm)] = dict(row)
    return out


async def fetch_ta_feature_diffs(
    *,
    conn,
    match_ids: list[Any],
) -> dict[str, dict[str, Any]]:
    if not match_ids:
        return {}
    ta_rows = (
        await conn.execute(
            text(
                """
                SELECT
                    match_id,
                    (p1_last5_hold_pct - p2_last5_hold_pct) AS d_last5_hold,
                    (p1_last5_break_pct - p2_last5_break_pct) AS d_last5_break,
                    (p1_last10_hold_pct - p2_last10_hold_pct) AS d_last10_hold,
                    (p1_last10_break_pct - p2_last10_break_pct) AS d_last10_break,
                    (p1_surface_last10_hold_pct - p2_surface_last10_hold_pct) AS d_surf_last10_hold,
                    (p1_surface_last10_break_pct - p2_surface_last10_break_pct) AS d_surf_last10_break,
                    (COALESCE(p1_last10_aces_per_svc_game,0) - COALESCE(p2_last10_aces_per_svc_game,0)) AS d_last10_aces_pg,
                    (COALESCE(p1_surface_last10_aces_per_svc_game,0) - COALESCE(p2_surface_last10_aces_per_svc_game,0)) AS d_surf_last10_aces_pg,
                    (COALESCE(p1_last10_df_per_svc_game,0) - COALESCE(p2_last10_df_per_svc_game,0)) AS d_last10_df_pg,
                    (COALESCE(p1_surface_last10_df_per_svc_game,0) - COALESCE(p2_surface_last10_df_per_svc_game,0)) AS d_surf_last10_df_pg,
                    (COALESCE(p1_last10_tb_match_rate,0) - COALESCE(p2_last10_tb_match_rate,0)) AS d_last10_tb_match_rate,
                    (COALESCE(p1_last10_tb_win_pct,0) - COALESCE(p2_last10_tb_win_pct,0)) AS d_last10_tb_win_pct,
                    (COALESCE(p1_surface_last10_tb_match_rate,0) - COALESCE(p2_surface_last10_tb_match_rate,0)) AS d_surf_last10_tb_match_rate,
                    (COALESCE(p1_surface_last10_tb_win_pct,0) - COALESCE(p2_surface_last10_tb_win_pct,0)) AS d_surf_last10_tb_win_pct
                FROM tennis_features_ta
                WHERE match_id = ANY(:ids)
                """
            ),
            {"ids": match_ids},
        )
    ).mappings().all()
    return {str(row["match_id"]): dict(row) for row in ta_rows}


async def get_h2h_from_db(
    *,
    conn,
    p1_canonical_id: Optional[int],
    p2_canonical_id: Optional[int],
    p1_name: Optional[str],
    p2_name: Optional[str],
    surface: Optional[str],
    as_of: date,
    lookback_years: int,
    normalize_surface_family: Callable[[Optional[str]], str],
    norm_name_simple: Callable[[Optional[str]], str],
) -> Tuple[int, int, int, int, int, int]:
    surface_norm = normalize_surface_family(surface)
    p1_norm = norm_name_simple(p1_name)
    p2_norm = norm_name_simple(p2_name)
    try:
        async with conn.begin_nested():
            q = text(
                """
                WITH h2h AS (
                  SELECT
                    match_date,
                    CASE
                      WHEN lower(surface) LIKE '%clay%' THEN 'clay'
                      WHEN lower(surface) LIKE '%grass%' THEN 'grass'
                      WHEN lower(surface) LIKE '%hard%' THEN 'hard'
                      ELSE 'unknown'
                    END AS surface,
                    winner_canonical_id
                  FROM tennis_matches
                  WHERE match_date < :as_of
                    AND match_date >= (:as_of - make_interval(years => :lookback_years))
                    AND (
                      (p1_canonical_id = :p1 AND p2_canonical_id = :p2)
                      OR
                      (p1_canonical_id = :p2 AND p2_canonical_id = :p1)
                    )
                )
                SELECT
                  COALESCE(SUM(CASE WHEN winner_canonical_id = :p1 THEN 1 ELSE 0 END), 0) AS p1_wins,
                  COALESCE(SUM(CASE WHEN winner_canonical_id = :p2 THEN 1 ELSE 0 END), 0) AS p2_wins,
                  COALESCE(COUNT(*), 0) AS total_matches,
                  COALESCE(SUM(CASE WHEN surface = :surface AND winner_canonical_id = :p1 THEN 1 ELSE 0 END), 0) AS p1_surface_wins,
                  COALESCE(SUM(CASE WHEN surface = :surface AND winner_canonical_id = :p2 THEN 1 ELSE 0 END), 0) AS p2_surface_wins,
                  COALESCE(SUM(CASE WHEN surface = :surface THEN 1 ELSE 0 END), 0) AS surface_matches
                FROM h2h
                """
            )
            r = await conn.execute(
                q,
                {
                    "as_of": as_of,
                    "lookback_years": lookback_years,
                    "p1": p1_canonical_id,
                    "p2": p2_canonical_id,
                    "surface": surface_norm,
                },
            )
            row = r.first()
            canonical = (0, 0, 0, 0, 0, 0) if not row else (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]))
            if canonical[2] > 0 or not (p1_norm and p2_norm):
                return canonical

        async with conn.begin_nested():
            q2 = text(
                """
                WITH h2h AS (
                  SELECT
                    match_date,
                    CASE
                      WHEN lower(surface) LIKE '%clay%' THEN 'clay'
                      WHEN lower(surface) LIKE '%grass%' THEN 'grass'
                      WHEN lower(surface) LIKE '%hard%' THEN 'hard'
                      ELSE 'unknown'
                    END AS surface,
                    winner_canonical_id,
                    p1_name,
                    p2_name
                  FROM tennis_matches
                  WHERE match_date < :as_of
                    AND match_date >= (:as_of - make_interval(years => :lookback_years))
                ),
                norms AS (
                  SELECT
                    *,
                    lower(regexp_replace(regexp_replace(unaccent(coalesce(p1_name,'')), '[^a-z0-9 ]', '', 'g'), '\\s+', ' ', 'g')) AS p1_norm,
                    lower(regexp_replace(regexp_replace(unaccent(coalesce(p2_name,'')), '[^a-z0-9 ]', '', 'g'), '\\s+', ' ', 'g')) AS p2_norm
                  FROM h2h
                ),
                filtered AS (
                  SELECT *
                  FROM norms
                  WHERE (
                    (p1_norm = :p1n AND p2_norm = :p2n)
                    OR
                    (p1_norm = :p2n AND p2_norm = :p1n)
                  )
                )
                SELECT
                  COALESCE(SUM(CASE WHEN winner_canonical_id = :p1 THEN 1 ELSE 0 END), 0) AS p1_wins,
                  COALESCE(SUM(CASE WHEN winner_canonical_id = :p2 THEN 1 ELSE 0 END), 0) AS p2_wins,
                  COALESCE(COUNT(*), 0) AS total_matches,
                  COALESCE(SUM(CASE WHEN surface = :surface AND winner_canonical_id = :p1 THEN 1 ELSE 0 END), 0) AS p1_surface_wins,
                  COALESCE(SUM(CASE WHEN surface = :surface AND winner_canonical_id = :p2 THEN 1 ELSE 0 END), 0) AS p2_surface_wins,
                  COALESCE(SUM(CASE WHEN surface = :surface THEN 1 ELSE 0 END), 0) AS surface_matches
                FROM filtered
                """
            )
            r2 = await conn.execute(
                q2,
                {
                    "as_of": as_of,
                    "lookback_years": lookback_years,
                    "p1": p1_canonical_id,
                    "p2": p2_canonical_id,
                    "p1n": p1_norm,
                    "p2n": p2_norm,
                    "surface": surface_norm,
                },
            )
            row2 = r2.first()
            if not row2:
                return canonical
            return int(row2[0]), int(row2[1]), int(row2[2]), int(row2[3]), int(row2[4]), int(row2[5])
    except DBAPIError as e:
        msg = str(e.orig) if getattr(e, "orig", None) is not None else str(e)
        if "tennis_matches" in msg and ("does not exist" in msg or "undefined" in msg):
            return 0, 0, 0, 0, 0, 0
        if "winner_canonical_id" in msg and ("does not exist" in msg or "undefined" in msg):
            return 0, 0, 0, 0, 0, 0
        raise


async def get_last_matchups_from_db(
    *,
    conn,
    p1_canonical_id: Optional[int],
    p2_canonical_id: Optional[int],
    p1_name: Optional[str],
    p2_name: Optional[str],
    limit: int,
    norm_name_simple: Callable[[Optional[str]], str],
    infer_winner_side_from_score: Callable[[Optional[str]], Optional[str]],
    format_db_score: Callable[[dict[str, Any]], Optional[str]],
) -> list[dict[str, Any]]:
    p1_norm = norm_name_simple(p1_name)
    p2_norm = norm_name_simple(p2_name)
    n = max(1, min(int(limit or 10), 20))

    def _rows_to_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in rows:
            wcid = r.get("winner_canonical_id")
            p1cid = r.get("p1_canonical_id")
            p2cid = r.get("p2_canonical_id")
            winner_name = None
            if wcid is not None:
                if p1cid is not None and int(wcid) == int(p1cid):
                    winner_name = r.get("p1_name")
                elif p2cid is not None and int(wcid) == int(p2cid):
                    winner_name = r.get("p2_name")
            if winner_name is None:
                winner_side = infer_winner_side_from_score(format_db_score(r))
                if winner_side == "p1":
                    winner_name = r.get("p1_name")
                elif winner_side == "p2":
                    winner_name = r.get("p2_name")
            out.append(
                {
                    "match_date": str(r.get("match_date")) if r.get("match_date") is not None else None,
                    "tournament": r.get("tournament"),
                    "round": r.get("round"),
                    "surface": r.get("surface"),
                    "p1_name": r.get("p1_name"),
                    "p2_name": r.get("p2_name"),
                    "winner_name": winner_name,
                    "score": format_db_score(r),
                }
            )
        return out

    try:
        if p1_canonical_id is not None and p2_canonical_id is not None:
            q = text(
                """
                SELECT
                  match_date,
                  tournament,
                  "round" AS round,
                  surface,
                  p1_name,
                  p2_name,
                  p1_canonical_id,
                  p2_canonical_id,
                  winner_canonical_id,
                  score,
                  score_raw
                FROM tennis_matches
                WHERE (
                    (p1_canonical_id = :p1 AND p2_canonical_id = :p2)
                    OR
                    (p1_canonical_id = :p2 AND p2_canonical_id = :p1)
                )
                ORDER BY match_date DESC
                LIMIT :n
                """
            )
            res = await conn.execute(q, {"p1": p1_canonical_id, "p2": p2_canonical_id, "n": n})
            rows = [dict(r) for r in res.mappings().all()]
            if rows:
                return _rows_to_payload(rows)
        if p1_norm and p2_norm:
            q2 = text(
                r"""
                WITH base AS (
                  SELECT
                    match_date,
                    tournament,
                    "round" AS round,
                    surface,
                    p1_name,
                    p2_name,
                    p1_canonical_id,
                    p2_canonical_id,
                    winner_canonical_id,
                    score,
                    score_raw,
                    lower(regexp_replace(regexp_replace(unaccent(coalesce(p1_name,'')), '[^a-z0-9 ]', '', 'g'), '\\s+', ' ', 'g')) AS p1_norm,
                    lower(regexp_replace(regexp_replace(unaccent(coalesce(p2_name,'')), '[^a-z0-9 ]', '', 'g'), '\\s+', ' ', 'g')) AS p2_norm
                  FROM tennis_matches
                )
                SELECT
                  match_date,
                  tournament,
                  round,
                  surface,
                  p1_name,
                  p2_name,
                  p1_canonical_id,
                  p2_canonical_id,
                  winner_canonical_id,
                  score,
                  score_raw
                FROM base
                WHERE (
                    (p1_norm = :p1n AND p2_norm = :p2n)
                    OR
                    (p1_norm = :p2n AND p2_norm = :p1n)
                )
                ORDER BY match_date DESC
                LIMIT :n
                """
            )
            res2 = await conn.execute(q2, {"p1n": p1_norm, "p2n": p2_norm, "n": n})
            rows2 = [dict(r) for r in res2.mappings().all()]
            return _rows_to_payload(rows2)
    except DBAPIError as e:
        msg = str(e.orig) if getattr(e, "orig", None) is not None else str(e)
        if "tennis_matches" in msg and ("does not exist" in msg or "undefined" in msg):
            return []
    return []


async def get_player_last_matches_from_db(
    *,
    conn,
    player_name: Optional[str],
    canonical_player_id: Optional[int],
    name_candidates: Optional[list[str]],
    limit: int,
    norm_name_simple: Callable[[Optional[str]], str],
    infer_winner_side_from_score: Callable[[Optional[str]], Optional[str]],
    format_db_score: Callable[[dict[str, Any]], Optional[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pn = norm_name_simple(player_name)
    n = max(1, min(int(limit or 10), 20))
    status_ok = ("finished", "completed", "ended")
    if not pn and canonical_player_id is None:
        return [], {"method": "not_found", "resolved_player_id": None, "resolved_name": None, "reason": "no_input"}

    def _rows_to_payload(
        rows: list[dict[str, Any]],
        *,
        side_from_norm: bool = False,
        candset: Optional[set[str]] = None,
        canonical_id_for_side: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        candset = candset or set()
        for row in rows:
            side = "p1"
            if canonical_id_for_side is not None:
                try:
                    if row.get("p2_canonical_id") is not None and int(row.get("p2_canonical_id")) == int(canonical_id_for_side):
                        side = "p2"
                    elif row.get("p1_canonical_id") is not None and int(row.get("p1_canonical_id")) == int(canonical_id_for_side):
                        side = "p1"
                except Exception:
                    side = "p1"
            elif side_from_norm:
                p1n = str(row.get("p1_norm") or "")
                p2n = str(row.get("p2_norm") or "")
                if p2n in candset and p1n not in candset:
                    side = "p2"
                elif p1n in candset:
                    side = "p1"
            player_side_name = row.get("p1_name") if side == "p1" else row.get("p2_name")
            opponent_name = row.get("p2_name") if side == "p1" else row.get("p1_name")
            p1cid = row.get("p1_canonical_id")
            p2cid = row.get("p2_canonical_id")
            wcid = row.get("winner_canonical_id")
            winner_name = None
            result = None
            try:
                if wcid is not None and p1cid is not None and int(wcid) == int(p1cid):
                    winner_name = row.get("p1_name")
                elif wcid is not None and p2cid is not None and int(wcid) == int(p2cid):
                    winner_name = row.get("p2_name")
                if winner_name is None:
                    winner_side = infer_winner_side_from_score(format_db_score(row))
                    if winner_side == "p1":
                        winner_name = row.get("p1_name")
                    elif winner_side == "p2":
                        winner_name = row.get("p2_name")
                if winner_name:
                    result = "W" if norm_name_simple(player_side_name) == norm_name_simple(winner_name) else "L"
            except Exception:
                winner_name = None
                result = None
            out.append(
                {
                    "match_date": str(row.get("match_date")) if row.get("match_date") is not None else None,
                    "tournament": row.get("tournament"),
                    "round": row.get("round"),
                    "surface": row.get("surface"),
                    "player_name": player_side_name,
                    "opponent_name": opponent_name,
                    "winner_name": winner_name,
                    "result": result,
                    "score": format_db_score(row),
                }
            )
        return out

    try:
        if canonical_player_id is not None:
            q_id = text(
                """
                SELECT
                  match_date,
                  tournament,
                  "round" AS round,
                  surface,
                  p1_name,
                  p2_name,
                  p1_canonical_id,
                  p2_canonical_id,
                  winner_canonical_id,
                  score,
                  score_raw
                FROM tennis_matches
                WHERE (p1_canonical_id = :cid OR p2_canonical_id = :cid)
                  AND COALESCE(lower(status), '') IN :statuses
                ORDER BY match_date DESC
                LIMIT :n
                """
            ).bindparams(bindparam("statuses", expanding=True))
            res_id = await conn.execute(q_id, {"cid": int(canonical_player_id), "statuses": list(status_ok), "n": n})
            rows_id = [dict(r) for r in res_id.mappings().all()]
            if rows_id:
                return _rows_to_payload(rows_id, canonical_id_for_side=int(canonical_player_id)), {
                    "method": "canonical_id",
                    "resolved_player_id": int(canonical_player_id),
                    "resolved_name": pn or None,
                    "reason": "ok",
                }

        candset: set[str] = set()
        if name_candidates:
            for c in name_candidates:
                cc = norm_name_simple(c)
                if cc:
                    candset.add(cc)
        if pn:
            candset.add(pn)
        if not candset:
            return [], {"method": "not_found", "resolved_player_id": int(canonical_player_id) if canonical_player_id is not None else None, "resolved_name": None, "reason": "no_name_candidates"}

        q = text(
            """
            SELECT
              match_date,
              tournament,
              "round" AS round,
              status,
              surface,
              p1_name,
              p2_name,
              p1_canonical_id,
              p2_canonical_id,
              winner_canonical_id,
              score,
              score_raw
            FROM tennis_matches
            WHERE COALESCE(lower(status), '') IN :statuses
            ORDER BY match_date DESC
            LIMIT :nscan
            """
        ).bindparams(bindparam("statuses", expanding=True))
        res = await conn.execute(q, {"statuses": list(status_ok), "nscan": 5000})
        rows = [dict(r) for r in res.mappings().all()]
        filtered: list[dict[str, Any]] = []
        for row in rows:
            p1n = norm_name_simple(row.get("p1_name"))
            p2n = norm_name_simple(row.get("p2_name"))
            if p1n in candset or p2n in candset:
                row["p1_norm"] = p1n
                row["p2_norm"] = p2n
                filtered.append(row)
                if len(filtered) >= n:
                    break
        if filtered:
            return _rows_to_payload(filtered, side_from_norm=True, candset=candset), {
                "method": "name_fallback",
                "resolved_player_id": int(canonical_player_id) if canonical_player_id is not None else None,
                "resolved_name": sorted(candset)[0],
                "reason": "ok",
            }
        return [], {
            "method": "not_found",
            "resolved_player_id": int(canonical_player_id) if canonical_player_id is not None else None,
            "resolved_name": sorted(candset)[0] if candset else None,
            "reason": "no_finished_matches_found",
        }
    except DBAPIError as e:
        msg = str(e.orig) if getattr(e, "orig", None) is not None else str(e)
        if "tennis_matches" in msg and ("does not exist" in msg or "undefined" in msg):
            return [], {
                "method": "not_found",
                "resolved_player_id": int(canonical_player_id) if canonical_player_id is not None else None,
                "resolved_name": pn or None,
                "reason": "tennis_matches_missing",
            }
        return [], {
            "method": "not_found",
            "resolved_player_id": int(canonical_player_id) if canonical_player_id is not None else None,
            "resolved_name": pn or None,
            "reason": "db_error",
        }


async def resolve_canonical_player_id(
    *,
    conn,
    player_name: Optional[str],
    name_norm_stats: Callable[[Optional[str]], Optional[str]],
    norm_name_simple: Callable[[Optional[str]], str],
    name_variants_for_stats: Callable[[str], list[str]],
) -> tuple[Optional[int], dict[str, Any]]:
    base = name_norm_stats(player_name) or norm_name_simple(player_name)
    if not base:
        return None, {"method": "not_found", "resolved_player_id": None, "resolved_name": None, "reason": "no_input", "candidates": []}
    candidates = name_variants_for_stats(base)
    if base not in candidates:
        candidates.insert(0, base)
    candidates = [c for i, c in enumerate(candidates) if c and c not in candidates[:i]]
    q = text(
        """
        SELECT alias_name_norm, canonical_player_id
        FROM public.tennis_player_aliases
        WHERE alias_name_norm IN :nms
          AND is_pending = false
          AND canonical_player_id IS NOT NULL
        """
    ).bindparams(bindparam("nms", expanding=True))
    try:
        res = await conn.execute(q, {"nms": candidates})
        rows = [dict(r) for r in res.mappings().all()]
        if not rows:
            return None, {
                "method": "not_found",
                "resolved_player_id": None,
                "resolved_name": base,
                "reason": "no_alias_match",
                "candidates": candidates,
            }
        order = {nm: i for i, nm in enumerate(candidates)}
        rows.sort(key=lambda r: (order.get(str(r.get("alias_name_norm") or ""), 999), int(r.get("canonical_player_id") or 0)))
        chosen = rows[0]
        cid = int(chosen.get("canonical_player_id"))
        return cid, {
            "method": "canonical_id",
            "resolved_player_id": cid,
            "resolved_name": str(chosen.get("alias_name_norm") or base),
            "reason": "canonical_from_aliases",
            "candidates": candidates,
        }
    except DBAPIError:
        return None, {
            "method": "not_found",
            "resolved_player_id": None,
            "resolved_name": base,
            "reason": "db_error",
            "candidates": candidates,
        }
