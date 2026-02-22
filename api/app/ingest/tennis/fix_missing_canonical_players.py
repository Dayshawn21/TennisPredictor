from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.db_session import engine


logger = logging.getLogger("fix_missing_canonical_players")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def _norm_name_py(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "")
    s = s.replace("\u00A0", " ")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_tours(raw: Optional[str]) -> tuple[str, ...]:
    v = (raw or "ATP,WTA").strip()
    items = [x.strip().upper() for x in v.split(",") if x.strip()]
    out = tuple(x for x in items if x in {"ATP", "WTA"})
    return out if out else ("ATP", "WTA")


def _load_target_list(path: Optional[str]) -> Optional[set[tuple[str, str]]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"target list file not found: {p}")

    targets: set[tuple[str, str]] = set()
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("target list CSV requires headers: name,tour")
        cols = {c.strip().lower(): c for c in reader.fieldnames}
        if "name" not in cols or "tour" not in cols:
            raise ValueError("target list CSV requires headers: name,tour")
        name_col = cols["name"]
        tour_col = cols["tour"]
        for row in reader:
            name = (row.get(name_col) or "").strip()
            tour = (row.get(tour_col) or "").strip().upper()
            if not name or tour not in {"ATP", "WTA"}:
                continue
            targets.add((_norm_name_py(name), tour))
    return targets


@dataclass(frozen=True)
class Candidate:
    tour: str
    match_name: str
    sofa_player_id: Optional[int]
    name_norm: str


FETCH_CANDIDATES_SQL = text(
    r"""
    WITH sides AS (
      SELECT
        upper(m.tour) AS tour,
        m.p1_name AS match_name,
        m.p1_sofascore_player_id AS sofa_player_id,
        m.p1_canonical_id AS canonical_id
      FROM tennis_matches m
      WHERE m.match_date BETWEEN :start AND :end
        AND upper(m.tour) IN ('ATP', 'WTA')
      UNION ALL
      SELECT
        upper(m.tour) AS tour,
        m.p2_name AS match_name,
        m.p2_sofascore_player_id AS sofa_player_id,
        m.p2_canonical_id AS canonical_id
      FROM tennis_matches m
      WHERE m.match_date BETWEEN :start AND :end
        AND upper(m.tour) IN ('ATP', 'WTA')
    )
    SELECT DISTINCT
      s.tour,
      s.match_name,
      s.sofa_player_id,
      lower(trim(regexp_replace(unaccent(replace(coalesce(s.match_name, ''), chr(160), ' ')), '\s+', ' ', 'g'))) AS name_norm
    FROM sides s
    WHERE s.canonical_id IS NULL
      AND s.tour = ANY(:tours)
    ORDER BY s.tour, s.match_name;
    """
)


FETCH_TA_HITS_SQL = text(
    r"""
    SELECT
      count(*)::int AS hit_cnt,
      min(el.player_id)::bigint AS ta_player_id
    FROM tennisabstract_elo_latest el
    WHERE upper(el.tour) = :tour
      AND lower(trim(regexp_replace(
            unaccent(replace(coalesce(el.player_name, ''), chr(160), ' ')),
            '\s+', ' ', 'g'
          ))) = :name_norm;
    """
)


UPSERT_PLAYER_SQL = text(
    """
    INSERT INTO tennis_players (name, gender)
    VALUES (:name, :gender)
    ON CONFLICT (name, gender) DO NOTHING;
    """
)


GET_PLAYER_SQL = text(
    """
    SELECT id
    FROM tennis_players
    WHERE name = :name
      AND gender = :gender
    LIMIT 1;
    """
)


GET_SOFA_MAPPING_SQL = text(
    """
    SELECT player_id
    FROM tennis_player_sources
    WHERE source = 'sofascore'
      AND source_player_id = :source_player_id
    LIMIT 1;
    """
)


UPSERT_SOFA_MAPPING_SQL = text(
    """
    INSERT INTO tennis_player_sources (player_id, source, source_player_id)
    VALUES (:player_id, 'sofascore', :source_player_id)
    ON CONFLICT (source, source_player_id)
    DO UPDATE SET player_id = EXCLUDED.player_id
    WHERE tennis_player_sources.player_id = EXCLUDED.player_id;
    """
)


BACKFILL_CANON_P1_SQL = text(
    """
    UPDATE tennis_matches m
    SET p1_canonical_id = s.player_id
    FROM tennis_player_sources s
    WHERE s.source = 'sofascore'
      AND NULLIF(btrim(s.source_player_id), '') ~ '^[0-9]+$'
      AND NULLIF(btrim(s.source_player_id), '')::bigint = m.p1_sofascore_player_id
      AND m.match_date BETWEEN :start AND :end
      AND m.p1_sofascore_player_id IS NOT NULL
      AND m.p1_canonical_id IS NULL;
    """
)


BACKFILL_CANON_P2_SQL = text(
    """
    UPDATE tennis_matches m
    SET p2_canonical_id = s.player_id
    FROM tennis_player_sources s
    WHERE s.source = 'sofascore'
      AND NULLIF(btrim(s.source_player_id), '') ~ '^[0-9]+$'
      AND NULLIF(btrim(s.source_player_id), '')::bigint = m.p2_sofascore_player_id
      AND m.match_date BETWEEN :start AND :end
      AND m.p2_sofascore_player_id IS NOT NULL
      AND m.p2_canonical_id IS NULL;
    """
)


def _gender_for_tour(tour: str) -> Optional[str]:
    t = (tour or "").upper()
    if t == "ATP":
        return "M"
    if t == "WTA":
        return "F"
    return None


async def _fetch_candidates(
    conn: AsyncConnection,
    start: dt.date,
    end: dt.date,
    tours: tuple[str, ...],
    targets: Optional[set[tuple[str, str]]],
) -> list[Candidate]:
    rows = (
        await conn.execute(
            FETCH_CANDIDATES_SQL,
            {"start": start, "end": end, "tours": list(tours)},
        )
    ).mappings().all()

    out: list[Candidate] = []
    for r in rows:
        c = Candidate(
            tour=str(r.get("tour") or "").upper(),
            match_name=str(r.get("match_name") or ""),
            sofa_player_id=int(r["sofa_player_id"]) if r.get("sofa_player_id") is not None else None,
            name_norm=str(r.get("name_norm") or ""),
        )
        if targets is not None and (c.name_norm, c.tour) not in targets:
            continue
        out.append(c)
    return out


def _init_stats(start: dt.date, end: dt.date, tours: Iterable[str], dry_run: bool) -> dict:
    return {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "tours": list(tours),
        "dry_run": bool(dry_run),
        "candidates_considered": 0,
        "canonicals_created": 0,
        "sofascore_mappings_upserted": 0,
        "canonicalized_without_ta_match": 0,
        "matches_backfilled": 0,
        "unresolved": {
            "NO_SOFA_ID": 0,
            "AMBIGUOUS_TA": 0,
            "SOURCE_CONFLICT": 0,
            "NO_TA_EXACT_MATCH": 0,
            "NO_GENDER_FOR_TOUR": 0,
            "NO_CANONICAL_AFTER_UPSERT": 0,
        },
    }


async def repair_missing_canonical_players(
    conn: AsyncConnection,
    *,
    start: dt.date,
    end: dt.date,
    tours: tuple[str, ...] = ("ATP", "WTA"),
    target_list_path: Optional[str] = None,
    dry_run: bool = False,
    strict: bool = True,
) -> dict:
    stats = _init_stats(start, end, tours, dry_run)
    if not strict:
        logger.warning("CANONICAL_REPAIR_STRICT=0 is not implemented; using strict exact matching.")

    targets = _load_target_list(target_list_path)
    candidates = await _fetch_candidates(conn, start=start, end=end, tours=tours, targets=targets)
    stats["candidates_considered"] = len(candidates)

    # One SAVEPOINT for dry runs. Roll back at the end.
    tx = await conn.begin_nested() if dry_run else None

    for c in candidates:
        if c.sofa_player_id is None:
            stats["unresolved"]["NO_SOFA_ID"] += 1
            continue

        no_ta_exact_match = False
        ta_row = (
            await conn.execute(FETCH_TA_HITS_SQL, {"tour": c.tour, "name_norm": c.name_norm})
        ).mappings().one()
        hit_cnt = int(ta_row.get("hit_cnt") or 0)
        if hit_cnt == 0:
            stats["unresolved"]["NO_TA_EXACT_MATCH"] += 1
            no_ta_exact_match = True
        if hit_cnt > 1:
            stats["unresolved"]["AMBIGUOUS_TA"] += 1
            continue

        gender = _gender_for_tour(c.tour)
        if not gender:
            stats["unresolved"]["NO_GENDER_FOR_TOUR"] += 1
            continue

        if not dry_run:
            ins = await conn.execute(UPSERT_PLAYER_SQL, {"name": c.match_name, "gender": gender})
            if (ins.rowcount or 0) > 0:
                stats["canonicals_created"] += int(ins.rowcount or 0)

        p = await conn.execute(GET_PLAYER_SQL, {"name": c.match_name, "gender": gender})
        row = p.first()
        if not row:
            stats["unresolved"]["NO_CANONICAL_AFTER_UPSERT"] += 1
            continue
        canonical_id = int(row[0])

        source_player_id = str(c.sofa_player_id)
        existing = await conn.execute(GET_SOFA_MAPPING_SQL, {"source_player_id": source_player_id})
        existing_row = existing.first()
        if existing_row is not None and int(existing_row[0]) != canonical_id:
            stats["unresolved"]["SOURCE_CONFLICT"] += 1
            continue

        if not dry_run:
            map_res = await conn.execute(
                UPSERT_SOFA_MAPPING_SQL,
                {"player_id": canonical_id, "source_player_id": source_player_id},
            )
            stats["sofascore_mappings_upserted"] += int(map_res.rowcount or 0)
            if no_ta_exact_match:
                stats["canonicalized_without_ta_match"] += 1

    if not dry_run:
        p1 = await conn.execute(BACKFILL_CANON_P1_SQL, {"start": start, "end": end})
        p2 = await conn.execute(BACKFILL_CANON_P2_SQL, {"start": start, "end": end})
        stats["matches_backfilled"] = int(p1.rowcount or 0) + int(p2.rowcount or 0)

    if tx is not None:
        await tx.rollback()

    return stats


def _log_stats(stats: dict) -> None:
    unresolved = stats.get("unresolved") or {}
    logger.info(
        "canonical_repair: window=%s..%s dry_run=%s candidates=%s created=%s mappings=%s backfilled=%s",
        stats.get("window_start"),
        stats.get("window_end"),
        stats.get("dry_run"),
        stats.get("candidates_considered"),
        stats.get("canonicals_created"),
        stats.get("sofascore_mappings_upserted"),
        stats.get("matches_backfilled"),
    )
    logger.info(
        "canonical_repair unresolved: NO_SOFA_ID=%s AMBIGUOUS_TA=%s SOURCE_CONFLICT=%s NO_TA_EXACT_MATCH=%s",
        unresolved.get("NO_SOFA_ID", 0),
        unresolved.get("AMBIGUOUS_TA", 0),
        unresolved.get("SOURCE_CONFLICT", 0),
        unresolved.get("NO_TA_EXACT_MATCH", 0),
    )


async def _run_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--tour", default="ATP,WTA", help="Comma-separated tours; default ATP,WTA")
    ap.add_argument("--target-list", default="", help="Optional CSV path with headers name,tour")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    tours = _parse_tours(args.tour)
    target_list_path = (args.target_list or "").strip() or None

    async with engine.begin() as conn:
        stats = await repair_missing_canonical_players(
            conn,
            start=start,
            end=end,
            tours=tours,
            target_list_path=target_list_path,
            dry_run=bool(args.dry_run),
            strict=True,
        )
    _log_stats(stats)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(_run_cli())
