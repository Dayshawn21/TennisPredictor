"""
flashscore_matchups.py

Flashscore tennis ingest (desktop odds feed via Playwright probe) with safe
player canonical mapping:
- external player id from detail URL first (source='flashscore')
- alias resolver fallback
- exact name_fold fallback (tour-gender constrained)
- conflict guards (same canonical on both sides -> drop weaker side)

Usage examples:
  py -3 api/app/ingest/tennis/flashscore_matchups.py --dry-run
  py -3 api/app/ingest/tennis/flashscore_matchups.py --day today --detail-all
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import os
import re
from typing import Any, Optional
from zoneinfo import ZoneInfo
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.sql import bindparam

from app.db_session import engine
from app.ingest.tennis.flashscore_ingest_probe import _probe_desktop
from app.ingest.tennis.player_aliases import resolve_player_id

load_dotenv()


FLASHSCORE_TZ = os.getenv("FLASHSCORE_TIMEZONE", "America/New_York")


ENSURE_SQL = [
    "CREATE EXTENSION IF NOT EXISTS pgcrypto",
    """
    CREATE TABLE IF NOT EXISTS tennis_player_sources (
      player_id INTEGER NOT NULL,
      source TEXT NOT NULL,
      source_player_id TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      PRIMARY KEY (source, source_player_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flashscore_events_raw (
      match_external_id TEXT PRIMARY KEY,
      payload JSONB NOT NULL,
      fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flashscore_event_odds_raw (
      match_external_id TEXT PRIMARY KEY,
      payload JSONB NOT NULL,
      fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_flashscore_events_raw_fetched_at ON flashscore_events_raw (fetched_at)",
    "CREATE INDEX IF NOT EXISTS ix_flashscore_event_odds_raw_fetched_at ON flashscore_event_odds_raw (fetched_at)",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS sofascore_p1_odds_american INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS sofascore_p2_odds_american INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS sofascore_odds_fetched_at TIMESTAMPTZ NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS flashscore_p1_odds_american INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS flashscore_p2_odds_american INTEGER NULL",
    "ALTER TABLE tennis_matches ADD COLUMN IF NOT EXISTS flashscore_odds_fetched_at TIMESTAMPTZ NULL",
]


UPSERT_FLASH_EVENT_RAW = (
    text(
        """
        INSERT INTO flashscore_events_raw (match_external_id, payload, fetched_at)
        VALUES (:match_external_id, :payload, NOW())
        ON CONFLICT (match_external_id)
        DO UPDATE SET
          payload = EXCLUDED.payload,
          fetched_at = NOW()
        """
    ).bindparams(bindparam("payload", type_=JSONB))
)


UPSERT_FLASH_ODDS_RAW = (
    text(
        """
        INSERT INTO flashscore_event_odds_raw (match_external_id, payload, fetched_at)
        VALUES (:match_external_id, :payload, NOW())
        ON CONFLICT (match_external_id)
        DO UPDATE SET
          payload = EXCLUDED.payload,
          fetched_at = NOW()
        """
    ).bindparams(bindparam("payload", type_=JSONB))
)


UPSERT_MATCH_SQL = text(
    """
    INSERT INTO tennis_matches (
      match_key,
      match_date,
      tour,
      tournament,
      "round",
      surface,
      p1_name,
      p2_name,
      start_time_utc,
      status,
      flashscore_id,
      score,
      score_raw,
      p1_canonical_id,
      p2_canonical_id,
      winner_canonical_id,
      p1_odds_american,
      p2_odds_american,
      odds_fetched_at,
      flashscore_p1_odds_american,
      flashscore_p2_odds_american,
      flashscore_odds_fetched_at,
      created_at,
      updated_at
    )
    VALUES (
      :match_key,
      :match_date,
      :tour,
      :tournament,
      :round,
      :surface,
      :p1_name,
      :p2_name,
      :start_time_utc,
      :status,
      :flashscore_id,
      :score,
      :score_raw,
      :p1_canonical_id,
      :p2_canonical_id,
      :winner_canonical_id,
      :p1_odds_american,
      :p2_odds_american,
      :odds_fetched_at,
      :flashscore_p1_odds_american,
      :flashscore_p2_odds_american,
      :flashscore_odds_fetched_at,
      NOW(),
      NOW()
    )
    ON CONFLICT (match_key)
    DO UPDATE SET
      match_date = EXCLUDED.match_date,
      tour = EXCLUDED.tour,
      tournament = COALESCE(EXCLUDED.tournament, tennis_matches.tournament),
      "round" = COALESCE(EXCLUDED."round", tennis_matches."round"),
      surface = COALESCE(EXCLUDED.surface, tennis_matches.surface),
      p1_name = EXCLUDED.p1_name,
      p2_name = EXCLUDED.p2_name,
      start_time_utc = COALESCE(EXCLUDED.start_time_utc, tennis_matches.start_time_utc),
      status = COALESCE(EXCLUDED.status, tennis_matches.status),
      flashscore_id = COALESCE(EXCLUDED.flashscore_id, tennis_matches.flashscore_id),
      score = COALESCE(EXCLUDED.score, tennis_matches.score),
      score_raw = COALESCE(EXCLUDED.score_raw, tennis_matches.score_raw),
      p1_canonical_id = COALESCE(EXCLUDED.p1_canonical_id, tennis_matches.p1_canonical_id),
      p2_canonical_id = COALESCE(EXCLUDED.p2_canonical_id, tennis_matches.p2_canonical_id),
      winner_canonical_id = COALESCE(EXCLUDED.winner_canonical_id, tennis_matches.winner_canonical_id),
      p1_odds_american = COALESCE(EXCLUDED.p1_odds_american, tennis_matches.p1_odds_american),
      p2_odds_american = COALESCE(EXCLUDED.p2_odds_american, tennis_matches.p2_odds_american),
      odds_fetched_at = COALESCE(EXCLUDED.odds_fetched_at, tennis_matches.odds_fetched_at),
      flashscore_p1_odds_american = COALESCE(EXCLUDED.flashscore_p1_odds_american, tennis_matches.flashscore_p1_odds_american),
      flashscore_p2_odds_american = COALESCE(EXCLUDED.flashscore_p2_odds_american, tennis_matches.flashscore_p2_odds_american),
      flashscore_odds_fetched_at = COALESCE(EXCLUDED.flashscore_odds_fetched_at, tennis_matches.flashscore_odds_fetched_at),
      updated_at = NOW()
    """
)

UPSERT_PLAYER_SOURCE_SQL = text(
    """
    INSERT INTO tennis_player_sources (player_id, source, source_player_id)
    VALUES (:player_id, :source, :source_player_id)
    ON CONFLICT (source, source_player_id)
    DO UPDATE SET player_id = EXCLUDED.player_id
    """
)

UPDATE_MATCH_ODDS_BY_CANONICAL_SQL = text(
    """
    UPDATE tennis_matches
    SET
      p1_odds_american = CASE
        WHEN p1_canonical_id = :p1_id AND p2_canonical_id = :p2_id THEN :p1_odds_american
        WHEN p1_canonical_id = :p2_id AND p2_canonical_id = :p1_id THEN :p2_odds_american
        ELSE p1_odds_american
      END,
      p2_odds_american = CASE
        WHEN p1_canonical_id = :p1_id AND p2_canonical_id = :p2_id THEN :p2_odds_american
        WHEN p1_canonical_id = :p2_id AND p2_canonical_id = :p1_id THEN :p1_odds_american
        ELSE p2_odds_american
      END,
      odds_fetched_at = :odds_fetched_at,
      flashscore_p1_odds_american = CASE
        WHEN p1_canonical_id = :p1_id AND p2_canonical_id = :p2_id THEN :p1_odds_american
        WHEN p1_canonical_id = :p2_id AND p2_canonical_id = :p1_id THEN :p2_odds_american
        ELSE flashscore_p1_odds_american
      END,
      flashscore_p2_odds_american = CASE
        WHEN p1_canonical_id = :p1_id AND p2_canonical_id = :p2_id THEN :p2_odds_american
        WHEN p1_canonical_id = :p2_id AND p2_canonical_id = :p1_id THEN :p1_odds_american
        ELSE flashscore_p2_odds_american
      END,
      flashscore_odds_fetched_at = :odds_fetched_at,
      flashscore_id = COALESCE(flashscore_id, :flashscore_id),
      updated_at = NOW()
    WHERE match_date = :match_date
      AND upper(tour) = :tour
      AND (
        (p1_canonical_id = :p1_id AND p2_canonical_id = :p2_id)
        OR
        (p1_canonical_id = :p2_id AND p2_canonical_id = :p1_id)
      )
    """
)


def _day_to_date(day: str, tz_name: str) -> dt.date:
    now_local = dt.datetime.now(ZoneInfo(tz_name)).date()
    if day == "yesterday":
        return now_local - dt.timedelta(days=1)
    if day == "tomorrow":
        return now_local + dt.timedelta(days=1)
    return now_local


def _parse_local_time_to_utc(v: Optional[str], day: str, tz_name: str) -> Optional[dt.datetime]:
    if not v:
        return None
    t = v.strip()
    try:
        z = ZoneInfo(tz_name)
    except Exception:
        z = ZoneInfo("America/New_York")

    # Example: "03:00 AM, February 14, 2026"
    for fmt in ("%I:%M %p, %B %d, %Y", "%I:%M %p, %b %d, %Y"):
        try:
            local_dt = dt.datetime.strptime(t, fmt).replace(tzinfo=z)
            return local_dt.astimezone(dt.timezone.utc)
        except Exception:
            pass

    # Example: "03:45 PM" (list row only)
    try:
        tm = dt.datetime.strptime(t, "%I:%M %p").time()
        base = _day_to_date(day, tz_name)
        local_dt = dt.datetime.combine(base, tm, tzinfo=z)
        return local_dt.astimezone(dt.timezone.utc)
    except Exception:
        return None


def _status_from_text(status_text: Optional[str]) -> str:
    s = (status_text or "").strip().lower()
    if not s:
        return "scheduled"
    if "final" in s or "ended" in s or "retired" in s or "walkover" in s:
        return "final"
    if "set" in s or "live" in s or "tiebreak" in s or "break" in s:
        return "live"
    return "scheduled"


def _tour_from_tournament(tournament: Optional[str]) -> str:
    t = (tournament or "").upper()
    if "WTA" in t:
        return "WTA"
    return "ATP"


def _surface_from_tournament(tournament: Optional[str]) -> Optional[str]:
    t = (tournament or "").lower()
    if "hard" in t:
        return "Hardcourt"
    if "clay" in t:
        return "Clay"
    if "grass" in t:
        return "Grass"
    if "carpet" in t:
        return "Carpet"
    return None


def _round_from_tournament(tournament: Optional[str]) -> Optional[str]:
    t = tournament or ""
    if ":" not in t:
        return None
    rhs = t.split(":", 1)[1].strip()
    if "," in rhs:
        rhs = rhs.rsplit(",", 1)[0].strip()
    return rhs or None


def _extract_player_external_ids(detail_url: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not detail_url:
        return None, None
    m = re.search(r"/game/tennis/([^/?#]+)/([^/?#]+)/", detail_url)
    if not m:
        return None, None

    def _id_from_seg(seg: str) -> Optional[str]:
        if "-" not in seg:
            return None
        tail = seg.rsplit("-", 1)[-1].strip()
        if re.match(r"^[A-Za-z0-9]{6,}$", tail):
            return tail
        return None

    return _id_from_seg(m.group(1)), _id_from_seg(m.group(2))


def _parse_score_pair(p1_score: Optional[str], p2_score: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    try:
        a = int(str(p1_score).strip()) if p1_score is not None else None
    except Exception:
        a = None
    try:
        b = int(str(p2_score).strip()) if p2_score is not None else None
    except Exception:
        b = None
    return a, b


async def _player_exists(conn: AsyncConnection, pid: int) -> bool:
    res = await conn.execute(text("SELECT 1 FROM tennis_players WHERE id = :id LIMIT 1"), {"id": pid})
    return res.first() is not None


async def _player_id_from_source(conn: AsyncConnection, source: str, source_player_id: Optional[str]) -> Optional[int]:
    sid = (source_player_id or "").strip()
    if not sid:
        return None
    res = await conn.execute(
        text(
            """
            SELECT player_id
            FROM tennis_player_sources
            WHERE source = :source
              AND btrim(source_player_id) = :sid
            LIMIT 1
            """
        ),
        {"source": source, "sid": sid},
    )
    row = res.first()
    return int(row[0]) if row else None


async def _player_id_from_name(conn: AsyncConnection, name: str, gender: Optional[str]) -> Optional[int]:
    if not name or not gender:
        return None
    res = await conn.execute(
        text(
            """
            SELECT id
            FROM tennis_players
            WHERE gender = :g
              AND name_fold = name_fold(:n)
            LIMIT 1
            """
        ),
        {"g": gender, "n": name},
    )
    row = res.first()
    return int(row[0]) if row else None


async def _resolve_one_player(
    conn: AsyncConnection,
    *,
    source: str,
    external_id: Optional[str],
    name: str,
    expected_gender: Optional[str],
) -> tuple[Optional[int], str]:
    pid = await _player_id_from_source(conn, source, external_id)
    if pid and await _player_exists(conn, pid):
        return pid, "source_id"

    r = await resolve_player_id(
        conn,
        source=source,
        alias_name=name,
        external_id=external_id,
        auto_create_pending=True,
    )
    if r.player_id and (not r.is_pending) and await _player_exists(conn, int(r.player_id)):
        return int(r.player_id), "alias"

    pid2 = await _player_id_from_name(conn, name, expected_gender)
    if pid2 and await _player_exists(conn, pid2):
        return pid2, "name_exact"

    return None, "unresolved"


def _method_rank(m: str) -> int:
    if m == "source_id":
        return 3
    if m == "alias":
        return 2
    if m == "name_exact":
        return 1
    return 0


async def _resolve_ids_for_item(conn: AsyncConnection, item: dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    p1_name = (item.get("p1_name") or "").strip()
    p2_name = (item.get("p2_name") or "").strip()
    p1_ext, p2_ext = _extract_player_external_ids(item.get("detail_url"))

    tour = _tour_from_tournament(item.get("tournament"))
    expected_gender = "M" if tour == "ATP" else "F"

    p1_id, p1_method = await _resolve_one_player(
        conn, source="flashscore", external_id=p1_ext, name=p1_name, expected_gender=expected_gender
    )
    p2_id, p2_method = await _resolve_one_player(
        conn, source="flashscore", external_id=p2_ext, name=p2_name, expected_gender=expected_gender
    )

    # Guard: impossible same player on both sides.
    if p1_id and p2_id and p1_id == p2_id:
        if _method_rank(p1_method) >= _method_rank(p2_method):
            p2_id = None
        else:
            p1_id = None

    return p1_id, p2_id


async def ensure_tables(conn: AsyncConnection) -> None:
    for stmt in ENSURE_SQL:
        await conn.execute(text(stmt))


async def apply_mapping_csv(conn: AsyncConnection, csv_path: str) -> dict[str, int]:
    out = {"rows_read": 0, "rows_applied": 0, "rows_skipped": 0}
    if not csv_path:
        return out
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {csv_path}")

    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out["rows_read"] += 1
            raw_pid = (row.get("canonical_player_id") or "").strip()
            source_player_id = (row.get("source_player_id") or row.get("flashscore_player_id") or "").strip()
            source = (row.get("source") or "flashscore").strip().lower()
            if not raw_pid or not source_player_id:
                out["rows_skipped"] += 1
                continue
            try:
                pid = int(raw_pid)
            except Exception:
                out["rows_skipped"] += 1
                continue
            await conn.execute(
                UPSERT_PLAYER_SOURCE_SQL,
                {"player_id": pid, "source": source, "source_player_id": source_player_id},
            )
            out["rows_applied"] += 1
    return out


def _write_mapping_template_csv(unresolved_rows: list[dict[str, Any]], out_path: str) -> None:
    if not out_path:
        return
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "canonical_player_id",
                "source",
                "source_player_id",
                "player_name",
                "tour",
                "match_external_id",
            ],
        )
        w.writeheader()
        seen: set[tuple[str, str]] = set()
        for r in unresolved_rows:
            tour = (r.get("tour") or "").strip()
            mid = (r.get("match_external_id") or "").strip()
            pairs = [
                ((r.get("p1_flashscore_player_id") or "").strip(), (r.get("p1_name") or "").strip()),
                ((r.get("p2_flashscore_player_id") or "").strip(), (r.get("p2_name") or "").strip()),
            ]
            for sid, nm in pairs:
                if not sid:
                    continue
                key = ("flashscore", sid)
                if key in seen:
                    continue
                seen.add(key)
                w.writerow(
                    {
                        "canonical_player_id": "",
                        "source": "flashscore",
                        "source_player_id": sid,
                        "player_name": nm,
                        "tour": tour,
                        "match_external_id": mid,
                    }
                )


def _odds_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "match_external_id": item.get("match_external_id"),
        "spread_p1_line": item.get("spread_p1_line"),
        "spread_p1_odds_american": item.get("spread_p1_odds_american"),
        "moneyline_p1_american": item.get("moneyline_p1_american"),
        "total_over_line": item.get("total_over_line"),
        "total_over_odds_american": item.get("total_over_odds_american"),
        "spread_p2_line": item.get("spread_p2_line"),
        "spread_p2_odds_american": item.get("spread_p2_odds_american"),
        "moneyline_p2_american": item.get("moneyline_p2_american"),
        "total_under_line": item.get("total_under_line"),
        "total_under_odds_american": item.get("total_under_odds_american"),
        "raw_odds_cells": item.get("raw_odds_cells"),
        "raw_odds_cells_list": item.get("raw_odds_cells_list"),
        "raw_odds_cells_detail": item.get("raw_odds_cells_detail"),
    }


async def ingest_flashscore(
    *,
    day: str,
    tour_filter: set[str],
    singles_only: bool,
    detail_odds: bool,
    detail_all: bool,
    detail_limit: int,
    dry_run: bool,
    unresolved_out: str = "",
    mapping_sql_out: str = "",
    mapping_csv_out: str = "",
    apply_mapping_csv_path: str = "",
    odds_only: bool = False,
) -> dict[str, Any]:
    payload = await _probe_desktop(day, tour_filter, singles_only, detail_odds, detail_all, detail_limit)
    items: list[dict[str, Any]] = payload.get("items") or []

    counts = {
        "fetched": len(items),
        "written_matches": 0,
        "written_raw": 0,
        "resolved_both": 0,
        "resolved_one": 0,
        "resolved_none": 0,
        "skipped_missing_external_id": 0,
        "odds_updated_existing": 0,
        "odds_skipped_no_target": 0,
    }
    unresolved_rows: list[dict[str, Any]] = []
    mapping_apply_counts = {"rows_read": 0, "rows_applied": 0, "rows_skipped": 0}

    if dry_run:
        async with engine.begin() as conn:
            await ensure_tables(conn)
            if apply_mapping_csv_path:
                mapping_apply_counts = await apply_mapping_csv(conn, apply_mapping_csv_path)
            for item in items:
                p1_ext, p2_ext = _extract_player_external_ids(item.get("detail_url"))
                if not item.get("match_external_id"):
                    counts["skipped_missing_external_id"] += 1
                    continue
                p1_id, p2_id = await _resolve_ids_for_item(conn, item)
                if p1_id and p2_id:
                    counts["resolved_both"] += 1
                elif p1_id or p2_id:
                    counts["resolved_one"] += 1
                else:
                    counts["resolved_none"] += 1
                    unresolved_rows.append(
                        {
                            "day": day,
                            "match_external_id": item.get("match_external_id"),
                            "tour": _tour_from_tournament(item.get("tournament")),
                            "tournament": item.get("tournament"),
                            "p1_name": item.get("p1_name"),
                            "p2_name": item.get("p2_name"),
                            "p1_flashscore_player_id": p1_ext,
                            "p2_flashscore_player_id": p2_ext,
                            "detail_url": item.get("detail_url"),
                        }
                    )
                if not p1_ext or not p2_ext:
                    counts["skipped_missing_external_id"] += 1
            if unresolved_out:
                p = Path(unresolved_out)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(unresolved_out, "w", encoding="utf-8") as f:
                    json.dump(unresolved_rows, f, ensure_ascii=True, indent=2)
            if mapping_sql_out:
                sql_lines: list[str] = []
                for r in unresolved_rows:
                    p1x = (r.get("p1_flashscore_player_id") or "").strip()
                    p2x = (r.get("p2_flashscore_player_id") or "").strip()
                    p1n = (r.get("p1_name") or "").replace("'", "''")
                    p2n = (r.get("p2_name") or "").replace("'", "''")
                    if p1x:
                        sql_lines.append(
                            f"-- {p1n}\n"
                            "INSERT INTO tennis_player_sources (player_id, source, source_player_id)\n"
                            f"VALUES (<canonical_player_id>, 'flashscore', '{p1x}')\n"
                            "ON CONFLICT (source, source_player_id) DO UPDATE SET player_id = EXCLUDED.player_id;"
                        )
                    if p2x:
                        sql_lines.append(
                            f"-- {p2n}\n"
                            "INSERT INTO tennis_player_sources (player_id, source, source_player_id)\n"
                            f"VALUES (<canonical_player_id>, 'flashscore', '{p2x}')\n"
                            "ON CONFLICT (source, source_player_id) DO UPDATE SET player_id = EXCLUDED.player_id;"
                        )
                p = Path(mapping_sql_out)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(mapping_sql_out, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(sql_lines))
            if mapping_csv_out:
                _write_mapping_template_csv(unresolved_rows, mapping_csv_out)
        return {
            "ok": True,
            "dry_run": True,
            "counts": counts,
            "mapping_apply_counts": mapping_apply_counts,
            "unresolved_count": len(unresolved_rows),
            "unresolved_out": unresolved_out or None,
            "mapping_sql_out": mapping_sql_out or None,
            "mapping_csv_out": mapping_csv_out or None,
        }

    async with engine.begin() as conn:
        await ensure_tables(conn)
        if apply_mapping_csv_path:
            mapping_apply_counts = await apply_mapping_csv(conn, apply_mapping_csv_path)
        for item in items:
            match_external_id = item.get("match_external_id")
            if not match_external_id:
                counts["skipped_missing_external_id"] += 1
                continue

            await conn.execute(
                UPSERT_FLASH_EVENT_RAW,
                {"match_external_id": str(match_external_id), "payload": item},
            )
            await conn.execute(
                UPSERT_FLASH_ODDS_RAW,
                {"match_external_id": str(match_external_id), "payload": _odds_payload(item)},
            )
            counts["written_raw"] += 1

            p1_id, p2_id = await _resolve_ids_for_item(conn, item)
            if p1_id and p2_id:
                counts["resolved_both"] += 1
            elif p1_id or p2_id:
                counts["resolved_one"] += 1
            else:
                counts["resolved_none"] += 1

            status = _status_from_text(item.get("status_text"))
            tour = _tour_from_tournament(item.get("tournament"))
            start_utc = _parse_local_time_to_utc(item.get("start_time_local"), day=day, tz_name=FLASHSCORE_TZ)
            match_date = start_utc.date() if start_utc else _day_to_date(day, FLASHSCORE_TZ)
            odds_fetched_at = dt.datetime.now(dt.timezone.utc)

            if odds_only:
                if not (p1_id and p2_id):
                    counts["odds_skipped_no_target"] += 1
                    continue

                res = await conn.execute(
                    UPDATE_MATCH_ODDS_BY_CANONICAL_SQL,
                    {
                        "match_date": match_date,
                        "tour": tour,
                        "p1_id": p1_id,
                        "p2_id": p2_id,
                        "p1_odds_american": item.get("moneyline_p1_american"),
                        "p2_odds_american": item.get("moneyline_p2_american"),
                        "odds_fetched_at": odds_fetched_at,
                        "flashscore_id": str(match_external_id),
                    },
                )
                matched = int(res.rowcount or 0)
                if matched > 0:
                    counts["odds_updated_existing"] += matched
                else:
                    counts["odds_skipped_no_target"] += 1
                continue

            p1s, p2s = _parse_score_pair(item.get("p1_score"), item.get("p2_score"))
            winner_canonical_id = None
            if status == "final" and p1s is not None and p2s is not None and p1_id and p2_id:
                if p1s > p2s:
                    winner_canonical_id = p1_id
                elif p2s > p1s:
                    winner_canonical_id = p2_id

            row = {
                "match_key": f"flashscore:{match_external_id}",
                "match_date": match_date,
                "tour": tour,
                "tournament": item.get("tournament") or "",
                "round": _round_from_tournament(item.get("tournament")),
                "surface": _surface_from_tournament(item.get("tournament")),
                "p1_name": item.get("p1_name") or "",
                "p2_name": item.get("p2_name") or "",
                "start_time_utc": start_utc,
                "status": status,
                "flashscore_id": str(match_external_id),
                "score": item.get("score_text"),
                "score_raw": item.get("score_text"),
                "p1_canonical_id": p1_id,
                "p2_canonical_id": p2_id,
                "winner_canonical_id": winner_canonical_id,
                "p1_odds_american": item.get("moneyline_p1_american"),
                "p2_odds_american": item.get("moneyline_p2_american"),
                "odds_fetched_at": odds_fetched_at,
                "flashscore_p1_odds_american": item.get("moneyline_p1_american"),
                "flashscore_p2_odds_american": item.get("moneyline_p2_american"),
                "flashscore_odds_fetched_at": odds_fetched_at,
            }

            if not row["p1_name"] or not row["p2_name"] or not row["tournament"]:
                continue

            await conn.execute(UPSERT_MATCH_SQL, row)
            counts["written_matches"] += 1

    return {"ok": True, "dry_run": False, "counts": counts, "mapping_apply_counts": mapping_apply_counts}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--day", choices=["today", "yesterday", "tomorrow"], default="today")
    p.add_argument("--tour-filter", type=str, default="ATP,WTA")
    p.add_argument("--include-doubles", action="store_true")
    p.add_argument("--detail-odds", action="store_true", default=True)
    p.add_argument("--detail-all", action="store_true", default=True)
    p.add_argument("--detail-limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--unresolved-out", type=str, default="")
    p.add_argument("--mapping-sql-out", type=str, default="")
    p.add_argument("--mapping-csv-out", type=str, default="")
    p.add_argument("--apply-mapping-csv", type=str, default="")
    p.add_argument("--odds-only", action="store_true")
    args = p.parse_args()

    tour_filter = {x.strip().upper() for x in (args.tour_filter or "").split(",") if x.strip()}
    result = asyncio.run(
        ingest_flashscore(
            day=args.day,
            tour_filter=tour_filter,
            singles_only=not args.include_doubles,
            detail_odds=bool(args.detail_odds),
            detail_all=bool(args.detail_all),
            detail_limit=max(0, int(args.detail_limit)),
            dry_run=bool(args.dry_run),
            unresolved_out=(args.unresolved_out or "").strip(),
            mapping_sql_out=(args.mapping_sql_out or "").strip(),
            mapping_csv_out=(args.mapping_csv_out or "").strip(),
            apply_mapping_csv_path=(args.apply_mapping_csv or "").strip(),
            odds_only=bool(args.odds_only),
        )
    )
    print(result)


if __name__ == "__main__":
    main()
