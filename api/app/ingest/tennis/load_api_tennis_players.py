# app/ingest/tennis/load_api_tennis_players.py
from __future__ import annotations

import os
import ssl
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


CSV_PATH = r"C:\Users\dayshawn\Desktop\SportApp\api\standings_wta.csv"


# -----------------------
# Helpers
# -----------------------
def _to_int(x: Any) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None


def _clean_player_key(x: Any) -> str:
    """CSV can come in as float (2382.0). Normalize to '2382'."""
    try:
        if pd.isna(x):
            return ""
        return str(int(float(x))).strip()
    except Exception:
        return str(x).strip()


def normalize_asyncpg_url_and_ssl(db_url: str) -> Tuple[str, Dict[str, Any]]:
    """
    Neon requires SSL. If URL has ?sslmode=require we create an SSL context.
    Also strip sslmode from URL query after converting to connect_args.
    """
    u = urlparse(db_url)
    q = parse_qs(u.query)
    sslmode = (q.get("sslmode", [""])[0] or "").lower()

    connect_args: Dict[str, Any] = {}
    if sslmode in {"require", "verify-ca", "verify-full"}:
        ctx = ssl.create_default_context()
        connect_args["ssl"] = ctx

    q.pop("sslmode", None)
    new_query = urlencode({k: v[0] for k, v in q.items()})
    new_url = urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))
    return new_url, connect_args


async def _get_table_columns(conn) -> List[str]:
    rows = (await conn.execute(
        text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'api_tennis_players'
        ORDER BY ordinal_position
        """)
    )).all()
    return [r[0] for r in rows]


# -----------------------
# SQL (minimal create + safe alters)
# -----------------------
CREATE_TABLE_MINIMAL = text("""
CREATE TABLE IF NOT EXISTS api_tennis_players (
  player_key text PRIMARY KEY
);
""")

# We’ll add these if missing — but we will NOT try to drop/rename existing cols.
ALTER_OPTIONAL_COLS = [
    "ADD COLUMN IF NOT EXISTS league text",
    "ADD COLUMN IF NOT EXISTS country text",
    "ADD COLUMN IF NOT EXISTS points int",
    "ADD COLUMN IF NOT EXISTS place int",
    "ADD COLUMN IF NOT EXISTS raw_row jsonb",
    "ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now()",
    "ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now()",
    # name columns (some schemas use display_name, some use player_name)
    "ADD COLUMN IF NOT EXISTS display_name text",
    "ADD COLUMN IF NOT EXISTS player_name text",
]


def _build_upsert_sql(cols: List[str]) -> str:
    """
    Build an UPSERT that matches the actual table columns.
    Key requirements:
    - must satisfy NOT NULL constraints (your schema requires display_name)
    - only insert/update columns that exist
    """
    has_display = "display_name" in cols
    has_player_name = "player_name" in cols
    has_league = "league" in cols
    has_country = "country" in cols
    has_points = "points" in cols
    has_place = "place" in cols
    has_raw = "raw_row" in cols

    insert_cols: List[str] = ["player_key"]
    values_expr: List[str] = [":player_key"]

    # Name column priority: display_name first if it exists (and your table enforces it)
    if has_display:
        insert_cols.append("display_name")
        values_expr.append(":display_name")
    if has_player_name:
        insert_cols.append("player_name")
        values_expr.append(":player_name")

    if has_league:
        insert_cols.append("league")
        values_expr.append(":league")
    if has_country:
        insert_cols.append("country")
        values_expr.append(":country")
    if has_points:
        insert_cols.append("points")
        values_expr.append(":points")
    if has_place:
        insert_cols.append("place")
        values_expr.append(":place")
    if has_raw:
        insert_cols.append("raw_row")
        values_expr.append("CAST(:raw_row AS jsonb)")

    # always update updated_at if present
    if "updated_at" in cols:
        insert_cols.append("updated_at")
        values_expr.append("now()")

    set_parts: List[str] = []
    # update the same fields we insert (except player_key)
    for c in insert_cols:
        if c in {"player_key", "updated_at"}:
            continue
        set_parts.append(f"{c} = EXCLUDED.{c}")

    if "updated_at" in cols:
        set_parts.append("updated_at = now()")

    sql = f"""
    INSERT INTO api_tennis_players ({", ".join(insert_cols)})
    VALUES ({", ".join(values_expr)})
    ON CONFLICT (player_key) DO UPDATE
    SET {", ".join(set_parts)};
    """
    return sql


# -----------------------
# Main
# -----------------------
async def main() -> None:
    db_url = os.getenv("DATABASE_URL_ASYNC") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Missing DATABASE_URL (or DATABASE_URL_ASYNC). Must include sslmode=require for Neon.")

    normalized_url, connect_args = normalize_asyncpg_url_and_ssl(db_url)

    df = pd.read_csv(CSV_PATH)
    print("CSV columns:", list(df.columns))
    print("Rows:", len(df))

    engine = create_async_engine(
        normalized_url,
        future=True,
        echo=False,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    async with engine.begin() as conn:
        # Ensure table exists
        await conn.execute(CREATE_TABLE_MINIMAL)

        # Ensure expected optional columns exist (won't overwrite existing schema)
        await conn.execute(text(f"ALTER TABLE api_tennis_players {', '.join(ALTER_OPTIONAL_COLS)};"))

        cols = await _get_table_columns(conn)

        # Build the right UPSERT for your real schema
        upsert_sql = text(_build_upsert_sql(cols))

        # Build parameter rows
        rows: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            player_key = _clean_player_key(r.get("player_key"))
            if not player_key:
                continue

            name = str(r.get("player") or "").strip()
            if not name:
                # skip bad rows (also prevents NOT NULL violations)
                continue

            payload = {
                "player_key": player_key,
                # supply BOTH; SQL will use whichever columns exist
                "display_name": name,
                "player_name": name,
                "league": str(r.get("league") or "").strip() or None,
                "country": str(r.get("country") or "").strip() or None,
                "points": _to_int(r.get("points")),
                "place": _to_int(r.get("place")),
                "raw_row": json.dumps(r.to_dict(), default=str),
            }
            rows.append(payload)

        # Bulk upsert in chunks
        chunk_size = 500
        for i in range(0, len(rows), chunk_size):
            await conn.execute(upsert_sql, rows[i:i + chunk_size])

    await engine.dispose()
    print(f"✅ Loaded/Upserted {len(rows)} rows into api_tennis_players")


if __name__ == "__main__":
    asyncio.run(main())
