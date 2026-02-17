from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.exc import IntegrityError

from app.db_session import engine


from dotenv import load_dotenv
load_dotenv()

# Your mapping sources (match your API endpoint join)
SRC_ATP = "tennisabstract_elo_atp"
SRC_WTA = "tennisabstract_elo_wta"

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def tokenize_ascii(s: str) -> List[str]:
    """Lowercase ASCII tokens (words/numbers), stripped of accents + punctuation."""
    if not s:
        return []
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    return re.findall(r"[a-z0-9]+", s)


def build_keys(tokens: List[str]) -> List[str]:
    """
    Build match keys in priority order:
      STRICT:
        - full join (catymcnally)
        - first+last (catherine+mcnally)
        - reversed for 2-token names (mcnallycaty)
        - suffix-drop variants (martindammjr -> martindamm)
      LOOSE (only auto-map if unique):
        - fi:<first_initial><last>  e.g. fi:cmcnally
        - ln:<last>                 e.g. ln:mcnally
    """
    if not tokens:
        return []

    variants: List[List[str]] = [tokens]
    if len(tokens) >= 2 and tokens[-1] in SUFFIXES:
        variants.append(tokens[:-1])

    keys: List[str] = []

    # strict keys first
    for v in variants:
        if not v:
            continue
        keys.append("".join(v))  # full
        if len(v) >= 2:
            keys.append(v[0] + v[-1])  # first+last
        if len(v) == 2:
            keys.append(v[1] + v[0])  # reversed (handles "Bu Yunchaokete")

    # loose keys (use base without suffix)
    base = tokens[:-1] if (len(tokens) >= 2 and tokens[-1] in SUFFIXES) else tokens
    if base:
        first_initial = base[0][0] if base[0] else ""
        last = base[-1]
        if first_initial and last:
            keys.append(f"fi:{first_initial}{last}")
        if last:
            keys.append(f"ln:{last}")

    # unique preserve order
    seen = set()
    out: List[str] = []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


@dataclass(frozen=True)
class NeededPlayer:
    tour: str  # ATP/WTA
    player_id: int  # canonical tennis_players.id
    name: str
    name_fold: Optional[str]


def source_for_tour(tour: str) -> str:
    return "tennisabstract_elo_atp" if tour.upper() == "ATP" else "tennisabstract_elo_wta"



LATEST_ASOF_SQL = text(
    r"""
    SELECT upper(tour) AS tour, max(as_of_date) AS as_of_date
    FROM tennisabstract_elo_snapshots
    WHERE upper(tour) IN ('ATP','WTA')
    GROUP BY 1
    ORDER BY 1;
    """
)

# NOTE: Build index using the latest row PER PLAYER (not just latest date overall)
# This helps include players not present on the most recent top list.
SNAP_INDEX_SQL = text(
    r"""
    WITH latest_per_player AS (
      SELECT upper(tour) AS tour, player_id, max(as_of_date) AS as_of_date
      FROM tennisabstract_elo_snapshots
      WHERE upper(tour) IN ('ATP','WTA')
        AND player_id IS NOT NULL
      GROUP BY 1,2
    ),
    snap AS (
      SELECT DISTINCT ON (upper(s.tour), s.player_id)
        upper(s.tour) AS tour,
        s.as_of_date,
        s.player_id,
        s.player_name,
        s.elo,
        s.created_at
      FROM tennisabstract_elo_snapshots s
      JOIN latest_per_player l
        ON upper(s.tour) = l.tour
       AND s.player_id = l.player_id
       AND s.as_of_date = l.as_of_date
      ORDER BY
        upper(s.tour),
        s.player_id,
        s.created_at DESC,
        s.player_name ASC
    )
    SELECT tour, as_of_date, player_id, player_name, elo
    FROM snap
    ORDER BY tour, player_name;
    """
)

NEEDED_PLAYERS_SQL = text(
    r"""
    WITH sides AS (
      SELECT upper(tour) AS tour, p1_canonical_id AS player_id
      FROM tennis_matches
      WHERE match_date BETWEEN :start AND :end
        AND upper(tour) IN ('ATP','WTA')
        AND p1_canonical_id IS NOT NULL

      UNION ALL

      SELECT upper(tour) AS tour, p2_canonical_id AS player_id
      FROM tennis_matches
      WHERE match_date BETWEEN :start AND :end
        AND upper(tour) IN ('ATP','WTA')
        AND p2_canonical_id IS NOT NULL
    ),
    uniq AS (
      SELECT DISTINCT tour, player_id
      FROM sides
    )
    SELECT
      u.tour,
      u.player_id,
      p.name,
      p.name_fold,
      tps.source_player_id AS existing_ta_id
    FROM uniq u
    JOIN tennis_players p
      ON p.id = u.player_id
    LEFT JOIN tennis_player_sources tps
      ON tps.player_id = u.player_id
     AND tps.source = CASE WHEN u.tour = 'ATP' THEN :src_atp ELSE :src_wta END
    WHERE tps.source_player_id IS NULL
    ORDER BY u.tour, p.name;
    """
)

# Single statement (CTE) so asyncpg prepared stmt is happy
UPSERT_TPS_SQL = text("""
INSERT INTO tennis_player_sources (player_id, source, source_player_id, source_name, created_at, updated_at)
VALUES (:player_id, :source, :ta_id, :source_name, now(), now())
ON CONFLICT (player_id, source)
DO UPDATE SET
  source_player_id = EXCLUDED.source_player_id,
  source_name = EXCLUDED.source_name,
  updated_at = now();
""")

CHECK_SOURCE_PLAYER_SQL = text("""
SELECT player_id, source_name
FROM tennis_player_sources
WHERE source = :source
  AND source_player_id = :ta_id
LIMIT 1;
""")


async def load_snapshot_index() -> Tuple[
    Dict[str, Dict[str, List[Tuple[int, str, Optional[float]]]]], Dict[str, date]
]:
    async with engine.begin() as conn:
        latest_rows = (await conn.execute(LATEST_ASOF_SQL)).mappings().all()
        snap_rows = (await conn.execute(SNAP_INDEX_SQL)).mappings().all()

    latest_asof: Dict[str, date] = {str(r["tour"]).upper(): r["as_of_date"] for r in latest_rows}

    by_tour_key: Dict[str, Dict[str, List[Tuple[int, str, Optional[float]]]]] = {"ATP": {}, "WTA": {}}

    for r in snap_rows:
        tour = (r["tour"] or "").upper()
        ta_id = int(r["player_id"])
        pname = str(r["player_name"] or "")
        elo = float(r["elo"]) if r["elo"] is not None else None

        tokens = tokenize_ascii(pname)
        for k in build_keys(tokens):
            by_tour_key.setdefault(tour, {}).setdefault(k, []).append((ta_id, pname, elo))

    return by_tour_key, latest_asof


async def load_needed_players(start: date, end: date) -> List[NeededPlayer]:
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                NEEDED_PLAYERS_SQL,
                {"start": start, "end": end, "src_atp": SRC_ATP, "src_wta": SRC_WTA},
            )
        ).mappings().all()

    return [
        NeededPlayer(
            tour=str(r["tour"] or "").upper(),
            player_id=int(r["player_id"]),
            name=str(r["name"] or ""),
            name_fold=r.get("name_fold"),
        )
        for r in rows
    ]


def pick_best_candidate(cands: List[Tuple[int, str, Optional[float]]]) -> Tuple[int, str, Optional[float]]:
    """Prefer highest Elo if available, otherwise lowest TA id."""
    if len(cands) == 1:
        return cands[0]
    with_elo = [c for c in cands if c[2] is not None]
    if with_elo:
        return sorted(with_elo, key=lambda x: x[2], reverse=True)[0]
    return sorted(cands, key=lambda x: x[0])[0]


def find_candidates(
    snap_index: Dict[str, Dict[str, List[Tuple[int, str, Optional[float]]]]],
    tour: str,
    name: str,
    name_fold: Optional[str],
) -> Tuple[List[Tuple[int, str, Optional[float]]], Optional[str]]:
    """
    Try strict keys first. For loose keys (fi:/ln:), only accept if unique.
    """
    tour = (tour or "").upper()
    idx = snap_index.get(tour, {})

    def try_text(s: str) -> Tuple[List[Tuple[int, str, Optional[float]]], Optional[str]]:
        toks = tokenize_ascii(s)
        for k in build_keys(toks):
            cands = idx.get(k, [])
            if not cands:
                continue
            if k.startswith("fi:") or k.startswith("ln:"):
                if len(cands) == 1:
                    return cands, k
                # too risky, keep searching
                continue
            return cands, k
        return [], None

    cands, key = try_text(name)
    if cands:
        return cands, key

    if name_fold:
        cands, key = try_text(name_fold)
        if cands:
            return cands, key

    return [], None


async def upsert_mapping(conn, player_id: int, tour: str, ta_id: int, dry_run: bool) -> None:
    src = source_for_tour(tour)
    ta_id_text = str(ta_id)  # source_player_id is TEXT

    # ✅ Guard: prevent (source, source_player_id) from pointing to multiple canonical players
    existing = (await conn.execute(
        CHECK_SOURCE_PLAYER_SQL, {"source": src, "ta_id": ta_id_text}
    )).first()

    if existing and existing[0] != player_id:
        print(
            f"⚠️  SKIP: {src} ta_id={ta_id_text} already mapped to player_id={existing[0]} "
            f"(wanted player_id={player_id})."
        )
        return

    if dry_run:
        print(f"DRY RUN: UPSERT tennis_player_sources (player_id={player_id}, source={src}, ta_id={ta_id_text})")
        return

    try:
        await conn.execute(
            UPSERT_TPS_SQL,
            {"player_id": player_id, "source": src, "ta_id": ta_id_text, "source_name": None},
        )
    except IntegrityError as e:
        # If anything still slips through, show who owns that (source, source_player_id)
        owner = (await conn.execute(
            CHECK_SOURCE_PLAYER_SQL, {"source": src, "ta_id": ta_id_text}
        )).first()
        print(f"❌ IntegrityError. Current owner of ({src}, {ta_id_text}) = {owner}")
        raise

# ─── Connection-aware helpers (for use inside an existing transaction) ────────

async def _load_snapshot_index_with_conn(
    conn: AsyncConnection,
) -> Tuple[Dict[str, Dict[str, List[Tuple[int, str, Optional[float]]]]], Dict[str, date]]:
    """Same as load_snapshot_index() but uses an existing connection."""
    latest_rows = (await conn.execute(LATEST_ASOF_SQL)).mappings().all()
    snap_rows = (await conn.execute(SNAP_INDEX_SQL)).mappings().all()

    latest_asof: Dict[str, date] = {
        str(r["tour"]).upper(): r["as_of_date"] for r in latest_rows
    }
    by_tour_key: Dict[str, Dict[str, List[Tuple[int, str, Optional[float]]]]] = {
        "ATP": {}, "WTA": {}
    }
    for r in snap_rows:
        tour = (r["tour"] or "").upper()
        ta_id = int(r["player_id"])
        pname = str(r["player_name"] or "")
        elo = float(r["elo"]) if r["elo"] is not None else None
        tokens = tokenize_ascii(pname)
        for k in build_keys(tokens):
            by_tour_key.setdefault(tour, {}).setdefault(k, []).append(
                (ta_id, pname, elo)
            )
    return by_tour_key, latest_asof


async def _load_needed_players_with_conn(
    conn: AsyncConnection, start: date, end: date,
) -> List[NeededPlayer]:
    """Same as load_needed_players() but uses an existing connection."""
    rows = (
        await conn.execute(
            NEEDED_PLAYERS_SQL,
            {"start": start, "end": end, "src_atp": SRC_ATP, "src_wta": SRC_WTA},
        )
    ).mappings().all()
    return [
        NeededPlayer(
            tour=str(r["tour"] or "").upper(),
            player_id=int(r["player_id"]),
            name=str(r["name"] or ""),
            name_fold=r.get("name_fold"),
        )
        for r in rows
    ]


# ─── Reusable entry-point (called by sofascore_matchups pipeline) ────────────

async def run_fix(
    conn: AsyncConnection,
    start: date,
    end: date,
    *,
    dry_run: bool = False,
    quiet: bool = False,
) -> Tuple[int, int, int]:
    """
    Fuzzy-match players in the date window who still lack a TA source mapping.

    Returns (fixed, not_found, ambiguous).
    Can be called inside an existing transaction (e.g. from sofascore ingest).
    """
    snap_index, latest_asof = await _load_snapshot_index_with_conn(conn)
    if not quiet:
        print(f"  fix_ta_elo: snapshot dates ATP={latest_asof.get('ATP')} WTA={latest_asof.get('WTA')}")

    needed = await _load_needed_players_with_conn(conn, start, end)
    if not quiet:
        print(f"  fix_ta_elo: {len(needed)} players missing TA mapping in {start}..{end}")

    fixed = 0
    not_found = 0
    ambiguous = 0

    for p in needed:
        cands, used_key = find_candidates(snap_index, p.tour, p.name, p.name_fold)

        if not cands:
            not_found += 1
            if not quiet:
                print(f"  fix_ta_elo: ❌ NOT FOUND [{p.tour}] cid={p.player_id} '{p.name}'")
            continue

        if (
            len(cands) > 1
            and used_key
            and not (used_key.startswith("fi:") or used_key.startswith("ln:"))
        ):
            ambiguous += 1

        ta_id, ta_name, ta_elo = pick_best_candidate(cands)
        await upsert_mapping(conn, p.player_id, p.tour, ta_id, dry_run)
        fixed += 1
        if not quiet:
            print(
                f"  fix_ta_elo: ✅ [{p.tour}] cid={p.player_id} '{p.name}'"
                f" -> TA {ta_id} '{ta_name}' (elo={ta_elo})"
            )

    if not quiet:
        print(f"  fix_ta_elo: done — fixed={fixed} not_found={not_found} ambiguous={ambiguous}")
    return fixed, not_found, ambiguous


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Match date start (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Match date end (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions but do not write to DB")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    async with engine.begin() as conn:
        fixed, not_found, ambiguous = await run_fix(
            conn, start, end, dry_run=args.dry_run
        )

    print(f"\n---- SUMMARY ----")
    print(f"fixed={fixed}  not_found={not_found}  ambiguous={ambiguous}  dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))
