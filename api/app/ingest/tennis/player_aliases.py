# app/ingest/tennis/player_aliases.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


@dataclass
class ResolveResult:
    alias_id: Optional[int]
    player_id: Optional[int]        # canonical players.id
    is_pending: bool


# These queries assume your schema:
# tennis_player_aliases(id, source, alias_name, alias_name_norm, external_id, is_pending, canonical_player_id, ...)
# player_source_map(source, source_player_id, canonical_player_id, source_name, ...)

Q_FIND_BY_EXTERNAL = text("""
    SELECT player_id AS canonical_player_id
    FROM tennis_player_sources
    WHERE source = :source
      AND source_player_id = :source_player_id
    LIMIT 1
""")


Q_FIND_ALIAS = text("""
    SELECT id, canonical_player_id, is_pending
    FROM tennis_player_aliases
    WHERE source = :source
      AND alias_name_norm = :alias_name_norm
    LIMIT 1
""")

Q_INSERT_ALIAS = text("""
    INSERT INTO tennis_player_aliases (
      source,
      alias_name,
      alias_name_norm,
      is_pending,
      canonical_player_id,
      created_at,
      updated_at
    )
    VALUES (
      :source,
      :alias_name,
      :alias_name_norm,
      :is_pending,
      :canonical_player_id,
      NOW(),
      NOW()
    )
    RETURNING id
""")

# Query removed - external_id column doesn't exist in database
# Q_UPDATE_ALIAS_EXTERNAL = text("""
#     UPDATE tennis_player_aliases
#     SET external_id = COALESCE(external_id, :external_id),
#         updated_at = NOW()
#     WHERE id = :id
# """)

def norm_name(name: str) -> str:
    # Keep consistent with your DB norm_name() function behavior:
    # lower, trim, collapse spaces, replace hyphens with spaces
    import re
    s = (name or "").strip().lower()
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


async def resolve_player_id(
    conn: AsyncConnection,
    source: str,
    alias_name: str,
    external_id: Optional[str],
    auto_create_pending: bool = True,
) -> ResolveResult:
    alias_name_norm = norm_name(alias_name)

    # 1) If we have an external_id, try player_source_map first (best/fastest)
    canonical_id: Optional[int] = None
    if external_id:
        # Keep source_player_id as string - database expects TEXT
        source_player_id = str(external_id)
        
        r = await conn.execute(
            Q_FIND_BY_EXTERNAL,
            {"source": source, "source_player_id": source_player_id},
        )
        canonical_id = r.scalar_one_or_none()
        if canonical_id is not None:
            # Ensure alias exists (or at least return success)
            r2 = await conn.execute(Q_FIND_ALIAS, {"source": source, "alias_name_norm": alias_name_norm})
            row = r2.mappings().first()
            if row:
                return ResolveResult(alias_id=row["id"], player_id=canonical_id, is_pending=False)

            if auto_create_pending:
                ins = await conn.execute(
                    Q_INSERT_ALIAS,
                    {
                        "source": source,
                        "alias_name": alias_name,
                        "alias_name_norm": alias_name_norm,
                        "is_pending": False,
                        "canonical_player_id": canonical_id,
                    },
                )
                new_id = ins.scalar_one()
                return ResolveResult(alias_id=new_id, player_id=canonical_id, is_pending=False)

            return ResolveResult(alias_id=None, player_id=canonical_id, is_pending=False)

    # 2) Look for existing alias row
    res = await conn.execute(Q_FIND_ALIAS, {"source": source, "alias_name_norm": alias_name_norm})
    row = res.mappings().first()
    if row:
        return ResolveResult(
            alias_id=row["id"],
            player_id=row["canonical_player_id"],
            is_pending=bool(row["is_pending"]) or row["canonical_player_id"] is None,
        )

    # 3) Not found — create pending alias if allowed
    if not auto_create_pending:
        return ResolveResult(alias_id=None, player_id=None, is_pending=True)

    ins = await conn.execute(
        Q_INSERT_ALIAS,
        {
            "source": source,
            "alias_name": alias_name,
            "alias_name_norm": alias_name_norm,
            "is_pending": True,
            "canonical_player_id": None,
        },
    )
    new_id = ins.scalar_one()
    return ResolveResult(alias_id=new_id, player_id=None, is_pending=True)
