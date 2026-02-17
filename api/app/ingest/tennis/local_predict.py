import os
import re
import unicodedata
import psycopg2
from typing import Optional, Dict, Any, Tuple

DATABASE_URL = os.getenv("DATABASE_URL")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(".", "")
    return s


def _parse_initial_last(name: str) -> Tuple[Optional[str], str]:
    n = _norm(name)
    parts = n.split()
    if len(parts) < 2:
        return None, n
    initial = parts[0][0] if parts[0] else None
    last = " ".join(parts[1:])
    return initial, last


def _find_player_id(conn, raw_name: str, gender: str) -> Optional[int]:
    """
    gender should match your DB values. If your DB stores 'M'/'F', pass those.
    If it stores 'ATP'/'WTA', pass those instead.
    """
    n = _norm(raw_name)
    initial, last = _parse_initial_last(raw_name)

    with conn.cursor() as cur:
        # 1) exact (normalized) match if your DB stores lower(name)
        cur.execute(
            """
            SELECT id
            FROM tennis_players
            WHERE gender = %s AND lower(name) = %s
            LIMIT 1
            """,
            (gender, n),
        )
        row = cur.fetchone()
        if row:
            return int(row[0])

        # 2) fallback: last name contains + name starts with initial
        if initial and last:
            cur.execute(
                """
                SELECT id
                FROM tennis_players
                WHERE gender = %s
                  AND lower(name) LIKE %s
                  AND lower(name) LIKE %s
                LIMIT 1
                """,
                (gender, f"%{last}%", f"{initial}%"),
            )
            row = cur.fetchone()
            if row:
                return int(row[0])

    return None


def predict_match_local(p1_name: str, p2_name: str, tour: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolves players from DB. If tour is None, tries both tours.
    Returns a structure compatible with your runner.
    """
    if not DATABASE_URL:
        return {"ok": False, "reason": "DATABASE_URL_not_set"}

    # Map tour -> gender in your DB
    # If your tennis_players.gender column stores "M"/"F":
    tour_to_gender = {"ATP": "M", "WTA": "F"}

    tours_to_try = [tour] if tour in ("ATP", "WTA") else ["ATP", "WTA"]

    with psycopg2.connect(DATABASE_URL) as conn:
        for t in tours_to_try:
            g = tour_to_gender[t]

            p1_canonical_id = _find_player_id(conn, p1_name, g)
            p2_canonical_id = _find_player_id(conn, p2_name, g)

            if p1_canonical_id and p2_canonical_id:
                # ✅ At this point you can fetch TA snapshot stats and run your model
                return {
                    "ok": True,
                    "reason": "resolved",
                    "tour": t,
                    "p1_canonical_id": p1_canonical_id,
                    "p2_canonical_id": p2_canonical_id,
                    # "proba_p1": ...,
                    # "features": ...,
                }

    return {"ok": False, "reason": "player_not_found", "p1_canonical_id": None, "p2_canonical_id": None}
