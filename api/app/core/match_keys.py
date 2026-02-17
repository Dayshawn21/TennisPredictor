# api/app/core/match_keys.py

import re
import unicodedata
from datetime import date

ROUND_MAP = {
    "final": "F",
    "f": "F",
    "semifinal": "SF",
    "semi-final": "SF",
    "sf": "SF",
    "quarterfinal": "QF",
    "quarter-final": "QF",
    "qf": "QF",
    "round of 16": "R16",
    "r16": "R16",
    "round of 32": "R32",
    "r32": "R32",
    "round of 64": "R64",
    "r64": "R64",
    "round of 128": "R128",
    "r128": "R128",
    "qualifying 1": "Q1",
    "q1": "Q1",
    "qualifying 2": "Q2",
    "q2": "Q2",
    "qualifying 3": "Q3",
    "q3": "Q3",
}

SURFACE_MAP = {
    "hard": "HARD",
    "clay": "CLAY",
    "grass": "GRASS",
    "carpet": "CARPET",
}

def normalize_slug(value: str | None) -> str:
    if not value:
        return "UNKNOWN"
    v = unicodedata.normalize("NFKD", value)
    v = v.encode("ascii", "ignore").decode("ascii")
    v = v.strip().upper()
    v = re.sub(r"['’`]", "", v)
    v = re.sub(r"[^A-Z0-9]+", "_", v)
    v = re.sub(r"_+", "_", v).strip("_")
    return v or "UNKNOWN"

def normalize_round(round_raw: str | None) -> str:
    if not round_raw:
        return "UNKNOWN"
    r = round_raw.strip().lower()
    r = re.sub(r"\s+", " ", r)
    return ROUND_MAP.get(r, normalize_slug(round_raw))

def normalize_surface(surface_raw: str | None) -> str:
    if not surface_raw:
        return "UNKNOWN"
    s = surface_raw.strip().lower()
    return SURFACE_MAP.get(s, normalize_slug(surface_raw))

def build_match_key(
    match_date: date,
    tour: str,
    tournament: str,
    round_raw: str | None,
    surface_raw: str | None,
    p1_name: str,
    p2_name: str,
) -> str:
    return (
        f"{match_date.isoformat()}|"
        f"{normalize_slug(tour)}|"
        f"{normalize_slug(tournament)}|"
        f"{normalize_round(round_raw)}|"
        f"{normalize_surface(surface_raw)}|"
        f"{normalize_slug(p1_name)}|"
        f"{normalize_slug(p2_name)}"
    )
