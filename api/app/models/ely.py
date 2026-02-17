import math
from typing import Any

def _to_float_or_none(x: Any) -> float | None:
    """Convert Decimal/int/str to float; return None for None/invalid."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def elo_prob(elo_a, elo_b, scale=400.0):
    if elo_a is None or elo_b is None:
        return None
    elo_a = float(elo_a)
    elo_b = float(elo_b)
    scale = float(scale)
    return 1.0 / (1.0 + 10 ** (-(elo_a - elo_b) / scale))



def _first_not_none(*vals: Any) -> Any:
    """Return the first value that is not None."""
    for v in vals:
        if v is not None:
            return v
    return None


def pick_surface_elo(row, prefix: str):
    surface = (row.get("surface") or "").lower()

    def pick(*keys):
        for k in keys:
            v = row.get(k)
            if v is not None:
                return v
        return None

    if "hard" in surface:
        return pick(f"{prefix}_helo", f"{prefix}_elo")
    if "clay" in surface:
        return pick(f"{prefix}_celo", f"{prefix}_elo")
    if "grass" in surface:
        return pick(f"{prefix}_gelo", f"{prefix}_elo")

    return row.get(f"{prefix}_elo")

