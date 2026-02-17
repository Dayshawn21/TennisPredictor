from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import asyncpg


def _freshness_band(age_min: float | None) -> str:
    if age_min is None:
        return "missing"
    if age_min <= 60:
        return "fresh_0_60m"
    if age_min <= 180:
        return "fresh_61_180m"
    return "stale_180m_plus"


async def main():
    db_url = os.getenv("DATABASE_URL", "")
    if db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")

    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            """
            SELECT
                ensemble_p1_prob,
                p1_market_no_vig_prob,
                p1_won,
                odds_age_minutes
            FROM tennis_prediction_logs
            WHERE p1_won IS NOT NULL
            """
        )
    finally:
        await conn.close()

    splits: Dict[str, Dict[str, Any]] = {
        "has_market": {"n": 0, "acc_sum": 0.0},
        "no_market": {"n": 0, "acc_sum": 0.0},
    }
    freshness = defaultdict(lambda: {"n": 0, "acc_sum": 0.0, "gap_sum": 0.0, "gap_n": 0})
    high_conf_gap_sum = 0.0
    high_conf_gap_n = 0

    for r in rows:
        p = r["ensemble_p1_prob"]
        m = r["p1_market_no_vig_prob"]
        y = bool(r["p1_won"])
        if p is None:
            continue
        p = float(p)
        correct = 1.0 if ((p >= 0.5 and y) or (p < 0.5 and not y)) else 0.0

        mk = "has_market" if m is not None else "no_market"
        splits[mk]["n"] += 1
        splits[mk]["acc_sum"] += correct

        band = _freshness_band(float(r["odds_age_minutes"]) if r["odds_age_minutes"] is not None else None)
        freshness[band]["n"] += 1
        freshness[band]["acc_sum"] += correct
        if m is not None:
            gap = abs(p - float(m))
            freshness[band]["gap_sum"] += gap
            freshness[band]["gap_n"] += 1
            if abs(p - 0.5) >= 0.10:
                high_conf_gap_sum += gap
                high_conf_gap_n += 1

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "market_split": {
            k: {
                "n": v["n"],
                "accuracy": (v["acc_sum"] / v["n"]) if v["n"] else None,
            }
            for k, v in splits.items()
        },
        "freshness_split": {
            k: {
                "n": v["n"],
                "accuracy": (v["acc_sum"] / v["n"]) if v["n"] else None,
                "mean_abs_prob_gap_vs_market": (v["gap_sum"] / v["gap_n"]) if v["gap_n"] else None,
            }
            for k, v in freshness.items()
        },
        "hidden_market_copy_monitor": {
            "mean_abs_gap_high_conf": (high_conf_gap_sum / high_conf_gap_n) if high_conf_gap_n else None,
            "alert_threshold": 0.015,
            "alert": ((high_conf_gap_sum / high_conf_gap_n) < 0.015) if high_conf_gap_n else False,
        },
    }

    out_dir = Path("api/app/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"market_dependency_report_{ts}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
