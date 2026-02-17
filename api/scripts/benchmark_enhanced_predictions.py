from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
from decimal import Decimal

import asyncpg


def _clamp01(x: float) -> float:
    return max(1e-9, min(1 - 1e-9, float(x)))


def _json_default(o: Any):
    if isinstance(o, Decimal):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _log_loss(prob: float, y: int) -> float:
    p = _clamp01(prob)
    return -(math.log(p) if y == 1 else math.log(1 - p))


def _bucket(abs_edge: float) -> str:
    if abs_edge < 0.01:
        return "<0.01"
    if abs_edge < 0.02:
        return "0.01-0.02"
    if abs_edge < 0.03:
        return "0.02-0.03"
    if abs_edge < 0.05:
        return "0.03-0.05"
    return ">=0.05"


def _unit_profit(side: str, p1_won: bool, p1_odds: int | None, p2_odds: int | None) -> float | None:
    def payout(odds: int | None) -> float | None:
        if odds is None:
            return None
        if odds > 0:
            return float(odds) / 100.0
        if odds < 0:
            return 100.0 / abs(float(odds))
        return None

    if side == "p1":
        if p1_won:
            return payout(p1_odds)
        return -1.0
    if side == "p2":
        if not p1_won:
            return payout(p2_odds)
        return -1.0
    return None


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
                match_id, match_date,
                ensemble_p1_prob, p1_market_no_vig_prob,
                edge_no_vig, edge_blended,
                bet_side, bet_eligible,
                p1_market_odds_american, p2_market_odds_american,
                odds_age_minutes,
                p1_won
            FROM tennis_prediction_logs
            WHERE p1_won IS NOT NULL
            ORDER BY match_date, match_id
            """
        )
    finally:
        await conn.close()

    if not rows:
        print("No settled rows found in tennis_prediction_logs.")
        return

    out_rows: List[Dict[str, Any]] = []
    log_losses: List[float] = []
    briers: List[float] = []
    picks_correct = 0
    picks_total = 0
    edge_vals: List[float] = []
    bucket_roi: Dict[str, List[float]] = {}
    clv_proxy_rows = 0

    for r in rows:
        p = r["ensemble_p1_prob"]
        y = 1 if r["p1_won"] else 0
        if p is not None:
            p = float(p)
            log_losses.append(_log_loss(p, y))
            briers.append((p - y) ** 2)
            picks_total += 1
            picks_correct += int((p >= 0.5 and y == 1) or (p < 0.5 and y == 0))

        edge = float(r["edge_blended"]) if r["edge_blended"] is not None else None
        if edge is not None:
            edge_vals.append(edge)

        side = (r["bet_side"] or "none") if r["bet_eligible"] else "none"
        profit = _unit_profit(side, bool(r["p1_won"]), r["p1_market_odds_american"], r["p2_market_odds_american"])
        if edge is not None and profit is not None:
            b = _bucket(abs(edge))
            bucket_roi.setdefault(b, []).append(float(profit))

        if r["p1_market_no_vig_prob"] is not None and p is not None:
            clv_proxy_rows += 1

        out_rows.append(
            {
                "match_id": r["match_id"],
                "match_date": str(r["match_date"]),
                "ensemble_p1_prob": p,
                "p1_won": y,
                "edge_no_vig": r["edge_no_vig"],
                "edge_blended": r["edge_blended"],
                "bet_eligible": r["bet_eligible"],
                "bet_side": side,
                "unit_profit": profit,
                "odds_age_minutes": r["odds_age_minutes"],
            }
        )

    summary = {
        "rows": len(out_rows),
        "brier": (sum(briers) / len(briers)) if briers else None,
        "log_loss": (sum(log_losses) / len(log_losses)) if log_losses else None,
        "pick_accuracy": (picks_correct / picks_total) if picks_total else None,
        "avg_edge_blended": (sum(edge_vals) / len(edge_vals)) if edge_vals else None,
        "roi_by_edge_bucket": {
            k: {"n": len(v), "roi_per_bet": (sum(v) / len(v)) if v else None}
            for k, v in sorted(bucket_roi.items())
        },
        "clv_proxy": {
            "available": clv_proxy_rows > 0,
            "note": "Uses no-vig market probability gap proxy; true CLV needs historical close snapshots."
        },
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("api/app/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"enhanced_benchmark_{ts}.json"
    csv_path = out_dir / f"enhanced_benchmark_rows_{ts}.csv"

    json_path.write_text(
        json.dumps({"summary": summary, "rows": out_rows}, indent=2, default=_json_default),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
