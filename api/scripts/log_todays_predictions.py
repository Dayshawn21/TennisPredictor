# api/scripts/log_todays_predictions.py
import argparse
import asyncio
from datetime import datetime
from typing import Any

import asyncpg
from dotenv import load_dotenv

from app.services.tennis.track_predictions import create_tracking_tables, log_predictions

load_dotenv()


async def main(
    *,
    days_ahead: int,
    include_incomplete: bool,
    bust_cache: bool,
    min_edge: float,
    max_odds_age_min: int,
    max_overround: float,
):
    import os

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    if database_url.startswith("postgresql+asyncpg://"):
        database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

    conn = await asyncpg.connect(database_url)
    try:
        await create_tracking_tables(conn)

        # Call enhanced endpoint function directly with explicit values
        # (avoid using FastAPI Query objects as defaults).
        from app.routers.tennis_predictions_today_enhanced import predictions_today_enhanced

        preds = await predictions_today_enhanced(
            days_ahead=days_ahead,
            include_incomplete=include_incomplete,
            bust_cache=bust_cache,
            min_edge=min_edge,
            max_odds_age_min=max_odds_age_min,
            max_overround=max_overround,
        )

        preds_data: dict[str, Any]
        if hasattr(preds, "model_dump"):
            preds_data = preds.model_dump(mode="json")
        elif isinstance(preds, dict):
            preds_data = preds
        else:
            preds_data = {}

        pred_dicts = []
        for p in preds_data.get("items", []):
            if isinstance(p.get("match_date"), str):
                match_date = datetime.fromisoformat(p["match_date"]).date()
            else:
                match_date = p.get("match_date")

            pred_dicts.append(
                {
                    "match_id": str(p.get("match_id")),
                    "match_date": match_date,
                    "tour": p.get("tour"),
                    "tournament": p.get("tournament"),
                    "round": p.get("round"),
                    "surface": p.get("surface"),
                    "p1_name": p.get("p1_name"),
                    "p2_name": p.get("p2_name"),
                    "p1_win_prob": p.get("p1_win_prob"),
                    "p1_market_no_vig_prob": p.get("p1_market_no_vig_prob"),
                    "p1_market_odds_american": p.get("p1_market_odds_american"),
                    "p2_market_odds_american": p.get("p2_market_odds_american"),
                    "edge_no_vig": p.get("edge_no_vig"),
                    "edge_blended": p.get("edge_blended"),
                    "bet_eligible": p.get("bet_eligible"),
                    "bet_side": p.get("bet_side"),
                    "kelly_fraction_capped": p.get("kelly_fraction_capped"),
                    "inputs": p.get("inputs", {}),
                }
            )

        await log_predictions(conn, pred_dicts)
        print(f"Successfully logged {len(pred_dicts)} predictions")
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log enhanced tennis predictions to tennis_prediction_logs.")
    parser.add_argument("--days-ahead", type=int, default=1, help="Include matches from today through this many days ahead.")
    parser.add_argument(
        "--include-incomplete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include non-finished/non-completed matches in the source endpoint query.",
    )
    parser.add_argument(
        "--bust-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bypass endpoint cache for fresh pulls.",
    )
    parser.add_argument("--min-edge", type=float, default=0.025, help="Minimum absolute edge for bet eligibility.")
    parser.add_argument("--max-odds-age-min", type=int, default=180, help="Maximum odds age in minutes for fresh-odds gating.")
    parser.add_argument("--max-overround", type=float, default=0.08, help="Maximum acceptable market overround.")
    args = parser.parse_args()

    asyncio.run(
        main(
            days_ahead=max(0, int(args.days_ahead)),
            include_incomplete=bool(args.include_incomplete),
            bust_cache=bool(args.bust_cache),
            min_edge=float(args.min_edge),
            max_odds_age_min=max(1, int(args.max_odds_age_min)),
            max_overround=float(args.max_overround),
        )
    )
