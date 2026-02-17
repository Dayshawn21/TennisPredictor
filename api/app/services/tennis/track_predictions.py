# FILE: api/app/services/tennis/track_predictions.py
"""
Track tennis prediction performance over time, including EV-oriented metrics.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import logging
import os
from datetime import date
from typing import Any, Dict, List, Optional

import asyncpg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _as_json(v: Any) -> str:
    try:
        return json.dumps(v or {})
    except Exception:
        return "{}"


def _individual_to_map(individual: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(individual, dict):
        for k, v in individual.items():
            fv = _safe_float(v)
            if fv is not None:
                out[str(k)] = fv
        return out
    if isinstance(individual, list):
        for row in individual:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            prob = _safe_float(row.get("p1_prob"))
            if name and prob is not None:
                out[name] = prob
    return out


async def create_tracking_tables(conn: asyncpg.Connection):
    """Create/upgrade tables for tracking predictions."""
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tennis_prediction_logs (
            id SERIAL PRIMARY KEY,
            match_id TEXT NOT NULL,
            match_date DATE NOT NULL,
            prediction_date DATE NOT NULL,
            tour TEXT,
            tournament TEXT,
            round TEXT,
            surface TEXT,
            p1_name TEXT,
            p2_name TEXT,

            -- Predictions
            elo_p1_prob NUMERIC,
            xgb_p1_prob NUMERIC,
            market_p1_prob NUMERIC,
            ensemble_p1_prob NUMERIC,
            p1_market_no_vig_prob NUMERIC,

            -- EV diagnostics
            edge_no_vig NUMERIC,
            edge_blended NUMERIC,
            bet_eligible BOOLEAN,
            bet_side TEXT,
            kelly_fraction_capped NUMERIC,
            odds_age_minutes NUMERIC,
            market_overround_main NUMERIC,
            predictor_quality_flags JSONB,
            effective_weights JSONB,
            p1_market_odds_american INT,
            p2_market_odds_american INT,

            -- Actual result
            p1_won BOOLEAN,
            result_updated_at TIMESTAMPTZ,

            -- Metadata
            method TEXT,
            num_predictors INT,
            created_at TIMESTAMPTZ DEFAULT NOW(),

            UNIQUE(match_id, prediction_date)
        )
        """
    )

    await conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_pred_logs_match_date
        ON tennis_prediction_logs(match_date)
        """
    )
    await conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_pred_logs_result
        ON tennis_prediction_logs(p1_won)
        WHERE p1_won IS NOT NULL
        """
    )
    await conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_pred_logs_bet_eligible
        ON tennis_prediction_logs(bet_eligible)
        """
    )
    # Ensure older existing tables gain new EV fields.
    alter_columns = [
        "ALTER TABLE tennis_prediction_logs ADD COLUMN IF NOT EXISTS p1_market_no_vig_prob NUMERIC",
        "ALTER TABLE tennis_prediction_logs ADD COLUMN IF NOT EXISTS edge_no_vig NUMERIC",
        "ALTER TABLE tennis_prediction_logs ADD COLUMN IF NOT EXISTS edge_blended NUMERIC",
        "ALTER TABLE tennis_prediction_logs ADD COLUMN IF NOT EXISTS bet_eligible BOOLEAN",
        "ALTER TABLE tennis_prediction_logs ADD COLUMN IF NOT EXISTS bet_side TEXT",
        "ALTER TABLE tennis_prediction_logs ADD COLUMN IF NOT EXISTS kelly_fraction_capped NUMERIC",
        "ALTER TABLE tennis_prediction_logs ADD COLUMN IF NOT EXISTS odds_age_minutes NUMERIC",
        "ALTER TABLE tennis_prediction_logs ADD COLUMN IF NOT EXISTS market_overround_main NUMERIC",
        "ALTER TABLE tennis_prediction_logs ADD COLUMN IF NOT EXISTS predictor_quality_flags JSONB",
        "ALTER TABLE tennis_prediction_logs ADD COLUMN IF NOT EXISTS effective_weights JSONB",
        "ALTER TABLE tennis_prediction_logs ADD COLUMN IF NOT EXISTS p1_market_odds_american INT",
        "ALTER TABLE tennis_prediction_logs ADD COLUMN IF NOT EXISTS p2_market_odds_american INT",
    ]
    for stmt in alter_columns:
        await conn.execute(stmt)
    logger.info("Tracking tables created/verified")


async def log_predictions(conn: asyncpg.Connection, predictions: List[Dict]):
    """Log predictions for today's matches."""
    logged = 0
    for pred in predictions:
        try:
            inputs = pred.get("inputs", {}) or {}
            ind_map = _individual_to_map(inputs.get("individual_predictions"))
            eff_weights = inputs.get("effective_weights", {}) or {}
            quality_flags = inputs.get("predictor_quality_flags", []) or []

            await conn.execute(
                """
                INSERT INTO tennis_prediction_logs (
                    match_id, match_date, prediction_date,
                    tour, tournament, round, surface,
                    p1_name, p2_name,
                    elo_p1_prob, xgb_p1_prob, market_p1_prob, ensemble_p1_prob,
                    p1_market_no_vig_prob,
                    edge_no_vig, edge_blended, bet_eligible, bet_side, kelly_fraction_capped,
                    odds_age_minutes, market_overround_main,
                    predictor_quality_flags, effective_weights,
                    p1_market_odds_american, p2_market_odds_american,
                    method, num_predictors
                )
                VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22::jsonb,$23::jsonb,$24,$25,$26,$27
                )
                ON CONFLICT (match_id, prediction_date)
                DO UPDATE SET
                    elo_p1_prob = EXCLUDED.elo_p1_prob,
                    xgb_p1_prob = EXCLUDED.xgb_p1_prob,
                    market_p1_prob = EXCLUDED.market_p1_prob,
                    ensemble_p1_prob = EXCLUDED.ensemble_p1_prob,
                    p1_market_no_vig_prob = EXCLUDED.p1_market_no_vig_prob,
                    edge_no_vig = EXCLUDED.edge_no_vig,
                    edge_blended = EXCLUDED.edge_blended,
                    bet_eligible = EXCLUDED.bet_eligible,
                    bet_side = EXCLUDED.bet_side,
                    kelly_fraction_capped = EXCLUDED.kelly_fraction_capped,
                    odds_age_minutes = EXCLUDED.odds_age_minutes,
                    market_overround_main = EXCLUDED.market_overround_main,
                    predictor_quality_flags = EXCLUDED.predictor_quality_flags,
                    effective_weights = EXCLUDED.effective_weights,
                    p1_market_odds_american = EXCLUDED.p1_market_odds_american,
                    p2_market_odds_american = EXCLUDED.p2_market_odds_american,
                    method = EXCLUDED.method,
                    num_predictors = EXCLUDED.num_predictors
                """,
                pred.get("match_id"),
                pred.get("match_date"),
                date.today(),
                pred.get("tour"),
                pred.get("tournament"),
                pred.get("round"),
                pred.get("surface"),
                pred.get("p1_name"),
                pred.get("p2_name"),
                ind_map.get("elo"),
                ind_map.get("xgb"),
                ind_map.get("market_no_vig"),
                pred.get("p1_win_prob"),
                pred.get("p1_market_no_vig_prob"),
                pred.get("edge_no_vig"),
                pred.get("edge_blended"),
                pred.get("bet_eligible"),
                pred.get("bet_side"),
                pred.get("kelly_fraction_capped"),
                inputs.get("odds_age_minutes"),
                inputs.get("market_overround_main"),
                _as_json(quality_flags),
                _as_json(eff_weights),
                pred.get("p1_market_odds_american"),
                pred.get("p2_market_odds_american"),
                inputs.get("method"),
                inputs.get("num_predictors"),
            )
            logged += 1
        except Exception as e:
            logger.error("Failed to log prediction for %s: %s", pred.get("match_id"), e)
    logger.info("Logged %d predictions", logged)
    return logged


def parse_winner_from_score(score: str) -> bool | None:
    if not score:
        return None
    try:
        sets = score.strip().split()
        p1_sets_won = 0
        p2_sets_won = 0
        for set_score in sets:
            if "-" not in set_score:
                continue
            set_score = set_score.split("(")[0]
            parts = set_score.split("-")
            if len(parts) != 2:
                continue
            p1_games = int(parts[0])
            p2_games = int(parts[1])
            if p1_games > p2_games:
                p1_sets_won += 1
            elif p2_games > p1_games:
                p2_sets_won += 1
        if p1_sets_won > p2_sets_won:
            return True
        if p2_sets_won > p1_sets_won:
            return False
        return None
    except Exception:
        return None


async def update_results(conn: asyncpg.Connection):
    pending = await conn.fetch(
        """
        SELECT id, match_id
        FROM tennis_prediction_logs
        WHERE p1_won IS NULL
          AND match_date < CURRENT_DATE
        ORDER BY match_date DESC
        LIMIT 1000
        """
    )
    updated = 0
    for row in pending:
        result = await conn.fetchrow(
            """
            SELECT COALESCE(score, score_raw) AS score, status
            FROM tennis_matches
            WHERE match_id::text = $1
              AND status IN ('finished', 'completed', 'ended')
              AND COALESCE(score, score_raw) IS NOT NULL
            """,
            row["match_id"],
        )
        if not result:
            continue
        p1_won = parse_winner_from_score(result["score"])
        if p1_won is None:
            continue
        await conn.execute(
            """
            UPDATE tennis_prediction_logs
            SET p1_won = $1,
                result_updated_at = NOW()
            WHERE id = $2
            """,
            p1_won,
            row["id"],
        )
        updated += 1
    logger.info("Updated %d results", updated)
    return updated


async def calculate_performance_metrics(conn: asyncpg.Connection) -> Dict:
    metrics: Dict[str, Any] = {}

    metrics["overall"] = dict(
        await conn.fetchrow(
            """
            SELECT
                COUNT(*) as total_predictions,
                COUNT(*) FILTER (WHERE p1_won IS NOT NULL) as completed_matches,
                AVG(CASE WHEN p1_won THEN ensemble_p1_prob ELSE (1 - ensemble_p1_prob) END) as ensemble_avg_prob,
                AVG(CASE WHEN (ensemble_p1_prob >= 0.5 AND p1_won) OR (ensemble_p1_prob < 0.5 AND NOT p1_won) THEN 1.0 ELSE 0.0 END) as ensemble_accuracy,
                AVG(CASE WHEN p1_won THEN elo_p1_prob ELSE (1 - elo_p1_prob) END) FILTER (WHERE elo_p1_prob IS NOT NULL) as elo_avg_prob,
                AVG(CASE WHEN (elo_p1_prob >= 0.5 AND p1_won) OR (elo_p1_prob < 0.5 AND NOT p1_won) THEN 1.0 ELSE 0.0 END) FILTER (WHERE elo_p1_prob IS NOT NULL) as elo_accuracy,
                AVG(CASE WHEN p1_won THEN xgb_p1_prob ELSE (1 - xgb_p1_prob) END) FILTER (WHERE xgb_p1_prob IS NOT NULL) as xgb_avg_prob,
                AVG(CASE WHEN (xgb_p1_prob >= 0.5 AND p1_won) OR (xgb_p1_prob < 0.5 AND NOT p1_won) THEN 1.0 ELSE 0.0 END) FILTER (WHERE xgb_p1_prob IS NOT NULL) as xgb_accuracy,
                AVG(CASE WHEN p1_won THEN market_p1_prob ELSE (1 - market_p1_prob) END) FILTER (WHERE market_p1_prob IS NOT NULL) as market_avg_prob,
                AVG(CASE WHEN (market_p1_prob >= 0.5 AND p1_won) OR (market_p1_prob < 0.5 AND NOT p1_won) THEN 1.0 ELSE 0.0 END) FILTER (WHERE market_p1_prob IS NOT NULL) as market_accuracy,
                AVG(edge_blended) FILTER (WHERE edge_blended IS NOT NULL) as avg_edge_blended
            FROM tennis_prediction_logs
            WHERE p1_won IS NOT NULL
            """
        )
        or {}
    )

    metrics["brier_scores"] = dict(
        await conn.fetchrow(
            """
            SELECT
                AVG(POWER(ensemble_p1_prob - CASE WHEN p1_won THEN 1.0 ELSE 0.0 END, 2)) as ensemble_brier,
                AVG(POWER(elo_p1_prob - CASE WHEN p1_won THEN 1.0 ELSE 0.0 END, 2)) FILTER (WHERE elo_p1_prob IS NOT NULL) as elo_brier,
                AVG(POWER(xgb_p1_prob - CASE WHEN p1_won THEN 1.0 ELSE 0.0 END, 2)) FILTER (WHERE xgb_p1_prob IS NOT NULL) as xgb_brier,
                AVG(POWER(market_p1_prob - CASE WHEN p1_won THEN 1.0 ELSE 0.0 END, 2)) FILTER (WHERE market_p1_prob IS NOT NULL) as market_brier
            FROM tennis_prediction_logs
            WHERE p1_won IS NOT NULL
            """
        )
        or {}
    )

    metrics["by_surface"] = [
        dict(r)
        for r in await conn.fetch(
            """
            SELECT
                surface,
                COUNT(*) as matches,
                AVG(CASE WHEN (ensemble_p1_prob >= 0.5 AND p1_won) OR (ensemble_p1_prob < 0.5 AND NOT p1_won) THEN 1.0 ELSE 0.0 END) as ensemble_accuracy
            FROM tennis_prediction_logs
            WHERE p1_won IS NOT NULL
              AND surface IS NOT NULL
            GROUP BY surface
            ORDER BY matches DESC
            """
        )
    ]

    metrics["market_availability"] = [
        dict(r)
        for r in await conn.fetch(
            """
            SELECT
                (p1_market_no_vig_prob IS NOT NULL) as has_market,
                COUNT(*) as matches,
                AVG(CASE WHEN (ensemble_p1_prob >= 0.5 AND p1_won) OR (ensemble_p1_prob < 0.5 AND NOT p1_won) THEN 1.0 ELSE 0.0 END) as ensemble_accuracy,
                AVG(edge_blended) FILTER (WHERE edge_blended IS NOT NULL) as avg_edge_blended
            FROM tennis_prediction_logs
            WHERE p1_won IS NOT NULL
            GROUP BY (p1_market_no_vig_prob IS NOT NULL)
            ORDER BY has_market DESC
            """
        )
    ]

    metrics["odds_freshness_bands"] = [
        dict(r)
        for r in await conn.fetch(
            """
            SELECT
                CASE
                    WHEN odds_age_minutes IS NULL THEN 'missing'
                    WHEN odds_age_minutes <= 60 THEN 'fresh_0_60m'
                    WHEN odds_age_minutes <= 180 THEN 'fresh_61_180m'
                    ELSE 'stale_180m_plus'
                END AS freshness_band,
                COUNT(*) as matches,
                AVG(CASE WHEN (ensemble_p1_prob >= 0.5 AND p1_won) OR (ensemble_p1_prob < 0.5 AND NOT p1_won) THEN 1.0 ELSE 0.0 END) as ensemble_accuracy,
                AVG(edge_blended) FILTER (WHERE edge_blended IS NOT NULL) as avg_edge_blended
            FROM tennis_prediction_logs
            WHERE p1_won IS NOT NULL
            GROUP BY freshness_band
            ORDER BY matches DESC
            """
        )
    ]

    metrics["ev_summary"] = dict(
        await conn.fetchrow(
            """
            WITH settled AS (
                SELECT
                    *,
                    CASE
                        WHEN bet_eligible IS NOT TRUE OR p1_won IS NULL THEN NULL
                        WHEN bet_side = 'p1' THEN
                            CASE
                                WHEN p1_won THEN
                                    CASE
                                        WHEN p1_market_odds_american > 0 THEN p1_market_odds_american::numeric / 100.0
                                        WHEN p1_market_odds_american < 0 THEN 100.0 / abs(p1_market_odds_american::numeric)
                                        ELSE NULL
                                    END
                                ELSE -1.0
                            END
                        WHEN bet_side = 'p2' THEN
                            CASE
                                WHEN NOT p1_won THEN
                                    CASE
                                        WHEN p2_market_odds_american > 0 THEN p2_market_odds_american::numeric / 100.0
                                        WHEN p2_market_odds_american < 0 THEN 100.0 / abs(p2_market_odds_american::numeric)
                                        ELSE NULL
                                    END
                                ELSE -1.0
                            END
                        ELSE NULL
                    END as unit_profit
                FROM tennis_prediction_logs
                WHERE p1_won IS NOT NULL
            )
            SELECT
                COUNT(*) as candidate_count,
                COUNT(*) FILTER (WHERE bet_eligible IS TRUE) as placed_bet_count,
                AVG(edge_blended) FILTER (WHERE bet_eligible IS TRUE AND edge_blended IS NOT NULL) as avg_edge_placed,
                AVG(unit_profit) FILTER (WHERE unit_profit IS NOT NULL) as roi_per_bet
            FROM settled
            """
        )
        or {}
    )

    market_copy = await conn.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE p1_market_no_vig_prob IS NOT NULL) as samples,
            AVG(ABS(ensemble_p1_prob - p1_market_no_vig_prob)) FILTER (WHERE p1_market_no_vig_prob IS NOT NULL) as mean_abs_gap,
            AVG(ABS(ensemble_p1_prob - p1_market_no_vig_prob)) FILTER (
                WHERE p1_market_no_vig_prob IS NOT NULL
                  AND ABS(ensemble_p1_prob - 0.5) >= 0.10
            ) as mean_abs_gap_high_conf
        FROM tennis_prediction_logs
        WHERE p1_won IS NOT NULL
        """
    )
    market_copy_d = dict(market_copy or {})
    alert = False
    try:
        g = _safe_float(market_copy_d.get("mean_abs_gap_high_conf"))
        if g is not None and g < 0.015:
            alert = True
    except Exception:
        alert = False
    market_copy_d["hidden_market_copy_alert"] = alert
    metrics["market_copy_monitor"] = market_copy_d

    return metrics


async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await create_tracking_tables(conn)
        await update_results(conn)
        metrics = await calculate_performance_metrics(conn)

        print("\n" + "=" * 64)
        print("TENNIS PREDICTION PERFORMANCE")
        print("=" * 64)
        overall = metrics.get("overall", {})
        if overall:
            print(f"Total predictions: {overall.get('total_predictions', 0)}")
            print(f"Completed:         {overall.get('completed_matches', 0)}")
            print(f"Ensemble accuracy: {overall.get('ensemble_accuracy', 0):.2%}")
            print(f"Avg edge blended:  {overall.get('avg_edge_blended', 0) or 0:.4f}")
        ev = metrics.get("ev_summary", {})
        if ev:
            print(f"Placed bets:       {ev.get('placed_bet_count', 0)}")
            print(f"ROI / bet:         {ev.get('roi_per_bet', 0) or 0:.4f}")
        mc = metrics.get("market_copy_monitor", {})
        if mc:
            print(f"Mean abs gap (high conf): {mc.get('mean_abs_gap_high_conf', 0) or 0:.4f}")
            if mc.get("hidden_market_copy_alert"):
                print("ALERT: Ensemble may be over-tracking market no-vig probabilities.")
        print("=" * 64 + "\n")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
