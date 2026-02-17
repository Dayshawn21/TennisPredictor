# FILE: api/app/services/tennis/optimize_ensemble_weights.py
"""
Optimize ensemble weights based on historical performance.
Uses grid search to find best combination.
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import logging
from itertools import product
import asyncpg
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")


async def get_historical_predictions(conn):
    """Get all completed predictions."""
    rows = await conn.fetch("""
        SELECT
            elo_p1_prob,
            xgb_p1_prob,
            market_p1_prob,
            p1_won
        FROM tennis_prediction_logs
        WHERE p1_won IS NOT NULL
          AND elo_p1_prob IS NOT NULL
          AND xgb_p1_prob IS NOT NULL
          AND market_p1_prob IS NOT NULL
    """)
    
    return [dict(r) for r in rows]


def calculate_metrics(predictions, w_elo, w_xgb, w_market):
    """
    Calculate ensemble performance for given weights.
    
    Returns: (accuracy, brier_score, log_loss)
    """
    if not predictions:
        return 0, 1, 1
    
    total_weight = w_elo + w_xgb + w_market
    
    accuracies = []
    brier_scores = []
    log_losses = []
    
    for pred in predictions:
        # Ensemble probability
        ensemble_prob = (
            w_elo * pred['elo_p1_prob'] +
            w_xgb * pred['xgb_p1_prob'] +
            w_market * pred['market_p1_prob']
        ) / total_weight
        
        # Clip to avoid log(0)
        ensemble_prob = np.clip(ensemble_prob, 0.001, 0.999)
        
        actual = 1.0 if pred['p1_won'] else 0.0
        
        # Accuracy
        predicted_winner = 1 if ensemble_prob >= 0.5 else 0
        correct = 1 if predicted_winner == actual else 0
        accuracies.append(correct)
        
        # Brier score
        brier_scores.append((ensemble_prob - actual) ** 2)
        
        # Log loss
        if actual == 1:
            log_losses.append(-np.log(ensemble_prob))
        else:
            log_losses.append(-np.log(1 - ensemble_prob))
    
    accuracy = np.mean(accuracies)
    brier = np.mean(brier_scores)
    logloss = np.mean(log_losses)
    
    return accuracy, brier, logloss


async def grid_search(predictions):
    """
    Find optimal weights using grid search.
    """
    logger.info(f"Running grid search on {len(predictions)} predictions...")
    
    # Define search space (percentages that sum to 100)
    elo_weights = [0.3, 0.4, 0.5, 0.6, 0.7]
    xgb_weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    market_weights = [0.0, 0.1, 0.2, 0.3]
    
    best_accuracy = 0
    best_brier = float('inf')
    best_weights = None
    
    results = []
    
    for w_elo, w_xgb, w_market in product(elo_weights, xgb_weights, market_weights):
        accuracy, brier, logloss = calculate_metrics(predictions, w_elo, w_xgb, w_market)
        
        results.append({
            'w_elo': w_elo,
            'w_xgb': w_xgb,
            'w_market': w_market,
            'accuracy': accuracy,
            'brier': brier,
            'logloss': logloss,
            # Combined score: 70% accuracy + 30% calibration (lower brier)
            'score': 0.7 * accuracy + 0.3 * (1 - brier)
        })
    
    # Sort by combined score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results


async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    
    try:
        predictions = await get_historical_predictions(conn)
        
        if len(predictions) < 50:
            logger.warning(f"Only {len(predictions)} completed predictions. Need more data for reliable optimization.")
            logger.info("Run predictions for a few days and update results, then try again.")
            return
        
        logger.info(f"Loaded {len(predictions)} completed predictions")
        
        # Run optimization
        results = await grid_search(predictions)
        
        print("\n" + "="*80)
        print("ENSEMBLE WEIGHT OPTIMIZATION RESULTS")
        print("="*80)
        
        print(f"\nTested on {len(predictions)} completed predictions")
        
        print("\n--- TOP 10 WEIGHT COMBINATIONS ---")
        print(f"{'Rank':<6} {'ELO':>6} {'XGBoost':>8} {'Market':>8} {'Accuracy':>10} {'Brier':>8} {'Score':>8}")
        print("-" * 80)
        
        for i, res in enumerate(results[:10], 1):
            print(f"{i:<6} {res['w_elo']:>6.1f} {res['w_xgb']:>8.1f} {res['w_market']:>8.1f} "
                  f"{res['accuracy']:>9.2%} {res['brier']:>8.4f} {res['score']:>8.4f}")
        
        best = results[0]
        total = best['w_elo'] + best['w_xgb'] + best['w_market']
        
        print("\n" + "="*80)
        print("RECOMMENDED WEIGHTS (normalized to sum to 1.0):")
        print(f"  ELO:     {best['w_elo']/total:.2%}")
        print(f"  XGBoost: {best['w_xgb']/total:.2%}")
        print(f"  Market:  {best['w_market']/total:.2%}")
        print("\nExpected Performance:")
        print(f"  Accuracy:    {best['accuracy']:.2%}")
        print(f"  Brier Score: {best['brier']:.4f}")
        print(f"  Log Loss:    {best['logloss']:.4f}")
        print("="*80 + "\n")
        
        print("\n💡 To use these weights, update app/services/tennis/combined_predictor.py:")
        print(f"   Line ~35: weights.append({best['w_elo']/total:.2f})  # ELO")
        print(f"   Line ~62: weights.append({best['w_xgb']/total:.2f})  # XGBoost")
        print(f"   Line ~85: weights.append({best['w_market']/total:.2f})  # Market")
        
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())