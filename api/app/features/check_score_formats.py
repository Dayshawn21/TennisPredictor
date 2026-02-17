# FILE: api/app/features/check_score_formats.py
"""Check what score formats exist in the database."""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import asyncpg

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")


async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    
    try:
        print("\n=== SCORE FORMATS ===\n")
        
        # Check different score formats
        result = await conn.fetch("""
            SELECT 
                score,
                score_raw,
                COUNT(*) as count
            FROM tennis_matches
            WHERE status = 'finished'
              AND match_date >= '2020-01-01'
              AND tour IN ('ATP', 'WTA')
            GROUP BY score, score_raw
            ORDER BY count DESC
            LIMIT 20
        """)
        
        print("Top score formats:")
        for row in result:
            print(f"  score: {row['score']!r:30s} score_raw: {row['score_raw']!r:30s} count: {row['count']}")
        
        # Check if we have winner_canonical_id
        print("\n\nChecking for winner field:")
        result = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total,
                COUNT(winner_canonical_id) as has_winner,
                COUNT(CASE WHEN score IS NOT NULL THEN 1 END) as has_score,
                COUNT(CASE WHEN score_raw IS NOT NULL THEN 1 END) as has_score_raw
            FROM tennis_matches
            WHERE status = 'finished'
              AND match_date >= '2020-01-01'
              AND tour IN ('ATP', 'WTA')
        """)
        print(f"  Total finished: {result['total']}")
        print(f"  Has winner_canonical_id: {result['has_winner']}")
        print(f"  Has score: {result['has_score']}")
        print(f"  Has score_raw: {result['has_score_raw']}")
        
        # Sample matches
        print("\n\nSample finished matches:")
        result = await conn.fetch("""
            SELECT match_date, tour, p1_name, p2_name, score, score_raw, 
                   winner_canonical_id, p1_canonical_id, p2_canonical_id
            FROM tennis_matches
            WHERE status = 'finished'
              AND match_date >= '2024-01-01'
              AND tour IN ('ATP', 'WTA')
            ORDER BY match_date DESC
            LIMIT 10
        """)
        for row in result:
            winner = "P1" if row['winner_canonical_id'] == row['p1_canonical_id'] else "P2" if row['winner_canonical_id'] == row['p2_canonical_id'] else "?"
            print(f"\n  {row['match_date']} {row['tour']}: {row['p1_name']} vs {row['p2_name']}")
            print(f"    score: {row['score']!r}, score_raw: {row['score_raw']!r}")
            print(f"    Winner: {winner}")
    
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())