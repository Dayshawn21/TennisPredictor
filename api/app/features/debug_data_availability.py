# FILE: api/app/features/debug_data_availability.py
"""
Debug script to check what data is available for training.
"""

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
        print("\n=== CHECKING DATA AVAILABILITY ===\n")
        
        # 1. Check matches table
        print("1. Matches with canonical IDs:")
        result = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_matches,
                COUNT(p1_canonical_id) as has_p1_canonical,
                COUNT(p2_canonical_id) as has_p2_canonical,
                COUNT(CASE WHEN p1_canonical_id IS NOT NULL AND p2_canonical_id IS NOT NULL THEN 1 END) as has_both_canonical
            FROM tennis_matches
            WHERE status IN ('finished', 'completed', 'ended')
              AND match_date >= '2020-01-01'
              AND tour IN ('ATP', 'WTA')
        """)
        print(f"  Total finished matches: {result['total_matches']}")
        print(f"  Has p1_canonical_id: {result['has_p1_canonical']}")
        print(f"  Has p2_canonical_id: {result['has_p2_canonical']}")
        print(f"  Has BOTH canonical IDs: {result['has_both_canonical']}")
        
        # 2. Check ELO snapshots
        print("\n2. ELO Snapshots:")
        result = await conn.fetch("""
            SELECT tour, COUNT(*) as count, MIN(as_of_date) as min_date, MAX(as_of_date) as max_date
            FROM tennisabstract_elo_snapshots
            GROUP BY tour
        """)
        for row in result:
            print(f"  {row['tour']}: {row['count']} records from {row['min_date']} to {row['max_date']}")
        
        # 3. Check tennis_player_sources
        print("\n3. Tennis Player Sources (TA ELO mappings):")
        result = await conn.fetch("""
            SELECT source, COUNT(*) as count
            FROM tennis_player_sources
            WHERE source LIKE '%tennisabstract%'
            GROUP BY source
        """)
        if result:
            for row in result:
                print(f"  {row['source']}: {row['count']} mappings")
        else:
            print("  ⚠️ NO tennis_player_sources for tennisabstract!")
        
        # 4. Sample match to see structure
        print("\n4. Sample recent match:")
        result = await conn.fetchrow("""
            SELECT match_id, match_date, tour, p1_name, p2_name, 
                   p1_canonical_id, p2_canonical_id, score, score_raw
            FROM tennis_matches
            WHERE status = 'finished'
              AND match_date >= '2025-01-01'
              AND tour IN ('ATP', 'WTA')
            ORDER BY match_date DESC
            LIMIT 1
        """)
        if result:
            print(f"  Date: {result['match_date']}")
            print(f"  Tour: {result['tour']}")
            print(f"  Match: {result['p1_name']} vs {result['p2_name']}")
            print(f"  P1 canonical: {result['p1_canonical_id']}")
            print(f"  P2 canonical: {result['p2_canonical_id']}")
            print(f"  Score: {result['score']}")
            print(f"  Score raw: {result['score_raw']}")
        
        # 5. Check if we can join anything
        print("\n5. Test join between matches and ELO:")
        result = await conn.fetchrow("""
            SELECT COUNT(*) as count
            FROM tennis_matches m
            WHERE m.match_date >= '2025-01-01'
              AND m.tour = 'ATP'
              AND m.p1_canonical_id IS NOT NULL
              AND EXISTS (
                SELECT 1 FROM tennisabstract_elo_snapshots s
                WHERE s.tour = 'ATP'
              )
            LIMIT 1
        """)
        print(f"  Recent ATP matches with canonical IDs: {result['count']}")
        
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())