import asyncio, asyncpg

async def go():
    conn = await asyncpg.connect(
        "postgresql://neondb_owner:npg_3bxYFijyoeD4@ep-dawn-wave-a8928yr9-pooler.eastus2.azure.neon.tech/neondb?sslmode=require"
    )
    rows = await conn.fetch("""
        SELECT date_part('year', m.match_date)::int AS yr, count(*) AS cnt
        FROM tennis_features_ta f
        JOIN tennis_matches m ON m.match_id = f.match_id
        GROUP BY 1 ORDER BY 1
    """)
    for r in rows:
        print(r['yr'], r['cnt'])
    await conn.close()

asyncio.run(go())
