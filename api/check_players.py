import asyncio
from sqlalchemy import text
from app.db import get_db

async def main():
    async for db in get_db():
        result = await db.execute(
            text("SELECT id, name FROM tennis_players ORDER BY id DESC LIMIT 20")
        )
        rows = result.fetchall()
        print("Players in DB:")
        for r in rows:
            print(r)
        break

asyncio.run(main())
