import asyncio
from sqlalchemy import text
from app.db import get_db

PLAYERS = [
    "Jaume Munar",
    "Sebastian Baez",
    "Jessica Bouzas Maneiro",
    "Solana Sierra",
]

async def main():
    async for db in get_db():
        for name in PLAYERS:
            await db.execute(
                text("""
                    INSERT INTO tennis_players (name)
                    VALUES (:name)
                """),
                {"name": name},
            )
        await db.commit()

        r = await db.execute(text("SELECT id, name FROM tennis_players ORDER BY id"))
        print("Players now in DB:")
        for row in r.fetchall():
            print(row)
        break

asyncio.run(main())
