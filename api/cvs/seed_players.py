
import csv
import asyncio
from pathlib import Path

from sqlalchemy import text
from app.db import get_db

BASE_DIR = Path(__file__).resolve().parent
ATP_CSV = BASE_DIR / "atp_players.csv"
WTA_CSV = BASE_DIR / "wta_players.csv"


def make_name(row: dict) -> str:
    first = (row.get("name_first") or "").strip()
    last = (row.get("name_last") or "").strip()
    return f"{first} {last}".strip()


def make_slug(name: str) -> str:
    # simple stable slug for your ta_slug column
    s = name.lower().strip()
    out = []
    last_was_dash = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            last_was_dash = False
        else:
            if not last_was_dash:
                out.append("-")
                last_was_dash = True
    slug = "".join(out).strip("-")
    return slug


async def seed_file(csv_path: Path, gender: str) -> int:
    if not csv_path.exists():
        print(f"❌ Missing: {csv_path}")
        return 0

    BATCH_SIZE = 2000
    count = 0
    processed = 0

    print(f"➡️ Seeding {csv_path.name} ({gender})...")

    async for db in get_db():
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            batch = []
            for row in reader:
                processed += 1

                name = make_name(row)
                if not name:
                    continue

                try:
                    ta_id = int(row["player_id"])
                except Exception:
                    continue

                slug = make_slug(f"{gender}-{ta_id}-{name}")

                batch.append(
                    {
                        "name": name,
                        "gender": gender,
                        "ta_player_id": ta_id,
                        "ta_slug": slug,
                    }
                )

                if processed % 5000 == 0:
                    print(f"   ...read {processed} rows, staged {len(batch)} in current batch, total written {count}")

                if len(batch) >= BATCH_SIZE:
                    await db.execute(
                        text("""
                            INSERT INTO tennis_players (name, gender, ta_player_id, ta_slug)
                            VALUES (:name, :gender, :ta_player_id, :ta_slug)
                            ON CONFLICT (name, gender)
                            DO UPDATE SET
                              ta_player_id = COALESCE(EXCLUDED.ta_player_id, tennis_players.ta_player_id),
                              ta_slug      = COALESCE(tennis_players.ta_slug, EXCLUDED.ta_slug),
                              name         = EXCLUDED.name
                        """),
                        batch,
                    )
                    await db.commit()
                    count += len(batch)
                    batch.clear()

            if batch:
                await db.execute(
                    text("""
                        INSERT INTO tennis_players (name, gender, ta_player_id, ta_slug)
                        VALUES (:name, :gender, :ta_player_id, :ta_slug)
                        ON CONFLICT (name, gender)
                        DO UPDATE SET
                          ta_player_id = COALESCE(EXCLUDED.ta_player_id, tennis_players.ta_player_id),
                          ta_slug      = COALESCE(tennis_players.ta_slug, EXCLUDED.ta_slug),
                          name         = EXCLUDED.name
                    """),
                    batch,
                )
                await db.commit()
                count += len(batch)

        break

    print(f"✅ Seeded {count} rows from {csv_path.name} ({gender})")
    return count


async def main():
    total = 0
    total += await seed_file(ATP_CSV, "M")
    total += await seed_file(WTA_CSV, "F")

    # quick counts
    async for db in get_db():
        m = await db.execute(text("SELECT COUNT(*) FROM tennis_players WHERE gender='M'"))
        f = await db.execute(text("SELECT COUNT(*) FROM tennis_players WHERE gender='F'"))
        print("Men:", m.scalar_one())
        print("Women:", f.scalar_one())
        break

    print(f"Done. Total processed: {total}")


if __name__ == "__main__":
    asyncio.run(main())
