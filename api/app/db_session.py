from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # ✅ make DATABASE_URL available at import time

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.db.engine import get_engine

engine = get_engine()

AsyncSessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
