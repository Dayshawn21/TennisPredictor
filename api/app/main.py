from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

# Your existing router imports
from app.routers import tennis_predictions_today_enhanced
from app.db_session import get_db
from .routers import nba, nfl, nhl, cfb, cbb, tennis, tennis_props, tennis_auto
from app.routers.api_tennis_router import router as api_tennis_router
from app.routers.tennis_admin_elo import router as tennis_admin_elo_router
from app.routers import tennis_value_bets

app = FastAPI(title="Dayshawn Sports Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import logging
import time as _time

_logger = logging.getLogger(__name__)


@app.on_event("startup")
async def _warmup():
    """Pre-load heavy artifacts so the first request isn't slow."""
    t0 = _time.perf_counter()

    # 1. XGBoost model (CPU-bound, ~200-400 ms)
    from app.services.tennis.tennis_predictor_simple import get_simple_predictor
    try:
        get_simple_predictor()
        _logger.info("startup: XGBoost model loaded")
    except Exception as e:
        _logger.warning("startup: XGBoost model failed to load: %s", e)

    # 2. Player surface stats CSV (~50-150 ms)
    from app.routers.tennis_predictions_today_enhanced import _load_player_surface_stats
    try:
        _load_player_surface_stats()
        _logger.info("startup: player surface stats loaded")
    except Exception as e:
        _logger.warning("startup: player surface stats failed: %s", e)

    _logger.info("startup: warmup done in %.0f ms", (_time.perf_counter() - t0) * 1000)


@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    await db.execute(text("SELECT 1"))
    return {"ok": True}

app.include_router(nba.router, prefix="/nba", tags=["NBA"])
app.include_router(nfl.router, prefix="/nfl", tags=["NFL"])
app.include_router(nhl.router, prefix="/nhl", tags=["NHL"])
app.include_router(cfb.router, prefix="/cfb", tags=["CFB"])
app.include_router(cbb.router, prefix="/cbb", tags=["CBB"])
# app.include_router(tennis.router, prefix="/tennis", tags=["Tennis"])
app.include_router(tennis_props.router, prefix="/tennis", tags=["Tennis Props"])
app.include_router(tennis_auto.router, prefix="/tennis", tags=["Tennis Auto"])
app.include_router(tennis_predictions_today_enhanced.router, prefix="/tennis")
app.include_router(api_tennis_router)
app.include_router(tennis_admin_elo_router)
app.include_router(tennis_value_bets.router, prefix="/tennis")
