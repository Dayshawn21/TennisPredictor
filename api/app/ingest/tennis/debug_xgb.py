import asyncio, datetime, traceback
from app.services.tennis.tennis_predictor_simple import get_simple_predictor, _build_features
from app.services.tennis.predictions_sql import SOFASCORE_MATCHES_ELO_SQL
from app.db_session import engine

async def main():
    async with engine.begin() as conn:
        res = await conn.execute(
            SOFASCORE_MATCHES_ELO_SQL,
            {"dates":[datetime.date(2026,1,27)], "include_incomplete": True},
        )
        rows = res.mappings().all()
        for r in rows:
            mk = r.get("match_key")
            if mk in ("sofascore:15345272","sofascore:15345445"):
                print("---- match_key:", mk)
                rdict = dict(r)
                feats = _build_features(rdict)
                pred = get_simple_predictor()
                print("feature_cols:", pred.feature_cols)
                print("len(feature_cols):", len(pred.feature_cols))
                print("built features:", feats)
                row = [float(feats.get(c, 0.0) or 0.0) for c in pred.feature_cols]
                print("len(row):", len(row))
                import numpy as np
                X = np.array(row, dtype=np.float32).reshape(1, -1)
                try:
                    print("predict_proba ->", pred.model.predict_proba(X)[0,1])
                except Exception as e:
                    print("EXCEPTION:", type(e).__name__, str(e))
                    traceback.print_exc()

asyncio.run(main())
