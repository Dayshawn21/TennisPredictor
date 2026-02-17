import os
from dotenv import load_dotenv
load_dotenv()
import json
import datetime as dt
from typing import Dict, Any, List
import requests

from app.ingest.tennis.sofascore_matchups import get_today_tomorrow_atp_wta
from app.ingest.tennis.db_upcoming_matches import upsert_upcoming_matches
from app.ingest.tennis.local_predict import predict_match_local

def trigger_predictions(enriched_matches):
    results = []

    for m in enriched_matches:
        try:
            pred = predict_match_local(
                m["p1_name"],
                m["p2_name"],
                m["tour"]
            )

            # skip non-TA players / bad matches
            if not pred.get("ok"):
                continue

            results.append({
                "match_key": m["match_key"],
                "prediction": pred
            })

        except Exception as e:
            results.append({
                "match_key": m["match_key"],
                "error": str(e)
            })

    return results 

def main():
    # Uses your local date (Central or whichever your server uses)
    today = dt.date.today()
    tomorrow = today + dt.timedelta(days=1)

    data = get_today_tomorrow_atp_wta(headless=True)

    # Upsert to DB
    today_rows = upsert_upcoming_matches(data["today"], match_date=today)
    tomorrow_rows = upsert_upcoming_matches(data["tomorrow"], match_date=tomorrow)

    all_rows = today_rows + tomorrow_rows

    print(f"DB upserted: today={len(today_rows)} tomorrow={len(tomorrow_rows)} total={len(all_rows)}")

    # Trigger predictions (local)
    preds = trigger_predictions(all_rows)
    print(f"Predictions generated: {len(preds)}")

    # Print a few
    for x in preds[:10]:
        print(json.dumps(x, indent=2))

if __name__ == "__main__":
    main()
