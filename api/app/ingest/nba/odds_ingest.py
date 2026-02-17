import os, psycopg2
from ingest.common.odds_api import get_odds

DB_URL = os.getenv("DATABASE_URL")

def upsert(cur, q, args): cur.execute(q, args)

def run():
    rows = get_odds("basketball_nba", market="h2h")
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    for g in rows:
        game_id = g["id"]
        home = g["home_team"]; away = g["away_team"]
        for b in g.get("bookmakers", []):
            book = b["title"]
            for m in b.get("markets", []):
                if m["key"] != "h2h": continue
                outs = m.get("outcomes", [])
                home_odds = next((o["price"] for o in outs if o["name"] == home), None)
                away_odds = next((o["price"] for o in outs if o["name"] == away), None)
                if home_odds is None or away_odds is None: continue
                upsert(cur, """
                    INSERT INTO nba_game_odds (game_id, book, moneyline_home, moneyline_away, spread_home, total_points, ts)
                    VALUES (%s,%s,%s,%s,NULL,NULL,NOW())
                    ON CONFLICT (game_id, book, ts) DO NOTHING;
                """, (game_id, book, int(home_odds), int(away_odds)))
    conn.commit(); cur.close(); conn.close()
    print(f"✅ NBA moneylines ingested: {len(rows)} games")

if __name__ == "__main__":
    run()
