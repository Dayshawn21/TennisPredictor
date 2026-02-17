import os, time, requests
from requests.adapters import HTTPAdapter, Retry
from datetime import datetime, timedelta, timezone
import psycopg2

DB_URL = os.getenv("DATABASE_URL")  # put in .env; rotate your Neon password now

# --- HTTP session with retries & UA ---
session = requests.Session()
retries = Retry(total=4, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))
HEADERS = {"User-Agent": "dayshawn-sportsbot/1.0 (+ingest)"}

def espn(url, params=None):
    r = session.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def parse_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def upsert(cursor, q, args): cursor.execute(q, args)

# --- pick dates: week 5 of 2024 (Sun/Mon/Thu) ---
dates = ["20241006", "20241007", "20241003"]

conn = None
try:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    games_seen = set()

    for d in dates:
        try:
            sb = espn("https://site.api.espn.com/apis/v2/sports/football/nfl/scoreboard",
                      params={"dates": d})
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print(f"No scoreboard for {d} (off day / offseason); skipping.")
                continue
            raise

        for ev in sb.get("events", []):
            game_id = parse_int(ev.get("id"))
            if not game_id: 
                continue
            comp = (ev.get("competitions") or [None])[0]
            if not comp:
                continue

            season = comp.get("season", {}).get("year")
            week   = comp.get("week", {}).get("number")
            date_utc = comp.get("date")

            home = next(t for t in comp["competitors"] if t["homeAway"] == "home")
            away = next(t for t in comp["competitors"] if t["homeAway"] == "away")
            home_id = parse_int(home["team"]["id"])
            away_id = parse_int(away["team"]["id"])
            home_score = parse_int(home.get("score"), 0)
            away_score = parse_int(away.get("score"), 0)

            upsert(cur, """
                INSERT INTO nfl_games (game_id, date_utc, home_team_id, away_team_id, home_score, away_score, week, season)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (game_id) DO UPDATE
                  SET date_utc=EXCLUDED.date_utc,
                      home_team_id=EXCLUDED.home_team_id,
                      away_team_id=EXCLUDED.away_team_id,
                      home_score=EXCLUDED.home_score,
                      away_score=EXCLUDED.away_score,
                      week=EXCLUDED.week,
                      season=EXCLUDED.season
            """, (game_id, date_utc, home_id, away_id, home_score, away_score, week, season))

            games_seen.add(game_id)

            # Boxscore / summary → player stats
            try:
                summ = espn("https://site.api.espn.com/apis/v2/sports/football/nfl/summary",
                            params={"event": game_id})
            except requests.exceptions.HTTPError as e:
                print(f"Summary not available yet for {game_id}: {e}")
                continue

            box = summ.get("boxscore", {})
            players_groups = box.get("players", [])
            team_ids = {home_id: away_id, away_id: home_id}

            for team_grp in players_groups:
                team_id = parse_int(team_grp.get("team", {}).get("id"))
                if not team_id:
                    continue
                opp_team_id = team_ids.get(team_id)
                for stat_block in team_grp.get("statistics", []):
                    for a in stat_block.get("athletes", []):
                        athlete = a.get("athlete", {})
                        player_id = parse_int(athlete.get("id"))
                        position = (athlete.get("position", {}) or {}).get("abbreviation") or None
                        if not player_id:
                            continue

                        stats = {s["name"]: s.get("value") for s in a.get("stats", []) if "name" in s}
                        pass_yds = parse_int(stats.get("passingYards") or stats.get("passYards") or 0, 0)
                        pass_td  = parse_int(stats.get("passingTouchdowns") or 0, 0)
                        rush_yds = parse_int(stats.get("rushingYards") or 0, 0)
                        rush_td  = parse_int(stats.get("rushingTouchdowns") or 0, 0)
                        rec_yds  = parse_int(stats.get("receivingYards") or 0, 0)
                        rec_td   = parse_int(stats.get("receivingTouchdowns") or 0, 0)
                        rec      = parse_int(stats.get("receptions") or 0, 0)
                        targets  = parse_int(stats.get("targets") or 0, 0)
                        att_pass = parse_int(stats.get("passingAttempts") or 0, 0)
                        att_rush = parse_int(stats.get("rushingAttempts") or 0, 0)

                        upsert(cur, """
                            INSERT INTO nfl_player_stats
                              (game_id, player_id, team_id, opp_team_id, position,
                               pass_yds, pass_td, rush_yds, rush_td, rec_yds, rec_td,
                               receptions, targets, attempts_pass, attempts_rush, snaps_pct)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (game_id, player_id) DO UPDATE
                              SET pass_yds=EXCLUDED.pass_yds,
                                  pass_td=EXCLUDED.pass_td,
                                  rush_yds=EXCLUDED.rush_yds,
                                  rush_td=EXCLUDED.rush_td,
                                  rec_yds=EXCLUDED.rec_yds,
                                  rec_td=EXCLUDED.rec_td,
                                  receptions=EXCLUDED.receptions,
                                  targets=EXCLUDED.targets,
                                  attempts_pass=EXCLUDED.attempts_pass,
                                  attempts_rush=EXCLUDED.attempts_rush
                        """, (
                            game_id, player_id, team_id, opp_team_id, position,
                            pass_yds, pass_td, rush_yds, rush_td, rec_yds, rec_td,
                            rec, targets, att_pass, att_rush, None
                        ))

            time.sleep(0.6)  # be polite

    # Basic team context from these games (quick – good starter)
    if games_seen:
        upsert(cur, """
            INSERT INTO nfl_team_ctx (team_id, as_of, pass_yds_allowed_pg, rush_yds_allowed_pg, plays_pg)
            SELECT
              t.team_id,
              CURRENT_DATE,
              COALESCE(AVG(t.pass_yds_allowed), 225),
              COALESCE(AVG(t.rush_yds_allowed), 105),
              NULL
            FROM (
              SELECT g.home_team_id AS team_id,
                     SUM(CASE WHEN s.team_id = g.away_team_id THEN s.pass_yds ELSE 0 END) AS pass_yds_allowed,
                     SUM(CASE WHEN s.team_id = g.away_team_id THEN s.rush_yds ELSE 0 END) AS rush_yds_allowed
              FROM nfl_games g
              JOIN nfl_player_stats s ON s.game_id = g.game_id
              WHERE g.game_id = ANY(%s)
              GROUP BY g.home_team_id

              UNION ALL

              SELECT g.away_team_id AS team_id,
                     SUM(CASE WHEN s.team_id = g.home_team_id THEN s.pass_yds ELSE 0 END),
                     SUM(CASE WHEN s.team_id = g.home_team_id THEN s.rush_yds ELSE 0 END)
              FROM nfl_games g
              JOIN nfl_player_stats s ON s.game_id = g.game_id
              WHERE g.game_id = ANY(%s)
              GROUP BY g.away_team_id
            ) t
            GROUP BY t.team_id
            ON CONFLICT (team_id) DO UPDATE
              SET as_of = EXCLUDED.as_of,
                  pass_yds_allowed_pg = EXCLUDED.pass_yds_allowed_pg,
                  rush_yds_allowed_pg = EXCLUDED.rush_yds_allowed_pg
        """, (list(games_seen), list(games_seen)))

    conn.commit()
    print(f"✅ Ingested NFL games: {len(games_seen)}")

except Exception as e:
    if conn:
        conn.rollback()
    raise
finally:
    if conn:
        conn.close()
