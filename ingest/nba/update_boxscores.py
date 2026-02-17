import pandas as pd
import psycopg2
from datetime import datetime, timedelta, timezone
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2, leaguedashteamstats
import os
import time

# ----------------------------------
# Neon connection (update this line)
# ----------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_3bxYFijyoeD4@ep-dawn-wave-a8928yr9-pooler.eastus2.azure.neon.tech/neondb?sslmode=require"
)

# ----------------------------------
# 1️⃣ Get recent completed games from last season
# ----------------------------------
print("Fetching most recent completed NBA games...")

# Get games from the 2023-24 season (last completed season)
gamefinder = leaguegamefinder.LeagueGameFinder(
    season_nullable="2023-24",
    season_type_nullable="Regular Season",
    league_id_nullable="00"
)
games_df = gamefinder.get_data_frames()[0]

# Get the most recent 10 games
games_df = games_df.sort_values('GAME_DATE', ascending=False).head(10)
print(f"Found {len(games_df)} recent games")

if games_df.empty:
    print("No games found.")
    exit()

# Convert to list of unique game IDs (each game appears twice - once per team)
game_ids = games_df['GAME_ID'].unique().tolist()
print(f"Processing {len(game_ids)} unique games...")

# ----------------------------------
# 2️⃣ Connect to Neon
# ----------------------------------
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# ----------------------------------
# 3️⃣ Upsert game info
# ----------------------------------
for game_id in game_ids:
    game_id_int = int(game_id)
    game_rows = games_df[games_df['GAME_ID'] == game_id]
    
    if len(game_rows) < 2:
        print(f"⚠️  Skipping game {game_id_int} - insufficient data")
        continue
    
    # Get home and away teams (MATCHUP format: "TOR @ BOS" or "BOS vs. TOR")
    home_rows = game_rows[game_rows['MATCHUP'].str.contains('vs.', na=False)]
    away_rows = game_rows[game_rows['MATCHUP'].str.contains('@', na=False)]
    
    if home_rows.empty or away_rows.empty:
        # Fallback: just use first two rows
        home_row = game_rows.iloc[0]
        away_row = game_rows.iloc[1]
    else:
        home_row = home_rows.iloc[0]
        away_row = away_rows.iloc[0]
    
    home_team_id = int(home_row['TEAM_ID'])
    away_team_id = int(away_row['TEAM_ID'])
    game_date = pd.to_datetime(home_row['GAME_DATE']).date()
    
    vegas_total = None
    vegas_spread = None

    cursor.execute(
        """
        INSERT INTO nba_games (game_id, date_utc, home_team_id, away_team_id, vegas_total, vegas_spread)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (game_id) DO UPDATE
          SET date_utc = EXCLUDED.date_utc,
              home_team_id = EXCLUDED.home_team_id,
              away_team_id = EXCLUDED.away_team_id;
        """,
        (game_id_int, game_date, home_team_id, away_team_id, vegas_total, vegas_spread)
    )
    print(f"✓ Inserted game {game_id_int}: {away_row['TEAM_ABBREVIATION']} @ {home_row['TEAM_ABBREVIATION']}")

# ----------------------------------
# 4️⃣ Upsert player box scores
# ----------------------------------
for game_id in game_ids:
    game_id_int = int(game_id)
    print(f"Fetching boxscore for {game_id_int}...")
    
    try:
        # Use traditional boxscore endpoint (works better for historical data)
        time.sleep(0.6)  # Rate limiting - NBA API allows ~2 requests per second
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        player_stats = box.player_stats.get_data_frame()
        
        if player_stats.empty:
            print(f"⚠️  No player stats found for game {game_id_int}")
            continue
        
        for _, player in player_stats.iterrows():
            player_id = int(player['PLAYER_ID'])
            team_id = int(player['TEAM_ID'])
            
            # Get opponent team ID from the game
            game_row = games_df[games_df['GAME_ID'] == game_id].iloc[0]
            opp_team_id = int(games_df[(games_df['GAME_ID'] == game_id) & (games_df['TEAM_ID'] != team_id)]['TEAM_ID'].iloc[0])
            
            # Parse minutes (format: "MM:SS")
            minutes_str = str(player.get('MIN', '0'))
            if ':' in minutes_str:
                mins, secs = minutes_str.split(':')
                minutes = float(mins) + float(secs) / 60
            else:
                minutes = float(minutes_str) if minutes_str else 0.0

            cursor.execute(
                """
                INSERT INTO nba_player_box
                  (game_id, player_id, team_id, opp_team_id, minutes, pts, reb, ast, fga, fta, three_pm, starter, usage_rate)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (game_id, player_id) DO UPDATE
                  SET pts = EXCLUDED.pts,
                      reb = EXCLUDED.reb,
                      ast = EXCLUDED.ast,
                      minutes = EXCLUDED.minutes;
                """,
                (
                    game_id_int,
                    player_id,
                    team_id,
                    opp_team_id,
                    minutes,
                    float(player.get('PTS', 0) or 0),
                    float(player.get('REB', 0) or 0),
                    float(player.get('AST', 0) or 0),
                    float(player.get('FGA', 0) or 0),
                    float(player.get('FTA', 0) or 0),
                    float(player.get('FG3M', 0) or 0),
                    player.get('START_POSITION', '') != '',
                    None  # usage_rate placeholder
                )
            )
        
        print(f"✓ Inserted {len(player_stats)} players for game {game_id_int}")
        
    except Exception as e:
        print(f"⚠️  Could not fetch boxscore for game {game_id_int}: {e}")
        continue

# ----------------------------------
# 5️⃣ Update team context (pace, rating)
# ----------------------------------
print("Updating team context...")
try:
    stats = leaguedashteamstats.LeagueDashTeamStats(season="2024-25", last_n_games=10)
    df = stats.get_data_frames()[0]
    
    # Debug: Print available columns
    print(f"Available columns: {df.columns.tolist()}")
    
    for _, row in df.iterrows():
        team_id = int(row["TEAM_ID"])
        # Use available columns - may need to adjust based on what's actually available
        off_rating = float(row.get("OFF_RATING", 0) or row.get("OFFENSIVE_RATING", 0))
        def_rating = float(row.get("DEF_RATING", 0) or row.get("DEFENSIVE_RATING", 0))
        pace = float(row.get("PACE", 0))

        cursor.execute(
            """
            INSERT INTO nba_team_ctx (team_id, as_of, off_rating_l10, def_rating_l10, pace_l10)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (team_id) DO UPDATE
              SET off_rating_l10 = EXCLUDED.off_rating_l10,
                  def_rating_l10 = EXCLUDED.def_rating_l10,
                  pace_l10 = EXCLUDED.pace_l10,
                  as_of = EXCLUDED.as_of;
            """,
            (team_id, datetime.now(timezone.utc).date(), off_rating, def_rating, pace)
        )
except Exception as e:
    print(f"⚠️  Could not update team context: {e}")

conn.commit()
cursor.close()
conn.close()

print("✅ Data successfully ingested into Neon.")
