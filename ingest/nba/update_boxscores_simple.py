import pandas as pd
import psycopg2
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2, leaguedashteamstats
import os
import time

# ----------------------------------
# Neon connection
# ----------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_3bxYFijyoeD4@ep-dawn-wave-a8928yr9-pooler.eastus2.azure.neon.tech/neondb?sslmode=require"
)

# ----------------------------------
# 1️⃣ Get specific completed game from last season
# ----------------------------------
print("Fetching recent NBA games from 2023-24 season...")

# Let's get games from April 14, 2024 (last day of 2023-24 regular season)
gamefinder = leaguegamefinder.LeagueGameFinder(
    season_nullable="2023-24",
    season_type_nullable="Regular Season",
    date_from_nullable="04/14/2024",
    date_to_nullable="04/14/2024"
)
games_df = gamefinder.get_data_frames()[0]

print(f"Found {len(games_df)} team-game records")

if games_df.empty:
    print("No games found for that date.")
    exit()

# Get unique game IDs
game_ids = games_df['GAME_ID'].unique().tolist()[:5]  # Process first 5 games
print(f"Processing {len(game_ids)} games...")

# ----------------------------------
# 2️⃣ Connect to Neon
# ----------------------------------
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# ----------------------------------
# 3️⃣ Process each game
# ----------------------------------
for game_id in game_ids:
    game_id_str = str(game_id)
    print(f"\n--- Processing game {game_id_str} ---")
    
    # Get both teams for this game
    game_data = games_df[games_df['GAME_ID'] == game_id]
    
    # Parse MATCHUP to determine home/away
    team1 = game_data.iloc[0]
    team2 = game_data.iloc[1] if len(game_data) > 1 else None
    
    # Determine home and away
    if '@' in team1['MATCHUP']:
        away_team_id = int(team1['TEAM_ID'])
        home_team_id = int(team2['TEAM_ID']) if team2 is not None else away_team_id
    else:
        home_team_id = int(team1['TEAM_ID'])
        away_team_id = int(team2['TEAM_ID']) if team2 is not None else home_team_id
    
    game_date = pd.to_datetime(team1['GAME_DATE']).date()
    
    # Insert game
    cursor.execute(
        """
        INSERT INTO nba_games (game_id, date_utc, home_team_id, away_team_id, vegas_total, vegas_spread)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (game_id) DO UPDATE
          SET date_utc = EXCLUDED.date_utc,
              home_team_id = EXCLUDED.home_team_id,
              away_team_id = EXCLUDED.away_team_id;
        """,
        (int(game_id_str[1:]), game_date, home_team_id, away_team_id, None, None)
    )
    print(f"✓ Inserted game record")
    
    # Get boxscore
    try:
        time.sleep(0.6)  # Rate limiting
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id_str)
        player_stats = box.player_stats.get_data_frame()
        
        if player_stats.empty:
            print(f"⚠️  No player stats available")
            continue
        
        # Insert player stats
        player_count = 0
        for _, player in player_stats.iterrows():
            try:
                player_id = int(player['PLAYER_ID'])
                team_id = int(player['TEAM_ID'])
                
                # Get opponent team ID
                opp_team_id = home_team_id if team_id == away_team_id else away_team_id
                
                # Parse minutes
                minutes_str = str(player['MIN']) if pd.notna(player['MIN']) else '0'
                if minutes_str and minutes_str != 'None' and ':' in minutes_str:
                    mins, secs = minutes_str.split(':')
                    minutes = float(mins) + float(secs) / 60
                else:
                    minutes = 0.0
                
                # Only insert players who actually played
                if minutes > 0:
                    cursor.execute(
                        """
                        INSERT INTO nba_player_box
                          (game_id, player_id, team_id, opp_team_id, minutes, pts, reb, ast, fga, fta, three_pm, starter, usage_rate)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (game_id, player_id) DO UPDATE
                          SET pts = EXCLUDED.pts,
                              reb = EXCLUDED.reb,
                              ast = EXCLUDED.ast,
                              minutes = EXCLUDED.minutes,
                              fga = EXCLUDED.fga,
                              fta = EXCLUDED.fta,
                              three_pm = EXCLUDED.three_pm;
                        """,
                        (
                            int(game_id_str[1:]),
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
                            None  # usage_rate
                        )
                    )
                    player_count += 1
            except Exception as e:
                print(f"⚠️  Error inserting player {player.get('PLAYER_NAME', 'unknown')}: {e}")
                continue
        
        print(f"✓ Inserted {player_count} players")
        
    except Exception as e:
        print(f"⚠️  Could not fetch boxscore: {e}")
        continue

# ----------------------------------
# 4️⃣ Update team context
# ----------------------------------
print("\nUpdating team context...")
try:
    stats = leaguedashteamstats.LeagueDashTeamStats(season="2023-24", last_n_games=10)
    df = stats.get_data_frames()[0]
    
    for _, row in df.iterrows():
        team_id = int(row["TEAM_ID"])
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
            (team_id, datetime.now().date(), 0.0, 0.0, 0.0)
        )
    print(f"✓ Updated team context for {len(df)} teams")
except Exception as e:
    print(f"⚠️  Could not update team context: {e}")

conn.commit()
cursor.close()
conn.close()

print("\n✅ Data successfully ingested into Neon!")
