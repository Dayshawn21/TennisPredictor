from sqlalchemy import Column, Integer, BigInteger, String, Float, DateTime, Date, ForeignKey
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class NFLGame(Base):
    __tablename__ = "nfl_games"
    game_id = Column(BigInteger, primary_key=True)
    date_utc = Column(DateTime)
    home_team_id = Column(BigInteger)
    away_team_id = Column(BigInteger)
    home_score = Column(Integer)
    away_score = Column(Integer)
    week = Column(Integer)
    season = Column(Integer)

class NFLPlayerStats(Base):
    __tablename__ = "nfl_player_stats"
    game_id = Column(BigInteger, primary_key=True)
    player_id = Column(BigInteger, primary_key=True)
    team_id = Column(BigInteger)
    opp_team_id = Column(BigInteger)
    position = Column(String)
    pass_yds = Column(Float)
    pass_td = Column(Float)
    rush_yds = Column(Float)
    rush_td = Column(Float)
    rec_yds = Column(Float)
    rec_td = Column(Float)
    receptions = Column(Float)
    targets = Column(Float)
    attempts_pass = Column(Float)
    attempts_rush = Column(Float)
    snaps_pct = Column(Float)

class NFLTeamCtx(Base):
    __tablename__ = "nfl_team_ctx"
    team_id = Column(BigInteger, primary_key=True)
    as_of = Column(Date)
    pass_yds_allowed_pg = Column(Float)
    rush_yds_allowed_pg = Column(Float)
    plays_pg = Column(Float)

class NFLOddsProp(Base):
    __tablename__ = "nfl_odds_props"
    prop_id = Column(String, primary_key=True)
    game_id = Column(BigInteger, ForeignKey("nfl_games.game_id"))
    player_id = Column(BigInteger)
    market = Column(String)     # e.g., PLAYER_PASS_YDS
    line = Column(Float)
    american_odds = Column(Integer)
    book = Column(String)
    ts = Column(DateTime)
  