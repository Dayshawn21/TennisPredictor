from sqlalchemy import Column, Integer, String, Float, DateTime, Date, ForeignKey
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class NBAGame(Base):
    __tablename__ = "nba_games"
    game_id = Column(Integer, primary_key=True)
    date_utc = Column(DateTime)
    home_team_id = Column(Integer)
    away_team_id = Column(Integer)
    vegas_total = Column(Float)
    vegas_spread = Column(Float)

class NBAOddsProp(Base):
    __tablename__ = "nba_odds_props"
    prop_id = Column(String, primary_key=True)
    player_id = Column(Integer)
    game_id = Column(Integer, ForeignKey("nba_games.game_id"))
    market = Column(String)          # e.g., 'PLAYER_POINTS'
    line = Column(Float)
    american_odds = Column(Integer)
    book = Column(String)
    ts = Column(DateTime)

class NBAPlayerBox(Base):
    __tablename__ = "nba_player_box"
    # PK in DB is (game_id, player_id); SQLAlchemy only needs columns here
    game_id = Column(Integer, primary_key=True)
    player_id = Column(Integer, primary_key=True)
    team_id = Column(Integer)
    opp_team_id = Column(Integer)
    minutes = Column(Float)
    pts = Column(Float)
    reb = Column(Float)
    ast = Column(Float)
    fga = Column(Float)
    fta = Column(Float)
    three_pm = Column(Float)
    starter = Column(String)  # or Boolean; keep String if your DB stored TRUE/FALSE text
    usage_rate = Column(Float)

class NBAOppVsPos(Base):
    __tablename__ = "nba_opp_vs_pos"
    team_id = Column(Integer, primary_key=True)
    as_of = Column(Date, primary_key=True)
    position = Column(String, primary_key=True)
    allowed_pts_l10 = Column(Float)
    allowed_reb_l10 = Column(Float)
    allowed_ast_l10 = Column(Float)
    allowed_3pm_l10 = Column(Float)

class NBATeamCtx(Base):
    __tablename__ = "nba_team_ctx"
    team_id = Column(Integer, primary_key=True)
    as_of = Column(Date)
    off_rating_l10 = Column(Float)
    def_rating_l10 = Column(Float)
    pace_l10 = Column(Float)
