CREATE TABLE IF NOT EXISTS predictions_history (
  match_id        bigint PRIMARY KEY,
  match_date      date NOT NULL,
  tour            text NOT NULL,
  surface         text,
  tournament_name text,
  round           text,

  p1_id           bigint NOT NULL,
  p2_id           bigint NOT NULL,

  p1_elo_overall  real,
  p2_elo_overall  real,
  p1_elo_surface  real,
  p2_elo_surface  real,
  d_elo_overall   real,
  d_elo_surface   real,

  p1_prob         real,
  p1_fair_american integer,

  winner_id       bigint,
  p1_win          integer,

  generated_at    timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);
