ALTER TABLE IF EXISTS tennis_prediction_logs
  ADD COLUMN IF NOT EXISTS p1_market_no_vig_prob NUMERIC,
  ADD COLUMN IF NOT EXISTS edge_no_vig NUMERIC,
  ADD COLUMN IF NOT EXISTS edge_blended NUMERIC,
  ADD COLUMN IF NOT EXISTS bet_eligible BOOLEAN,
  ADD COLUMN IF NOT EXISTS bet_side TEXT,
  ADD COLUMN IF NOT EXISTS kelly_fraction_capped NUMERIC,
  ADD COLUMN IF NOT EXISTS odds_age_minutes NUMERIC,
  ADD COLUMN IF NOT EXISTS market_overround_main NUMERIC,
  ADD COLUMN IF NOT EXISTS predictor_quality_flags JSONB,
  ADD COLUMN IF NOT EXISTS effective_weights JSONB,
  ADD COLUMN IF NOT EXISTS p1_market_odds_american INT,
  ADD COLUMN IF NOT EXISTS p2_market_odds_american INT;

CREATE INDEX IF NOT EXISTS idx_pred_logs_bet_eligible
  ON tennis_prediction_logs (bet_eligible);
