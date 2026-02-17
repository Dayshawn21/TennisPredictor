from __future__ import annotations

ENSURE_API_TENNIS_TABLES = [
    # 1) Base table (idempotent)
    """
    CREATE TABLE IF NOT EXISTS api_tennis_fixtures (
        fixture_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

        -- stable key from API-Tennis
        event_key TEXT NOT NULL UNIQUE,

        -- convenience match_key you can join on elsewhere
        match_key TEXT NOT NULL UNIQUE,

        tour TEXT NOT NULL,                 -- ATP / WTA
        match_date DATE NULL,
        match_time TEXT NULL,
        timezone TEXT NULL,

        tournament_name TEXT NULL,
        tournament_round TEXT NULL,
        surface TEXT NULL,

        player1_name TEXT NOT NULL,
        player2_name TEXT NOT NULL,

        -- ✅ NEW: API-Tennis source player keys (DO NOT confuse with canonical ids)
        player1_api_id BIGINT NULL,
        player2_api_id BIGINT NULL,

        -- ✅ NEW: canonical ids (tennis_players.id)
        player1_id BIGINT NULL,
        player2_id BIGINT NULL,

        status TEXT NULL,                   -- Scheduled / Finished / etc (depends on API response)
        score_raw TEXT NULL,                -- store final score if present
        winner_name TEXT NULL,              -- if present

        raw_payload JSONB NOT NULL,         -- store the entire response row

        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """,

    # 2) Column adds (safe if table already existed without them)
    "ALTER TABLE api_tennis_fixtures ADD COLUMN IF NOT EXISTS player1_api_id BIGINT NULL;",
    "ALTER TABLE api_tennis_fixtures ADD COLUMN IF NOT EXISTS player2_api_id BIGINT NULL;",
    "ALTER TABLE api_tennis_fixtures ADD COLUMN IF NOT EXISTS player1_id BIGINT NULL;",
    "ALTER TABLE api_tennis_fixtures ADD COLUMN IF NOT EXISTS player2_id BIGINT NULL;",

    # 3) Helpful indexes
    "CREATE INDEX IF NOT EXISTS ix_api_tennis_fixtures_date ON api_tennis_fixtures(match_date);",
    "CREATE INDEX IF NOT EXISTS ix_api_tennis_fixtures_tour ON api_tennis_fixtures(tour);",
    "CREATE INDEX IF NOT EXISTS ix_api_tennis_fixtures_tournament ON api_tennis_fixtures(tournament_name);",
    "CREATE INDEX IF NOT EXISTS ix_api_tennis_fixtures_p1_api ON api_tennis_fixtures(player1_api_id);",
    "CREATE INDEX IF NOT EXISTS ix_api_tennis_fixtures_p2_api ON api_tennis_fixtures(player2_api_id);",
    "CREATE INDEX IF NOT EXISTS ix_api_tennis_fixtures_p1_id ON api_tennis_fixtures(player1_id);",
    "CREATE INDEX IF NOT EXISTS ix_api_tennis_fixtures_p2_id ON api_tennis_fixtures(player2_id);",
]
