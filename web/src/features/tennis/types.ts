export type UiState = 'idle' | 'loading' | 'loaded' | 'error'
export type LookupSource = 'csv_rows' | 'rolling' | 'elo_latest'

export interface H2HMatch {
    date?: string
    tournament?: string
    round?: string
    surface?: string
    home?: string
    away?: string
    score?: string
    winnerCode?: number
    winner_code?: number
}

export interface TotalsSetsSim {
    expected_total_games?: number
    expectedTotalGames?: number
    expected_sets?: number
    expectedSets?: number
    expected_set_games?: number
    expectedSetGames?: number
    p_set_win_p1?: number
    pSetWinP1?: number
    p_two_sets?: number
    pTwoSets?: number
    p_three_sets?: number
    pThreeSets?: number
    p_four_sets?: number
    pFourSets?: number
    p_five_sets?: number
    pFiveSets?: number
    p_straight_sets?: number
    pStraightSets?: number
    set_score_probs?: Record<string, number>
    setScoreProbs?: Record<string, number>
}

export interface ProjectedProps {
    aces?: {
        p1_expected?: number
        p2_expected?: number
        p1Expected?: number
        p2Expected?: number
    }
    break_points?: {
        p1_breaks?: number
        p2_breaks?: number
        p1Breaks?: number
        p2Breaks?: number
    }
    breakPoints?: {
        p1_breaks?: number
        p2_breaks?: number
        p1Breaks?: number
        p2Breaks?: number
    }
}

export interface EnhancedPredictionInputs {
    method?: string
    numPredictors?: number
    num_predictors?: number
    individualPredictions?: {
        elo?: number | null
        xgbSimple?: number | null
        market?: number | null
    }
    individual_predictions?:
        | { name: string; p1_prob: number | null; weight?: number | null }[]
        | {
              elo?: number | null
              xgb_simple?: number | null
              market?: number | null
          }
    hasElo?: boolean
    has_elo?: boolean
    hasRolling?: boolean
    has_rolling?: boolean
    hasOdds?: boolean
    has_odds?: boolean
    eloUsedMedian?: boolean
    elo_used_median?: boolean
    p1EloUsedMedian?: boolean
    p1_elo_used_median?: boolean
    p2EloUsedMedian?: boolean
    p2_elo_used_median?: boolean
    totalsSetsSim?: TotalsSetsSim
    totals_sets_sim?: TotalsSetsSim
    projectedProps?: ProjectedProps
    projected_props?: ProjectedProps
    projected_props_v2?: ProjectedProps
    style_summary?: {
        p1?: { name?: string; label?: string | null }
        p2?: { name?: string; label?: string | null }
    }
    pick_summary?: string | null
    tweet_text?: string | null
    [key: string]: unknown
}

export interface EnhancedPrediction {
    matchId?: string
    match_id?: string
    matchKey?: string
    match_key?: string
    matchDate?: string
    match_date?: string
    tour?: string
    tournament?: string
    matchStartUtc?: string
    match_start_utc?: string
    round?: string
    surface?: string
    bestOf?: number
    best_of?: number
    p1Name?: string
    p1_name?: string
    p2Name?: string
    p2_name?: string
    p1_player_id?: number
    p2_player_id?: number
    p1_ta_id?: number
    p2_ta_id?: number
    p1WinProb?: number | null
    p1_win_prob?: number | null
    p2WinProb?: number | null
    p2_win_prob?: number | null
    p1FairAmerican?: number | null
    p1_fair_american?: number | null
    p2FairAmerican?: number | null
    p2_fair_american?: number | null
    p1_market_odds_american?: number | null
    p2_market_odds_american?: number | null
    odds_fetched_at?: string | null
    p1_market_implied_prob?: number | null
    p2_market_implied_prob?: number | null
    p1_market_no_vig_prob?: number | null
    p2_market_no_vig_prob?: number | null
    predictedWinner?: string | null
    predicted_winner?: string | null
    p1_blended_win_prob?: number | null
    p2_blended_win_prob?: number | null
    predicted_winner_blended?: string | null
    missing_reason?: string | null
    h2hP1Wins?: number
    h2h_p1_wins?: number
    h2hP2Wins?: number
    h2h_p2_wins?: number
    h2hTotalMatches?: number
    h2h_total_matches?: number
    h2hSurfaceP1Wins?: number
    h2h_surface_p1_wins?: number
    h2hSurfaceP2Wins?: number
    h2h_surface_p2_wins?: number
    h2hSurfaceMatches?: number
    h2h_surface_matches?: number
    h2hMatches?: H2HMatch[]
    h2h_matches?: H2HMatch[]
    projected_total_games?: number | null
    projected_spread_p1?: number | null
    projected_sets?: string | number | null
    inputs?: EnhancedPredictionInputs
    [key: string]: unknown
}

export interface EnhancedPredictionsResponse {
    as_of: string
    source: string
    cached: boolean
    count: number
    items: EnhancedPrediction[]
}

export interface SurfaceRow {
    svc_pts: number | null
    ret_pts: number | null
    svc_n: number | null
    ret_n: number | null
    svc_aces_pg: number | null
    svc_dfs_pg: number | null
    svc_first_pct: number | null
    svc_first_win: number | null
    svc_second_win: number | null
    ret_first_win: number | null
    ret_second_win: number | null
    ret_opp_aces_pg: number | null
    svc_bp_save_pct: number | null
    ret_bp_win_pct: number | null
    svc_hold_pct: number | null
    ret_opp_hold_pct: number | null
}

export interface PlayerStatsResponse {
    player: string
    tour: string
    surface_requested: string
    normalized: {
        base: string | null
        stats_key: string | null
        stats_name_variants: string[]
        db_alias_variants: string[]
    }
    ids: {
        ta_id: number | null
    }
    match_status: Record<LookupSource, boolean>
    match_status_detail: Record<LookupSource, string>
    resolution: Record<LookupSource, string>
    quality_flags: string[]
    rolling_stats: {
        win_rate_last_20: number | null
        matches_played: number | null
        hard_win_rate_last_10: number | null
        clay_win_rate_last_10: number | null
        grass_win_rate_last_10: number | null
    } | null
    elo_latest: {
        player_id: number | null
        as_of_date: string | null
        elo: number | null
        helo: number | null
        celo: number | null
        gelo: number | null
        official_rank: number | null
        age: number | null
        player_name: string | null
    } | null
    current_surface_rates: {
        service_points_won: number | null
        return_points_won: number | null
    }
    style: {
        label: string | null
        metrics: Record<string, number | null>
    }
    surface_rows: Record<
        'hard' | 'clay' | 'grass',
        {
            row_12m: SurfaceRow | null
            row_all_time: SurfaceRow | null
        }
    >
    stats_path: string | null
    debug: unknown
}

export interface Snapshot {
    query: string
    requested_at_iso: string
    request_id: number
}

export interface CompareResponse {
    meta: Record<string, unknown>
    players: {
        left: Record<string, unknown>
        right: Record<string, unknown>
    }
    compare: Record<string, unknown>
    h2h: Record<string, unknown>
    last_10_matches: Record<string, unknown>
    last_10_resolution: Record<string, unknown>
    last_10_matchups: H2HMatch[]
    quality: Record<string, unknown>
    debug: unknown
}

export interface ParlayLeg {
    match_id: string
    pick: string
    odds_american: number | null
    model_prob: number | null
    no_vig_prob: number | null
    edge: number | null
    summary?: string | null
}

export interface ParlaySuggestion {
    legs: ParlayLeg[]
    parlay_decimal: number | null
    parlay_american: number | null
    win_prob: number | null
    ev: number | null
}
