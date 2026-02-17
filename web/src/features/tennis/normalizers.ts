import {
    CompareResponse,
    EnhancedPrediction,
    EnhancedPredictionsResponse,
    H2HMatch,
    LookupSource,
    ParlaySuggestion,
    PlayerStatsResponse,
    SurfaceRow
} from './types'
import { getField, toNumOrNull, toStrOrNull } from './formatters'
type InputShape = NonNullable<EnhancedPrediction['inputs']>

const SURFACE_NUM_KEYS = [
    'svc_pts',
    'ret_pts',
    'svc_n',
    'ret_n',
    'svc_aces_pg',
    'svc_dfs_pg',
    'svc_first_pct',
    'svc_first_win',
    'svc_second_win',
    'ret_first_win',
    'ret_second_win',
    'ret_opp_aces_pg',
    'svc_bp_save_pct',
    'ret_bp_win_pct',
    'svc_hold_pct',
    'ret_opp_hold_pct'
] as const

export const normalizeEnhancedPrediction = (raw: Record<string, unknown>): EnhancedPrediction => {
    const inputs = (raw.inputs as Record<string, unknown>) || {}
    const totals = (inputs.totals_sets_sim ?? inputs.totalsSetsSim) as InputShape['totalsSetsSim']
    const projectedProps = (inputs.projected_props ?? inputs.projectedProps ?? inputs.projected_props_v2) as InputShape['projectedProps']

    return {
        ...raw,
        matchId: (raw.match_id ?? raw.matchId) as string | undefined,
        matchKey: (raw.match_key ?? raw.matchKey) as string | undefined,
        matchDate: (raw.match_date ?? raw.matchDate) as string | undefined,
        matchStartUtc: (raw.match_start_utc ?? raw.matchStartUtc) as string | undefined,
        bestOf: toNumOrNull(raw.best_of ?? raw.bestOf) ?? undefined,
        p1Name: (raw.p1_name ?? raw.p1Name) as string | undefined,
        p2Name: (raw.p2_name ?? raw.p2Name) as string | undefined,
        p1WinProb: toNumOrNull(raw.p1_win_prob ?? raw.p1WinProb ?? raw.p1_blended_win_prob ?? raw.p1BlendedWinProb),
        p2WinProb: toNumOrNull(raw.p2_win_prob ?? raw.p2WinProb ?? raw.p2_blended_win_prob ?? raw.p2BlendedWinProb),
        p1FairAmerican: toNumOrNull(raw.p1_fair_american ?? raw.p1FairAmerican),
        p2FairAmerican: toNumOrNull(raw.p2_fair_american ?? raw.p2FairAmerican),
        predictedWinner: (raw.predicted_winner_blended ?? raw.predicted_winner ?? raw.predictedWinner) as string | null | undefined,
        inputs: {
            ...(inputs as InputShape),
            method: inputs.method as string | undefined,
            numPredictors: toNumOrNull(inputs.num_predictors ?? inputs.numPredictors) ?? undefined,
            individualPredictions: (inputs.individual_predictions ?? inputs.individualPredictions) as InputShape['individualPredictions'],
            hasElo: Boolean(inputs.has_elo ?? inputs.hasElo),
            hasRolling: Boolean(inputs.has_rolling ?? inputs.hasRolling),
            hasOdds: Boolean(inputs.has_odds ?? inputs.hasOdds),
            eloUsedMedian: Boolean(inputs.elo_used_median ?? inputs.eloUsedMedian),
            p1EloUsedMedian: Boolean(inputs.p1_elo_used_median ?? inputs.p1EloUsedMedian),
            p2EloUsedMedian: Boolean(inputs.p2_elo_used_median ?? inputs.p2EloUsedMedian),
            totalsSetsSim: totals,
            projectedProps,
            style_summary: (inputs.style_summary ?? inputs.styleSummary) as InputShape['style_summary'],
            pick_summary: (inputs.pick_summary ?? inputs.pickSummary) as string | null | undefined
        }
    }
}

export const normalizeEnhancedPredictionsResponse = (raw: Record<string, unknown>): EnhancedPredictionsResponse => ({
    as_of: String(raw.as_of ?? raw.asOf ?? ''),
    source: String(raw.source ?? ''),
    cached: !!raw.cached,
    count: Number(raw.count ?? (Array.isArray(raw.items) ? raw.items.length : 0)),
    items: Array.isArray(raw.items) ? raw.items.map((item) => normalizeEnhancedPrediction((item || {}) as Record<string, unknown>)) : []
})

export const normalizeCompareResponse = (raw: Record<string, unknown>): CompareResponse => ({
    meta: (getField(raw, 'meta') || {}) as Record<string, unknown>,
    players: {
        left: (getField(getField(raw, 'players') || {}, 'left') || {}) as Record<string, unknown>,
        right: (getField(getField(raw, 'players') || {}, 'right') || {}) as Record<string, unknown>
    },
    compare: (getField(raw, 'compare') || {}) as Record<string, unknown>,
    h2h: (getField(raw, 'h2h') || {}) as Record<string, unknown>,
    last_10_matches: (getField(raw, 'last_10_matches', 'last10Matches') || {}) as Record<string, unknown>,
    last_10_resolution: (getField(raw, 'last_10_resolution', 'last10Resolution') || {}) as Record<string, unknown>,
    last_10_matchups: ((getField(raw, 'last_10_matchups', 'last10Matchups') || []) as H2HMatch[]) ?? [],
    quality: (getField(raw, 'quality') || {}) as Record<string, unknown>,
    debug: getField(raw, 'debug') || null
})

const normalizeSurfaceRow = (raw: unknown): SurfaceRow | null => {
    if (!raw || typeof raw !== 'object') return null
    const out: Partial<SurfaceRow> = {}
    for (const key of SURFACE_NUM_KEYS) out[key] = toNumOrNull(getField(raw, key))
    return out as SurfaceRow
}

export const normalizePlayerStatsResponse = (raw: Record<string, unknown>): PlayerStatsResponse => {
    const sources: LookupSource[] = ['csv_rows', 'rolling', 'elo_latest']
    const surfaceRowsRaw = (getField(raw, 'surface_rows', 'surfaceRows') || {}) as Record<string, unknown>
    const matchStatusRaw = (getField(raw, 'match_status', 'matchStatus') || {}) as Record<string, unknown>
    const matchDetailRaw = (getField(raw, 'match_status_detail', 'matchStatusDetail') || {}) as Record<string, unknown>
    const resolutionRaw = (getField(raw, 'resolution') || {}) as Record<string, unknown>
    const normalizedRaw = (getField(raw, 'normalized') || {}) as Record<string, unknown>
    const idsRaw = (getField(raw, 'ids') || {}) as Record<string, unknown>
    const rollingRaw = (getField(raw, 'rolling_stats', 'rollingStats') || null) as Record<string, unknown> | null
    const eloRaw = (getField(raw, 'elo_latest', 'eloLatest') || null) as Record<string, unknown> | null
    const currentRatesRaw = (getField(raw, 'current_surface_rates', 'currentSurfaceRates') || {}) as Record<string, unknown>
    const styleRaw = (getField(raw, 'style') || {}) as Record<string, unknown>
    const styleMetricsRaw = (getField(styleRaw, 'metrics') || {}) as Record<string, unknown>

    const matchStatus: Record<LookupSource, boolean> = {
        csv_rows: !!getField(matchStatusRaw, 'csv_rows', 'csvRows'),
        rolling: !!getField(matchStatusRaw, 'rolling'),
        elo_latest: !!getField(matchStatusRaw, 'elo_latest', 'eloLatest')
    }
    const matchStatusDetail: Record<LookupSource, string> = {
        csv_rows: String(getField(matchDetailRaw, 'csv_rows', 'csvRows') ?? 'not_found'),
        rolling: String(getField(matchDetailRaw, 'rolling') ?? 'not_found'),
        elo_latest: String(getField(matchDetailRaw, 'elo_latest', 'eloLatest') ?? 'not_found')
    }
    const resolution: Record<LookupSource, string> = {
        csv_rows: String(getField(resolutionRaw, 'csv_rows', 'csvRows') ?? 'not_found'),
        rolling: String(getField(resolutionRaw, 'rolling') ?? 'not_found'),
        elo_latest: String(getField(resolutionRaw, 'elo_latest', 'eloLatest') ?? 'not_found')
    }

    for (const source of sources) {
        if (matchStatus[source] && (matchStatusDetail[source] === 'not_found' || !matchStatusDetail[source])) {
            matchStatusDetail[source] = 'ok'
        }
    }

    return {
        player: String(getField(raw, 'player') ?? ''),
        tour: String(getField(raw, 'tour') ?? ''),
        surface_requested: String(getField(raw, 'surface_requested', 'surfaceRequested') ?? ''),
        normalized: {
            base: toStrOrNull(getField(normalizedRaw, 'base')),
            stats_key: toStrOrNull(getField(normalizedRaw, 'stats_key', 'statsKey')),
            stats_name_variants: Array.isArray(getField(normalizedRaw, 'stats_name_variants', 'statsNameVariants'))
                ? (getField(normalizedRaw, 'stats_name_variants', 'statsNameVariants') as unknown[]).map((x) => String(x))
                : [],
            db_alias_variants: Array.isArray(getField(normalizedRaw, 'db_alias_variants', 'dbAliasVariants'))
                ? (getField(normalizedRaw, 'db_alias_variants', 'dbAliasVariants') as unknown[]).map((x) => String(x))
                : []
        },
        ids: {
            ta_id: toNumOrNull(getField(idsRaw, 'ta_id', 'taId'))
        },
        match_status: matchStatus,
        match_status_detail: matchStatusDetail,
        resolution,
        quality_flags: Array.isArray(getField(raw, 'quality_flags', 'qualityFlags'))
            ? (getField(raw, 'quality_flags', 'qualityFlags') as unknown[]).map((x) => String(x))
            : [],
        rolling_stats: rollingRaw
            ? {
                  win_rate_last_20: toNumOrNull(getField(rollingRaw, 'win_rate_last_20', 'winRateLast20')),
                  matches_played: toNumOrNull(getField(rollingRaw, 'matches_played', 'matchesPlayed')),
                  hard_win_rate_last_10: toNumOrNull(getField(rollingRaw, 'hard_win_rate_last_10', 'hardWinRateLast10')),
                  clay_win_rate_last_10: toNumOrNull(getField(rollingRaw, 'clay_win_rate_last_10', 'clayWinRateLast10')),
                  grass_win_rate_last_10: toNumOrNull(getField(rollingRaw, 'grass_win_rate_last_10', 'grassWinRateLast10'))
              }
            : null,
        elo_latest: eloRaw
            ? {
                  player_id: toNumOrNull(getField(eloRaw, 'player_id', 'playerId')),
                  as_of_date: toStrOrNull(getField(eloRaw, 'as_of_date', 'asOfDate')),
                  elo: toNumOrNull(getField(eloRaw, 'elo')),
                  helo: toNumOrNull(getField(eloRaw, 'helo')),
                  celo: toNumOrNull(getField(eloRaw, 'celo')),
                  gelo: toNumOrNull(getField(eloRaw, 'gelo')),
                  official_rank: toNumOrNull(getField(eloRaw, 'official_rank', 'officialRank')),
                  age: toNumOrNull(getField(eloRaw, 'age')),
                  player_name: toStrOrNull(getField(eloRaw, 'player_name', 'playerName'))
              }
            : null,
        current_surface_rates: {
            service_points_won: toNumOrNull(getField(currentRatesRaw, 'service_points_won', 'servicePointsWon')),
            return_points_won: toNumOrNull(getField(currentRatesRaw, 'return_points_won', 'returnPointsWon'))
        },
        style: {
            label: toStrOrNull(getField(styleRaw, 'label')),
            metrics:
                styleMetricsRaw && typeof styleMetricsRaw === 'object'
                    ? Object.fromEntries(Object.entries(styleMetricsRaw).map(([key, value]) => [key, toNumOrNull(value)]))
                    : {}
        },
        surface_rows: {
            hard: {
                row_12m: normalizeSurfaceRow(getField(getField(surfaceRowsRaw, 'hard') || {}, 'row_12m', 'row12m')),
                row_all_time: normalizeSurfaceRow(getField(getField(surfaceRowsRaw, 'hard') || {}, 'row_all_time', 'rowAllTime'))
            },
            clay: {
                row_12m: normalizeSurfaceRow(getField(getField(surfaceRowsRaw, 'clay') || {}, 'row_12m', 'row12m')),
                row_all_time: normalizeSurfaceRow(getField(getField(surfaceRowsRaw, 'clay') || {}, 'row_all_time', 'rowAllTime'))
            },
            grass: {
                row_12m: normalizeSurfaceRow(getField(getField(surfaceRowsRaw, 'grass') || {}, 'row_12m', 'row12m')),
                row_all_time: normalizeSurfaceRow(getField(getField(surfaceRowsRaw, 'grass') || {}, 'row_all_time', 'rowAllTime'))
            }
        },
        stats_path: toStrOrNull(getField(raw, 'stats_path', 'statsPath')),
        debug: getField(raw, 'debug') ?? null
    }
}

export const normalizeParlaySuggestions = (raw: unknown): ParlaySuggestion[] => {
    if (!Array.isArray(raw)) return []
    return raw.map((item) => {
        const parlay = (item || {}) as Record<string, unknown>
        const legsRaw = (parlay.legs ?? parlay.leg_list ?? []) as Record<string, unknown>[]
        return {
            legs: legsRaw.map((leg) => ({
                match_id: String(leg.match_id ?? leg.matchId ?? ''),
                pick: String(leg.pick ?? leg.player ?? ''),
                odds_american: toNumOrNull(leg.odds_american ?? leg.oddsAmerican),
                model_prob: toNumOrNull(leg.model_prob ?? leg.modelProb),
                no_vig_prob: toNumOrNull(leg.no_vig_prob ?? leg.noVigProb),
                edge: toNumOrNull(leg.edge),
                summary: toStrOrNull(leg.summary)
            })),
            parlay_decimal: toNumOrNull(parlay.parlay_decimal ?? parlay.parlayDecimal),
            parlay_american: toNumOrNull(parlay.parlay_american ?? parlay.parlayAmerican),
            win_prob: toNumOrNull(parlay.win_prob ?? parlay.winProb),
            ev: toNumOrNull(parlay.ev)
        }
    })
}
