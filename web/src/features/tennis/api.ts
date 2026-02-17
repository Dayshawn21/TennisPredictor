import CustomAxios from '../../utility/customAxios'
import { normalizeCompareResponse, normalizeEnhancedPredictionsResponse, normalizeParlaySuggestions, normalizePlayerStatsResponse } from './normalizers'
import { CompareResponse, EnhancedPredictionsResponse, H2HMatch, ParlaySuggestion, PlayerStatsResponse } from './types'

export const fetchEnhancedPredictions = async (
    params?: {
        days_ahead?: number
        include_incomplete?: boolean
        bust_cache?: boolean
        min_edge?: number
        max_odds_age_min?: number
        max_overround?: number
    }
): Promise<EnhancedPredictionsResponse> => {
    const response = await CustomAxios.get('/tennis/predictions/today/enhanced', {
        params: {
            days_ahead: 1,
            include_incomplete: true,
            bust_cache: true,
            min_edge: 0.025,
            max_odds_age_min: 180,
            max_overround: 0.08,
            ...(params || {})
        }
    })
    return normalizeEnhancedPredictionsResponse((response.data || {}) as Record<string, unknown>)
}

export const fetchH2HByEvent = async (eventId: number): Promise<H2HMatch[]> => {
    const response = await CustomAxios.get('/tennis/predictions/today/enhanced/h2h', { params: { event_id: eventId } })
    return Array.isArray(response.data) ? (response.data as H2HMatch[]) : []
}

export const fetchPlayerStats = async (params: { player: string; tour: string; surface: string; debug: boolean; req_ts: number }): Promise<PlayerStatsResponse> => {
    const response = await CustomAxios.get('/tennis/players/stats', { params })
    return normalizePlayerStatsResponse((response.data || {}) as Record<string, unknown>)
}

export const fetchPlayerCompare = async (params: {
    player1: string
    player2: string
    tour: string
    surface: string
    event_id?: number
    debug: boolean
    req_ts: number
}): Promise<CompareResponse> => {
    const response = await CustomAxios.get('/tennis/players/compare', { params })
    return normalizeCompareResponse((response.data || {}) as Record<string, unknown>)
}

export const fetchSuggestedParlays = async (params: Record<string, unknown>): Promise<ParlaySuggestion[]> => {
    const response = await CustomAxios.get('/tennis/parlay/suggested', { params })
    const raw = Array.isArray(response.data) ? response.data : response.data?.parlays ?? response.data?.data ?? []
    return normalizeParlaySuggestions(raw)
}
