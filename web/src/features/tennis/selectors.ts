import { EnhancedPrediction } from './types'
import { getField, toNumOrNull } from './formatters'

export const getPredictionName = (record: EnhancedPrediction, side: 'p1' | 'p2'): string =>
    String(getField(record, side === 'p1' ? 'p1_name' : 'p2_name', side === 'p1' ? 'p1Name' : 'p2Name') ?? (side === 'p1' ? 'P1' : 'P2'))

export const getPredictionWinProb = (record: EnhancedPrediction, side: 'p1' | 'p2'): number | null =>
    toNumOrNull(getField(record, side === 'p1' ? 'p1_blended_win_prob' : 'p2_blended_win_prob', side === 'p1' ? 'p1BlendedWinProb' : 'p2BlendedWinProb')) ??
    toNumOrNull(getField(record, side === 'p1' ? 'p1_win_prob' : 'p2_win_prob', side === 'p1' ? 'p1WinProb' : 'p2WinProb'))

export const getPredictionModelWinProb = (record: EnhancedPrediction, side: 'p1' | 'p2'): number | null =>
    toNumOrNull(getField(record, side === 'p1' ? 'p1_win_prob' : 'p2_win_prob', side === 'p1' ? 'p1WinProb' : 'p2WinProb'))

export const getPredictionBlendedWinProb = (record: EnhancedPrediction, side: 'p1' | 'p2'): number | null =>
    toNumOrNull(getField(record, side === 'p1' ? 'p1_blended_win_prob' : 'p2_blended_win_prob', side === 'p1' ? 'p1BlendedWinProb' : 'p2BlendedWinProb'))

export const getPredictionFairOdds = (record: EnhancedPrediction, side: 'p1' | 'p2'): number | null =>
    toNumOrNull(getField(record, side === 'p1' ? 'p1_fair_american' : 'p2_fair_american', side === 'p1' ? 'p1FairAmerican' : 'p2FairAmerican'))

export const getPredictionMarketOdds = (record: EnhancedPrediction, side: 'p1' | 'p2'): number | null => {
    const fromRoot = toNumOrNull(
        getField(record, side === 'p1' ? 'p1_market_odds_american' : 'p2_market_odds_american', side === 'p1' ? 'p1MarketOddsAmerican' : 'p2MarketOddsAmerican')
    )
    if (fromRoot !== null) return fromRoot
    return toNumOrNull(
        getField(
            record.inputs || {},
            side === 'p1' ? 'p1_market_odds_american' : 'p2_market_odds_american',
            side === 'p1' ? 'p1MarketOddsAmerican' : 'p2MarketOddsAmerican'
        )
    )
}

export const getSofaEventId = (record: EnhancedPrediction): number | null => {
    const key = getField<string>(record, 'match_key', 'matchKey')
    if (!key || !key.startsWith('sofascore:')) return null
    const id = Number(key.split(':')[1])
    return Number.isFinite(id) ? id : null
}

export const getStyleSummary = (record: EnhancedPrediction) =>
    getField<{ p1?: { name?: string; label?: string | null }; p2?: { name?: string; label?: string | null } }>(record.inputs, 'style_summary', 'style_summary')

export const getPickSummary = (record: EnhancedPrediction): string | undefined =>
    getField<string | undefined>(record.inputs, 'pick_summary', 'pick_summary')
