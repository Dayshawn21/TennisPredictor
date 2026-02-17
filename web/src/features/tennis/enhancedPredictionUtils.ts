import { EnhancedPrediction } from './types'
import { americanToProb, fmtPct, getField, toNumOrNull } from './formatters'
import { getPredictionFairOdds, getPredictionMarketOdds, getPredictionName, getPredictionWinProb } from './selectors'

export interface FavoredPick {
    side: 'p1' | 'p2'
    name: string
    prob: number | null
    fairOdds: number | null
}

export interface MarketEdge {
    marketFavProb: number | null
    edge: number | null
    bookFavProb: number | null
    bookEdge: number | null
}

export interface ConfidenceScore {
    score: number
    label: 'HIGH' | 'MED' | 'LOW'
    color: 'green' | 'gold' | 'red'
}

export interface PredictorComponent {
    name: string
    p1_prob: number | null
    weight: number | null
}

export const toNum = (value: unknown): number | null => {
    if (value == null) return null
    const n = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN
    return Number.isFinite(n) ? n : null
}

const normalizeScoreKey = (key: unknown): string | null => {
    const s = String(key).trim()
    if (!s) return null
    if (s.includes('-')) return s
    if (/^\d{2}$/.test(s)) return `${s[0]}-${s[1]}`
    return null
}

export const normalizeSetScoreProbs = (raw: unknown): Record<string, number> | undefined => {
    if (!raw || typeof raw !== 'object') return undefined
    const out: Record<string, number> = {}
    for (const [k, v] of Object.entries(raw)) {
        const nk = normalizeScoreKey(k)
        const nv = toNum(v)
        if (nk && nv != null) out[nk] = nv
    }
    return Object.keys(out).length ? out : undefined
}

export const deriveScoreProbsFromSetP = (pSetWinP1: number, bestOf: number): Record<string, number> => {
    const p = Math.min(0.999999, Math.max(0.000001, pSetWinP1))
    const q = 1 - p
    if (Number(bestOf) === 5) {
        return {
            '3-0': p ** 3,
            '3-1': 3 * p ** 3 * q,
            '3-2': 6 * p ** 3 * q ** 2,
            '0-3': q ** 3,
            '1-3': 3 * q ** 3 * p,
            '2-3': 6 * q ** 3 * p ** 2
        }
    }
    return { '2-0': p ** 2, '2-1': 2 * p ** 2 * q, '0-2': q ** 2, '1-2': 2 * q ** 2 * p }
}

export const pickTopScorelines = (scoreProbs: Record<string, number> | undefined, topN: number = 2): string | null => {
    if (!scoreProbs) return null
    const entries = Object.entries(scoreProbs)
        .map(([k, v]) => ({ k, v: toNum(v) }))
        .filter((x): x is { k: string; v: number } => x.v != null)
        .sort((a, b) => b.v - a.v)
    if (!entries.length) return null
    return entries
        .slice(0, topN)
        .map((entry) => `${entry.k} (${fmtPct(entry.v)})`)
        .join(' • ')
}

export const getComponents = (record: EnhancedPrediction): PredictorComponent[] => {
    const inp = (record?.inputs || {}) as Record<string, unknown>
    const arr = (inp.individual_predictions ?? inp.individualPredictions) as unknown
    if (Array.isArray(arr)) {
        return arr
            .map((component) => {
                const row = (component || {}) as Record<string, unknown>
                return {
                    name: String(row.name ?? '').toLowerCase(),
                    p1_prob: toNumOrNull(row.p1_prob ?? row.p1Prob ?? row.p1_probability),
                    weight: toNumOrNull(row.weight)
                }
            })
            .filter((component) => component.name)
    }

    const obj = inp.individualPredictions as Record<string, unknown> | undefined
    if (obj && typeof obj === 'object') {
        const out: PredictorComponent[] = []
        if (obj.elo != null) out.push({ name: 'elo', p1_prob: toNumOrNull(obj.elo), weight: null })
        if (obj.xgbSimple != null) out.push({ name: 'xgb', p1_prob: toNumOrNull(obj.xgbSimple), weight: null })
        if (obj.market != null) out.push({ name: 'market_no_vig', p1_prob: toNumOrNull(obj.market), weight: null })
        return out
    }

    return []
}

export const compProb = (components: PredictorComponent[], names: string[]): number | null => {
    const wanted = names.map((name) => name.toLowerCase())
    const hit = components.find((component) => wanted.includes(String(component?.name ?? '').toLowerCase()))
    return toNumOrNull(hit?.p1_prob)
}

export const getFavored = (record: EnhancedPrediction): FavoredPick => {
    const p1 = getPredictionWinProb(record, 'p1')
    const p2 = getPredictionWinProb(record, 'p2')
    const useP1 = p1 != null && (p2 == null || p1 >= p2)
    const side: 'p1' | 'p2' = useP1 ? 'p1' : 'p2'
    return { side, name: getPredictionName(record, side), prob: side === 'p1' ? p1 : p2, fairOdds: getPredictionFairOdds(record, side) }
}

export const getMarketEdge = (record: EnhancedPrediction): MarketEdge => {
    const components = getComponents(record)
    const marketP1FromComponents = compProb(components, ['market_no_vig', 'market'])
    const marketP1Top = toNum(getField(record, 'p1_market_no_vig_prob', 'p1MarketNoVigProb'))
    const marketP2Top = toNum(getField(record, 'p2_market_no_vig_prob', 'p2MarketNoVigProb'))
    const marketP1Implied = toNum(getField(record, 'p1_market_implied_prob', 'p1MarketImpliedProb'))
    const marketP2Implied = toNum(getField(record, 'p2_market_implied_prob', 'p2MarketImpliedProb'))
    const p1BookOdds = getPredictionMarketOdds(record, 'p1')
    const p2BookOdds = getPredictionMarketOdds(record, 'p2')
    const favored = getFavored(record)

    if (favored.prob == null) return { marketFavProb: null, edge: null, bookFavProb: null, bookEdge: null }

    const baseP1 = marketP1FromComponents ?? marketP1Top ?? marketP1Implied
    const baseP2 = marketP1FromComponents == null ? (marketP2Top ?? marketP2Implied) : null
    const marketFavProb = favored.side === 'p1' ? (baseP1 ?? null) : (baseP2 ?? (baseP1 != null ? 1 - baseP1 : null))
    const bookP1 = americanToProb(p1BookOdds)
    const bookP2 = americanToProb(p2BookOdds)
    const bookFavProb = favored.side === 'p1' ? bookP1 : bookP2

    if (marketFavProb == null) {
        return { marketFavProb: null, edge: null, bookFavProb, bookEdge: bookFavProb == null ? null : favored.prob - bookFavProb }
    }

    return { marketFavProb, edge: favored.prob - marketFavProb, bookFavProb, bookEdge: bookFavProb == null ? null : favored.prob - bookFavProb }
}

export const getConfidence = (record: EnhancedPrediction): ConfidenceScore => {
    const inp = (record.inputs || {}) as Record<string, unknown>
    const numPredictors = Number(inp.numPredictors ?? inp.num_predictors ?? 0)
    const hasElo = !!(inp.hasElo ?? inp.has_elo)
    const hasRolling = !!(inp.hasRolling ?? inp.has_rolling)
    const hasOdds = !!(inp.hasOdds ?? inp.has_odds)

    const favored = getFavored(record)
    const probabilityStrength = favored.prob == null ? 0 : Math.min(1, Math.abs(favored.prob - 0.5) * 2)
    const predictorScore = Math.min(1, Math.max(0, numPredictors / 3))
    const coverageScore = (Number(hasElo) + Number(hasRolling) + Number(hasOdds)) / 3
    const score = 0.62 * probabilityStrength + 0.23 * predictorScore + 0.15 * coverageScore

    if (score >= 0.72) return { score, label: 'HIGH', color: 'green' }
    if (score >= 0.58) return { score, label: 'MED', color: 'gold' }
    return { score, label: 'LOW', color: 'red' }
}

export const getCoverageFlags = (record: EnhancedPrediction) => {
    const inp = (record.inputs || {}) as Record<string, unknown>
    return {
        hasElo: !!(inp.hasElo ?? inp.has_elo),
        hasRolling: !!(inp.hasRolling ?? inp.has_rolling),
        hasOdds: !!(inp.hasOdds ?? inp.has_odds),
        eloUsedMedian: !!(inp.eloUsedMedian ?? inp.elo_used_median)
    }
}

export const isUpcoming = (record: EnhancedPrediction): boolean => {
    const raw = getField(record, 'match_start_utc', 'matchStartUtc') ?? getField(record, 'match_date', 'matchDate')
    const t = raw ? Date.parse(String(raw)) : NaN
    const graceMs = 3 * 60 * 60 * 1000
    return !Number.isFinite(t) || t + graceMs >= Date.now()
}
