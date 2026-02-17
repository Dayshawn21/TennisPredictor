// FILE: web/src/pages/TennisEnhancedPredictionsPage.tsx
import { CheckCircleOutlined, LineChartOutlined, ThunderboltOutlined } from '@ant-design/icons'
import { Alert, Card, Descriptions, Divider, Spin, Switch, Table, Tag, Tooltip } from 'antd'
import React, { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { fetchEnhancedPredictions, fetchH2HByEvent } from '../features/tennis/api'
import {
    compProb,
    getComponents,
    getConfidence,
    getCoverageFlags,
    getFavored,
    getMarketEdge,
    isUpcoming,
    toNum
} from '../features/tennis/enhancedPredictionUtils'
import {
    getPickSummary,
    getPredictionFairOdds as getFairOdds,
    getPredictionMarketOdds as getMarketOdds,
    getPredictionModelWinProb as getModelWinProb,
    getPredictionName as getName,
    getPredictionWinProb as getWinProb,
    getSofaEventId,
    getStyleSummary
} from '../features/tennis/selectors'
import { EnhancedPrediction, EnhancedPredictionsResponse, H2HMatch } from '../features/tennis/types'
import { ROUTES } from '../resources/routes-constants'

const TennisEnhancedPredictionsPage: React.FC = () => {
    const navigate = useNavigate()
    const [data, setData] = useState<EnhancedPredictionsResponse | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [showUpcomingOnly, setShowUpcomingOnly] = useState(true)
    const [h2hDetails, setH2hDetails] = useState<Record<string, H2HMatch[]>>({})
    const [h2hLoading, setH2hLoading] = useState<Record<string, boolean>>({})

    const fetchH2H = async (record: EnhancedPrediction) => {
        const evId = getSofaEventId(record)
        if (!evId) return
        const k = String(evId)
        if (h2hDetails[k] || h2hLoading[k]) return
        setH2hLoading((prev) => ({ ...prev, [k]: true }))
        try {
            const rows = await fetchH2HByEvent(evId)
            setH2hDetails((prev) => ({ ...prev, [k]: rows }))
        } catch {
            setH2hDetails((prev) => ({ ...prev, [k]: [] }))
        } finally {
            setH2hLoading((prev) => ({ ...prev, [k]: false }))
        }
    }

    useEffect(() => {
        const fetchPredictions = async () => {
            try {
                setLoading(true)
                setError(null)
                setData(await fetchEnhancedPredictions())
            } catch (e: unknown) {
                setError((e as { message?: string })?.message || 'Failed to load predictions')
            } finally {
                setLoading(false)
            }
        }

        fetchPredictions()
    }, [])

    const tournamentGroups = useMemo(() => {
        const items = data?.items || []
        const startTs = (m: EnhancedPrediction) => {
            const raw = (m as any)?.matchStartUtc ?? (m as any)?.match_start_utc ?? (m as any)?.matchDate ?? (m as any)?.match_date ?? null
            const t = raw ? Date.parse(String(raw)) : NaN
            return Number.isFinite(t) ? t : Number.POSITIVE_INFINITY
        }
        const map = new Map<string, { key: string; title: string; matches: EnhancedPrediction[] }>()

        for (const m of items) {
            const tournament = String(m.tournament || 'Unknown Tournament').trim() || 'Unknown Tournament'
            const tour = String(m.tour || '').trim()
            const key = `${tournament}||${tour}`.toLowerCase()
            const title = tour ? `${tournament} (${tour.toUpperCase()})` : tournament

            const existing = map.get(key)
            if (existing) {
                existing.matches.push(m)
            } else {
                map.set(key, { key, title, matches: [m] })
            }
        }

        for (const group of map.values()) {
            group.matches.sort((a, b) => startTs(a) - startTs(b))
        }

        return Array.from(map.values()).sort((a, b) => a.title.localeCompare(b.title))
    }, [data])

    // ----------------------------
    // UI helpers
    // ----------------------------
    const getMethodIcon = (numPredictors: number) => {
        if (numPredictors >= 3) return <CheckCircleOutlined style={{ color: '#52c41a' }} />
        if (numPredictors === 2) return <ThunderboltOutlined style={{ color: '#1890ff' }} />
        return <LineChartOutlined style={{ color: '#faad14' }} />
    }

    const getMethodColor = (numPredictors: number) => {
        switch (numPredictors) {
            case 3:
                return 'green'
            case 2:
                return 'blue'
            default:
                return 'orange'
        }
    }

    const fmtPct = (val: number | null | undefined) => (val == null ? '—' : `${(val * 100).toFixed(1)}%`)

    const fmtOdds = (val: number | null | undefined) => {
        if (val == null) return '—'
        return val > 0 ? `+${val}` : `${val}`
    }

    const fmtNum = (val: number | null | undefined, digits: number = 2) => {
        if (val == null) return '—'
        const n = Number(val)
        if (!Number.isFinite(n)) return '—'
        return n.toFixed(digits)
    }

    const fmtPp = (val: number | null | undefined) => {
        if (val == null) return '—'
        const pp = val * 100
        const s = pp >= 0 ? `+${pp.toFixed(1)}pp` : `${pp.toFixed(1)}pp`
        return s
    }

    const probToAmerican = (p: number | null | undefined): number | null => {
        if (p == null) return null
        if (p <= 0 || p >= 1) return null
        if (p >= 0.5) return -Math.round((p / (1 - p)) * 100)
        return Math.round(((1 - p) / p) * 100)
    }
    const fmtCentralDateTime = (isoUtc?: string | null) => {
        if (!isoUtc) return '—'
        const d = new Date(isoUtc)
        if (Number.isNaN(d.getTime())) return '—'
        return d.toLocaleString('en-US', {
            timeZone: 'America/Chicago',
            month: '2-digit',
            day: '2-digit',
            year: 'numeric',
            hour: 'numeric',
            minute: '2-digit'
        })
    }

    const ts = (v?: string | null) => {
        const t = v ? Date.parse(v) : NaN
        return Number.isFinite(t) ? t : Number.POSITIVE_INFINITY
    }

    const g = (obj: any, camel: string, snake?: string) => obj?.[camel] ?? (snake ? obj?.[snake] : undefined)

    const openPlayerStats = (record: EnhancedPrediction, side: 'p1' | 'p2') => {
        const name = getName(record, side)
        const tour = String(record?.tour || 'ATP').toUpperCase()
        const surface = String(record?.surface || 'hard').toLowerCase()
        const q = new URLSearchParams({
            player: name,
            tour,
            surface
        })
        navigate(`${ROUTES.TENNIS_PLAYER_STATS_ROUTE}?${q.toString()}`)
    }

    const openCompare = (record: EnhancedPrediction) => {
        const player1 = getName(record, 'p1')
        const player2 = getName(record, 'p2')
        const tour = String(record?.tour || 'ATP').toUpperCase()
        const surface = String(record?.surface || 'hard').toLowerCase()
        const evId = getSofaEventId(record)
        const q = new URLSearchParams({ player1, player2, tour, surface })
        if (evId) q.set('event_id', String(evId))
        navigate(`${ROUTES.TENNIS_PLAYER_COMPARE_ROUTE}?${q.toString()}`)
    }

    const coverageTags = (record: EnhancedPrediction) => {
        const { hasElo, hasRolling, hasOdds, eloUsedMedian } = getCoverageFlags(record)

        const mk = (ok: boolean, label: string) => (
            <Tag key={label} color={ok ? 'green' : 'red'} style={{ marginInlineEnd: 4 }}>
                {label}
            </Tag>
        )
        return (
            <div className="flex flex-wrap gap-1">
                {mk(hasElo, 'ELO')}
                {eloUsedMedian ? (
                    <Tag key="MED-ELO" color="gold" style={{ marginInlineEnd: 4 }}>
                        MED-ELO
                    </Tag>
                ) : null}
                {mk(hasRolling, 'ROLL')}
                {mk(hasOdds, 'ODDS')}
            </div>
        )
    }

    // ----------------------------
    // Summary cards (missing info surfaced)
    // ----------------------------
    const filteredItems = useMemo(() => {
        const items = data?.items || []
        if (!showUpcomingOnly) return items
        return items.filter(isUpcoming)
    }, [data?.items, showUpcomingOnly])

    const stats = useMemo(() => {
        const items = filteredItems
        const np = (m: EnhancedPrediction) => Number(m.inputs?.numPredictors ?? (m.inputs as any)?.num_predictors ?? 0)
        const has = (m: EnhancedPrediction, k: 'hasElo' | 'hasRolling' | 'hasOdds') =>
            !!((m.inputs as any)?.[k] ?? (m.inputs as any)?.[k.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`)])
        const missingOdds = items.filter((m) => !has(m, 'hasOdds')).length
        const missingElo = items.filter((m) => !has(m, 'hasElo')).length
        const missingRolling = items.filter((m) => !has(m, 'hasRolling')).length

        return {
            total: items.length,
            fullEnsemble: items.filter((m) => np(m) >= 3).length,
            partial: items.filter((m) => np(m) === 2).length,
            single: items.filter((m) => np(m) <= 1).length,
            missingOdds,
            missingElo,
            missingRolling
        }
    }, [filteredItems])

    const ladderEdges = useMemo(() => {
        const items = filteredItems
        const rows = items
            .map((m) => {
                const favored = getFavored(m)
                const { marketFavProb, edge, bookFavProb, bookEdge } = getMarketEdge(m)
                if ((edge == null && bookEdge == null) || favored.prob == null) return null
                const marketFavAmerican = probToAmerican(marketFavProb)
                const bookFavAmerican = probToAmerican(bookFavProb)
                return {
                    matchKey: String(g(m as any, 'matchKey', 'match_key') ?? ''),
                    tour: g(m as any, 'tour', 'tour'),
                    p1: getName(m, 'p1'),
                    p2: getName(m, 'p2'),
                    favoredName: favored.name,
                    modelFavProb: favored.prob,
                    marketFavProb,
                    marketFavAmerican,
                    bookFavProb,
                    bookFavAmerican,
                    edge,
                    bookEdge,
                    start: g(m as any, 'matchStartUtc', 'match_start_utc')
                }
            })
            .filter(Boolean) as any[]

        const minEdge = -0.5 // allow small negative edge
        const ladderEligible = rows
            .filter((r) => {
                const inBook = r.bookFavAmerican != null && r.bookFavAmerican <= -140 && r.bookFavAmerican >= -180
                const inNoVig = r.marketFavAmerican != null && r.marketFavAmerican <= -140 && r.marketFavAmerican >= -180
                const edgeOk = (r.bookEdge ?? r.edge ?? -999) >= minEdge
                return (inBook || inNoVig) && edgeOk
            })
            .sort((a, b) => {
                const ta = ts(a.start)
                const tb = ts(b.start)
                if (ta !== tb) return ta - tb
                return (b.bookEdge ?? b.edge ?? 0) - (a.bookEdge ?? a.edge ?? 0)
            })

        const otherEdges = rows
            .filter((r) => {
                const inBook = r.bookFavAmerican != null && r.bookFavAmerican <= -140 && r.bookFavAmerican >= -180
                const inNoVig = r.marketFavAmerican != null && r.marketFavAmerican <= -140 && r.marketFavAmerican >= -180
                const edgeOk = (r.bookEdge ?? r.edge ?? -999) >= minEdge
                return !((inBook || inNoVig) && edgeOk)
            })
            .sort((a, b) => (b.bookEdge ?? b.edge ?? 0) - (a.bookEdge ?? a.edge ?? 0))

        const twoMan = rows
            .filter((r) => {
                const inBook = r.bookFavAmerican != null && r.bookFavAmerican <= -110 && r.bookFavAmerican >= -180
                const edgeOk = (r.bookEdge ?? r.edge ?? -999) >= minEdge
                return inBook && edgeOk
            })
            .sort((a, b) => (b.bookEdge ?? b.edge ?? 0) - (a.bookEdge ?? a.edge ?? 0))
            .slice(0, 2)

        return {
            ladderEligible,
            otherEdges,
            twoMan
        }
    }, [filteredItems])

    const parlayLadder = useMemo(() => {
        const items = filteredItems
        const rows = items
            .map((m) => {
                const favored = getFavored(m)
                const { edge, bookFavProb, bookEdge } = getMarketEdge(m)
                if ((edge == null && bookEdge == null) || favored.prob == null) return null
                const bookFavAmerican = probToAmerican(bookFavProb)
                return {
                    matchKey: String(g(m as any, 'matchKey', 'match_key') ?? ''),
                    tour: g(m as any, 'tour', 'tour'),
                    p1: getName(m, 'p1'),
                    p2: getName(m, 'p2'),
                    favoredName: favored.name,
                    modelFavProb: favored.prob,
                    bookFavProb,
                    bookFavAmerican,
                    edge,
                    bookEdge,
                    start: g(m as any, 'matchStartUtc', 'match_start_utc')
                }
            })
            .filter(Boolean) as any[]

        const minEdge = 0.0
        const eligible = rows
            .filter((r) => (r.bookEdge ?? r.edge ?? -999) >= minEdge)
            .sort((a, b) => (b.bookEdge ?? b.edge ?? 0) - (a.bookEdge ?? a.edge ?? 0))
            .slice(0, 30)

        const pairs: any[] = []
        for (let i = 0; i < eligible.length; i++) {
            for (let j = i + 1; j < eligible.length; j++) {
                const a = eligible[i]
                const b = eligible[j]
                if (!a.bookFavProb || !b.bookFavProb) continue
                const bookParlayProb = a.bookFavProb * b.bookFavProb
                const modelParlayProb = (a.modelFavProb ?? 0) * (b.modelFavProb ?? 0)
                const parlayOdds = probToAmerican(bookParlayProb)
                const parlayEdge = modelParlayProb - bookParlayProb
                if (parlayOdds != null && parlayOdds >= -180 && parlayOdds <= 120) {
                    pairs.push({
                        key: `${a.matchKey}__${b.matchKey}`,
                        legs: [a, b],
                        bookParlayProb,
                        modelParlayProb,
                        parlayOdds,
                        parlayEdge
                    })
                }
            }
        }

        pairs.sort((x, y) => y.parlayEdge - x.parlayEdge)

        const usedPlayers = new Set<string>()
        const uniquePairs: any[] = []
        for (const p of pairs) {
            const aName = String(p.legs?.[0]?.favoredName ?? '').toLowerCase()
            const bName = String(p.legs?.[1]?.favoredName ?? '').toLowerCase()
            if (!aName || !bName) continue
            if (usedPlayers.has(aName) || usedPlayers.has(bName)) continue
            usedPlayers.add(aName)
            usedPlayers.add(bName)
            uniquePairs.push(p)
            if (uniquePairs.length >= 5) break
        }

        return uniquePairs
    }, [filteredItems])
    // ----------------------------
    // Columns
    // ----------------------------
    const columns: any[] = [
        {
            title: 'Match',
            key: 'match',
            render: (_: any, record: EnhancedPrediction) => {
                const inp: any = record.inputs || {}
                const p1Med = !!(inp.p1EloUsedMedian ?? inp.p1_elo_used_median)
                const p2Med = !!(inp.p2EloUsedMedian ?? inp.p2_elo_used_median)
                return (
                    <div>
                        <div className="font-semibold">
                            <button
                                type="button"
                                onClick={() => openPlayerStats(record, 'p1')}
                                className="bg-transparent border-0 p-0 cursor-pointer underline"
                                style={{ color: '#91caff' }}
                            >
                                {record.p1Name}
                            </button>
                            {p1Med ? (
                                <Tag color="gold" style={{ marginInlineStart: 6 }}>
                                    MED-ELO
                                </Tag>
                            ) : null}
                            <span> vs </span>
                            <button
                                type="button"
                                onClick={() => openPlayerStats(record, 'p2')}
                                className="bg-transparent border-0 p-0 cursor-pointer underline"
                                style={{ color: '#91caff' }}
                            >
                                {record.p2Name}
                            </button>
                            {p2Med ? (
                                <Tag color="gold" style={{ marginInlineStart: 6 }}>
                                    MED-ELO
                                </Tag>
                            ) : null}
                        </div>
                        <div className="text-xs text-gray-500 mt-1">
                            {record.tournament} {record.round} BO{record.bestOf || 3}
                        </div>
                        <div className="text-xs text-gray-500 mt-1">
                            <span>{fmtCentralDateTime(record.matchStartUtc)}</span>
                        </div>
                        <div className="mt-2">
                            <button
                                type="button"
                                onClick={() => openCompare(record)}
                                className="bg-transparent border border-gray-500 rounded px-2 py-1 text-xs cursor-pointer"
                                style={{ color: '#91caff' }}
                            >
                                Compare
                            </button>
                        </div>
                    </div>
                )
            },
            width: 320
        },

        {
            title: 'Surface',
            dataIndex: 'surface',
            key: 'surface',
            render: (surface: string) => (
                <Tag color={surface?.toLowerCase().includes('hard') ? 'blue' : surface?.toLowerCase().includes('clay') ? 'orange' : 'green'}>{surface}</Tag>
            ),
            width: 120
        },
        {
            title: 'Coverage',
            key: 'coverage',
            render: (_: any, record: EnhancedPrediction) => coverageTags(record),
            width: 170
        },
        {
            title: 'Method',
            key: 'method',
            render: (_: any, record: EnhancedPrediction) => {
                const inp: any = record.inputs || {}
                const numPredictors = inp.numPredictors ?? inp.num_predictors ?? 0
                const comps = getComponents(record)

                const eloP = compProb(comps, ['elo'])
                const xgbP = compProb(comps, ['xgb', 'xgb_simple', 'xgbsimple'])
                const mktP = compProb(comps, ['market_no_vig', 'market'])

                const tooltipContent = (
                    <div className="text-xs">
                        <div className="font-semibold mb-2">Ensemble Components:</div>
                        {eloP != null && <div>• ELO: {fmtPct(eloP)}</div>}
                        {xgbP != null && <div>• XGBoost: {fmtPct(xgbP)}</div>}
                        {mktP != null && <div>• Market (no-vig): {fmtPct(mktP)}</div>}
                        <div className="mt-2 text-gray-400 border-t pt-2">
                            {numPredictors} predictor{numPredictors > 1 ? 's' : ''} active
                        </div>
                    </div>
                )

                const label = numPredictors >= 3 ? 'ELO + XGB + MKT' : numPredictors === 2 ? 'ELO + XGB' : 'SINGLE'

                return (
                    <Tooltip title={tooltipContent}>
                        <Tag color={getMethodColor(numPredictors)} icon={getMethodIcon(numPredictors)}>
                            {label}
                        </Tag>
                    </Tooltip>
                )
            },
            width: 160
        },
        {
            title: 'Pick',
            key: 'prediction',
            render: (_: any, record: EnhancedPrediction) => {
                // Use getFavored (blended?model fallback) so the name matches the summary
                const favored = getFavored(record)
                const side = favored.side
                const oppSide: 'p1' | 'p2' = side === 'p1' ? 'p2' : 'p1'
                const oppName = getName(record, oppSide)
                const modelProb = getModelWinProb(record, side)
                const fairOdds = getFairOdds(record, side)
                const mktOdds = getMarketOdds(record, side)
                const props: any = g(record.inputs, 'projectedProps', 'projected_props') ?? g(record.inputs, 'projected_props_v2', 'projected_props_v2')
                const aces: any = g(props, 'aces', 'aces')
                const bp: any = g(props, 'breakPoints', 'break_points')
                const p1A = g(aces, 'p1Expected', 'p1_expected') as number | undefined
                const p2A = g(aces, 'p2Expected', 'p2_expected') as number | undefined
                const bp1 = g(bp, 'p1Breaks', 'p1_breaks') as number | undefined
                const bp2 = g(bp, 'p2Breaks', 'p2_breaks') as number | undefined

                return (
                    <div>
                        <div className="font-semibold">{favored.name}</div>
                        <div className="text-xs text-gray-400">over {oppName}</div>
                        <div className="text-sm mt-1">
                            <span className="text-gray-500">Model:</span>{' '}
                            <span className="text-green-600 font-medium">{modelProb == null ? '--' : fmtPct(modelProb)}</span>
                            <span className="text-gray-400 mx-2">•</span>
                            <span className="text-blue-600">{fmtOdds(fairOdds)}</span>
                        </div>
                        {mktOdds != null && (
                            <div className="text-sm mt-1">
                                <span className="text-gray-500">Market:</span> <span className="text-blue-600">{fmtOdds(mktOdds)}</span>
                            </div>
                        )}
                        <div className="text-xs mt-1 text-gray-500">
                            <div>Aces: {p1A == null || p2A == null ? '--' : `${fmtNum(p1A, 1)} / ${fmtNum(p2A, 1)}`}</div>
                            <div>BP: {bp1 == null || bp2 == null ? '--' : `${fmtNum(bp1, 1)} / ${fmtNum(bp2, 1)}`}</div>
                        </div>
                    </div>
                )
            },
            width: 240
        },
        {
            title: 'Model vs Market',
            key: 'edge',
            sorter: (a: EnhancedPrediction, b: EnhancedPrediction) => {
                const ea = getMarketEdge(a)
                const eb = getMarketEdge(b)
                const aScore = Math.max(Math.abs(ea.edge ?? 0), Math.abs(ea.bookEdge ?? 0))
                const bScore = Math.max(Math.abs(eb.edge ?? 0), Math.abs(eb.bookEdge ?? 0))
                return aScore - bScore
            },
            render: (_: any, record: EnhancedPrediction) => {
                const favored = getFavored(record)
                const { marketFavProb, edge } = getMarketEdge(record)

                if (marketFavProb == null || edge == null || favored.prob == null) return <span className="text-gray-400">—</span>

                const color = (edge ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'
                return (
                    <div className="text-xs space-y-1">
                        <div className="flex justify-between gap-3">
                            <span className="text-gray-500">No-vig:</span>
                            <span className="font-medium">{fmtPct(marketFavProb)}</span>
                        </div>
                        <div className="flex justify-between gap-3">
                            <span className="text-gray-500">Edge:</span>
                            <span className={`font-semibold ${color}`}>{fmtPp(edge)}</span>
                        </div>
                    </div>
                )
            },
            width: 160
        },
        {
            title: 'Confidence',
            key: 'conf',
            sorter: (a: EnhancedPrediction, b: EnhancedPrediction) => getConfidence(a).score - getConfidence(b).score,
            render: (_: any, record: EnhancedPrediction) => {
                const c = getConfidence(record)
                return (
                    <Tooltip title={`Composite score: ${fmtNum(c.score, 3)} (prob strength + #models + coverage)`}>
                        <Tag color={c.color}>{c.label}</Tag>
                    </Tooltip>
                )
            },
            width: 120
        },
        {
            title: 'Summary',
            key: 'summary',
            render: (_: any, record: EnhancedPrediction) => {
                const summary = getPickSummary(record)
                if (!summary) return <span className="text-gray-400">—</span>

                // Parse structured sections: intro, Key factors, Risks
                const factorsIdx = summary.indexOf('Key factors:')
                const risksIdx = summary.indexOf('Risks:')

                // Determine where intro ends
                const sectionStart = Math.min(factorsIdx >= 0 ? factorsIdx : Infinity, risksIdx >= 0 ? risksIdx : Infinity)
                const intro = sectionStart < Infinity ? summary.slice(0, sectionStart).trim() : summary.trim()

                // Extract Key factors section
                let factors: string[] = []
                if (factorsIdx >= 0) {
                    const factorsEnd = risksIdx > factorsIdx ? risksIdx : summary.length
                    const factorsRaw = summary
                        .slice(factorsIdx + 'Key factors:'.length, factorsEnd)
                        .replace(/\.$/, '')
                        .trim()
                    factors = factorsRaw
                        .split(';')
                        .map((f) => f.trim())
                        .filter(Boolean)
                }

                // Extract Risks section
                let risks: string[] = []
                if (risksIdx >= 0) {
                    const risksRaw = summary
                        .slice(risksIdx + 'Risks:'.length)
                        .replace(/\.$/, '')
                        .trim()
                    risks = risksRaw
                        .split(';')
                        .map((f) => f.trim())
                        .filter(Boolean)
                }

                if (!factors.length && !risks.length) {
                    return <div className="text-xs text-gray-500 leading-relaxed">{summary}</div>
                }

                return (
                    <div className="text-xs leading-relaxed space-y-2">
                        {intro && <div className="text-gray-600 font-medium">{intro}</div>}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                            <div>
                                <div className="text-[10px] uppercase tracking-wide text-gray-400 mb-1">Factors</div>
                                {factors.length === 0 ? (
                                    <div className="text-gray-400">—</div>
                                ) : (
                                    <div className="flex flex-wrap gap-2">
                                        {factors.map((f, i) => (
                                            <span key={`f${i}`} className="inline-block bg-green-50 text-green-700 rounded px-2 py-0.5 text-[10px]">
                                                <span className="mr-1">?</span>
                                                {f}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>
                            <div>
                                <div className="text-[10px] uppercase tracking-wide text-gray-400 mb-1">Risks</div>
                                {risks.length === 0 ? (
                                    <div className="text-gray-400">—</div>
                                ) : (
                                    <div className="flex flex-wrap gap-2">
                                        {risks.map((r, i) => (
                                            <span key={`r${i}`} className="inline-block bg-red-50 text-red-600 rounded px-2 py-0.5 text-[10px]">
                                                <span className="mr-1">?</span>
                                                {r}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )
            },
            width: 300
        }
    ]

    // Expanded row: nested details + component table
    const expandedRowRender = (record: EnhancedPrediction) => {
        const comps = getComponents(record)

        const favored = getFavored(record)
        const { marketFavProb, edge, bookFavProb, bookEdge } = getMarketEdge(record)
        const conf = getConfidence(record)

        const sim = g(record.inputs, 'totalsSetsSim', 'totals_sets_sim') as Record<string, unknown> | undefined
        const expGames = g(sim, 'expectedTotalGames', 'expected_total_games') as number | undefined
        const expSets = g(sim, 'expectedSets', 'expected_sets') as number | undefined
        const eSetGames = g(sim, 'expectedSetGames', 'expected_set_games') as number | undefined
        const hasTotalLine = Boolean(g(record.inputs, 'hasTotalGamesLine', 'has_total_games_line'))
        const marketTotalLine = toNum(g(record.inputs, 'sofascoreTotalGamesLine', 'sofascore_total_games_line'))
        const marketTotalOver = toNum(g(record.inputs, 'sofascoreTotalGamesOverAmerican', 'sofascore_total_games_over_american'))
        const marketTotalUnder = toNum(g(record.inputs, 'sofascoreTotalGamesUnderAmerican', 'sofascore_total_games_under_american'))
        const totalGamesEdge = expGames != null && marketTotalLine != null ? expGames - marketTotalLine : null
        const hasGameSpread = Boolean(g(record.inputs, 'hasGameSpread', 'has_game_spread'))
        const hasModelGameSpread = Boolean(g(record.inputs, 'hasModelGameSpread', 'has_model_game_spread'))
        const modelSpreadP1Line = toNum(g(record.inputs, 'modelSpreadP1Line', 'model_spread_p1_line'))
        const modelSpreadP2Line = toNum(g(record.inputs, 'modelSpreadP2Line', 'model_spread_p2_line'))
        const spreadP1Line = toNum(g(record.inputs, 'sofascoreSpreadP1Line', 'sofascore_spread_p1_line'))
        const spreadP2Line = toNum(g(record.inputs, 'sofascoreSpreadP2Line', 'sofascore_spread_p2_line'))
        const spreadP1Odds = toNum(g(record.inputs, 'sofascoreSpreadP1OddsAmerican', 'sofascore_spread_p1_odds_american'))
        const spreadP2Odds = toNum(g(record.inputs, 'sofascoreSpreadP2OddsAmerican', 'sofascore_spread_p2_odds_american'))

        const props: any = g(record.inputs, 'projectedProps', 'projected_props') ?? g(record.inputs, 'projected_props_v2', 'projected_props_v2')
        const aces: any = g(props, 'aces', 'aces')
        const bp: any = g(props, 'breakPoints', 'break_points')

        const p1A = g(aces, 'p1Expected', 'p1_expected') as number | undefined
        const p2A = g(aces, 'p2Expected', 'p2_expected') as number | undefined
        const bp1 = g(bp, 'p1Breaks', 'p1_breaks') as number | undefined
        const bp2 = g(bp, 'p2Breaks', 'p2_breaks') as number | undefined

        const h2hP1Wins = toNum(g(record as any, 'h2hP1Wins', 'h2h_p1_wins'))
        const h2hP2Wins = toNum(g(record as any, 'h2hP2Wins', 'h2h_p2_wins'))
        const h2hTotal = toNum(g(record as any, 'h2hTotalMatches', 'h2h_total_matches'))
        const h2hSurfP1 = toNum(g(record as any, 'h2hSurfaceP1Wins', 'h2h_surface_p1_wins'))
        const h2hSurfP2 = toNum(g(record as any, 'h2hSurfaceP2Wins', 'h2h_surface_p2_wins'))
        const h2hSurfTotal = toNum(g(record as any, 'h2hSurfaceMatches', 'h2h_surface_matches'))

        const h2hOverall = h2hTotal != null && h2hTotal > 0 ? `${h2hP1Wins ?? 0}-${h2hP2Wins ?? 0}` : '—'
        const h2hSurface = h2hSurfTotal != null && h2hSurfTotal > 0 ? `${h2hSurfP1 ?? 0}-${h2hSurfP2 ?? 0}` : '—'
        const h2hOverallPct = h2hTotal && h2hTotal > 0 ? Number(h2hP1Wins ?? 0) / h2hTotal : null
        const h2hSurfacePct = h2hSurfTotal && h2hSurfTotal > 0 ? Number(h2hSurfP1 ?? 0) / h2hSurfTotal : null
        const evId = getSofaEventId(record)
        const h2hList = evId != null ? (h2hDetails[String(evId)] ?? []) : []
        const h2hIsLoading = evId != null ? Boolean(h2hLoading[String(evId)]) : false

        const styleSummary = getStyleSummary(record)
        const p1Style = styleSummary?.p1?.label
        const p2Style = styleSummary?.p2?.label
        const fmtStyle = (v?: string | null) => {
            if (!v) return '—'
            return String(v).replace(/_/g, ' ')
        }

        const componentRows = comps.map((c: any, i: number) => ({
            key: `${c.name}-${i}`,
            name: String(c.name ?? '').toUpperCase(),
            p1: toNum(c.p1_prob),
            p2: toNum(c.p1_prob) == null ? null : 1 - Number(c.p1_prob),
            weight: toNum(c.weight)
        }))

        const h2hRows = h2hList.map((m, i) => ({
            key: `h2h-${i}`,
            date: m.date,
            tournament: m.tournament,
            round: m.round,
            surface: m.surface,
            home: m.home,
            away: m.away,
            score: m.score,
            winnerCode: m.winnerCode ?? m.winner_code
        }))

        const h2hColumns = [
            {
                title: 'Date',
                dataIndex: 'date',
                key: 'date',
                width: 90,
                render: (v: any) => v ?? '—'
            },
            {
                title: 'Match',
                key: 'match',
                render: (_: any, row: any) => (
                    <span>
                        {row.home ?? '—'} <span className="text-gray-500">vs</span> {row.away ?? '—'}
                    </span>
                )
            },
            {
                title: 'Round',
                dataIndex: 'round',
                key: 'round',
                width: 110,
                render: (v: any) => v ?? '—'
            },
            {
                title: 'Surface',
                dataIndex: 'surface',
                key: 'surface',
                width: 120,
                render: (v: any) => v ?? '—'
            },
            {
                title: 'Score',
                dataIndex: 'score',
                key: 'score',
                width: 100,
                render: (v: any) => v ?? '—'
            }
        ]

        return (
            <div className="p-2">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    <Card size="small" title="Match details" bordered={false}>
                        <Descriptions size="small" column={1}>
                            <Descriptions.Item label="Match key">{record.matchKey}</Descriptions.Item>
                            <Descriptions.Item label="Start (CT)">{fmtCentralDateTime(record.matchStartUtc)}</Descriptions.Item>
                            <Descriptions.Item label="Tournament">{record.tournament}</Descriptions.Item>
                            <Descriptions.Item label="Round">{record.round}</Descriptions.Item>
                            <Descriptions.Item label="Surface">{record.surface}</Descriptions.Item>
                            <Descriptions.Item label="H2H">
                                <div className="text-xs">
                                    <div className="flex items-center gap-2">
                                        <span className="font-medium">{h2hOverall}</span>
                                        <span className="text-gray-500">({h2hTotal ?? 0} matches)</span>
                                        <span className="text-gray-500">{h2hOverallPct == null ? '—' : fmtPct(h2hOverallPct)}</span>
                                    </div>
                                    <div className="flex items-center gap-2 text-gray-500">
                                        <span>Surface:</span>
                                        <span className="font-medium text-gray-700">{h2hSurface}</span>
                                        <span>({h2hSurfTotal ?? 0} matches)</span>
                                        <span>{h2hSurfacePct == null ? '—' : fmtPct(h2hSurfacePct)}</span>
                                    </div>
                                </div>
                            </Descriptions.Item>
                            <Descriptions.Item label="Style">
                                {getName(record, 'p1')}: {fmtStyle(p1Style)} • {getName(record, 'p2')}: {fmtStyle(p2Style)}
                            </Descriptions.Item>
                            <Descriptions.Item label="Coverage">{coverageTags(record)}</Descriptions.Item>
                            <Descriptions.Item label="Confidence">
                                <Tag color={conf.color}>{conf.label}</Tag> <span className="text-gray-500">({fmtNum(conf.score, 3)})</span>
                            </Descriptions.Item>
                        </Descriptions>
                    </Card>

                    <Card size="small" title="Win probs & pricing" bordered={false}>
                        <div className="text-xs space-y-2">
                            <div className="flex justify-between">
                                <span className="text-gray-500">{getName(record, 'p1')}:</span>
                                <span className="font-medium">
                                    {fmtPct(getWinProb(record, 'p1'))} • fair {fmtOdds(getFairOdds(record, 'p1'))}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-500">{getName(record, 'p2')}:</span>
                                <span className="font-medium">
                                    {fmtPct(getWinProb(record, 'p2'))} • fair {fmtOdds(getFairOdds(record, 'p2'))}
                                </span>
                            </div>

                            <Divider className="my-2" />

                            <div className="flex justify-between">
                                <span className="text-gray-500">Favored:</span>
                                <span className="font-semibold">{favored.name}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-500">Model favored prob:</span>
                                <span className="font-semibold text-green-600">{fmtPct(favored.prob)}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-500">Market favored prob (no-vig):</span>
                                <span className="font-semibold">{fmtPct(marketFavProb)}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-500">Edge (no-vig):</span>
                                <span className={`font-semibold ${edge == null ? 'text-gray-400' : edge >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                    {fmtPp(edge)}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-500">Market favored prob (book):</span>
                                <span className="font-semibold">{fmtPct(bookFavProb)}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-500">Edge (book):</span>
                                <span className={`font-semibold ${bookEdge == null ? 'text-gray-400' : bookEdge >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                    {fmtPp(bookEdge)}
                                </span>
                            </div>

                            <div className="text-[11px] text-gray-400 pt-2 border-t mt-2">
                                Tip: “Edge” is **model prob – market prob** for the favored side (no-vig or book).
                            </div>
                        </div>
                    </Card>

                    <Card size="small" title="Totals, scorelines, props" bordered={false}>
                        <div className="text-xs space-y-2">
                            <div className="flex justify-between">
                                <span className="text-gray-500">Exp total games:</span>
                                <span className="font-medium">{fmtNum(expGames, 1)}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-500">Exp sets:</span>
                                <span className="font-medium">{fmtNum(expSets, 2)}</span>
                            </div>
                            {eSetGames != null && (
                                <div className="flex justify-between">
                                    <span className="text-gray-500">Exp games/set:</span>
                                    <span className="font-medium">{fmtNum(eSetGames, 2)}</span>
                                </div>
                            )}
                            <div className="pt-2 border-t">
                                <div className="text-gray-500 mb-1">Market total games</div>
                                <div className="flex justify-between">
                                    <span className="text-gray-500">Line:</span>
                                    <span className="font-medium">{fmtNum(marketTotalLine, 1)}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-500">Over / Under:</span>
                                    <span className="font-medium">{`${fmtOdds(marketTotalOver)} / ${fmtOdds(marketTotalUnder)}`}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-500">Model edge (games):</span>
                                    <span
                                        className={`font-semibold ${
                                            totalGamesEdge == null ? 'text-gray-400' : totalGamesEdge >= 0 ? 'text-green-600' : 'text-red-600'
                                        }`}
                                    >
                                        {totalGamesEdge == null ? '—' : `${totalGamesEdge >= 0 ? '+' : ''}${fmtNum(totalGamesEdge, 1)}`}
                                    </span>
                                </div>
                                {!hasTotalLine && <div className="text-[11px] text-gray-400 mt-1">No total-games line available for this match.</div>}
                            </div>
                            <div className="pt-2 border-t">
                                <div className="text-gray-500 mb-1">Model game spread (derived)</div>
                                <div className="flex justify-between">
                                    <span className="text-gray-500">{getName(record, 'p1')}:</span>
                                    <span className="font-medium">{modelSpreadP1Line == null ? '—' : fmtNum(modelSpreadP1Line, 1)}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-500">{getName(record, 'p2')}:</span>
                                    <span className="font-medium">{modelSpreadP2Line == null ? '—' : fmtNum(modelSpreadP2Line, 1)}</span>
                                </div>
                                {!hasModelGameSpread && <div className="text-[11px] text-gray-400 mt-1">No derived spread available from model probability.</div>}
                            </div>
                            <div className="pt-2 border-t">
                                <div className="text-gray-500 mb-1">Book game spread (if available)</div>
                                <div className="flex justify-between">
                                    <span className="text-gray-500">{getName(record, 'p1')}:</span>
                                    <span className="font-medium">
                                        {spreadP1Line == null ? '—' : `${fmtNum(spreadP1Line, 1)} (${fmtOdds(spreadP1Odds)})`}
                                    </span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-500">{getName(record, 'p2')}:</span>
                                    <span className="font-medium">
                                        {spreadP2Line == null ? '—' : `${fmtNum(spreadP2Line, 1)} (${fmtOdds(spreadP2Odds)})`}
                                    </span>
                                </div>
                                {!hasGameSpread && <div className="text-[11px] text-gray-400 mt-1">No game-spread market available for this match.</div>}
                            </div>
                            <div className="pt-2 border-t">
                                <div className="text-gray-500 mb-1">Projected props</div>
                                <div className="flex justify-between">
                                    <span className="text-gray-500">Aces (p1/p2):</span>
                                    <span className="font-medium">{p1A == null || p2A == null ? '—' : `${fmtNum(p1A, 1)} / ${fmtNum(p2A, 1)}`}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-500">BP created (p1/p2):</span>
                                    <span className="font-medium">{bp1 == null || bp2 == null ? '—' : `${fmtNum(bp1, 1)} / ${fmtNum(bp2, 1)}`}</span>
                                </div>
                            </div>
                        </div>
                    </Card>
                </div>

                <div className="mt-4">
                    <Card size="small" title="Ensemble components (nested)" bordered={false}>
                        <Table
                            size="small"
                            dataSource={componentRows}
                            pagination={false}
                            columns={[
                                { title: 'Component', dataIndex: 'name', key: 'name', width: 160 },
                                {
                                    title: 'P1',
                                    dataIndex: 'p1',
                                    key: 'p1',
                                    render: (v: number | null) => <span className="font-medium">{fmtPct(v)}</span>,
                                    width: 120
                                },
                                {
                                    title: 'P2',
                                    dataIndex: 'p2',
                                    key: 'p2',
                                    render: (v: number | null) => <span className="font-medium">{fmtPct(v)}</span>,
                                    width: 120
                                },
                                {
                                    title: 'Weight',
                                    dataIndex: 'weight',
                                    key: 'weight',
                                    render: (v: number | null) => (v == null ? '—' : fmtNum(v, 3)),
                                    width: 120
                                }
                            ]}
                            rowKey="key"
                            locale={{ emptyText: 'No component breakdown provided by API.' }}
                        />
                        <div className="text-[11px] text-gray-400 mt-2">
                            If market is missing, “Model vs Market” edge will show —. That’s usually “missing odds” coverage.
                        </div>
                    </Card>

                    <Card size="small" title="H2H Matches" className="lg:col-span-3" bordered={false}>
                        {h2hIsLoading ? (
                            <div className="text-xs text-gray-500">Loading?</div>
                        ) : h2hRows.length === 0 ? (
                            <div className="text-xs text-gray-500">?</div>
                        ) : (
                            <Table size="small" columns={h2hColumns} dataSource={h2hRows} pagination={false} />
                        )}
                    </Card>
                </div>
            </div>
        )
    }

    if (loading) {
        return (
            <div className="flex justify-center items-center h-screen">
                <Spin size="large" />
            </div>
        )
    }

    if (error) {
        return (
            <div className="p-4">
                <Alert type="error" message="Error" description={error} />
            </div>
        )
    }

    return (
        <div className="p-4 max-w-7xl mx-auto">
            <h1 className="text-3xl font-bold mb-6 text-white">Enhanced Tennis Predictions</h1>

            <Alert
                message="What’s new on this page"
                description={
                    <div className="text-sm">
                        <ul className="list-disc list-inside space-y-1">
                            <li>
                                <strong>Coverage</strong> shows if the match had ELO, rolling form, and market odds.
                            </li>
                            <li>
                                <strong>Model vs Market</strong> shows the probability gap vs market no-vig (favored side).
                            </li>
                            <li>
                                Expand a row for nested details: <strong>both-player pricing, scorelines, props, and component table</strong>.
                            </li>
                        </ul>
                    </div>
                }
                type="info"
                className="mb-6"
            />

            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                <div className="flex items-center gap-3 text-sm">
                    <Switch checked={showUpcomingOnly} onChange={setShowUpcomingOnly} />
                    <span className="text-gray-300">{showUpcomingOnly ? 'Upcoming only' : 'Show all matches'}</span>
                </div>
                <div className="text-xs text-gray-400">
                    Showing {filteredItems.length} of {data?.items?.length ?? 0}
                </div>
            </div>

            <div className="grid grid-cols-6 md:grid-cols-6 gap-4 mb-6">
                <Card bordered={false}>
                    <div className="text-2xl font-bold">{stats.total}</div>
                    <div className="text-gray-500">Total Matches</div>
                </Card>
                <Card bordered={false}>
                    <div className="text-2xl font-bold text-green-600">{stats.fullEnsemble}</div>
                    <div className="text-gray-500">Full Ensemble</div>
                </Card>
                <Card bordered={false}>
                    <div className="text-2xl font-bold text-blue-600">{stats.partial}</div>
                    <div className="text-gray-500">Partial Ensemble</div>
                </Card>
                <Card bordered={false}>
                    <div className="text-2xl font-bold text-orange-600">{stats.single}</div>
                    <div className="text-gray-500">Single Model</div>
                </Card>
                <Card bordered={false}>
                    <div className="text-2xl font-bold">{stats.missingOdds}</div>
                    <div className="text-gray-500">Missing Odds</div>
                </Card>
                <Card bordered={false}>
                    <div className="text-2xl font-bold">{stats.missingRolling}</div>
                    <div className="text-gray-500">Missing Rolling</div>
                </Card>
            </div>

            {data && (
                <Card className="mb-6" bordered={false}>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                            <span className="text-gray-500">Source:</span>
                            <span className="ml-2 font-medium">{data.source}</span>
                        </div>
                        <div>
                            <span className="text-gray-500">As of:</span>
                            <span className="ml-2 font-medium">{data.as_of}</span>
                        </div>
                        <div>
                            <span className="text-gray-500">Cached:</span>
                            <span className="ml-2 font-medium">{data.cached ? 'Yes' : 'No'}</span>
                        </div>
                    </div>
                </Card>
            )}

            <Card className="mb-6" title={<span className="text-lg font-semibold">Ladder Picks (Singles ML -180 to -140)</span>} bordered={false}>
                <div className="text-xs text-gray-400 mb-2">Includes book or no-vig odds in range; allows small negative edge.</div>
                {ladderEdges.ladderEligible.length === 0 ? (
                    <div className="text-sm text-gray-500">No ladder-eligible picks found.</div>
                ) : (
                    <div className="grid grid-cols-6 md:grid-cols-6 gap-4 items-center">
                        {ladderEdges.ladderEligible.map((d) => (
                            <Card key={d.matchKey} size="small">
                                <div className="text-xs space-y-1">
                                    <div className="text-gray-400">[{String(d.tour).toUpperCase()}]</div>
                                    <div className="font-semibold">
                                        {d.p1} vs {d.p2}
                                    </div>
                                    <div className="text-gray-500">{fmtCentralDateTime(d.start)}</div>
                                    <div className="text-gray-500">Favorite : {d.favoredName}</div>

                                    <div className="text-gray-400">Market: {fmtOdds(d.marketFavAmerican ?? d.bookFavAmerican)}</div>

                                    <div className="text-gray-400">Edge: {fmtPp(d.bookEdge ?? d.edge)}</div>
                                </div>
                            </Card>
                        ))}
                    </div>
                )}
            </Card>

            <Card className="mb-6" title={<span className="text-lg font-semibold">2-Leg Parlay Ladder (Parlay -180 to +120)</span>} bordered={false}>
                <div className="text-xs text-gray-400 mb-2">Top 5 parlay edges where the combined odds are in range.</div>
                {parlayLadder.length === 0 ? (
                    <div className="text-sm text-gray-500">No parlay candidates found.</div>
                ) : (
                    <div className="grid grid-cols-6 md:grid-cols-6 gap-4 items-center">
                        {parlayLadder.map((p) => (
                            <Card key={p.key} size="small">
                                <div className="text-xs space-y-1">
                                    <div className="font-semibold">
                                        {p.legs[0].favoredName} + {p.legs[1].favoredName}
                                    </div>
                                    <div className="text-gray-400">Parlay odds: {fmtOdds(p.parlayOdds)}</div>
                                    <div className="text-gray-400">Edge: {fmtPp(p.parlayEdge)}</div>
                                </div>
                                <div className="mt-2 space-y-2 text-xs">
                                    {p.legs.map((l: any) => (
                                        <Card key={l.matchKey} bordered={false} size="small">
                                            <div className="text-xs space-y-1">
                                                <div className="text-gray-400">[{String(l.tour).toUpperCase()}]</div>
                                                <div className="font-semibold">
                                                    {l.p1} vs {l.p2}
                                                </div>
                                                <div className="text-gray-500">{fmtCentralDateTime(l.start)}</div>
                                                <div className="text-gray-500">Favorite</div>
                                                <div className="font-semibold">{l.favoredName}</div>
                                                <div className="text-gray-400">Market: {fmtOdds(l.bookFavAmerican ?? l.marketFavAmerican)}</div>
                                                <div className="text-gray-400">Edge: {fmtPp(l.bookEdge ?? l.edge)}</div>
                                            </div>
                                        </Card>
                                    ))}
                                </div>
                            </Card>
                        ))}
                    </div>
                )}
            </Card>

            {filteredItems.length === 0 ? (
                <Card bordered={false}>
                    <div className="text-sm text-gray-500">No matches found.</div>
                </Card>
            ) : (
                tournamentGroups.map((group) => {
                    const matches = showUpcomingOnly ? group.matches.filter(isUpcoming) : group.matches
                    if (matches.length === 0) return null
                    return (
                        <Card key={group.key} title={<span className="text-lg font-semibold">{group.title}</span>} className="mb-6" bordered={false}>
                            <Table
                                dataSource={matches}
                                columns={columns}
                                rowKey={(r: any) => String(g(r, 'matchId', 'match_id') ?? '')}
                                pagination={{ pageSize: 10 }}
                                locale={{ emptyText: 'No matches found' }}
                                expandable={{
                                    expandedRowRender,
                                    onExpand: (expanded, record) => {
                                        if (expanded) fetchH2H(record)
                                    }
                                }}
                                size="middle"
                                sticky
                                scroll={{ x: 1200 }}
                            />
                        </Card>
                    )
                })
            )}
        </div>
    )
}

export default TennisEnhancedPredictionsPage
