import React, { useEffect, useMemo, useState } from 'react'
import { Alert, Button, Card, Divider, InputNumber, Select, Spin, Tooltip } from 'antd'
import { FundOutlined, ReloadOutlined } from '@ant-design/icons'
import { fetchSuggestedParlays } from '../features/tennis/api'
import { fmtAmericanOdds, fmtNum, fmtPct } from '../features/tennis/formatters'
import { ParlaySuggestion } from '../features/tennis/types'

const toDecimalFromAmerican = (odds: number | null | undefined) => {
    if (odds == null || !Number.isFinite(odds) || odds === 0) return null
    return odds > 0 ? 1 + odds / 100 : 1 + 100 / Math.abs(odds)
}
const toAmericanFromDecimal = (dec: number | null | undefined) => {
    if (dec == null || !Number.isFinite(dec) || dec <= 1) return null
    if (dec >= 2) return Math.round((dec - 1) * 100)
    return Math.round(-100 / (dec - 1))
}
const impliedProbFromDecimal = (dec: number | null | undefined) => {
    if (dec == null || !Number.isFinite(dec) || dec <= 1) return null
    return 1 / dec
}

const TennisParlaysPage: React.FC = () => {
    const [parlays, setParlays] = useState<ParlaySuggestion[] | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    // controls (match your API params)
    const [legs, setLegs] = useState(3)
    const [topN, setTopN] = useState(5)
    const [daysAhead, setDaysAhead] = useState(1)
    const [objective, setObjective] = useState<'win_prob' | 'ev'>('win_prob')
    const [maxOverlap, setMaxOverlap] = useState(0)
    const [minLegOdds, setMinLegOdds] = useState(-500)
    const [minParlayPayout, setMinParlayPayout] = useState(-125)
    const [candidatePool, setCandidatePool] = useState(200)

    const params = useMemo(
        () => ({
            legs,
            top_n: topN,
            days_ahead: daysAhead,
            include_incomplete: true,
            objective,
            max_overlap: maxOverlap,
            min_leg_odds: minLegOdds,
            min_parlay_payout: minParlayPayout,
            candidate_pool: candidatePool
        }),
        [legs, topN, daysAhead, objective, maxOverlap, minLegOdds, minParlayPayout, candidatePool]
    )

    const fetchParlays = async () => {
        try {
            setLoading(true)
            setError(null)
            const normalized = await fetchSuggestedParlays(params)
            setParlays(normalized)
        } catch (e: unknown) {
            setError((e as { message?: string })?.message || 'Failed to load parlay suggestions')
            setParlays(null)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchParlays()
    }, [
        params.legs,
        params.top_n,
        params.days_ahead,
        params.objective,
        params.max_overlap,
        params.min_leg_odds,
        params.min_parlay_payout,
        params.candidate_pool
    ])

    return (
        <div className="p-4 max-w-7xl mx-auto">
            <h1 className="text-3xl font-bold mb-6 text-white">
                <FundOutlined className="mr-2" />
                Parlays
            </h1>

            <Card
                className="mb-6"
                title="Parlay Builder Controls"
                extra={
                    <Button icon={<ReloadOutlined />} onClick={fetchParlays}>
                        Refresh
                    </Button>
                }
            >
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                        <div className="text-xs text-gray-400 mb-1">Legs</div>
                        <InputNumber min={2} max={8} value={legs} onChange={(v) => setLegs(Number(v ?? 3))} className="w-full" />
                    </div>
                    <div>
                        <div className="text-xs text-gray-400 mb-1">Top N</div>
                        <InputNumber min={1} max={25} value={topN} onChange={(v) => setTopN(Number(v ?? 5))} className="w-full" />
                    </div>
                    <div>
                        <div className="text-xs text-gray-400 mb-1">Days ahead</div>
                        <InputNumber min={0} max={7} value={daysAhead} onChange={(v) => setDaysAhead(Number(v ?? 1))} className="w-full" />
                    </div>

                    <div>
                        <div className="text-xs text-gray-400 mb-1">Objective</div>
                        <Select
                            value={objective}
                            onChange={(v) => setObjective(v)}
                            className="w-full"
                            options={[
                                { value: 'win_prob', label: 'Max win probability' },
                                { value: 'ev', label: 'Max EV' }
                            ]}
                        />
                    </div>
                    <div>
                        <div className="text-xs text-gray-400 mb-1">Max overlap</div>
                        <InputNumber min={0} max={legs - 1} value={maxOverlap} onChange={(v) => setMaxOverlap(Number(v ?? 0))} className="w-full" />
                    </div>
                    <div>
                        <div className="text-xs text-gray-400 mb-1">Candidate pool</div>
                        <InputNumber min={25} max={1000} value={candidatePool} onChange={(v) => setCandidatePool(Number(v ?? 200))} className="w-full" />
                    </div>

                    <div>
                        <div className="text-xs text-gray-400 mb-1">Min leg odds (American)</div>
                        <InputNumber value={minLegOdds} onChange={(v) => setMinLegOdds(Number(v ?? -500))} className="w-full" />
                    </div>
                    <div>
                        <div className="text-xs text-gray-400 mb-1">Min parlay payout (American)</div>
                        <InputNumber value={minParlayPayout} onChange={(v) => setMinParlayPayout(Number(v ?? -125))} className="w-full" />
                    </div>
                </div>

                <Divider className="my-4" />

                <div className="text-xs text-gray-400">
                    Tip: If this page feels same-y, switch objective to EV and raise min payout, or raise max overlap.
                </div>
            </Card>

            {loading ? (
                <div className="flex justify-center py-10">
                    <Spin />
                </div>
            ) : error ? (
                <Alert type="error" message="Parlay suggestions failed" description={error} />
            ) : !parlays || parlays.length === 0 ? (
                <Alert type="info" message="No parlays returned" description="Try lowering min payout, increasing candidate pool, or allowing overlap." />
            ) : (
                <Card title="Suggested Parlays">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        {parlays.map((p, idx) => {
                            const decimal = Number.isFinite(p.parlay_decimal) ? p.parlay_decimal : toDecimalFromAmerican(p.parlay_american)
                            const american = p.parlay_american ?? toAmericanFromDecimal(p.parlay_decimal)
                            const impliedWinProb = impliedProbFromDecimal(decimal)
                            const winProb = p.win_prob ?? impliedWinProb

                            return (
                            <div key={idx} className="border rounded-xl p-4 bg-white/5">
                                <div className="flex items-center justify-between gap-4">
                                    <div className="text-sm text-gray-400">Parlay #{idx + 1}</div>
                                    <div className="text-xs text-gray-400">{p.legs?.length ?? 0} legs</div>
                                </div>

                                <div className="mt-3 grid grid-cols-2 gap-3">
                                    <div className="rounded-lg border border-white/10 p-3">
                                        <div className="text-xs text-gray-400">Payout (American)</div>
                                        <div className="text-lg font-semibold">{fmtAmericanOdds(american)}</div>
                                    </div>
                                    <div className="rounded-lg border border-white/10 p-3">
                                        <div className="text-xs text-gray-400">Payout (Decimal)</div>
                                        <div className="text-lg font-semibold">{fmtNum(decimal, 3)}</div>
                                    </div>
                                    <div className="rounded-lg border border-white/10 p-3">
                                        <div className="text-xs text-gray-400">Win Prob</div>
                                        <div className="text-lg font-semibold flex items-baseline gap-2">
                                            <span>{winProb == null ? '--' : fmtPct(winProb)}</span>
                                            {winProb != null && p.win_prob == null ? (
                                                <Tooltip title="Implied from payout">
                                                    <span className="text-[10px] uppercase tracking-wide text-gray-500">implied</span>
                                                </Tooltip>
                                            ) : null}
                                        </div>
                                    </div>
                                    <div className="rounded-lg border border-white/10 p-3">
                                        <div className="text-xs text-gray-400">EV</div>
                                        <div className="text-lg font-semibold">{fmtNum(p.ev, 4)}</div>
                                    </div>
                                </div>

                                <div className="mt-4">
                                    <div className="text-xs text-gray-400 mb-2">Legs</div>
                                    <div className="space-y-2">
                                        {p.legs?.map((leg, j) => (
                                            <div key={j} className="rounded-lg border border-white/10 p-3">
                                                <div className="text-sm font-medium">{leg.pick}</div>
                                                <div className="mt-2 flex flex-wrap gap-3 text-xs">
                                                    <span className="text-blue-400">odds {fmtAmericanOdds(leg.odds_american)}</span>
                                                    <Tooltip title="Model probability for this leg">
                                                        <span className="text-green-400">model {fmtPct(leg.model_prob)}</span>
                                                    </Tooltip>
                                                    <Tooltip title="No-vig probability implied by market">
                                                        <span className="text-gray-300">mkt {fmtPct(leg.no_vig_prob)}</span>
                                                    </Tooltip>
                                                    <span className="text-gray-400">edge {fmtNum(leg.edge, 4)}</span>
                                                </div>
                                                {leg.summary ? (
                                                    <div className="mt-2 text-xs text-gray-300 leading-relaxed">
                                                        <span className="text-[10px] uppercase tracking-wide text-gray-500 mr-1">Why:</span>
                                                        {leg.summary}
                                                    </div>
                                                ) : null}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                            )
                        })}
                    </div>
                </Card>
            )}
        </div>
    )
}

export default TennisParlaysPage
