import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Alert, Card, Collapse, Spin, Table, Tag } from 'antd'
import { useLocation, useNavigate } from 'react-router-dom'
import { ROUTES } from '../resources/routes-constants'
import { fetchH2HByEvent, fetchPlayerCompare } from '../features/tennis/api'
import { diffFmt, displayValue, fmtNum, fmtPct, getField, last10MethodColor, toNumOrNull } from '../features/tennis/formatters'
import { CompareResponse, H2HMatch, Snapshot, UiState } from '../features/tennis/types'

const h2hWinnerDisplay = (r: any): string => {
    const winnerName = getField(r, 'winner_name', 'winnerName')
    if (winnerName) return displayValue(winnerName)
    const wc = toNumOrNull(getField(r, 'winner_code', 'winnerCode'))
    const home = getField(r, 'home') ?? getField(r, 'p1_name', 'p1Name')
    const away = getField(r, 'away') ?? getField(r, 'p2_name', 'p2Name')
    if (wc === 1) return displayValue(home)
    if (wc === 2) return displayValue(away)
    return '-'
}

const TennisPlayerComparePage: React.FC = () => {
    const location = useLocation()
    const navigate = useNavigate()
    const query = useMemo(() => new URLSearchParams(location.search), [location.search])
    const player1 = (query.get('player1') || '').trim()
    const player2 = (query.get('player2') || '').trim()
    const tour = (query.get('tour') || 'ATP').toUpperCase()
    const surface = (query.get('surface') || 'hard').toLowerCase()
    const eventId = Number(query.get('event_id') || 0)

    const [uiState, setUiState] = useState<UiState>('idle')
    const [error, setError] = useState<string | null>(null)
    const [data, setData] = useState<CompareResponse | null>(null)
    const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
    const [h2hRows, setH2hRows] = useState<H2HMatch[]>([])
    const requestSeq = useRef(0)

    useEffect(() => {
        const fetchCompare = async () => {
            const reqId = ++requestSeq.current
            if (!player1 || !player2) {
                setUiState('error')
                setError('Missing player1 or player2 query param')
                setData(null)
                return
            }
            setUiState('loading')
            setError(null)
            setData(null)
            const reqTs = Date.now()
            try {
                const res = await fetchPlayerCompare({
                    player1,
                    player2,
                    tour,
                    surface,
                    event_id: eventId || undefined,
                    debug: true,
                    req_ts: reqTs
                })
                if (reqId !== requestSeq.current) return
                setData(res)
                setH2hRows(
                    Array.isArray(getField(getField(res, 'h2h') || {}, 'last_10_matchups', 'last10Matchups'))
                        ? (getField(getField(res, 'h2h') || {}, 'last_10_matchups', 'last10Matchups') as H2HMatch[])
                        : []
                )
                if (eventId > 0) {
                    try {
                        const h2hRes = await fetchH2HByEvent(eventId)
                        if (reqId === requestSeq.current && Array.isArray(h2hRes)) setH2hRows(h2hRes)
                    } catch {
                        // keep DB fallback rows from compare response
                    }
                }
                setSnapshot({
                    query: `/tennis/players/compare?${new URLSearchParams({
                        player1,
                        player2,
                        tour,
                        surface,
                        event_id: String(eventId || ''),
                        debug: 'true',
                        req_ts: String(reqTs)
                    }).toString()}`,
                    requested_at_iso: new Date(reqTs).toISOString(),
                    request_id: reqId
                })
                setUiState('loaded')
            } catch (e: unknown) {
                if (reqId !== requestSeq.current) return
                const err = e as { response?: { data?: { detail?: string } }; message?: string }
                setError(err?.response?.data?.detail || err?.message || 'Failed to load compare')
                setUiState('error')
            }
        }
        fetchCompare()
    }, [location.search, player1, player2, tour, surface, eventId])

    if (uiState === 'idle' || uiState === 'loading') {
        return (
            <div className="p-6">
                <Spin />
            </div>
        )
    }

    const left = getField(data, 'players') ? (getField(getField(data, 'players') || {}, 'left') || {}) : {}
    const right = getField(data, 'players') ? (getField(getField(data, 'players') || {}, 'right') || {}) : {}
    const compare = getField(data, 'compare') || {}
    const h2h = getField(data, 'h2h') || {}
    const leftLast10 = Array.isArray(getField(getField(data, 'last_10_matches', 'last10Matches') || {}, 'left'))
        ? ((getField(getField(data, 'last_10_matches', 'last10Matches') || {}, 'left') as unknown[]) ?? [])
        : []
    const rightLast10 = Array.isArray(getField(getField(data, 'last_10_matches', 'last10Matches') || {}, 'right'))
        ? ((getField(getField(data, 'last_10_matches', 'last10Matches') || {}, 'right') as unknown[]) ?? [])
        : []
    const leftLast10Res = getField(getField(data, 'last_10_resolution', 'last10Resolution') || {}, 'left') || {}
    const rightLast10Res = getField(getField(data, 'last_10_resolution', 'last10Resolution') || {}, 'right') || {}
    const quality = getField(data, 'quality') || {}
    const leftName = displayValue(getField(left, 'player'))
    const rightName = displayValue(getField(right, 'player'))
    const h2hOverall = getField(h2h, 'overall') || {}
    const h2hSurface = getField(h2h, 'surface') || {}
    const h2hRecord = getField(h2h, 'record') || {}

    const summaryRows = [
        {
            key: 'elo',
            metric: 'Overall ELO',
            left: fmtNum(getField(getField(left, 'elo_latest', 'eloLatest'), 'elo'), 1),
            right: fmtNum(getField(getField(right, 'elo_latest', 'eloLatest'), 'elo'), 1),
            diff: diffFmt(getField(getField(compare, 'elo') || {}, 'elo'))
        },
        {
            key: 'wr20',
            metric: 'Win Rate Last 20',
            left: fmtPct(getField(getField(left, 'rolling_stats', 'rollingStats'), 'win_rate_last_20', 'winRateLast20')),
            right: fmtPct(getField(getField(right, 'rolling_stats', 'rollingStats'), 'win_rate_last_20', 'winRateLast20')),
            diff: diffFmt(getField(getField(compare, 'rolling') || {}, 'win_rate_last_20', 'winRateLast20'), true)
        },
        {
            key: 'svc',
            metric: 'Current Surface Service Points Won',
            left: fmtPct(getField(getField(left, 'current_surface_rates', 'currentSurfaceRates'), 'service_points_won', 'servicePointsWon')),
            right: fmtPct(getField(getField(right, 'current_surface_rates', 'currentSurfaceRates'), 'service_points_won', 'servicePointsWon')),
            diff: diffFmt(getField(getField(compare, 'current_surface_rates', 'currentSurfaceRates') || {}, 'service_points_won', 'servicePointsWon'), true)
        },
        {
            key: 'ret',
            metric: 'Current Surface Return Points Won',
            left: fmtPct(getField(getField(left, 'current_surface_rates', 'currentSurfaceRates'), 'return_points_won', 'returnPointsWon')),
            right: fmtPct(getField(getField(right, 'current_surface_rates', 'currentSurfaceRates'), 'return_points_won', 'returnPointsWon')),
            diff: diffFmt(getField(getField(compare, 'current_surface_rates', 'currentSurfaceRates') || {}, 'return_points_won', 'returnPointsWon'), true)
        }
    ]

    const surfaceRows = (surf: string, window: 'row_12m' | 'row_all_time') => {
        const l = getField(getField(getField(left, 'surface_rows', 'surfaceRows') || {}, surf) || {}, window, window === 'row_12m' ? 'row12m' : 'rowAllTime') || {}
        const r = getField(getField(getField(right, 'surface_rows', 'surfaceRows') || {}, surf) || {}, window, window === 'row_12m' ? 'row12m' : 'rowAllTime') || {}
        const d = getField(getField(getField(compare, 'surface_rows', 'surfaceRows') || {}, surf) || {}, window, window === 'row_12m' ? 'row12m' : 'rowAllTime') || {}
        return [
            {
                key: 'svc_pts',
                metric: 'Service Points Won',
                left: fmtPct(getField(l, 'svc_pts')),
                right: fmtPct(getField(r, 'svc_pts')),
                diff: diffFmt(getField(d, 'svc_pts'), true)
            },
            {
                key: 'ret_pts',
                metric: 'Return Points Won',
                left: fmtPct(getField(l, 'ret_pts')),
                right: fmtPct(getField(r, 'ret_pts')),
                diff: diffFmt(getField(d, 'ret_pts'), true)
            },
            {
                key: 'hold',
                metric: 'Hold %',
                left: fmtPct(getField(l, 'svc_hold_pct', 'svcHoldPct')),
                right: fmtPct(getField(r, 'svc_hold_pct', 'svcHoldPct')),
                diff: diffFmt(getField(d, 'svc_hold_pct', 'svcHoldPct'), true)
            },
            {
                key: 'bp_win',
                metric: 'Break Point Win %',
                left: fmtPct(getField(l, 'ret_bp_win_pct', 'retBpWinPct')),
                right: fmtPct(getField(r, 'ret_bp_win_pct', 'retBpWinPct')),
                diff: diffFmt(getField(d, 'ret_bp_win_pct', 'retBpWinPct'), true)
            }
        ]
    }

    return (
        <div className="p-4 md:p-6 space-y-4">
            <div className="flex items-center justify-between gap-3 flex-wrap">
                <h1 className="text-xl font-semibold">Player Compare</h1>
                <Tag color="blue">
                    {tour} | {surface}
                </Tag>
            </div>

            {uiState === 'error' ? (
                <Alert
                    type="error"
                    showIcon
                    message="Could not load comparison"
                    description={displayValue(error)}
                    action={
                        <button
                            type="button"
                            className="bg-transparent border-0 p-0 cursor-pointer underline"
                            onClick={() => navigate(ROUTES.TENNIS_ENHANCED_PREDICTIONS_ROUTE)}
                            style={{ color: '#91caff' }}
                        >
                            Back to predictions
                        </button>
                    }
                />
            ) : null}

            {getField(quality, 'partial_data', 'partialData') ? <Alert type="warning" showIcon message="Partial data for one or both players" /> : null}

            {uiState === 'loaded' ? (
                <>
                    <div className="grid grid-cols-2 lg:grid-cols-2 gap-4">
                        <Card title={displayValue(getField(left, 'player'))}>
                            <div className="text-sm">TA ID: {displayValue(getField(getField(left, 'ids') || {}, 'ta_id', 'taId'))}</div>
                            <div className="text-sm">Rank: {displayValue(getField(getField(left, 'elo_latest', 'eloLatest') || {}, 'official_rank', 'officialRank'))}</div>
                            <div className="text-sm">Style: {displayValue(getField(getField(left, 'style') || {}, 'label'))}</div>
                        </Card>
                        <Card title={displayValue(getField(right, 'player'))}>
                            <div className="text-sm">TA ID: {displayValue(getField(getField(right, 'ids') || {}, 'ta_id', 'taId'))}</div>
                            <div className="text-sm">Rank: {displayValue(getField(getField(right, 'elo_latest', 'eloLatest') || {}, 'official_rank', 'officialRank'))}</div>
                            <div className="text-sm">Style: {displayValue(getField(getField(right, 'style') || {}, 'label'))}</div>
                        </Card>
                    </div>

                    <Card title="Comparison Summary">
                        <Table
                            size="small"
                            pagination={false}
                            dataSource={summaryRows}
                            columns={[
                                { title: 'Metric', dataIndex: 'metric', key: 'metric' },
                                { title: leftName, dataIndex: 'left', key: 'left' },
                                { title: rightName, dataIndex: 'right', key: 'right' },
                                { title: 'Diff (P1-P2)', dataIndex: 'diff', key: 'diff' }
                            ]}
                        />
                    </Card>

                    {['hard', 'clay', 'grass'].map((surf) => (
                        <Card key={surf} title={`${surf.toUpperCase()} Surface`}>
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                                <Card size="small" title="12 Month">
                                    <Table
                                        size="small"
                                        pagination={false}
                                        dataSource={surfaceRows(surf, 'row_12m')}
                                        columns={[
                                            { title: 'Metric', dataIndex: 'metric', key: 'metric' },
                                            { title: leftName, dataIndex: 'left', key: 'left' },
                                            { title: rightName, dataIndex: 'right', key: 'right' },
                                            { title: 'Diff', dataIndex: 'diff', key: 'diff' }
                                        ]}
                                    />
                                </Card>
                                <Card size="small" title="All Time">
                                    <Table
                                        size="small"
                                        pagination={false}
                                        dataSource={surfaceRows(surf, 'row_all_time')}
                                        columns={[
                                            { title: 'Metric', dataIndex: 'metric', key: 'metric' },
                                            { title: leftName, dataIndex: 'left', key: 'left' },
                                            { title: rightName, dataIndex: 'right', key: 'right' },
                                            { title: 'Diff', dataIndex: 'diff', key: 'diff' }
                                        ]}
                                    />
                                </Card>
                            </div>
                        </Card>
                    ))}

                    <Card title="H2H Last 10 Matchups">
                        <div className="mb-3 text-sm text-gray-300">
                            <span className="mr-4">
                                Overall Record ({leftName}-{rightName}):{' '}
                                {displayValue(getField(h2hRecord, 'overall')) !== '-'
                                    ? displayValue(getField(h2hRecord, 'overall'))
                                    : `${displayValue(getField(h2hOverall, 'p1_wins'))}-${displayValue(getField(h2hOverall, 'p2_wins'))}`}
                            </span>
                            <span>
                                {surface.toUpperCase()} Record: {displayValue(getField(h2hRecord, 'surface')) !== '-'
                                    ? displayValue(getField(h2hRecord, 'surface'))
                                    : `${displayValue(getField(h2hSurface, 'p1_wins'))}-${displayValue(getField(h2hSurface, 'p2_wins'))}`}
                            </span>
                        </div>
                        <Table
                            size="small"
                            pagination={false}
                            rowKey={(r: unknown, i?: number) => `${getField(r, 'date') || getField(r, 'match_date', 'matchDate') || 'x'}_${getField(r, 'tournament') || 'x'}_${i ?? 0}`}
                            dataSource={h2hRows}
                            columns={[
                                { title: 'Date', key: 'date', render: (_: unknown, r: unknown) => displayValue(getField(r, 'date') ?? getField(r, 'match_date', 'matchDate')) },
                                { title: 'Tournament', key: 'tournament', render: (_: unknown, r: unknown) => displayValue(getField(r, 'tournament')) },
                                { title: 'Round', key: 'round', render: (_: unknown, r: unknown) => displayValue(getField(r, 'round')) },
                                { title: 'Surface', key: 'surface', render: (_: unknown, r: unknown) => displayValue(getField(r, 'surface')) },
                                {
                                    title: `${leftName} vs ${rightName}`,
                                    key: 'players',
                                    render: (_: unknown, r: unknown) =>
                                        `${displayValue(getField(r, 'home') ?? getField(r, 'p1_name', 'p1Name'))} vs ${displayValue(getField(r, 'away') ?? getField(r, 'p2_name', 'p2Name'))}`
                                },
                                {
                                    title: 'Result',
                                    key: 'result',
                                    render: (_: unknown, r: unknown) => displayValue(getField(r, 'score') ?? getField(r, 'winner_name', 'winnerName'))
                                },
                                {
                                    title: 'Winner',
                                    key: 'winner',
                                    render: (_: unknown, r: unknown) => h2hWinnerDisplay(r)
                                }
                            ]}
                        />
                    </Card>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        <Card title={`${leftName} Last 10 Matches`}>
                            <div className="mb-2">
                                <Tag color={last10MethodColor(getField(leftLast10Res, 'method'))}>{displayValue(getField(leftLast10Res, 'method'))}</Tag>
                                <span className="text-xs text-gray-400 ml-2">reason: {displayValue(getField(leftLast10Res, 'reason'))}</span>
                            </div>
                            <Table
                                size="small"
                                pagination={false}
                                rowKey={(r: unknown, i?: number) => `${getField(r, 'match_date', 'matchDate') || 'x'}_${getField(r, 'tournament') || 'x'}_${i ?? 0}`}
                                dataSource={leftLast10}
                                columns={[
                                    { title: 'Date', key: 'date', render: (_: unknown, r: unknown) => displayValue(getField(r, 'match_date', 'matchDate')) },
                                    { title: 'Opponent', key: 'opponent', render: (_: unknown, r: unknown) => displayValue(getField(r, 'opponent_name', 'opponentName')) },
                                    { title: 'Score', key: 'score', render: (_: unknown, r: unknown) => displayValue(getField(r, 'score')) },
                                    { title: 'Surface', key: 'surface', render: (_: unknown, r: unknown) => displayValue(getField(r, 'surface')) },
                                    { title: 'Winner', key: 'winner', render: (_: unknown, r: unknown) => displayValue(getField(r, 'winner_name', 'winnerName')) }
                                ]}
                            />
                            {!leftLast10.length ? <div className="text-xs text-gray-400 mt-2">No finished matches found for this player.</div> : null}
                        </Card>
                        <Card title={`${rightName} Last 10 Matches`}>
                            <div className="mb-2">
                                <Tag color={last10MethodColor(getField(rightLast10Res, 'method'))}>{displayValue(getField(rightLast10Res, 'method'))}</Tag>
                                <span className="text-xs text-gray-400 ml-2">reason: {displayValue(getField(rightLast10Res, 'reason'))}</span>
                            </div>
                            <Table
                                size="small"
                                pagination={false}
                                rowKey={(r: unknown, i?: number) => `${getField(r, 'match_date', 'matchDate') || 'x'}_${getField(r, 'tournament') || 'x'}_${i ?? 0}`}
                                dataSource={rightLast10}
                                columns={[
                                    { title: 'Date', key: 'date', render: (_: unknown, r: unknown) => displayValue(getField(r, 'match_date', 'matchDate')) },
                                    { title: 'Opponent', key: 'opponent', render: (_: unknown, r: unknown) => displayValue(getField(r, 'opponent_name', 'opponentName')) },
                                    { title: 'Score', key: 'score', render: (_: unknown, r: unknown) => displayValue(getField(r, 'score')) },
                                    { title: 'Surface', key: 'surface', render: (_: unknown, r: unknown) => displayValue(getField(r, 'surface')) },
                                    { title: 'Winner', key: 'winner', render: (_: unknown, r: unknown) => displayValue(getField(r, 'winner_name', 'winnerName')) }
                                ]}
                            />
                            {!rightLast10.length ? <div className="text-xs text-gray-400 mt-2">No finished matches found for this player.</div> : null}
                        </Card>
                    </div>

                    <Card title="Debug">
                        <Collapse
                            items={[
                                {
                                    key: '1',
                                    label: 'Response Snapshot',
                                    children: (
                                        <div className="space-y-2 text-xs text-gray-400">
                                            <div>request_url: {displayValue(snapshot?.query)}</div>
                                            <div>requested_at: {displayValue(snapshot?.requested_at_iso)}</div>
                                            <div>request_id: {displayValue(snapshot?.request_id)}</div>
                                            <div>both_resolved: {String(!!getField(quality, 'both_resolved', 'bothResolved'))}</div>
                                            <div>partial_data: {String(!!getField(quality, 'partial_data', 'partialData'))}</div>
                                            <div>any_ambiguous: {String(!!getField(quality, 'any_ambiguous', 'anyAmbiguous'))}</div>
                                        </div>
                                    )
                                }
                            ]}
                        />
                    </Card>
                </>
            ) : null}
        </div>
    )
}

export default TennisPlayerComparePage
