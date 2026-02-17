import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Alert, Card, Collapse, Descriptions, Spin, Table, Tag } from 'antd'
import { useLocation, useNavigate } from 'react-router-dom'
import { ROUTES } from '../resources/routes-constants'
import { fetchPlayerStats } from '../features/tennis/api'
import { displayValue, fmtNum, fmtPct, getField, methodColor } from '../features/tennis/formatters'
import { LookupSource, PlayerStatsResponse, Snapshot, UiState } from '../features/tennis/types'

const TennisPlayerStatsPage: React.FC = () => {
    const location = useLocation()
    const navigate = useNavigate()
    const [uiState, setUiState] = useState<UiState>('idle')
    const [error, setError] = useState<string | null>(null)
    const [data, setData] = useState<PlayerStatsResponse | null>(null)
    const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
    const requestSeq = useRef(0)

    const query = useMemo(() => new URLSearchParams(location.search), [location.search])
    const player = (query.get('player') || '').trim()
    const tour = (query.get('tour') || 'ATP').toUpperCase()
    const surface = (query.get('surface') || 'hard').toLowerCase()

    useEffect(() => {
        const fetchStats = async () => {
            const reqId = ++requestSeq.current
            if (!player) {
                setUiState('error')
                setError('Missing player query param')
                setData(null)
                return
            }

            setUiState('loading')
            setError(null)
            setData(null)

            const reqTs = Date.now()
            const params = { player, tour, surface, debug: true, req_ts: reqTs }

            try {
                const normalized = await fetchPlayerStats(params)
                if (reqId !== requestSeq.current) return
                setData(normalized)
                setSnapshot({
                    query: `/tennis/players/stats?${new URLSearchParams({
                        player,
                        tour,
                        surface,
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
                setError(err?.response?.data?.detail || err?.message || 'Failed to load player stats')
                setUiState('error')
            }
        }
        fetchStats()
    }, [location.search, player, tour, surface])

    const surfaceRows = useMemo(() => {
        const src = (data?.surface_rows || {}) as Record<string, { row_12m?: unknown; row_all_time?: unknown }>
        return ['hard', 'clay', 'grass'].map((s) => ({
            key: s,
            surface: s,
            row12: src?.[s]?.row_12m || null,
            rowAll: src?.[s]?.row_all_time || null
        }))
    }, [data])

    const surfaceMetricRows = (surfaceKey: string, row12: any, rowAll: any) => {
        const s = String(surfaceKey || '').toLowerCase()
        const surfaceElo =
            s === 'hard'
                ? data?.elo_latest?.helo
                : s === 'clay'
                  ? data?.elo_latest?.celo
                  : s === 'grass'
                    ? data?.elo_latest?.gelo
                    : null
        return [
            { key: 'surface_elo', metric: 'Surface ELO', m12: fmtNum(surfaceElo, 1), all: '-' },
            { key: 'overall_elo', metric: 'Overall ELO', m12: fmtNum(data?.elo_latest?.elo, 1), all: '-' },
            { key: 'svc_pts', metric: 'Service Points Won', m12: fmtPct(row12?.svc_pts), all: fmtPct(rowAll?.svc_pts) },
            { key: 'ret_pts', metric: 'Return Points Won', m12: fmtPct(row12?.ret_pts), all: fmtPct(rowAll?.ret_pts) },
            { key: 'hold', metric: 'Hold %', m12: fmtPct(row12?.svc_hold_pct), all: fmtPct(rowAll?.svc_hold_pct) },
            { key: 'bp_win', metric: 'Break Point Win %', m12: fmtPct(row12?.ret_bp_win_pct), all: fmtPct(rowAll?.ret_bp_win_pct) },
            { key: 'aces', metric: 'Aces / Game', m12: fmtNum(row12?.svc_aces_pg, 2), all: fmtNum(rowAll?.svc_aces_pg, 2) },
            { key: 'dfs', metric: 'Double Faults / Game', m12: fmtNum(row12?.svc_dfs_pg, 2), all: fmtNum(rowAll?.svc_dfs_pg, 2) }
        ]
    }

    const statusReason = (k: LookupSource) => {
        const matched = !!data?.match_status?.[k]
        if (matched) return null
        const detail = data?.match_status_detail?.[k]
        return detail ? String(detail) : 'not_found'
    }

    const resolutionTag = (k: LookupSource) => {
        const v = data?.resolution?.[k]
        if (!v) return <Tag color="default">not_found</Tag>
        return <Tag color={methodColor(v)}>{String(v)}</Tag>
    }

    if (uiState === 'loading' || uiState === 'idle') {
        return (
            <div className="p-6">
                <Spin />
            </div>
        )
    }

    return (
        <div className="p-4 md:p-6 space-y-4">
            <div className="flex items-center justify-between gap-3 flex-wrap">
                <h1 className="text-xl font-semibold">Player Stats</h1>
                <Tag color="blue">
                    {tour} | {surface}
                </Tag>
            </div>

            {uiState === 'error' ? (
                <Alert
                    type="error"
                    showIcon
                    message="Could not load player stats"
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

            {data && uiState === 'loaded' ? (
                <>
                    <Card title={displayValue(data.player || player)}>
                        <Descriptions size="small" column={2} bordered>
                            <Descriptions.Item label="Tour">{displayValue(data.tour || tour)}</Descriptions.Item>
                            <Descriptions.Item label="TA ID">{displayValue(data.ids?.ta_id)}</Descriptions.Item>
                            <Descriptions.Item label="ELO">
                                <div>{fmtNum(data.elo_latest?.elo, 1)}</div>
                                <div className="mt-1">{resolutionTag('elo_latest')}</div>
                                {statusReason('elo_latest') ? <div className="text-xs text-gray-400 mt-1">reason: {statusReason('elo_latest')}</div> : null}
                            </Descriptions.Item>
                            <Descriptions.Item label="Official Rank">{displayValue(data.elo_latest?.official_rank)}</Descriptions.Item>
                            <Descriptions.Item label="Service Points Won">
                                <div>{fmtPct(data.current_surface_rates?.service_points_won)}</div>
                                <div className="mt-1">{resolutionTag('csv_rows')}</div>
                                {statusReason('csv_rows') ? <div className="text-xs text-gray-400 mt-1">reason: {statusReason('csv_rows')}</div> : null}
                            </Descriptions.Item>
                            <Descriptions.Item label="Return Points Won">{fmtPct(data.current_surface_rates?.return_points_won)}</Descriptions.Item>
                            <Descriptions.Item label="Style">{displayValue(data.style?.label)}</Descriptions.Item>
                            <Descriptions.Item label="Rolling Win Rate (Last 20)">
                                <div>{fmtPct(data.rolling_stats?.win_rate_last_20)}</div>
                                <div className="mt-1">{resolutionTag('rolling')}</div>
                                {statusReason('rolling') ? <div className="text-xs text-gray-400 mt-1">reason: {statusReason('rolling')}</div> : null}
                            </Descriptions.Item>
                        </Descriptions>
                    </Card>

                    <Card title="Surface Stats (12m vs All-Time)">
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                            {surfaceRows.map((r: any) => (
                                <Card key={r.surface} size="small" title={String(r.surface).toUpperCase()}>
                                    <Table
                                        size="small"
                                        pagination={false}
                                        dataSource={surfaceMetricRows(r?.surface, r?.row12, r?.rowAll)}
                                        columns={[
                                            { title: 'Metric', dataIndex: 'metric', key: 'metric' },
                                            { title: '12m', dataIndex: 'm12', key: 'm12' },
                                            { title: 'All', dataIndex: 'all', key: 'all' }
                                        ]}
                                    />
                                </Card>
                            ))}
                        </div>
                    </Card>

                    <Card title="Lookup Quality">
                        <div className="flex flex-wrap gap-2">
                            {(data.quality_flags.length ? data.quality_flags : ['none']).map((f: string) => (
                                <Tag key={f} color={f === 'none' ? 'default' : 'orange'}>
                                    {f}
                                </Tag>
                            ))}
                        </div>
                    </Card>

                    <Card title="Lookup Debug">
                        <Collapse
                            items={[
                                {
                                    key: '1',
                                    label: 'Show Resolution Details',
                                    children: (
                                        <div className="space-y-3">
                                            <div className="text-xs text-gray-400">stats_path: {displayValue(data.stats_path)}</div>
                                            <div className="text-xs text-gray-400">base_norm: {displayValue(getField(getField(data.debug, 'normalization'), 'base_norm', 'baseNorm') ?? data.normalized?.base)}</div>
                                            <div className="text-xs text-gray-400">stats_norm: {displayValue(getField(getField(data.debug, 'normalization'), 'stats_norm', 'statsNorm') ?? data.normalized?.stats_key)}</div>
                                            <Table
                                                size="small"
                                                pagination={false}
                                                rowKey="source"
                                                dataSource={[
                                                    {
                                                        source: 'csv_rows',
                                                        method: data.resolution?.csv_rows || '-',
                                                        chosen:
                                                            getField((getField(getField(getField(data.debug, 'sources'), 'csv_rows', 'csvRows'), 'per_surface', 'perSurface') as Record<string, unknown> | undefined)?.[surface], 'chosen_name', 'chosenName') ||
                                                            '-',
                                                        reason:
                                                            data.match_status_detail?.csv_rows ||
                                                            getField((getField(getField(getField(data.debug, 'sources'), 'csv_rows', 'csvRows'), 'per_surface', 'perSurface') as Record<string, unknown> | undefined)?.[surface], 'reason') ||
                                                            '-'
                                                    },
                                                    {
                                                        source: 'rolling',
                                                        method: data.resolution?.rolling || '-',
                                                        chosen: getField(getField(data.debug, 'sources'), 'rolling') ? getField(getField(getField(data.debug, 'sources'), 'rolling'), 'chosen_name', 'chosenName') || '-' : '-',
                                                        reason:
                                                            data.match_status_detail?.rolling ||
                                                            (getField(getField(data.debug, 'sources'), 'rolling') ? getField(getField(getField(data.debug, 'sources'), 'rolling'), 'reason') : '-') ||
                                                            '-'
                                                    },
                                                    {
                                                        source: 'elo_latest',
                                                        method: data.resolution?.elo_latest || '-',
                                                        chosen: getField(getField(data.debug, 'sources'), 'elo_latest', 'eloLatest')
                                                            ? getField(getField(getField(data.debug, 'sources'), 'elo_latest', 'eloLatest'), 'chosen_name', 'chosenName') || '-'
                                                            : '-',
                                                        reason:
                                                            data.match_status_detail?.elo_latest ||
                                                            (getField(getField(data.debug, 'sources'), 'elo_latest', 'eloLatest')
                                                                ? getField(getField(getField(data.debug, 'sources'), 'elo_latest', 'eloLatest'), 'reason')
                                                                : '-') ||
                                                            '-'
                                                    }
                                                ]}
                                                columns={[
                                                    { title: 'Source', dataIndex: 'source', key: 'source' },
                                                    {
                                                        title: 'Method',
                                                        dataIndex: 'method',
                                                        key: 'method',
                                                        render: (v: string) => <Tag color={methodColor(v)}>{v}</Tag>
                                                    },
                                                    { title: 'Chosen Name', dataIndex: 'chosen', key: 'chosen' },
                                                    { title: 'Reason', dataIndex: 'reason', key: 'reason' }
                                                ]}
                                            />
                                        </div>
                                    )
                                },
                                {
                                    key: '2',
                                    label: 'Response Snapshot',
                                    children: (
                                        <div className="space-y-2 text-xs text-gray-400">
                                            <div>request_url: {displayValue(snapshot?.query)}</div>
                                            <div>requested_at: {displayValue(snapshot?.requested_at_iso)}</div>
                                            <div>request_id: {displayValue(snapshot?.request_id)}</div>
                                            <div>status_csv_rows: {String(!!data.match_status?.csv_rows)}</div>
                                            <div>status_rolling: {String(!!data.match_status?.rolling)}</div>
                                            <div>status_elo_latest: {String(!!data.match_status?.elo_latest)}</div>
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

export default TennisPlayerStatsPage
