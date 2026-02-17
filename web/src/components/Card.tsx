import Card from 'antd/es/card'
import Row from 'antd/es/row'
import Col from 'antd/es/col'

// -----------------------------
// Helpers (handle camel OR snake keys)
// -----------------------------
const g = (obj: any, camel: string, snake: string) => obj?.[camel] ?? obj?.[snake]

const fmtNum = (v: any, digits = 1) => (v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(digits))
const fmtPct = (v: any, digits = 1) => (v == null || Number.isNaN(v) ? '—' : (Number(v) * 100).toFixed(digits) + '%')
const fmtAmerican = (v: any) => {
    if (v == null || Number.isNaN(v)) return '—'
    const n = Number(v)
    if (!Number.isFinite(n) || n === 0) return '—'
    return n > 0 ? `+${Math.trunc(n)}` : `${Math.trunc(n)}`
}

const renderParlayCard = (p: any, title: string, key: string) => {
    if (!p || !Array.isArray(g(p, 'legs', 'legs')) || g(p, 'legs', 'legs').length === 0) return null

    const legs = g(p, 'legs', 'legs')
    const parlayAmerican = g(p, 'parlayAmerican', 'parlay_american')
    const parlayDecimal = g(p, 'parlayDecimal', 'parlay_decimal')
    const winProb = g(p, 'winProb', 'win_prob')
    const ev = g(p, 'ev', 'ev')

    return (
        <Col key={key} xs={24} sm={12} lg={8}>
            <Card title={title} variant="borderless" style={{ width: '100%' }} bodyStyle={{ paddingTop: 8 }}>
                <div className="text-sm text-gray-200">
                    {legs.map((l: any, idx: number) => (
                        <div key={idx} className="mb-1">
                            • {g(l, 'pick', 'pick')} ({fmtAmerican(g(l, 'oddsAmerican', 'odds_american'))}) — edge{' '}
                            {(Number(g(l, 'edge', 'edge')) * 100).toFixed(1)}%
                        </div>
                    ))}

                    <div className="mt-2 text-gray-300">
                        <div>
                            Approx payout: {parlayAmerican == null ? '—' : fmtAmerican(parlayAmerican)} (decimal {Number(parlayDecimal).toFixed(2)})
                        </div>
                        <div>
                            Win% {fmtPct(winProb, 1)} • EV {fmtNum(ev, 3)}
                        </div>
                    </div>
                </div>
            </Card>
        </Col>
    )
}

export const renderParlayCards = (list: any[], titlePrefix: string) => {
    if (!Array.isArray(list) || list.length === 0) return null

    return <Row gutter={[12, 12]}>{list.map((p, i) => renderParlayCard(p, `${titlePrefix} #${i + 1}`, `${titlePrefix}-${i}`))}</Row>
}
