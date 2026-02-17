export const getField = <T = unknown>(obj: unknown, snake: string, camel?: string): T | undefined => {
    if (!obj || typeof obj !== 'object') return undefined
    const source = obj as Record<string, unknown>
    const camelKey = camel || snake.replace(/_([a-z])/g, (_, x: string) => x.toUpperCase())
    return (source[snake] ?? source[camelKey]) as T | undefined
}

export const toNumOrNull = (value: unknown): number | null => {
    if (value === null || value === undefined) return null
    const n = typeof value === 'number' ? value : Number(value)
    return Number.isFinite(n) ? n : null
}

export const toStrOrNull = (value: unknown): string | null => {
    if (value === null || value === undefined) return null
    const text = String(value)
    return text.length ? text : null
}

export const displayValue = (value: unknown): string => {
    if (value === null || value === undefined) return '-'
    if (typeof value === 'string' && !value.trim()) return '-'
    return String(value)
}

export const fmtNum = (value: unknown, digits: number = 2): string => {
    const n = toNumOrNull(value)
    if (n === null) return '-'
    return n.toFixed(digits)
}

export const fmtPct = (value: unknown, digits: number = 1): string => {
    const n = toNumOrNull(value)
    if (n === null) return '-'
    return `${(n * 100).toFixed(digits)}%`
}

export const fmtPp = (value: unknown, digits: number = 1): string => {
    const n = toNumOrNull(value)
    if (n === null) return '-'
    const pp = n * 100
    return `${pp >= 0 ? '+' : ''}${pp.toFixed(digits)}pp`
}

export const fmtAmericanOdds = (value: unknown): string => {
    const n = toNumOrNull(value)
    if (n === null || n === 0) return '-'
    return n > 0 ? `+${Math.trunc(n)}` : `${Math.trunc(n)}`
}

export const fmtCentralDateTime = (isoUtc?: string | null): string => {
    if (!isoUtc) return '-'
    const date = new Date(isoUtc)
    if (Number.isNaN(date.getTime())) return '-'
    return date.toLocaleString('en-US', {
        timeZone: 'America/Chicago',
        month: '2-digit',
        day: '2-digit',
        year: 'numeric',
        hour: 'numeric',
        minute: '2-digit'
    })
}

export const diffFmt = (value: unknown, pct: boolean = false): string => {
    const n = toNumOrNull(value)
    if (n === null) return '-'
    if (pct) return fmtPp(n)
    return `${n >= 0 ? '+' : ''}${n.toFixed(1)}`
}

export const methodColor = (method?: string): string => {
    const x = String(method || '').toLowerCase()
    if (x.includes('exact')) return 'green'
    if (x.includes('variant') || x.includes('swapped') || x.includes('initials')) return 'blue'
    if (x.includes('fuzzy')) return 'gold'
    if (x.includes('mixed')) return 'purple'
    return 'default'
}

export const last10MethodColor = (method: unknown): string => {
    const x = String(method || '').toLowerCase()
    if (x === 'canonical_id') return 'green'
    if (x === 'name_fallback') return 'gold'
    return 'default'
}

export const americanToProb = (odds: unknown): number | null => {
    const o = toNumOrNull(odds)
    if (o === null || o === 0) return null
    if (o < 0) return -o / (-o + 100)
    return 100 / (o + 100)
}

export const probToAmerican = (probability: unknown): number | null => {
    const p = toNumOrNull(probability)
    if (p === null || p <= 0 || p >= 1) return null
    if (p >= 0.5) return -Math.round((p / (1 - p)) * 100)
    return Math.round(((1 - p) / p) * 100)
}

export const tsOrInfinity = (value?: string | null): number => {
    const t = value ? Date.parse(value) : NaN
    return Number.isFinite(t) ? t : Number.POSITIVE_INFINITY
}
