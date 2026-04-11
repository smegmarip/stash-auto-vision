import { useEffect, useState, useCallback } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Cpu, HardDrive, Activity, MemoryStick } from 'lucide-react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts'

interface MetricSample {
  timestamp: number
  gpu_utilization_pct: number
  vram_used_mb: number
  vram_total_mb: number
  cpu_utilization_pct: number
  ram_used_mb: number
  ram_total_mb: number
}

interface MetricsResponse {
  current: MetricSample | null
  averages: {
    gpu_utilization_pct: number
    vram_used_mb: number
    vram_total_mb: number
    cpu_utilization_pct: number
    ram_used_mb: number
    ram_total_mb: number
  } | null
  history: MetricSample[]
  sample_count: number
  interval_seconds: number
}

const POLL_INTERVAL = 5000

function formatTime(timestamp: number): string {
  const d = new Date(timestamp * 1000)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function formatMB(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${Math.round(mb)} MB`
}

// ---------------------------------------------------------------------------
// Chart component
// ---------------------------------------------------------------------------

interface MetricChartProps {
  data: MetricSample[]
  dataKey: string
  color: string
  domain?: [number, number]
  unit?: string
  formatValue?: (v: number) => string
}

function MetricChart({ data, dataKey, color, domain, unit = '', formatValue }: MetricChartProps) {
  const fmt = formatValue || ((v: number) => `${v.toFixed(1)}${unit}`)

  return (
    <ResponsiveContainer width="100%" height={140}>
      <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id={`grad-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={color} stopOpacity={0.3} />
            <stop offset="95%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
        <XAxis
          dataKey="timestamp"
          tickFormatter={formatTime}
          tick={{ fontSize: 10 }}
          stroke="hsl(var(--muted-foreground))"
          minTickGap={60}
        />
        <YAxis
          domain={domain}
          tickFormatter={(v) => fmt(v)}
          tick={{ fontSize: 10 }}
          stroke="hsl(var(--muted-foreground))"
          width={52}
        />
        <Tooltip
          labelFormatter={(label) => formatTime(Number(label))}
          formatter={(value) => [fmt(Number(value)), '']}
          contentStyle={{
            backgroundColor: 'hsl(var(--card))',
            border: '1px solid hsl(var(--border))',
            borderRadius: '6px',
            fontSize: '12px',
          }}
        />
        <Area
          type="monotone"
          dataKey={dataKey}
          stroke={color}
          fill={`url(#grad-${dataKey})`}
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

// ---------------------------------------------------------------------------
// Stat card
// ---------------------------------------------------------------------------

interface StatCardProps {
  icon: React.ReactNode
  label: string
  current: string
  currentPct: string
  average: string
  averagePct: string
  color: string
  data: MetricSample[]
  dataKey: string
  chartDomain?: [number, number]
  chartUnit?: string
  chartFormat?: (v: number) => string
}

function StatCard({
  icon,
  label,
  current,
  currentPct,
  average,
  averagePct,
  color,
  data,
  dataKey,
  chartDomain,
  chartUnit,
  chartFormat,
}: StatCardProps) {
  return (
    <Card>
      <CardContent className="p-4 space-y-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded" style={{ backgroundColor: `${color}20`, color }}>
            {icon}
          </div>
          <h3 className="font-medium text-sm">{label}</h3>
        </div>
        <MetricChart
          data={data}
          dataKey={dataKey}
          color={color}
          domain={chartDomain}
          unit={chartUnit}
          formatValue={chartFormat}
        />
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <p className="text-muted-foreground text-xs">Current</p>
            <p className="font-semibold">{current}</p>
            <p className="text-xs text-muted-foreground">{currentPct}</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">1h Average</p>
            <p className="font-semibold">{average}</p>
            <p className="text-xs text-muted-foreground">{averagePct}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Overview card (compact current-value display)
// ---------------------------------------------------------------------------

function OverviewCard({
  icon,
  label,
  value,
  sub,
  color,
}: {
  icon: React.ReactNode
  label: string
  value: string
  sub?: string
  color: string
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg" style={{ backgroundColor: `${color}20`, color }}>
            {icon}
          </div>
          <div>
            <p className="text-2xl font-bold">{value}</p>
            <p className="text-xs text-muted-foreground">
              {label}{sub ? ` \u00b7 ${sub}` : ''}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function ResourcesPage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const fetchMetrics = useCallback(async () => {
    try {
      const res = await fetch('/api/resources/metrics')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: MetricsResponse = await res.json()
      setMetrics(data)
      setError(null)
    } catch (err) {
      setError((err as Error).message)
    }
  }, [])

  useEffect(() => {
    fetchMetrics()
    const id = setInterval(fetchMetrics, POLL_INTERVAL)
    return () => clearInterval(id)
  }, [fetchMetrics])

  if (error && !metrics) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold tracking-tight">Resources</h1>
        <Card>
          <CardContent className="py-8 text-center text-destructive">
            Failed to load metrics: {error}
          </CardContent>
        </Card>
      </div>
    )
  }

  if (!metrics || !metrics.current || !metrics.averages) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold tracking-tight">Resources</h1>
        <div className="grid gap-4 md:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i}>
              <CardContent className="p-4">
                <Skeleton className="h-48 w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  const { current: c, averages: a, history } = metrics

  const gpuPct = c.gpu_utilization_pct
  const vramPct = c.vram_total_mb > 0 ? (c.vram_used_mb / c.vram_total_mb) * 100 : 0
  const cpuPct = c.cpu_utilization_pct
  const ramPct = c.ram_total_mb > 0 ? (c.ram_used_mb / c.ram_total_mb) * 100 : 0

  const avgVramPct = a.vram_total_mb > 0 ? (a.vram_used_mb / a.vram_total_mb) * 100 : 0
  const avgRamPct = a.ram_total_mb > 0 ? (a.ram_used_mb / a.ram_total_mb) * 100 : 0

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Resource Monitor</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Measurements taken at {metrics.interval_seconds}s intervals
          {error && <span className="text-destructive ml-2">(polling error: {error})</span>}
        </p>
      </div>

      {/* Overview cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <OverviewCard
          icon={<Activity className="h-5 w-5" />}
          label="GPU"
          value={`${gpuPct.toFixed(1)}%`}
          color="#f59e0b"
        />
        <OverviewCard
          icon={<HardDrive className="h-5 w-5" />}
          label="VRAM"
          value={formatMB(c.vram_used_mb)}
          sub={`${vramPct.toFixed(1)}%`}
          color="#8b5cf6"
        />
        <OverviewCard
          icon={<Cpu className="h-5 w-5" />}
          label="CPU"
          value={`${cpuPct.toFixed(1)}%`}
          color="#3b82f6"
        />
        <OverviewCard
          icon={<MemoryStick className="h-5 w-5" />}
          label="RAM"
          value={formatMB(c.ram_used_mb)}
          sub={`${ramPct.toFixed(1)}%`}
          color="#10b981"
        />
      </div>

      {/* Detail charts */}
      <div className="grid gap-4 md:grid-cols-2">
        <StatCard
          icon={<Activity className="h-4 w-4" />}
          label="GPU Utilization"
          current={`${gpuPct.toFixed(1)}%`}
          currentPct=""
          average={`${a.gpu_utilization_pct.toFixed(1)}%`}
          averagePct=""
          color="#f59e0b"
          data={history}
          dataKey="gpu_utilization_pct"
          chartDomain={[0, 100]}
          chartUnit="%"
        />

        <StatCard
          icon={<HardDrive className="h-4 w-4" />}
          label="VRAM Usage"
          current={formatMB(c.vram_used_mb)}
          currentPct={`${vramPct.toFixed(1)}% of ${formatMB(c.vram_total_mb)}`}
          average={formatMB(a.vram_used_mb)}
          averagePct={`${avgVramPct.toFixed(1)}% of ${formatMB(a.vram_total_mb)}`}
          color="#8b5cf6"
          data={history}
          dataKey="vram_used_mb"
          chartDomain={[0, c.vram_total_mb || 16384]}
          chartFormat={formatMB}
        />

        <StatCard
          icon={<Cpu className="h-4 w-4" />}
          label="CPU Utilization"
          current={`${cpuPct.toFixed(1)}%`}
          currentPct=""
          average={`${a.cpu_utilization_pct.toFixed(1)}%`}
          averagePct=""
          color="#3b82f6"
          data={history}
          dataKey="cpu_utilization_pct"
          chartDomain={[0, 100]}
          chartUnit="%"
        />

        <StatCard
          icon={<MemoryStick className="h-4 w-4" />}
          label="RAM Usage"
          current={formatMB(c.ram_used_mb)}
          currentPct={`${ramPct.toFixed(1)}% of ${formatMB(c.ram_total_mb)}`}
          average={formatMB(a.ram_used_mb)}
          averagePct={`${avgRamPct.toFixed(1)}% of ${formatMB(a.ram_total_mb)}`}
          color="#10b981"
          data={history}
          dataKey="ram_used_mb"
          chartDomain={[0, c.ram_total_mb || 65536]}
          chartFormat={formatMB}
        />
      </div>
    </div>
  )
}
