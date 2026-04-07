import { useState, useEffect } from 'react'

interface VttEntry {
  start: number
  end: number
  x: number
  y: number
  w: number
  h: number
}

function parseVtt(vttText: string): VttEntry[] {
  const entries: VttEntry[] = []
  const lines = vttText.split('\n')
  let i = 0
  while (i < lines.length && !lines[i].includes('-->')) i++
  while (i < lines.length) {
    const line = lines[i].trim()
    const timeMatch = line.match(/(\d+):(\d+):(\d+)\.(\d+)\s*-->\s*(\d+):(\d+):(\d+)\.(\d+)/)
    if (timeMatch) {
      const start = parseInt(timeMatch[1]) * 3600 + parseInt(timeMatch[2]) * 60 + parseInt(timeMatch[3]) + parseInt(timeMatch[4]) / 1000
      const end = parseInt(timeMatch[5]) * 3600 + parseInt(timeMatch[6]) * 60 + parseInt(timeMatch[7]) + parseInt(timeMatch[8]) / 1000
      i++
      if (i < lines.length) {
        const coordMatch = lines[i].trim().match(/#xywh=(\d+),(\d+),(\d+),(\d+)/)
        if (coordMatch) {
          entries.push({
            start, end,
            x: parseInt(coordMatch[1]), y: parseInt(coordMatch[2]),
            w: parseInt(coordMatch[3]), h: parseInt(coordMatch[4]),
          })
        }
      }
    }
    i++
  }
  return entries
}

function findEntryForTimestamp(entries: VttEntry[], timestamp: number): VttEntry | null {
  for (const entry of entries) {
    if (timestamp >= entry.start && timestamp < entry.end) return entry
  }
  if (entries.length > 0) {
    let closest = entries[0]
    let minDiff = Math.abs(timestamp - entries[0].start)
    for (const entry of entries) {
      const diff = Math.abs(timestamp - entry.start)
      if (diff < minDiff) { minDiff = diff; closest = entry }
    }
    return closest
  }
  return null
}

// VTT cache shared across all instances
const vttCache = new Map<string, VttEntry[]>()

interface SpriteFrameProps {
  spriteUrl: string
  vttUrl: string
  timestamp: number
  width?: number
  height?: number
  className?: string
}

export function SpriteFrame({
  spriteUrl,
  vttUrl,
  timestamp,
  width = 120,
  height = 68,
  className,
}: SpriteFrameProps) {
  const [entry, setEntry] = useState<VttEntry | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  // Proxy URLs to avoid CORS
  const proxiedVtt = `/api/proxy?url=${encodeURIComponent(vttUrl)}`
  const proxiedSprite = `/api/proxy?url=${encodeURIComponent(spriteUrl)}`

  useEffect(() => {
    if (!vttUrl) { setError(true); setLoading(false); return }

    if (vttCache.has(vttUrl)) {
      setEntry(findEntryForTimestamp(vttCache.get(vttUrl)!, timestamp))
      setLoading(false)
      return
    }

    setLoading(true)
    fetch(proxiedVtt)
      .then(res => { if (!res.ok) throw new Error(`VTT ${res.status}`); return res.text() })
      .then(text => {
        const entries = parseVtt(text)
        vttCache.set(vttUrl, entries)
        setEntry(findEntryForTimestamp(entries, timestamp))
        setLoading(false)
      })
      .catch(() => { setError(true); setLoading(false) })
  }, [vttUrl, timestamp, proxiedVtt])

  if (loading || error || !entry) {
    return <div className={className} style={{ width, height, backgroundColor: 'hsl(var(--muted))' }} />
  }

  const scale = Math.min(width / entry.w, height / entry.h)

  return (
    <div className={className} style={{ width, height, overflow: 'hidden', position: 'relative' }}>
      <div
        style={{
          backgroundImage: `url(${proxiedSprite})`,
          backgroundPosition: `-${entry.x}px -${entry.y}px`,
          backgroundRepeat: 'no-repeat',
          width: entry.w,
          height: entry.h,
          transform: `scale(${scale})`,
          transformOrigin: 'top left',
        }}
      />
    </div>
  )
}
