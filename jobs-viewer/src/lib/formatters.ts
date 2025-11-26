import { format, formatDistanceToNow, parseISO } from 'date-fns'

export function formatDate(dateString?: string | null): string {
  if (!dateString) return '-'
  try {
    const date = parseISO(dateString)
    return format(date, 'MMM d, yyyy HH:mm')
  } catch {
    return dateString
  }
}

export function formatRelativeTime(dateString?: string | null): string {
  if (!dateString) return '-'
  try {
    const date = parseISO(dateString)
    return formatDistanceToNow(date, { addSuffix: true })
  } catch {
    return dateString
  }
}

export function formatDuration(seconds?: number | null): string {
  if (seconds === null || seconds === undefined) return '-'

  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`
  }

  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60

  if (minutes < 60) {
    return `${minutes}m ${Math.round(remainingSeconds)}s`
  }

  const hours = Math.floor(minutes / 60)
  const remainingMinutes = minutes % 60

  return `${hours}h ${remainingMinutes}m`
}

export function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

export function formatProgress(progress: number): string {
  return `${Math.round(progress * 100)}%`
}

export function formatNumber(num?: number | null): string {
  if (num === null || num === undefined) return '-'
  return num.toLocaleString()
}

export function truncatePath(path: string, maxLength = 40): string {
  if (path.length <= maxLength) return path

  const parts = path.split('/')
  const filename = parts.pop() || ''

  if (filename.length >= maxLength - 3) {
    return '...' + filename.slice(-(maxLength - 3))
  }

  return '.../' + filename
}
