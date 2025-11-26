import { Badge } from '@/components/ui/badge'
import type { JobStatus } from '@/api/types'

interface JobStatusBadgeProps {
  status: JobStatus
  className?: string
}

const statusConfig: Record<JobStatus, { label: string; variant: 'success' | 'processing' | 'warning' | 'destructive' | 'secondary' }> = {
  completed: { label: 'Completed', variant: 'success' },
  processing: { label: 'Processing', variant: 'processing' },
  queued: { label: 'Queued', variant: 'secondary' },
  failed: { label: 'Failed', variant: 'destructive' },
}

export function JobStatusBadge({ status, className }: JobStatusBadgeProps) {
  const config = statusConfig[status] || { label: status, variant: 'secondary' as const }

  return (
    <Badge variant={config.variant} className={className}>
      {config.label}
    </Badge>
  )
}
