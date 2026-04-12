import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { JobProgress } from './JobProgress'
import { JobStatusBadge } from './JobStatusBadge'
import { formatDate } from '@/lib/formatters'
import type { JobStatus, JobStatusResponse } from '@/api/types'

interface JobOverviewCardProps {
  status: JobStatusResponse
}

export function JobOverviewCard({ status }: JobOverviewCardProps) {
  const isActive = status.status === 'processing' || status.status === 'queued'

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle>Job Overview</CardTitle>
          <JobStatusBadge status={status.status as JobStatus} />
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {isActive && (
          <div className="space-y-2">
            <JobProgress progress={status.progress} />
            {status.message && (
              <p className="text-sm text-muted-foreground">{status.message}</p>
            )}
            {status.stage && (
              <Badge variant="outline">Stage: {status.stage}</Badge>
            )}
          </div>
        )}

        <dl className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <dt className="text-muted-foreground">Job ID</dt>
            <dd className="font-mono text-xs mt-1">{status.job_id}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Processing Mode</dt>
            <dd className="mt-1">{status.processing_mode || 'sequential'}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Created</dt>
            <dd className="mt-1">{formatDate(status.created_at)}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Completed</dt>
            <dd className="mt-1">{formatDate(status.completed_at)}</dd>
          </div>
        </dl>

        {status.error && (
          <div className="pt-2 border-t">
            <pre className="text-xs text-destructive bg-destructive/10 p-3 rounded overflow-auto">
              {status.error}
            </pre>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
