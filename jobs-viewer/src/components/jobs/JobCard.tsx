import { Link } from 'react-router-dom'
import { Film, User, Clock, FileVideo } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { JobStatusBadge } from './JobStatusBadge'
import { JobProgress } from './JobProgress'
import { formatRelativeTime, truncatePath } from '@/lib/formatters'
import type { JobSummary } from '@/api/types'

interface JobCardProps {
  job: JobSummary
}

const serviceIcons: Record<string, React.ReactNode> = {
  vision: <Film className="h-4 w-4" />,
  faces: <User className="h-4 w-4" />,
  scenes: <Clock className="h-4 w-4" />,
}

const serviceColors: Record<string, string> = {
  vision: 'bg-purple-500',
  faces: 'bg-blue-500',
  scenes: 'bg-green-500',
}

export function JobCard({ job }: JobCardProps) {
  const isActive = job.status === 'processing' || job.status === 'queued'

  return (
    <Link to={`/jobs/${job.job_id}`}>
      <Card className="hover:bg-accent/50 transition-colors cursor-pointer h-full">
        <CardHeader className="pb-2">
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-center gap-2 min-w-0">
              <div className={`p-1.5 rounded ${serviceColors[job.service]} text-white`}>
                {serviceIcons[job.service]}
              </div>
              <div className="min-w-0">
                <CardTitle className="text-sm font-mono truncate" title={job.job_id}>
                  {job.job_id.slice(0, 8)}...
                </CardTitle>
                <Badge variant="outline" className="text-xs mt-1">
                  {job.service}
                </Badge>
              </div>
            </div>
            <JobStatusBadge status={job.status} />
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* Source path */}
          {job.source && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <FileVideo className="h-4 w-4 flex-shrink-0" />
              <span className="truncate" title={job.source}>
                {truncatePath(job.source)}
              </span>
            </div>
          )}

          {/* Scene ID */}
          {job.scene_id && (
            <div className="text-xs text-muted-foreground">
              Scene: <span className="font-mono">{job.scene_id}</span>
            </div>
          )}

          {/* Progress bar for active jobs */}
          {isActive && (
            <JobProgress progress={job.progress} />
          )}

          {/* Result summary for completed jobs */}
          {job.status === 'completed' && job.result_summary && (
            <div className="flex gap-2 text-xs">
              {(job.result_summary as Record<string, number>).faces_count !== undefined && (
                <Badge variant="outline">
                  {(job.result_summary as Record<string, number>).faces_count} faces
                </Badge>
              )}
              {(job.result_summary as Record<string, number>).scenes_count !== undefined && (
                <Badge variant="outline">
                  {(job.result_summary as Record<string, number>).scenes_count} scenes
                </Badge>
              )}
            </div>
          )}

          {/* Timestamp */}
          <div className="text-xs text-muted-foreground">
            {job.completed_at
              ? `Completed ${formatRelativeTime(job.completed_at)}`
              : job.started_at
              ? `Started ${formatRelativeTime(job.started_at)}`
              : `Created ${formatRelativeTime(job.created_at)}`}
          </div>
        </CardContent>
      </Card>
    </Link>
  )
}
