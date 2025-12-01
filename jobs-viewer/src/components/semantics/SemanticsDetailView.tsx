import { useJobResults } from '@/hooks/useJobs'
import { Card, CardContent } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { FrameCard } from './FrameCard'
import { formatDuration, formatNumber } from '@/lib/formatters'
import { Film, Clock, Tags, Settings, Sparkles } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

interface SemanticsDetailViewProps {
  jobId: string
}

export function SemanticsDetailView({ jobId }: SemanticsDetailViewProps) {
  const { data: results, isLoading, error } = useJobResults(jobId)

  if (isLoading) {
    return <SemanticsDetailSkeleton />
  }

  if (error) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-destructive">
          Failed to load semantics results: {error.message}
        </CardContent>
      </Card>
    )
  }

  const semanticsResult = results?.semantics
  if (!semanticsResult || !semanticsResult.frames) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          No semantic data available for this job
        </CardContent>
      </Card>
    )
  }

  const { frames, metadata, scene_summaries } = semanticsResult

  const framesAnalyzed = metadata?.frames_analyzed ?? frames?.length ?? 0
  const processingTime = metadata?.processing_time_seconds ?? 0
  const totalTags = metadata?.total_tags_generated ?? 0
  const model = metadata?.model ?? 'unknown'
  const device = metadata?.device ?? 'unknown'
  const videoPath = metadata?.source || ''

  // Collect all unique tags across frames
  const allTags = new Set<string>()
  frames.forEach((frame) => {
    frame.tags.forEach((tag) => {
      allTags.add(tag.tag)
    })
  })

  return (
    <div className="space-y-6">
      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          icon={<Film className="h-5 w-5" />}
          label="Frames Analyzed"
          value={formatNumber(framesAnalyzed)}
        />
        <StatCard
          icon={<Tags className="h-5 w-5" />}
          label="Unique Tags"
          value={formatNumber(allTags.size)}
        />
        <StatCard
          icon={<Sparkles className="h-5 w-5" />}
          label="Total Classifications"
          value={formatNumber(totalTags)}
        />
        <StatCard
          icon={<Clock className="h-5 w-5" />}
          label="Processing Time"
          value={formatDuration(processingTime)}
        />
      </div>

      {/* Model Info */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center gap-4">
            <Settings className="h-5 w-5 text-muted-foreground" />
            <div className="flex-1">
              <p className="text-sm font-medium">Model: {model}</p>
              <p className="text-xs text-muted-foreground">Device: {device}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Scene Summaries (if available) */}
      {scene_summaries && Array.isArray(scene_summaries) && scene_summaries.length > 0 && (
        <Card>
          <CardContent className="p-4">
            <h3 className="font-medium mb-4">Scene Summaries</h3>
            <div className="space-y-3">
              {scene_summaries.map((summary, idx) => (
                <div
                  key={idx}
                  className="border rounded-lg p-3 space-y-2"
                >
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">
                      Scene {idx + 1}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {formatDuration(summary.start_timestamp)} - {formatDuration(summary.end_timestamp)}
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {summary.dominant_tags.map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {summary.frame_count} frames • Avg confidence: {(summary.avg_confidence * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Frame Results */}
      {frames.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            No frames analyzed
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          <h3 className="font-medium">
            {frames.length} Frame{frames.length !== 1 ? 's' : ''} Analyzed
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {frames.map((frame) => (
              <FrameCard
                key={frame.frame_index}
                frame={frame}
                videoPath={videoPath}
              />
            ))}
          </div>
        </div>
      )}

      {/* Metadata footer */}
      <div className="text-sm text-muted-foreground">
        Model: {model} • Device: {device.toUpperCase()} • {framesAnalyzed} frames analyzed
      </div>
    </div>
  )
}

function StatCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode
  label: string
  value: string
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-lg text-primary">{icon}</div>
          <div>
            <p className="text-2xl font-bold">{value}</p>
            <p className="text-xs text-muted-foreground">{label}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function SemanticsDetailSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i}>
            <CardContent className="p-4">
              <Skeleton className="h-16 w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
      <Card>
        <CardContent className="p-4">
          <Skeleton className="h-24 w-full" />
        </CardContent>
      </Card>
    </div>
  )
}
