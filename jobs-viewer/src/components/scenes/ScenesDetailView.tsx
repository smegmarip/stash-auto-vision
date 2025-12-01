import { useJobResults } from '@/hooks/useJobs'
import { Card, CardContent } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { SceneCard } from './SceneCard'
import { SceneTimeline } from './SceneTimeline'
import { formatDuration, formatNumber } from '@/lib/formatters'
import { Film, Clock, Layers, Settings } from 'lucide-react'
import { ScenesResult } from '@/api/types'

interface ScenesDetailViewProps {
  jobId: string
}

export function ScenesDetailView({ jobId }: ScenesDetailViewProps) {
  const { data: jobResults, isLoading, error } = useJobResults(jobId)
  const results = jobResults as ScenesResult | undefined

  if (isLoading) {
    return <ScenesDetailSkeleton />
  }

  if (error) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-destructive">
          Failed to load scene results: {error.message}
        </CardContent>
      </Card>
    )
  }

  const scenesResult = results?.scenes
  if (!scenesResult || !Array.isArray(scenesResult)) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          No scene data available for this job
        </CardContent>
      </Card>
    )
  }

  const scenes = scenesResult
  const metadata = results.metadata

  // Handle different field naming conventions from backend
  const totalScenes = metadata?.total_scenes ?? scenes?.length ?? 0
  const videoDuration = metadata?.video_duration_seconds ?? metadata?.total_duration_seconds ?? 0
  const processingTime = metadata?.processing_time_seconds ?? 0
  const threshold = metadata?.threshold
  const detector = metadata?.detection_method ?? metadata?.detector ?? 'unknown'

  return (
    <div className="space-y-6">
      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          icon={<Layers className="h-5 w-5" />}
          label="Scenes Detected"
          value={formatNumber(totalScenes)}
        />
        <StatCard
          icon={<Film className="h-5 w-5" />}
          label="Video Duration"
          value={formatDuration(videoDuration)}
        />
        <StatCard
          icon={<Clock className="h-5 w-5" />}
          label="Processing Time"
          value={formatDuration(processingTime)}
        />
        {threshold !== undefined && (
          <StatCard
            icon={<Settings className="h-5 w-5" />}
            label="Threshold"
            value={threshold.toFixed(1)}
          />
        )}
      </div>

      {/* Timeline */}
      <Card>
        <CardContent className="p-4">
          <h3 className="font-medium mb-4">Scene Timeline</h3>
          <SceneTimeline
            scenes={scenes}
            totalDuration={videoDuration}
          />
        </CardContent>
      </Card>

      {/* Scene cards */}
      {scenes.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            No scene boundaries detected
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          <h3 className="font-medium">
            {scenes.length} Scene{scenes.length !== 1 ? 's' : ''} Detected
          </h3>
          <div className="space-y-4">
            {scenes.map((scene) => (
              <SceneCard
                key={scene.scene_number}
                scene={scene}
                videoPath={metadata.video_path}
              />
            ))}
          </div>
        </div>
      )}

      {/* Metadata footer */}
      <div className="text-sm text-muted-foreground">
        Detector: {detector}
        {threshold !== undefined && ` • Threshold: ${threshold}`}
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

function ScenesDetailSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i}>
            <CardContent className="p-4">
              <Skeleton className="h-12 w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
      <Card>
        <CardContent className="p-4">
          <Skeleton className="h-8 w-full" />
        </CardContent>
      </Card>
      <div className="space-y-4">
        {Array.from({ length: 3 }).map((_, i) => (
          <Card key={i}>
            <CardContent className="p-4">
              <Skeleton className="h-24 w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
