import { useState } from 'react'
import { useJobResults } from '@/hooks/useJobs'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { FaceCard } from './FaceCard'
import { formatDuration, formatNumber } from '@/lib/formatters'
import { Eye, EyeOff, Users, Clock, Film, Layers } from 'lucide-react'

interface FacesDetailViewProps {
  jobId: string
}

export function FacesDetailView({ jobId }: FacesDetailViewProps) {
  const { data: results, isLoading, error } = useJobResults(jobId)
  const [showOverlays, setShowOverlays] = useState(true)

  if (isLoading) {
    return <FacesDetailSkeleton />
  }

  if (error) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-destructive">
          Failed to load face results: {error.message}
        </CardContent>
      </Card>
    )
  }

  // Handle different result structures:
  // - Vision orchestrated job: results.faces = { faces: [...], metadata: {...} }
  // - Faces-service job: results.faces = [...] (array directly)
  const facesResult = results?.faces

  let faces: unknown[] = []
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let metadata: any = {}

  if (Array.isArray(facesResult)) {
    // Faces-service job returns faces array directly
    faces = facesResult
    // Use top-level metadata if available
    metadata = results?.metadata || {}
  } else if (facesResult && typeof facesResult === 'object') {
    // Vision orchestrated job returns nested structure
    const fr = facesResult as { faces?: unknown[]; metadata?: unknown }
    faces = fr.faces || []
    metadata = fr.metadata || {}
  }

  if (faces.length === 0) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          No face data available for this job
        </CardContent>
      </Card>
    )
  }

  // Extract metadata fields with fallbacks
  const uniqueFaces = metadata?.unique_faces ?? faces.length
  const totalDetections = metadata?.total_detections ?? faces.reduce<number>((sum, f) => {
    const face = f as { detections?: unknown[] }
    return sum + (face.detections?.length ?? 1)
  }, 0)
  const framesProcessed = metadata?.frames_processed
  const processingTime = metadata?.processing_time_seconds
  const model = metadata?.model || 'buffalo_l'
  const method = metadata?.method || 'insightface'
  const videoSource = metadata?.source || metadata?.video_path || ''

  return (
    <div className="space-y-6">
      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          icon={<Users className="h-5 w-5" />}
          label="Unique Faces"
          value={formatNumber(uniqueFaces)}
        />
        <StatCard
          icon={<Layers className="h-5 w-5" />}
          label="Total Detections"
          value={formatNumber(totalDetections)}
        />
        {framesProcessed !== undefined && (
          <StatCard
            icon={<Film className="h-5 w-5" />}
            label="Frames Processed"
            value={formatNumber(framesProcessed)}
          />
        )}
        {processingTime !== undefined && (
          <StatCard
            icon={<Clock className="h-5 w-5" />}
            label="Processing Time"
            value={formatDuration(processingTime)}
          />
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4">
        <Button
          variant={showOverlays ? 'default' : 'outline'}
          size="sm"
          onClick={() => setShowOverlays(!showOverlays)}
        >
          {showOverlays ? <Eye className="h-4 w-4 mr-2" /> : <EyeOff className="h-4 w-4 mr-2" />}
          {showOverlays ? 'Overlays On' : 'Overlays Off'}
        </Button>
        <span className="text-sm text-muted-foreground">
          Model: {model} • Method: {method}
        </span>
      </div>

      {/* Faces grid */}
      <div className="grid gap-4 md:grid-cols-2">
        {faces.map((face) => (
          <FaceCard
            key={(face as { face_id: string }).face_id}
            face={face as import('@/api/types').Face}
            videoPath={videoSource}
            showOverlay={showOverlays}
          />
        ))}
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

function FacesDetailSkeleton() {
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
      <div className="grid gap-4 md:grid-cols-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i}>
            <CardContent className="p-4">
              <div className="flex gap-4">
                <Skeleton className="h-20 w-20 rounded" />
                <div className="flex-1 space-y-2">
                  <Skeleton className="h-4 w-24" />
                  <Skeleton className="h-3 w-32" />
                  <Skeleton className="h-3 w-20" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
