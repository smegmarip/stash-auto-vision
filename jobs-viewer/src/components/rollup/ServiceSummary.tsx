import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Users, Layers, Sparkles, Box, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ServiceSummaryProps {
  service: string
  label: string
  data?: Record<string, unknown> | null
  className?: string
}

const serviceIcons: Record<string, React.ReactNode> = {
  faces: <Users className="h-5 w-5" />,
  scenes: <Layers className="h-5 w-5" />,
  semantics: <Sparkles className="h-5 w-5" />,
  objects: <Box className="h-5 w-5" />,
}

const serviceColors: Record<string, string> = {
  faces: 'bg-blue-500',
  scenes: 'bg-green-500',
  semantics: 'bg-purple-500',
  objects: 'bg-orange-500',
}

export function ServiceSummary({ service, label, data, className }: ServiceSummaryProps) {
  const hasData = data && Object.keys(data).length > 0
  const isNotImplemented = data && (data as { status?: string }).status === 'not_implemented'

  const getSummary = () => {
    if (!hasData || isNotImplemented) return null

    switch (service) {
      case 'faces': {
        const faces = data as { faces?: unknown[]; metadata?: { unique_faces?: number; total_detections?: number } }
        return (
          <div className="flex gap-2 mt-2">
            <Badge variant="outline">
              {faces.metadata?.unique_faces || faces.faces?.length || 0} faces
            </Badge>
            {faces.metadata?.total_detections && (
              <Badge variant="outline">
                {faces.metadata.total_detections} detections
              </Badge>
            )}
          </div>
        )
      }
      case 'scenes': {
        const scenes = data as { scenes?: unknown[]; metadata?: { total_scenes?: number } }
        return (
          <div className="flex gap-2 mt-2">
            <Badge variant="outline">
              {scenes.metadata?.total_scenes || scenes.scenes?.length || 0} scenes
            </Badge>
          </div>
        )
      }
      case 'semantics': {
        type TagItem = { tag_name: string; score: number; decode_type: string }
        const sem = data as {
          semantics?: { tags?: TagItem[]; scene_summary?: string | null; suggested_title?: string | null };
          metadata?: { classifier_model?: string; processing_time_seconds?: number }
        }
        const tags = sem.semantics?.tags || []
        const directCount = tags.filter((t: TagItem) => t.decode_type === 'direct').length
        const hasSummary = sem.semantics?.scene_summary != null
        const hasTitle = sem.semantics?.suggested_title != null
        return (
          <div className="flex flex-wrap gap-2 mt-2">
            {tags.length > 0 ? (
              <>
                <Badge variant="outline">
                  {tags.length} tags
                </Badge>
                {directCount > 0 && directCount !== tags.length && (
                  <Badge variant="secondary">
                    {directCount} direct
                  </Badge>
                )}
              </>
            ) : null}
            {hasSummary && <Badge variant="outline">summary</Badge>}
            {hasTitle && <Badge variant="outline">title</Badge>}
            {sem.metadata?.classifier_model && (
              <Badge variant="secondary">
                {sem.metadata.classifier_model}
              </Badge>
            )}
          </div>
        )
      }
      default:
        return null
    }
  }

  return (
    <Card
      className={cn(
        'transition-all',
        hasData && !isNotImplemented ? 'hover:ring-2 hover:ring-primary/50 cursor-pointer' : 'opacity-60',
        className
      )}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={cn('p-2 rounded-lg text-white', serviceColors[service] || 'bg-gray-500')}>
              {serviceIcons[service]}
            </div>
            <CardTitle className="text-lg">{label}</CardTitle>
          </div>
          {hasData && !isNotImplemented && (
            <ChevronRight className="h-5 w-5 text-muted-foreground" />
          )}
        </div>
      </CardHeader>
      <CardContent>
        {isNotImplemented ? (
          <Badge variant="secondary">Not Implemented</Badge>
        ) : hasData ? (
          <>
            <Badge variant="success">Completed</Badge>
            {getSummary()}
          </>
        ) : (
          <Badge variant="secondary">No Data</Badge>
        )}
      </CardContent>
    </Card>
  )
}
