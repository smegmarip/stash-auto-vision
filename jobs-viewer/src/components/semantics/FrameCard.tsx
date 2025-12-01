import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { formatDuration } from '@/lib/formatters'
import { Clock, Tag } from 'lucide-react'
import type { FrameSemantics } from '@/api/types'
import { FrameViewer } from '../shared/FrameViewer'

interface FrameCardProps {
  frame: FrameSemantics
  videoPath?: string
}

export function FrameCard({ frame, videoPath }: FrameCardProps) {
  // Get source color for tags
  const getSourceColor = (source: string) => {
    switch (source) {
      case 'predefined':
        return 'bg-blue-500/10 text-blue-700 border-blue-500/20'
      case 'custom_prompt':
        return 'bg-purple-500/10 text-purple-700 border-purple-500/20'
      case 'zero_shot':
        return 'bg-green-500/10 text-green-700 border-green-500/20'
      default:
        return 'bg-gray-500/10 text-gray-700 border-gray-500/20'
    }
  }

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="p-4 space-y-3">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">
              {formatDuration(frame.timestamp)}
            </span>
          </div>
          <span className="text-xs text-muted-foreground">
            Frame {frame.frame_index}
          </span>
        </div>

        <div className="flex items-center justify-between">
          {/* Tags */}
          {frame.tags && frame.tags.length > 0 ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Tag className="h-3 w-3 text-muted-foreground" />
                <span className="text-xs text-muted-foreground">
                  {frame.tags.length} tag{frame.tags.length !== 1 ? 's' : ''}
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {frame.tags.map((tag, idx) => (
                  <div
                    key={idx}
                    className="flex flex-col gap-0.5"
                  >
                    <Badge
                      variant="outline"
                      className={`text-xs ${getSourceColor(tag.source)}`}
                    >
                      {tag.tag}
                    </Badge>
                    <span className="text-[10px] text-muted-foreground text-center">
                      {(tag.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-xs text-muted-foreground italic">
              No tags detected
            </div>
          )}
          {/* Thumbnail */}
          {videoPath && (
            <div className="mt-2">
              <FrameViewer
                videoPath={videoPath}
                timestamp={frame.timestamp}
                className="w-full h-16 rounded border bg-black object-cover"
                alt={`Frame at ${formatDuration(frame.timestamp)}`}
              />
            </div>
          )}
        </div>

        {/* Embedding indicator */}
        {frame.embedding && (
          <div className="text-[10px] text-muted-foreground pt-2 border-t">
            Embedding: {frame.embedding.length}-D vector
          </div>
        )}
      </CardContent>
    </Card>
  )
}
