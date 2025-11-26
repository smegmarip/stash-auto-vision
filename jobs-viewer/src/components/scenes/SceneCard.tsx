import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { FrameViewer } from '@/components/shared/FrameViewer'
import { formatTimestamp, formatDuration } from '@/lib/formatters'
import type { Scene } from '@/api/types'

interface SceneCardProps {
  scene: Scene
  videoPath: string
}

export function SceneCard({ scene, videoPath }: SceneCardProps) {
  // Handle different field naming conventions from backend
  const startTime = scene.start_timestamp ?? scene.start_time ?? 0
  const endTime = scene.end_timestamp ?? scene.end_time ?? 0

  return (
    <Card className="overflow-hidden">
      <CardContent className="p-0">
        <div className="flex flex-col md:flex-row">
          {/* Start frame thumbnail */}
          <div className="relative w-full md:w-48 aspect-video md:aspect-auto md:h-32 bg-muted flex-shrink-0">
            <FrameViewer
              videoPath={videoPath}
              timestamp={startTime}
              className="w-full h-full"
              alt={`Scene ${scene.scene_number} start`}
            />
            <Badge className="absolute bottom-2 left-2 text-xs">
              Start
            </Badge>
          </div>

          {/* Scene info */}
          <div className="flex-1 p-4 space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">Scene {scene.scene_number}</h3>
              <Badge variant="outline">
                {formatDuration(scene.duration)}
              </Badge>
            </div>

            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm text-muted-foreground">
              <div className="flex justify-between">
                <span>Start:</span>
                <span className="font-mono">{formatTimestamp(startTime)}</span>
              </div>
              <div className="flex justify-between">
                <span>End:</span>
                <span className="font-mono">{formatTimestamp(endTime)}</span>
              </div>
              <div className="flex justify-between">
                <span>Start Frame:</span>
                <span className="font-mono">{scene.start_frame}</span>
              </div>
              <div className="flex justify-between">
                <span>End Frame:</span>
                <span className="font-mono">{scene.end_frame}</span>
              </div>
            </div>
          </div>

          {/* End frame thumbnail */}
          <div className="relative w-full md:w-48 aspect-video md:aspect-auto md:h-32 bg-muted flex-shrink-0 hidden md:block">
            <FrameViewer
              videoPath={videoPath}
              timestamp={endTime}
              className="w-full h-full"
              alt={`Scene ${scene.scene_number} end`}
            />
            <Badge className="absolute bottom-2 right-2 text-xs">
              End
            </Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
