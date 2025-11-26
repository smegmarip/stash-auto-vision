import { formatTimestamp } from '@/lib/formatters'
import type { Scene } from '@/api/types'
import { cn } from '@/lib/utils'

interface SceneTimelineProps {
  scenes: Scene[]
  totalDuration: number
  className?: string
}

// Generate distinct colors for scenes
const SCENE_COLORS = [
  'bg-blue-500',
  'bg-green-500',
  'bg-yellow-500',
  'bg-purple-500',
  'bg-pink-500',
  'bg-indigo-500',
  'bg-teal-500',
  'bg-orange-500',
  'bg-cyan-500',
  'bg-emerald-500',
]

export function SceneTimeline({ scenes, totalDuration, className }: SceneTimelineProps) {
  if (scenes.length === 0 || totalDuration === 0) {
    return null
  }

  // Helper to get timestamps with fallback for different field names
  const getStartTime = (scene: Scene) => scene.start_timestamp ?? scene.start_time ?? 0
  const getEndTime = (scene: Scene) => scene.end_timestamp ?? scene.end_time ?? 0

  return (
    <div className={cn('space-y-2', className)}>
      {/* Timeline bar */}
      <div className="relative h-8 bg-muted rounded-lg overflow-hidden flex">
        {scenes.map((scene, index) => {
          const startTime = getStartTime(scene)
          const endTime = getEndTime(scene)
          const startPercent = (startTime / totalDuration) * 100
          const widthPercent = ((endTime - startTime) / totalDuration) * 100
          const color = SCENE_COLORS[index % SCENE_COLORS.length]

          return (
            <div
              key={scene.scene_number}
              className={cn(
                color,
                'h-full relative group cursor-pointer hover:brightness-110 transition-all'
              )}
              style={{
                width: `${widthPercent}%`,
                marginLeft: index === 0 ? `${startPercent}%` : 0,
              }}
              title={`Scene ${scene.scene_number}: ${formatTimestamp(startTime)} - ${formatTimestamp(endTime)}`}
            >
              {/* Scene number label (only if wide enough) */}
              {widthPercent > 5 && (
                <span className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white">
                  {scene.scene_number}
                </span>
              )}

              {/* Tooltip on hover */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-popover text-popover-foreground text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
                Scene {scene.scene_number}
                <br />
                {formatTimestamp(startTime)} - {formatTimestamp(endTime)}
              </div>
            </div>
          )
        })}
      </div>

      {/* Time labels */}
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>0:00</span>
        <span>{formatTimestamp(totalDuration / 2)}</span>
        <span>{formatTimestamp(totalDuration)}</span>
      </div>
    </div>
  )
}
