import { useRef, useEffect, useState } from 'react'
import { Skeleton } from '@/components/ui/skeleton'
import { getImageUrl } from '@/api/client'
import type { BoundingBox } from '@/api/types'
import { cn } from '@/lib/utils'

interface FaceThumbnailProps {
  videoPath: string
  timestamp: number
  bbox: BoundingBox
  size?: number
  className?: string
  enhanced?: boolean
}

export function FaceThumbnail({
  videoPath,
  timestamp,
  bbox,
  size = 80,
  className,
  enhanced = false,
}: FaceThumbnailProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.crossOrigin = 'anonymous'

    img.onload = () => {
      // Calculate crop region with padding
      const padding = 10
      const x = Math.max(0, bbox.x_min - padding)
      const y = Math.max(0, bbox.y_min - padding)
      const w = (bbox.x_max - bbox.x_min) + padding * 2
      const h = (bbox.y_max - bbox.y_min) + padding * 2

      // Ensure we don't go outside the image bounds
      const cropX = Math.max(0, x)
      const cropY = Math.max(0, y)
      const cropW = Math.min(w, img.naturalWidth - cropX)
      const cropH = Math.min(h, img.naturalHeight - cropY)

      // Set canvas size maintaining aspect ratio
      const aspectRatio = cropW / cropH
      if (aspectRatio > 1) {
        canvas.width = size
        canvas.height = size / aspectRatio
      } else {
        canvas.width = size * aspectRatio
        canvas.height = size
      }

      // Draw cropped region
      ctx.drawImage(
        img,
        cropX, cropY, cropW, cropH, // Source rect
        0, 0, canvas.width, canvas.height // Dest rect
      )

      setLoading(false)
    }

    img.onerror = () => {
      setLoading(false)
      setError(true)
    }

    img.src = getImageUrl(videoPath, timestamp, { enhance: enhanced })
  }, [videoPath, timestamp, bbox, size, enhanced])

  return (
    <div
      className={cn('relative inline-block', className)}
      style={{ width: size, height: size }}
    >
      {loading && <Skeleton className="absolute inset-0 rounded" />}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-muted rounded text-xs text-muted-foreground">
          Error
        </div>
      )}
      <canvas
        ref={canvasRef}
        className={cn(
          'rounded border object-contain',
          loading && 'hidden'
        )}
        style={{ maxWidth: size, maxHeight: size }}
      />
    </div>
  )
}
