import { useRef, useEffect } from 'react'
import type { Detection } from '@/api/types'
import { cn } from '@/lib/utils'

interface FaceOverlayProps {
  detection: Detection
  frameWidth: number
  frameHeight: number
  containerWidth: number
  containerHeight: number
  showBbox?: boolean
  showLandmarks?: boolean
  className?: string
}

export function FaceOverlay({
  detection,
  frameWidth,
  frameHeight,
  containerWidth,
  containerHeight,
  showBbox = true,
  showLandmarks = true,
  className,
}: FaceOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size to match container
    canvas.width = containerWidth
    canvas.height = containerHeight

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Calculate rendered image dimensions with object-contain behavior
    // The image maintains aspect ratio and is centered within the container
    const containerAspect = containerWidth / containerHeight
    const imageAspect = frameWidth / frameHeight

    let renderedWidth: number
    let renderedHeight: number
    let offsetX: number
    let offsetY: number

    if (imageAspect > containerAspect) {
      // Image is wider than container - fits width, letterboxed top/bottom
      renderedWidth = containerWidth
      renderedHeight = containerWidth / imageAspect
      offsetX = 0
      offsetY = (containerHeight - renderedHeight) / 2
    } else {
      // Image is taller than container - fits height, letterboxed left/right
      renderedHeight = containerHeight
      renderedWidth = containerHeight * imageAspect
      offsetX = (containerWidth - renderedWidth) / 2
      offsetY = 0
    }

    // Calculate scale factors based on rendered image size
    const scaleX = renderedWidth / frameWidth
    const scaleY = renderedHeight / frameHeight

    const { bbox, landmarks, confidence, quality } = detection

    // Draw bounding box
    if (showBbox) {
      // Color based on quality
      const hue = quality.composite > 0.7 ? 120 : quality.composite > 0.4 ? 60 : 0
      ctx.strokeStyle = `hsl(${hue}, 80%, 50%)`
      ctx.lineWidth = 2

      const x = offsetX + bbox.x_min * scaleX
      const y = offsetY + bbox.y_min * scaleY
      const w = (bbox.x_max - bbox.x_min) * scaleX
      const h = (bbox.y_max - bbox.y_min) * scaleY

      ctx.strokeRect(x, y, w, h)

      // Draw confidence label
      ctx.fillStyle = ctx.strokeStyle
      ctx.font = '12px monospace'
      const label = `${(confidence * 100).toFixed(0)}%`
      const labelWidth = ctx.measureText(label).width + 8
      ctx.fillRect(x, y - 18, labelWidth, 18)
      ctx.fillStyle = '#fff'
      ctx.fillText(label, x + 4, y - 5)
    }

    // Draw landmarks
    if (showLandmarks && landmarks) {
      ctx.fillStyle = '#3b82f6' // blue
      const radius = 3

      const landmarkPoints = [
        landmarks.left_eye,
        landmarks.right_eye,
        landmarks.nose,
        landmarks.mouth_left,
        landmarks.mouth_right,
      ]

      landmarkPoints.forEach((point) => {
        if (point) {
          ctx.beginPath()
          ctx.arc(offsetX + point[0] * scaleX, offsetY + point[1] * scaleY, radius, 0, Math.PI * 2)
          ctx.fill()
        }
      })
    }
  }, [detection, frameWidth, frameHeight, containerWidth, containerHeight, showBbox, showLandmarks])

  return (
    <canvas
      ref={canvasRef}
      className={cn('absolute inset-0 pointer-events-none', className)}
      style={{ width: '100%', height: '100%' }}
    />
  )
}
