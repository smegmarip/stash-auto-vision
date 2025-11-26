import { useState, useRef, useEffect } from 'react'
import { Skeleton } from '@/components/ui/skeleton'
import { getFrameUrl } from '@/api/client'
import { cn } from '@/lib/utils'

interface FrameViewerProps {
  videoPath: string
  timestamp: number
  className?: string
  alt?: string
  onLoad?: (dimensions: { width: number; height: number }) => void
  enhance?: boolean
}

export function FrameViewer({
  videoPath,
  timestamp,
  className,
  alt = 'Video frame',
  onLoad,
  enhance = false,
}: FrameViewerProps) {
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState(false)
  const [isVisible, setIsVisible] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  // Lazy loading with IntersectionObserver
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setIsVisible(true)
          observer.disconnect()
        }
      },
      { threshold: 0.1 }
    )

    if (containerRef.current) {
      observer.observe(containerRef.current)
    }

    return () => observer.disconnect()
  }, [])

  const frameUrl = getFrameUrl(videoPath, timestamp, { enhance })

  return (
    <div
      ref={containerRef}
      className={cn('relative bg-muted overflow-hidden', className)}
    >
      {!loaded && !error && (
        <Skeleton className="absolute inset-0" />
      )}

      {error && (
        <div className="absolute inset-0 flex items-center justify-center text-muted-foreground text-sm">
          Failed to load frame
        </div>
      )}

      {isVisible && (
        <img
          src={frameUrl}
          alt={alt}
          className={cn(
            'w-full h-full object-contain transition-opacity',
            loaded ? 'opacity-100' : 'opacity-0'
          )}
          onLoad={(e) => {
            const img = e.target as HTMLImageElement
            setLoaded(true)
            onLoad?.({ width: img.naturalWidth, height: img.naturalHeight })
          }}
          onError={() => {
            setLoaded(true)
            setError(true)
          }}
        />
      )}
    </div>
  )
}
