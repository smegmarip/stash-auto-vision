import { useState, useRef, useEffect } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { FaceThumbnail } from './FaceThumbnail'
import { FaceOverlay } from './FaceOverlay'
import { FrameViewer } from '@/components/shared/FrameViewer'
import { formatTimestamp } from '@/lib/formatters'
import type { Face } from '@/api/types'
import { User, Sparkles, Eye, EyeOff, ZoomIn } from 'lucide-react'

interface FaceCardProps {
  face: Face
  videoPath: string
  showOverlay?: boolean
}

export function FaceCard({ face, videoPath, showOverlay = true }: FaceCardProps) {
  const [dialogOpen, setDialogOpen] = useState(false)
  const [showBbox, setShowBbox] = useState(true)
  const [showLandmarks, setShowLandmarks] = useState(true)
  const [frameDimensions, setFrameDimensions] = useState({ width: 1920, height: 1080 })
  const [containerDimensions, setContainerDimensions] = useState({ width: 800, height: 450 })
  const containerRef = useRef<HTMLDivElement>(null)

  // Measure actual container dimensions
  useEffect(() => {
    if (!dialogOpen || !containerRef.current) return

    const measureContainer = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        setContainerDimensions({ width: rect.width, height: rect.height })
      }
    }

    measureContainer()
    window.addEventListener('resize', measureContainer)
    return () => window.removeEventListener('resize', measureContainer)
  }, [dialogOpen])

  const rep = face.representative_detection
  const demographics = face.demographics
  const quality = rep.quality


  return (
    <>
      <Card className="overflow-hidden hover:ring-2 hover:ring-primary/50 transition-all cursor-pointer">
        <CardContent className="p-4">
          <div className="flex gap-4">
            {/* Face thumbnail */}
            <div className="flex-shrink-0" onClick={() => setDialogOpen(true)}>
              <FaceThumbnail
                videoPath={videoPath}
                timestamp={rep.timestamp}
                bbox={rep.bbox}
                size={80}
                enhanced={rep.enhanced}
                className="hover:ring-2 hover:ring-primary"
              />
            </div>

            {/* Face info */}
            <div className="flex-1 min-w-0 space-y-2">
              {/* Demographics */}
              <div className="flex items-center gap-2 flex-wrap">
                {demographics && (
                  <>
                    <Badge variant="outline">
                      <User className="h-3 w-3 mr-1" />
                      {demographics.gender === 'M' ? 'Male' : 'Female'}
                    </Badge>
                    <Badge variant="outline">Age ~{demographics.age}</Badge>
                  </>
                )}
                {rep.enhanced && (
                  <Badge variant="secondary" className="text-xs">
                    <Sparkles className="h-3 w-3 mr-1" />
                    Enhanced
                  </Badge>
                )}
              </div>

              {/* Quality scores */}
              <div className="text-xs text-muted-foreground space-y-1">
                <div className="flex items-center gap-2">
                  <span>Quality:</span>
                  <QualityBar value={quality.composite} />
                  <span>{(quality.composite * 100).toFixed(0)}%</span>
                </div>
                <div className="flex items-center gap-2">
                  <span>Confidence:</span>
                  <span className="font-mono">{(rep.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>

              {/* Detection count and time */}
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span>{face.detections.length} detection{face.detections.length !== 1 ? 's' : ''}</span>
                {rep.timestamp != null && rep.timestamp >= 0 && (
                  <>
                    <span>•</span>
                    <span>@ {formatTimestamp(rep.timestamp)}</span>
                  </>
                )}
              </div>
            </div>

            {/* View button */}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setDialogOpen(true)}
              className="flex-shrink-0"
            >
              <ZoomIn className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Face detail dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              Face Details
              {rep.enhanced && (
                <Badge variant="secondary">
                  <Sparkles className="h-3 w-3 mr-1" />
                  Enhanced
                </Badge>
              )}
            </DialogTitle>
          </DialogHeader>

          {/* Controls */}
          <div className="flex items-center gap-2">
            <Button
              variant={showBbox ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowBbox(!showBbox)}
            >
              {showBbox ? <Eye className="h-4 w-4 mr-1" /> : <EyeOff className="h-4 w-4 mr-1" />}
              Bbox
            </Button>
            <Button
              variant={showLandmarks ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowLandmarks(!showLandmarks)}
            >
              {showLandmarks ? <Eye className="h-4 w-4 mr-1" /> : <EyeOff className="h-4 w-4 mr-1" />}
              Landmarks
            </Button>
          </div>

          {/* Frame with overlay */}
          <div ref={containerRef} className="relative aspect-video bg-black rounded-lg overflow-hidden">
            <FrameViewer
              videoPath={videoPath}
              timestamp={rep.timestamp}
              enhance={rep.enhanced}
              className="w-full h-full"
              onLoad={setFrameDimensions}
            />
            {showOverlay && (
              <FaceOverlay
                detection={rep}
                frameWidth={frameDimensions.width}
                frameHeight={frameDimensions.height}
                containerWidth={containerDimensions.width}
                containerHeight={containerDimensions.height}
                showBbox={showBbox}
                showLandmarks={showLandmarks}
              />
            )}
          </div>

          {/* Detailed metadata */}
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">Demographics</h4>
                <dl className="space-y-1 text-muted-foreground">
                  {demographics ? (
                    <>
                      <div className="flex justify-between">
                        <dt>Gender</dt>
                        <dd>{demographics.gender === 'M' ? 'Male' : 'Female'}</dd>
                      </div>
                      <div className="flex justify-between">
                        <dt>Age</dt>
                        <dd>~{demographics.age}</dd>
                      </div>
                      <div className="flex justify-between">
                        <dt>Emotion</dt>
                        <dd className="capitalize">{demographics.emotion}</dd>
                      </div>
                    </>
                  ) : (
                    <div>Demographics not available</div>
                  )}
                </dl>
              </div>

              <div>
                <h4 className="font-medium mb-2">Face Data</h4>
                <dl className="space-y-1 text-muted-foreground">
                  <div className="flex justify-between">
                    <dt>Bounding Box</dt>
                    <dd>{Math.round(rep.bbox.x_max - rep.bbox.x_min)}px x {Math.round(rep.bbox.y_max - rep.bbox.y_min)}px</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt>Pose</dt>
                    <dd>{rep.pose.replaceAll('-', ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}</dd>
                  </div>
                </dl>
              </div>
            </div>

            <div>
              <h4 className="font-medium mb-2">Quality Scores</h4>
              <dl className="space-y-1 text-muted-foreground">
                <div className="flex justify-between">
                  <dt>Overall</dt>
                  <dd>{(quality.composite * 100).toFixed(1)}%</dd>
                </div>
                <div className="flex justify-between">
                  <dt>Sharpness</dt>
                  <dd>{(quality.components.sharpness * 100).toFixed(1)}%</dd>
                </div>
                <div className="flex justify-between">
                  <dt>Pose</dt>
                  <dd>{(quality.components.pose * 100).toFixed(1)}%</dd>
                </div>
                <div className="flex justify-between">
                  <dt>Size</dt>
                  <dd>{(quality.components.size * 100).toFixed(1)}%</dd>
                </div>
                <div className="flex justify-between">
                  <dt>Occlusion</dt>
                  <dd>{rep.occlusion.occluded ? 'Yes' : 'No'} ({(rep.occlusion.probability * 100).toFixed(0)}%)</dd>
                </div>
                <div className="flex justify-between">
                  <dt>Confidence</dt>
                  <dd>{(rep.confidence * 100).toFixed(1)}%</dd>
                </div>
              </dl>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}

function QualityBar({ value }: { value: number }) {
  const percentage = Math.round(value * 100)
  const hue = value > 0.7 ? 120 : value > 0.4 ? 60 : 0

  return (
    <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
      <div
        className="h-full rounded-full"
        style={{
          width: `${percentage}%`,
          backgroundColor: `hsl(${hue}, 80%, 50%)`,
        }}
      />
    </div>
  )
}
