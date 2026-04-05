import { useJobResults } from '@/hooks/useJobs'
import { Card, CardContent } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { formatDuration, formatNumber } from '@/lib/formatters'
import { Film, Clock, Tags, Settings, Sparkles, FileText, ChevronDown, ChevronUp } from 'lucide-react'
import { useState } from 'react'
import type { ClassifierTag, FrameCaptionResult } from '@/api/types'

interface SemanticsDetailViewProps {
  jobId: string
}

export function SemanticsDetailView({ jobId }: SemanticsDetailViewProps) {
  const { data: results, isLoading, error } = useJobResults(jobId)
  const [showCaptions, setShowCaptions] = useState(false)
  const [showSummary, setShowSummary] = useState(false)

  if (isLoading) {
    return <SemanticsDetailSkeleton />
  }

  if (error) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-destructive">
          Failed to load semantics results: {error.message}
        </CardContent>
      </Card>
    )
  }

  const result = results?.semantics
  if (!result) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          No semantics data available for this job
        </CardContent>
      </Card>
    )
  }

  // Handle both nested (semantics.semantics) and flat layouts
  const outcome = result.semantics || result
  const metadata = result.metadata
  const tags: ClassifierTag[] = outcome.tags || []
  const frameCaptions: FrameCaptionResult[] = outcome.frame_captions || []
  const sceneSummary: string = outcome.scene_summary || ''

  const directTags = tags.filter(t => t.decode_type === 'direct')
  const parentTags = tags.filter(t => t.decode_type === 'parent_only')

  return (
    <div className="space-y-6">
      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          icon={<Tags className="h-5 w-5" />}
          label="Tags Predicted"
          value={formatNumber(tags.length)}
        />
        <StatCard
          icon={<Sparkles className="h-5 w-5" />}
          label="Direct Tags"
          value={formatNumber(directTags.length)}
        />
        <StatCard
          icon={<Film className="h-5 w-5" />}
          label="Frames Captioned"
          value={formatNumber(metadata?.frames_captioned ?? frameCaptions.length)}
        />
        <StatCard
          icon={<Clock className="h-5 w-5" />}
          label="Processing Time"
          value={formatDuration(metadata?.processing_time_seconds ?? 0)}
        />
      </div>

      {/* Model Info */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center gap-4">
            <Settings className="h-5 w-5 text-muted-foreground" />
            <div className="flex-1 space-y-1">
              <p className="text-sm font-medium">
                Classifier: {metadata?.classifier_model ?? 'unknown'}
              </p>
              <p className="text-xs text-muted-foreground">
                Device: {metadata?.device ?? 'unknown'} | Taxonomy: {metadata?.taxonomy_size ?? '?'} tags
              </p>
              <div className="flex gap-2 mt-1">
                {metadata?.has_promo && (
                  <Badge variant="secondary" className="text-xs">Promo available</Badge>
                )}
                {outcome.scene_embedding && (
                  <Badge variant="secondary" className="text-xs">
                    {outcome.scene_embedding.length}-D embedding
                  </Badge>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Predicted Tags */}
      {tags.length > 0 && (
        <Card>
          <CardContent className="p-4">
            <h3 className="font-medium mb-4">
              Predicted Tags ({tags.length})
            </h3>
            <div className="space-y-4">
              {/* Direct predictions */}
              <div className="flex flex-wrap gap-2">
                {directTags.map((tag) => (
                  <TagBadge key={tag.tag_id} tag={tag} />
                ))}
              </div>
              {/* Parent-only activations */}
              {parentTags.length > 0 && (
                <div>
                  <p className="text-xs text-muted-foreground mb-2">
                    Parent activations ({parentTags.length})
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {parentTags.map((tag) => (
                      <TagBadge key={tag.tag_id} tag={tag} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Scene Summary */}
      {sceneSummary && (
        <Card>
          <CardContent className="p-4">
            <button
              onClick={() => setShowSummary(!showSummary)}
              className="flex items-center gap-2 w-full text-left"
            >
              <FileText className="h-4 w-4 text-muted-foreground" />
              <h3 className="font-medium flex-1">Scene Summary</h3>
              {showSummary ? (
                <ChevronUp className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              )}
            </button>
            {showSummary && (
              <p className="mt-3 text-sm text-muted-foreground whitespace-pre-wrap leading-relaxed border-l-2 border-muted pl-4">
                {sceneSummary}
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {/* Frame Captions */}
      {frameCaptions.length > 0 && (
        <Card>
          <CardContent className="p-4">
            <button
              onClick={() => setShowCaptions(!showCaptions)}
              className="flex items-center gap-2 w-full text-left"
            >
              <Film className="h-4 w-4 text-muted-foreground" />
              <h3 className="font-medium flex-1">
                Frame Captions ({frameCaptions.length})
              </h3>
              {showCaptions ? (
                <ChevronUp className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              )}
            </button>
            {showCaptions && (
              <div className="mt-3 space-y-3">
                {frameCaptions.map((fc) => (
                  <div
                    key={fc.frame_index}
                    className="border rounded-lg p-3 space-y-1"
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-medium">
                        Frame {fc.frame_index}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {formatDuration(fc.timestamp)}
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {fc.caption}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Metadata footer */}
      <div className="text-sm text-muted-foreground">
        Classifier: {metadata?.classifier_model ?? '?'} |
        Caption model: {metadata?.caption_model ?? '?'} |
        Summary model: {metadata?.summary_model ?? '?'} |
        Device: {(metadata?.device ?? '?').toUpperCase()}
      </div>
    </div>
  )
}

function TagBadge({ tag }: { tag: ClassifierTag }) {
  const isDirect = tag.decode_type === 'direct'
  return (
    <div className="flex flex-col gap-0.5">
      <Badge
        variant="outline"
        className={
          isDirect
            ? 'bg-emerald-500/10 text-emerald-700 border-emerald-500/20'
            : 'bg-slate-500/10 text-slate-600 border-slate-500/20'
        }
        title={tag.path || tag.tag_name}
      >
        {tag.tag_name}
      </Badge>
      <span className="text-[10px] text-muted-foreground text-center">
        {(tag.score * 100).toFixed(1)}%
      </span>
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

function SemanticsDetailSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i}>
            <CardContent className="p-4">
              <Skeleton className="h-16 w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
      <Card>
        <CardContent className="p-4">
          <Skeleton className="h-24 w-full" />
        </CardContent>
      </Card>
    </div>
  )
}
