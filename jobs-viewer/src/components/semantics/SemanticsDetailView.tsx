import { useJobResults } from '@/hooks/useJobs'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { formatNumber } from '@/lib/formatters'
import {
  Film, Clock, Tags, Sparkles, FileText,
  ChevronDown, ChevronUp, Image as ImageIcon, Users,
} from 'lucide-react'
import { useState, useMemo, useEffect } from 'react'
import type { ClassifierTag, FrameCaptionResult, SemanticsMetadata } from '@/api/types'
import { SpriteFrame } from './SpriteFrame'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Fetch STASH_URL from server config endpoint */
let _stashUrl: string | undefined
async function getStashUrl(): Promise<string> {
  if (_stashUrl !== undefined) return _stashUrl
  try {
    const res = await fetch('/api/config')
    const data = await res.json()
    _stashUrl = data.stashUrl || ''
  } catch {
    _stashUrl = ''
  }
  return _stashUrl!
}

/** Build proxied URL */
function proxyUrl(url: string): string {
  return `/api/proxy?url=${encodeURIComponent(url)}`
}

/** Extract scene title from source path */
function sceneTitle(scene?: { title?: string }, source?: string, sourceId?: string): string {
  if (scene?.title) return scene.title
  if (source) {
    const filename = source.split('/').pop() || source
    return filename.replace(/\.[^.]+$/, '')
  }
  return sourceId ? `Scene ${sourceId}` : 'Unknown Scene'
}

/** Format duration as m:ss or h:mm:ss */
function formatSceneDuration(seconds?: number | null): string {
  if (!seconds) return '—'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  return `${m}:${s.toString().padStart(2, '0')}`
}

/** Group tags by common ancestor at depth cutoff. */
function groupTagsByAncestor(tags: ClassifierTag[], tagNameToId?: Record<string, string>, maxDepth = 3) {
  const groups: Record<string, { groupTagId: string | null; tags: ClassifierTag[] }> = {}
  const shallow: ClassifierTag[] = []

  for (const tag of tags) {
    const parts = tag.path ? tag.path.split(' > ') : []
    if (parts.length <= maxDepth) {
      shallow.push(tag)
    } else {
      const groupKey = parts[2] || parts[parts.length - 1]
      if (!groups[groupKey]) {
        groups[groupKey] = { groupTagId: tagNameToId?.[groupKey] || null, tags: [] }
      }
      groups[groupKey].tags.push(tag)
    }
  }

  const sorted = Object.entries(groups).sort((a, b) => b[1].tags.length - a[1].tags.length)
  return { groups: sorted, shallow }
}

// ---------------------------------------------------------------------------
// Color palette for tag decode types
// ---------------------------------------------------------------------------
const TAG_COLORS: Record<string, string> = {
  direct: 'bg-emerald-500/10 text-emerald-700 border-emerald-500/20 dark:text-emerald-400',
  competition: 'bg-amber-500/10 text-amber-700 border-amber-500/20 dark:text-amber-400',
  parent_only: 'bg-slate-500/10 text-slate-600 border-slate-500/20 dark:text-slate-400',
}

const TAG_DOT_COLORS: Record<string, string> = {
  direct: 'bg-emerald-500',
  competition: 'bg-amber-500',
  parent_only: 'bg-slate-400',
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface SemanticsDetailViewProps {
  jobId: string
}

export function SemanticsDetailView({ jobId }: SemanticsDetailViewProps) {
  const { data: results, isLoading, error } = useJobResults(jobId)
  const [showCaptions, setShowCaptions] = useState(false)
  const [showSummary, setShowSummary] = useState(true)
  const [stashUrl, setStashUrl] = useState<string>('')

  useEffect(() => { getStashUrl().then(setStashUrl) }, [])

  if (isLoading) return <SemanticsDetailSkeleton />

  if (error) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-destructive">
          Failed to load semantics results: {error.message}
        </CardContent>
      </Card>
    )
  }

  // Data shape varies:
  // - Semantics-only job: { semantics: {tags, frame_captions, ...}, metadata: SemanticsMetadata }
  // - Rollup job: { semantics: { semantics: {...}, metadata: {...} }, metadata: JobMetadata }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw = results as any
  const semanticsBlock = raw?.semantics
  // Handle both nested and flat layouts
  const outcome = semanticsBlock?.tags ? semanticsBlock : semanticsBlock?.semantics
  const metadata: SemanticsMetadata | undefined = raw?.metadata?.classifier_model ? raw.metadata : semanticsBlock?.metadata

  if (!outcome) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          No semantics data available for this job
        </CardContent>
      </Card>
    )
  }

  const tags: ClassifierTag[] = outcome.tags || []
  const frameCaptions: FrameCaptionResult[] = outcome.frame_captions || []
  const sceneSummary: string = outcome.scene_summary || ''
  const scene = metadata?.scene

  const directTags = tags.filter(t => t.decode_type === 'direct')
  const competitionTags = tags.filter(t => t.decode_type === 'competition')
  const parentTags = tags.filter(t => t.decode_type === 'parent_only')

  const title = sceneTitle(scene, metadata?.source, metadata?.source_id)
  const screenshotUrl = stashUrl && metadata?.source_id ? proxyUrl(`${stashUrl}/scene/${metadata.source_id}/screenshot`) : null

  return (
    <div className="space-y-6">
      {/* Model info */}
      <Card>
        <CardContent className="p-4">
          <dl className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <dt className="text-muted-foreground">Classifier</dt>
              <dd className="mt-1 font-medium">{metadata?.classifier_model || '—'}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Device</dt>
              <dd className="mt-1">{(metadata?.device || '—').toUpperCase()}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Taxonomy</dt>
              <dd className="mt-1">{metadata?.taxonomy_size ?? '—'} tags</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Processing Time</dt>
              <dd className="mt-1">{metadata?.processing_time_seconds ? `${metadata.processing_time_seconds.toFixed(1)}s` : '—'}</dd>
            </div>
          </dl>
        </CardContent>
      </Card>

      {/* Stats row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard icon={<Tags className="h-5 w-5" />} label="Tags Predicted" value={formatNumber(tags.length)} />
        <StatCard icon={<Sparkles className="h-5 w-5" />} label="Direct Tags" value={formatNumber(directTags.length)} />
        <StatCard icon={<Film className="h-5 w-5" />} label="Frames Captioned" value={formatNumber(metadata?.frames_captioned ?? frameCaptions.length)} />
        <StatCard icon={<Clock className="h-5 w-5" />} label="Frames Extracted" value={formatNumber(metadata?.total_frames_extracted ?? 0)} />
      </div>

      {/* Scene header: title + cover */}
      <div className="space-y-4">
        <div>
          <h2 className="text-xl font-semibold">{title}</h2>
          {metadata?.source_id && (
            <p className="text-sm text-muted-foreground">Scene ID: {metadata.source_id}</p>
          )}
        </div>

        {/* Cover image */}
        {screenshotUrl && (
          <div className="aspect-video max-w-3xl bg-muted rounded-lg overflow-hidden">
            <img
              src={screenshotUrl}
              alt={title}
              className="w-full h-full object-contain"
              onError={(e) => { (e.currentTarget.parentElement as HTMLElement).style.display = 'none' }}
            />
          </div>
        )}

        {/* Scene metadata */}
        {scene && (
          <dl className="flex flex-wrap gap-x-6 gap-y-2 text-sm text-muted-foreground">
            {scene.duration != null && scene.duration > 0 && (
              <div className="flex items-center gap-1.5">
                <Clock className="h-3.5 w-3.5" />
                <dd>{formatSceneDuration(scene.duration)}</dd>
              </div>
            )}
            {scene.resolution && (
              <div className="flex items-center gap-1.5">
                <Film className="h-3.5 w-3.5" />
                <dd>{scene.resolution}{scene.frame_rate ? ` @ ${scene.frame_rate.toFixed(1)}fps` : ''}</dd>
              </div>
            )}
            {scene.performer_count > 0 && (
              <div className="flex items-center gap-1.5">
                <Users className="h-3.5 w-3.5" />
                <dd>{scene.performer_count} performer{scene.performer_count !== 1 ? 's' : ''}</dd>
              </div>
            )}
          </dl>
        )}
      </div>

      {/* Tags grouped by taxonomy */}
      {tags.length > 0 && (
        <TagsSection
          tags={tags}
          directCount={directTags.length}
          competitionCount={competitionTags.length}
          parentCount={parentTags.length}
          stashUrl={stashUrl}
          tagNameToId={metadata?.tag_name_to_id}
        />
      )}

      {/* Scene summary */}
      {sceneSummary && (
        <Card>
          <CardContent className="p-4">
            <button onClick={() => setShowSummary(!showSummary)} className="flex items-center gap-2 w-full text-left">
              <FileText className="h-4 w-4 text-muted-foreground" />
              <h3 className="font-medium flex-1">Scene Summary</h3>
              {showSummary ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
            </button>
            {showSummary && (
              <p className="mt-3 text-sm text-muted-foreground whitespace-pre-wrap leading-relaxed border-l-2 border-muted pl-4">
                {sceneSummary}
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {/* Frame captions */}
      {frameCaptions.length > 0 && (
        <Card>
          <CardContent className="p-4">
            <button onClick={() => setShowCaptions(!showCaptions)} className="flex items-center gap-2 w-full text-left">
              <Film className="h-4 w-4 text-muted-foreground" />
              <h3 className="font-medium flex-1">Frame Captions ({frameCaptions.length})</h3>
              {showCaptions ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
            </button>
            {showCaptions && (
              <div className="mt-3 space-y-2">
                {frameCaptions.map((fc) => (
                  <FrameCaptionRow key={fc.frame_index} caption={fc} spriteUrl={metadata?.sprite_image_url} vttUrl={metadata?.sprite_vtt_url} />
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Footer */}
      <div className="text-xs text-muted-foreground flex flex-wrap gap-x-3 gap-y-1">
        <span>Classifier: {metadata?.classifier_model || '—'}</span>
        <span>Caption: {metadata?.caption_model || '—'}</span>
        <span>Summary: {metadata?.summary_model || '—'}</span>
        <span>Device: {(metadata?.device || '—').toUpperCase()}</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tags section — grouped by taxonomy ancestor, each group in a Card
// ---------------------------------------------------------------------------

function TagsSection({
  tags,
  directCount,
  competitionCount,
  parentCount,
  stashUrl,
  tagNameToId,
}: {
  tags: ClassifierTag[]
  directCount: number
  competitionCount: number
  parentCount: number
  tagNameToId?: Record<string, string>
  stashUrl: string
}) {
  const { groups, shallow } = useMemo(() => groupTagsByAncestor(tags, tagNameToId), [tags, tagNameToId])

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-medium">Predicted Tags ({tags.length})</h3>
        <div className="flex gap-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <span className={`inline-block w-2 h-2 rounded-full ${TAG_DOT_COLORS.direct}`} />
            Direct ({directCount})
          </span>
          {competitionCount > 0 && (
            <span className="flex items-center gap-1">
              <span className={`inline-block w-2 h-2 rounded-full ${TAG_DOT_COLORS.competition}`} />
              Competition ({competitionCount})
            </span>
          )}
          {parentCount > 0 && (
            <span className="flex items-center gap-1">
              <span className={`inline-block w-2 h-2 rounded-full ${TAG_DOT_COLORS.parent_only}`} />
              Parent ({parentCount})
            </span>
          )}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {groups.map(([groupName, { groupTagId, tags: groupTags }]) => (
          <Card key={groupName}>
            <CardHeader className="pb-2 pt-3 px-4">
              <CardTitle className="text-sm flex items-center gap-2">
                {stashUrl && groupTagId && (
                  <img
                    src={proxyUrl(`${stashUrl}/tag/${groupTagId}/image`)}
                    alt={`${groupName} Icon`}
                    className="w-5 h-5 rounded object-cover"
                    onError={(e) => { e.currentTarget.style.display = 'none' }}
                  />
                )}
                {groupName}
                <Badge variant="secondary" className="text-[10px] ml-auto">{groupTags.length}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-3">
              <div className="flex flex-wrap gap-1.5">
                {groupTags.map((tag) => <TagBadge key={tag.tag_id} tag={tag} />)}
              </div>
            </CardContent>
          </Card>
        ))}

        {shallow.length > 0 && (
          <Card>
            <CardHeader className="pb-2 pt-3 px-4">
              <CardTitle className="text-sm">
                Other
                <Badge variant="secondary" className="text-[10px] ml-2">{shallow.length}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-3">
              <div className="flex flex-wrap gap-1.5">
                {shallow.map((tag) => <TagBadge key={tag.tag_id} tag={tag} />)}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tag badge
// ---------------------------------------------------------------------------

function TagBadge({ tag }: { tag: ClassifierTag }) {
  const colorClass = TAG_COLORS[tag.decode_type] || TAG_COLORS.direct
  return (
    <div className="flex flex-col gap-0.5">
      <Badge
        variant="outline"
        className={colorClass}
        title={`${tag.path || tag.tag_name}\n${tag.decode_type} (${(tag.score * 100).toFixed(1)}%)`}
      >
        {tag.tag_name}
      </Badge>
      <span className="text-[10px] text-muted-foreground text-center">
        {(tag.score * 100).toFixed(1)}%
      </span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Frame caption row with sprite thumbnail
// ---------------------------------------------------------------------------

function FrameCaptionRow({ caption, spriteUrl, vttUrl }: { caption: FrameCaptionResult; spriteUrl?: string; vttUrl?: string }) {
  const hasSprite = !!(spriteUrl && vttUrl)
  return (
    <div className="flex gap-3 border rounded-lg p-3">
      <div className="shrink-0">
        {hasSprite ? (
          <SpriteFrame spriteUrl={spriteUrl} vttUrl={vttUrl} timestamp={caption.timestamp} width={120} height={68} className="rounded" />
        ) : (
          <div className="w-[120px] h-[68px] bg-muted rounded flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <ImageIcon className="h-4 w-4 mx-auto mb-0.5 opacity-50" />
              <span className="text-[9px]">Frame {caption.frame_index}</span>
            </div>
          </div>
        )}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex justify-between items-center mb-1">
          <span className="text-xs font-medium">Frame {caption.frame_index}</span>
          <span className="text-xs text-muted-foreground">{formatTimestamp(caption.timestamp)}</span>
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed">{caption.caption}</p>
      </div>
    </div>
  )
}

function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

// ---------------------------------------------------------------------------
// Stat card
// ---------------------------------------------------------------------------

function StatCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
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

// ---------------------------------------------------------------------------
// Loading skeleton
// ---------------------------------------------------------------------------

function SemanticsDetailSkeleton() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-8 w-64" />
      <Skeleton className="aspect-video max-w-3xl rounded-lg" />
      <Card><CardContent className="p-4"><Skeleton className="h-16 w-full" /></CardContent></Card>
      <div className="grid gap-4 md:grid-cols-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i}><CardContent className="p-4"><Skeleton className="h-20 w-full" /></CardContent></Card>
        ))}
      </div>
    </div>
  )
}
