import { useJobResults } from "@/hooks/useJobs";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { CaptionFrameCard } from "./CaptionFrameCard";
import { formatDuration, formatNumber } from "@/lib/formatters";
import { Film, Clock, Tags, Settings, MessageSquare, Sparkles, Users } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { CaptionsResult } from "@/api/types";

interface CaptionsDetailViewProps {
  jobId: string;
}

export function CaptionsDetailView({ jobId }: CaptionsDetailViewProps) {
  const { data: results, isLoading, error } = useJobResults(jobId);

  if (isLoading) {
    return <CaptionsDetailSkeleton />;
  }

  if (error) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-destructive">
          Failed to load captioning results: {error.message}
        </CardContent>
      </Card>
    );
  }

  const captionsResult = results as unknown as CaptionsResult | null;
  if (!captionsResult || !captionsResult.captions?.frames) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          No caption data available for this job
        </CardContent>
      </Card>
    );
  }

  const { frames, scene_summaries } = captionsResult.captions;
  const metadata = captionsResult.metadata || captionsResult.captions.metadata;

  const framesCaptioned = metadata?.frames_captioned ?? frames?.length ?? 0;
  const processingTime = metadata?.processing_time_seconds ?? 0;
  const model = metadata?.model ?? "unknown";
  const modelVariant = metadata?.model_variant ?? "";
  const device = metadata?.device ?? "unknown";
  const quantization = metadata?.quantization ?? "none";
  const promptType = metadata?.prompt_type ?? "unknown";
  const vramPeak = metadata?.vram_peak_mb;
  const videoPath = metadata?.source || "";
  const framesAnalyzed = metadata?.frames_analyzed;
  const sharpnessFiltered = metadata?.sharpness_filtered;
  const gpuWaitTime = metadata?.gpu_wait_time_seconds;

  // Collect all unique tags across frames
  const allTags = new Set<string>();
  let totalTags = 0;
  let framesWithSummary = 0;
  let totalPersons = 0;
  frames.forEach((frame) => {
    frame.tags?.forEach((tag) => {
      allTags.add(tag.tag);
      totalTags++;
    });
    if (frame.summary) {
      framesWithSummary++;
      if (frame.summary.persons?.count) {
        totalPersons += frame.summary.persons.count;
      }
    }
  });

  return (
    <div className="space-y-6">
      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          icon={<Film className="h-5 w-5" />}
          label="Frames Captioned"
          value={formatNumber(framesCaptioned)}
          subtitle={framesAnalyzed && sharpnessFiltered ? `${framesAnalyzed} analyzed` : undefined}
        />
        <StatCard
          icon={<Tags className="h-5 w-5" />}
          label="Unique Tags"
          value={formatNumber(allTags.size)}
          subtitle={`${formatNumber(totalTags)} total`}
        />
        {framesWithSummary > 0 && (
          <StatCard
            icon={<Sparkles className="h-5 w-5" />}
            label="Scene Summaries"
            value={formatNumber(framesWithSummary)}
          />
        )}
        {totalPersons > 0 && (
          <StatCard
            icon={<Users className="h-5 w-5" />}
            label="Persons Detected"
            value={formatNumber(totalPersons)}
          />
        )}
        <StatCard
          icon={<Clock className="h-5 w-5" />}
          label="Processing Time"
          value={formatDuration(processingTime)}
          subtitle={gpuWaitTime ? `+${formatDuration(gpuWaitTime)} GPU wait` : undefined}
        />
      </div>

      {/* Model Info */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center gap-4">
            <Settings className="h-5 w-5 text-muted-foreground" />
            <div className="flex-1">
              <p className="text-sm font-medium">
                Model: {model} {modelVariant && `(${modelVariant})`}
              </p>
              <p className="text-xs text-muted-foreground">
                Device: {device.toUpperCase()} | Quantization: {quantization} |
                Prompt: {promptType}
              </p>
              <div className="flex flex-wrap gap-2 mt-2">
                {vramPeak && (
                  <Badge variant="outline" className="text-xs">
                    Peak VRAM: {formatNumber(Math.round(vramPeak))} MB
                  </Badge>
                )}
                {sharpnessFiltered && (
                  <Badge variant="secondary" className="text-xs">
                    Sharpness Filtered
                  </Badge>
                )}
                {promptType === "scene_summary" && (
                  <Badge variant="secondary" className="text-xs">
                    Structured Output
                  </Badge>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Scene Summaries (if available) */}
      {scene_summaries &&
        Array.isArray(scene_summaries) &&
        scene_summaries.length > 0 && (
          <Card>
            <CardContent className="p-4">
              <h3 className="font-medium mb-4">Scene Summaries</h3>
              <div className="space-y-3">
                {scene_summaries.map((summary, idx) => (
                  <div key={idx} className="border rounded-lg p-3 space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">
                        Scene {summary.scene_index + 1}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {formatDuration(summary.start_timestamp)} -{" "}
                        {formatDuration(summary.end_timestamp)}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {summary.dominant_tags.map((tag) => (
                        <Badge
                          key={tag}
                          variant="secondary"
                          className="text-xs"
                        >
                          {tag}
                        </Badge>
                      ))}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {summary.frame_count} frames | Avg confidence:{" "}
                      {(summary.avg_confidence * 100).toFixed(1)}%
                    </div>
                    {summary.combined_caption && (
                      <p className="text-xs text-muted-foreground italic mt-2">
                        {summary.combined_caption}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

      {/* Frame Results */}
      {frames.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            No frames captioned
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          <h3 className="font-medium">
            {frames.length} Frame{frames.length !== 1 ? "s" : ""} Captioned
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {frames.map((frame) => (
              <CaptionFrameCard
                key={frame.frame_index}
                frame={frame}
                videoPath={videoPath}
              />
            ))}
          </div>
        </div>
      )}

      {/* Metadata footer */}
      <div className="text-sm text-muted-foreground">
        Model: {model} | Device: {device.toUpperCase()} | {framesCaptioned}{" "}
        frames captioned
      </div>
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  subtitle,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  subtitle?: string;
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-lg text-primary">
            {icon}
          </div>
          <div>
            <p className="text-2xl font-bold">{value}</p>
            <p className="text-xs text-muted-foreground">{label}</p>
            {subtitle && (
              <p className="text-xs text-muted-foreground/70">{subtitle}</p>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function CaptionsDetailSkeleton() {
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
  );
}
