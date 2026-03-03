import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { formatDuration } from "@/lib/formatters";
import type { FrameCaption } from "@/api/types";

interface CaptionFrameCardProps {
  frame: FrameCaption;
  videoPath?: string;
}

export function CaptionFrameCard({
  frame,
  videoPath: _videoPath,
}: CaptionFrameCardProps) {
  const {
    frame_index,
    timestamp,
    raw_caption,
    tags,
    scene_index,
    prompt_type_used,
  } = frame;

  return (
    <Card className="overflow-hidden">
      <CardContent className="p-4 space-y-3">
        {/* Header */}
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Frame {frame_index}</span>
          <span className="text-xs text-muted-foreground">
            {formatDuration(timestamp)}
          </span>
        </div>

        {/* Scene info */}
        {scene_index !== null && scene_index !== undefined && (
          <Badge variant="outline" className="text-xs">
            Scene {scene_index + 1}
          </Badge>
        )}

        {/* Raw caption */}
        {raw_caption && (
          <div className="bg-muted/50 rounded p-2">
            <p className="text-xs text-muted-foreground line-clamp-3">
              {raw_caption || "(no caption)"}
            </p>
          </div>
        )}

        {/* Tags */}
        {tags && tags.length > 0 ? (
          <div className="flex flex-wrap gap-1">
            {tags.slice(0, 10).map((tag, idx) => (
              <Badge
                key={`${tag.tag}-${idx}`}
                variant="secondary"
                className="text-xs"
                title={`Confidence: ${(tag.confidence * 100).toFixed(1)}%`}
              >
                {tag.tag}
              </Badge>
            ))}
            {tags.length > 10 && (
              <Badge variant="outline" className="text-xs">
                +{tags.length - 10} more
              </Badge>
            )}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground italic">
            No tags extracted
          </p>
        )}

        {/* Prompt type */}
        <div className="text-xs text-muted-foreground">
          Prompt: {prompt_type_used}
        </div>
      </CardContent>
    </Card>
  );
}
