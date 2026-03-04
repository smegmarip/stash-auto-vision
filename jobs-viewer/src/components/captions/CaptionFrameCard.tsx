import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatDuration, formatNumber } from "@/lib/formatters";
import { ChevronDown, ChevronUp, MapPin, Users, Camera, Palette, Activity } from "lucide-react";
import type { FrameCaption, SceneSummaryData } from "@/api/types";

interface CaptionFrameCardProps {
  frame: FrameCaption;
  videoPath?: string;
}

export function CaptionFrameCard({
  frame,
  videoPath: _videoPath,
}: CaptionFrameCardProps) {
  const [expanded, setExpanded] = useState(false);
  const {
    frame_index,
    timestamp,
    raw_caption,
    tags,
    summary,
    scene_index,
    prompt_type_used,
    sharpness_score,
  } = frame;

  const hasSummary = summary && Object.keys(summary).length > 0;

  return (
    <Card className="overflow-hidden">
      <CardContent className="p-4 space-y-3">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Frame {frame_index}</span>
            {sharpness_score !== undefined && (
              <Badge variant="outline" className="text-xs">
                Quality: {(sharpness_score * 100).toFixed(0)}%
              </Badge>
            )}
          </div>
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

        {/* Summary data (when using scene_summary prompt) */}
        {hasSummary && (
          <div className="space-y-2">
            <SummarySection summary={summary} expanded={expanded} />
            <Button
              variant="ghost"
              size="sm"
              className="w-full h-6 text-xs"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? (
                <>
                  <ChevronUp className="h-3 w-3 mr-1" /> Show Less
                </>
              ) : (
                <>
                  <ChevronDown className="h-3 w-3 mr-1" /> Show More Details
                </>
              )}
            </Button>
          </div>
        )}

        {/* Raw caption (for non-summary prompts) */}
        {!hasSummary && raw_caption && (
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
                variant={tag.stash_tag_id ? "default" : "secondary"}
                className="text-xs"
                title={`Confidence: ${(tag.confidence * 100).toFixed(1)}%${tag.stash_tag_id ? " (Aligned)" : ""}`}
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

function SummarySection({ summary, expanded }: { summary: SceneSummaryData; expanded: boolean }) {
  return (
    <div className="bg-muted/30 rounded p-3 space-y-2">
      {/* Location - Always visible */}
      {(summary.setting || summary.locale) && (
        <div className="flex items-start gap-2">
          <MapPin className="h-3 w-3 mt-0.5 text-muted-foreground shrink-0" />
          <div className="text-xs">
            <span className="font-medium">{summary.setting}</span>
            {summary.locale && (
              <span className="text-muted-foreground"> ({summary.locale})</span>
            )}
            {summary.location_details && expanded && (
              <p className="text-muted-foreground mt-1">{summary.location_details}</p>
            )}
          </div>
        </div>
      )}

      {/* Persons - Always visible if present */}
      {summary.persons && summary.persons.count > 0 && (
        <div className="flex items-start gap-2">
          <Users className="h-3 w-3 mt-0.5 text-muted-foreground shrink-0" />
          <div className="text-xs">
            <span className="font-medium">{summary.persons.count} person{summary.persons.count !== 1 ? "s" : ""}</span>
            {expanded && summary.persons.details && summary.persons.details.length > 0 && (
              <div className="mt-1 space-y-1">
                {summary.persons.details.map((person, idx) => (
                  <div key={idx} className="text-muted-foreground">
                    {[person.gender, person.age_range, person.body_type, person.expression]
                      .filter(Boolean)
                      .join(", ")}
                    {person.pose && ` - ${person.pose}`}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Activities - Always visible if present */}
      {summary.activities && summary.activities.length > 0 && (
        <div className="flex items-start gap-2">
          <Activity className="h-3 w-3 mt-0.5 text-muted-foreground shrink-0" />
          <div className="text-xs">
            <span className="font-medium">Activities:</span>{" "}
            <span className="text-muted-foreground">
              {summary.activities.slice(0, expanded ? undefined : 3).join(", ")}
              {!expanded && summary.activities.length > 3 && "..."}
            </span>
          </div>
        </div>
      )}

      {/* Expanded content */}
      {expanded && (
        <>
          {/* Cinematography */}
          {summary.cinematography && (
            <div className="flex items-start gap-2">
              <Camera className="h-3 w-3 mt-0.5 text-muted-foreground shrink-0" />
              <div className="text-xs">
                <span className="font-medium">Shot:</span>{" "}
                <span className="text-muted-foreground">
                  {[
                    summary.cinematography.shot_type,
                    summary.cinematography.camera_angle,
                    summary.cinematography.framing,
                  ]
                    .filter(Boolean)
                    .join(", ")}
                </span>
              </div>
            </div>
          )}

          {/* Visual Style */}
          {summary.visual_style && (
            <div className="flex items-start gap-2">
              <Palette className="h-3 w-3 mt-0.5 text-muted-foreground shrink-0" />
              <div className="text-xs">
                <span className="font-medium">Style:</span>{" "}
                <span className="text-muted-foreground">
                  {[
                    summary.visual_style.visual_style,
                    summary.visual_style.color_grading,
                    summary.visual_style.era_aesthetic,
                  ]
                    .filter(Boolean)
                    .join(", ")}
                </span>
                {summary.visual_style.color_palette && summary.visual_style.color_palette.length > 0 && (
                  <div className="flex gap-1 mt-1">
                    {summary.visual_style.color_palette.map((color, idx) => (
                      <span
                        key={idx}
                        className="inline-block px-1.5 py-0.5 rounded text-xs bg-muted"
                      >
                        {color}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Mood & Genre */}
          {(summary.mood || summary.genre) && (
            <div className="flex flex-wrap gap-1 mt-2">
              {summary.mood && (
                <Badge variant="outline" className="text-xs">
                  Mood: {summary.mood}
                </Badge>
              )}
              {summary.genre && (
                <Badge variant="outline" className="text-xs">
                  Genre: {summary.genre}
                </Badge>
              )}
              {summary.tension_level && (
                <Badge variant="outline" className="text-xs">
                  Tension: {summary.tension_level}
                </Badge>
              )}
            </div>
          )}

          {/* Objects */}
          {summary.objects && summary.objects.length > 0 && (
            <div className="text-xs">
              <span className="font-medium">Objects:</span>{" "}
              <span className="text-muted-foreground">
                {summary.objects.join(", ")}
              </span>
            </div>
          )}

          {/* Attire */}
          {summary.attire && summary.attire.length > 0 && (
            <div className="text-xs">
              <span className="font-medium">Attire:</span>{" "}
              <span className="text-muted-foreground">
                {summary.attire.join(", ")}
              </span>
            </div>
          )}

          {/* Narrative Context */}
          {summary.narrative_context && (
            <div className="text-xs italic text-muted-foreground mt-2 border-l-2 border-muted pl-2">
              {summary.narrative_context}
            </div>
          )}
        </>
      )}
    </div>
  );
}
