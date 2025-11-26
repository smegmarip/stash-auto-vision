import { Link } from "react-router-dom";
import { useJobResults, useJobStatus } from "@/hooks/useJobs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { ServiceSummary } from "./ServiceSummary";
import { JobProgress } from "@/components/jobs/JobProgress";
import { JobStatusBadge } from "@/components/jobs/JobStatusBadge";
import { formatDate, formatDuration } from "@/lib/formatters";
import type { JobStatus } from "@/api/types";

interface RollupDetailViewProps {
  jobId: string;
}

export function RollupDetailView({ jobId }: RollupDetailViewProps) {
  const { data: status, isLoading: statusLoading } = useJobStatus(jobId);
  const { data: results, isLoading: resultsLoading } = useJobResults(jobId, {
    enabled: status?.status === "completed",
  });

  if (statusLoading) {
    return <RollupDetailSkeleton />;
  }

  const isActive =
    status?.status === "processing" || status?.status === "queued";
  const isCompleted = status?.status === "completed";

  const services = [
    { key: "scenes", label: "Scene Detection" },
    { key: "faces", label: "Face Recognition" },
    { key: "semantics", label: "Semantic Analysis" },
    { key: "objects", label: "Object Detection" },
  ];

  return (
    <div className="space-y-6">
      {/* Job Status Header */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle>Job Overview</CardTitle>
            {status && <JobStatusBadge status={status.status as JobStatus} />}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Progress for active jobs */}
          {isActive && status && (
            <div className="space-y-2">
              <JobProgress progress={status.progress} />
              {status.message && (
                <p className="text-sm text-muted-foreground">
                  {status.message}
                </p>
              )}
              {status.stage && (
                <Badge variant="outline">Stage: {status.stage}</Badge>
              )}
            </div>
          )}

          {/* Job metadata */}
          <dl className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <dt className="text-muted-foreground">Job ID</dt>
              <dd className="font-mono text-xs mt-1">{jobId}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Processing Mode</dt>
              <dd className="mt-1">
                {status?.processing_mode || "sequential"}
              </dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Created</dt>
              <dd className="mt-1">{formatDate(status?.created_at)}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Completed</dt>
              <dd className="mt-1">{formatDate(status?.completed_at)}</dd>
            </div>
          </dl>

          {/* Processing time for completed jobs */}
          {isCompleted && results?.metadata && (
            <div className="pt-2 border-t">
              <span className="text-sm text-muted-foreground">
                Total processing time:{" "}
                <span className="font-medium text-foreground">
                  {formatDuration(results.metadata.processing_time_seconds)}
                </span>
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Service Summaries - links to detail views */}
      <div>
        <h3 className="font-medium mb-4">Service Results</h3>
        {resultsLoading ? (
          <div className="grid gap-4 md:grid-cols-2">
            {services.map((svc) => (
              <Card key={svc.key}>
                <CardContent className="p-4">
                  <Skeleton className="h-20 w-full" />
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {services.map((svc) => {
              const rawData = results?.[svc.key as keyof typeof results];

              // Normalize data structure:
              // - If array (service-specific job), wrap it: { faces: [...] } or { scenes: [...] }
              // - If dict (orchestrated job), use as-is
              // - If null/undefined, leave as undefined
              let serviceData: Record<string, unknown> | undefined;
              if (Array.isArray(rawData)) {
                // Service-specific job returns array directly
                serviceData = {
                  [svc.key]: rawData,
                  metadata: results?.metadata,
                };
              } else if (rawData && typeof rawData === "object") {
                serviceData = rawData as Record<string, unknown>;
              }

              const hasData =
                serviceData &&
                (Array.isArray(serviceData[svc.key])
                  ? (serviceData[svc.key] as unknown[]).length > 0
                  : Object.keys(serviceData).length > 0);
              const isNotImplemented =
                serviceData &&
                (serviceData as { status?: string }).status ===
                  "not_implemented";

              // Only link to detail view if we have actual data
              if (hasData && !isNotImplemented) {
                return (
                  <Link key={svc.key} to={`/jobs/${jobId}/${svc.key}`}>
                    <ServiceSummary
                      service={svc.key}
                      label={svc.label}
                      data={serviceData}
                    />
                  </Link>
                );
              }

              return (
                <ServiceSummary
                  key={svc.key}
                  service={svc.key}
                  label={svc.label}
                  data={serviceData}
                />
              );
            })}
          </div>
        )}
      </div>

      {/* Source info */}
      {results?.source_id && (
        <Card>
          <CardContent className="p-4">
            <dl className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <dt className="text-muted-foreground">Scene ID</dt>
                <dd className="font-mono mt-1">{results.source_id}</dd>
              </div>
              {status?.services && status.services.length > 0 && (
                <div>
                  <dt className="text-muted-foreground">Services Used</dt>
                  <dd className="mt-1 flex gap-1">
                    {status.services.map((svc) => (
                      <Badge
                        key={svc.service}
                        variant="outline"
                        className="text-xs"
                      >
                        {svc.service}
                      </Badge>
                    ))}
                  </dd>
                </div>
              )}
            </dl>
          </CardContent>
        </Card>
      )}

      {/* Error display */}
      {status?.error && (
        <Card className="border-destructive">
          <CardHeader className="pb-2">
            <CardTitle className="text-destructive text-sm">Error</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-xs text-destructive bg-destructive/10 p-3 rounded overflow-auto">
              {status.error}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function RollupDetailSkeleton() {
  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="p-4">
          <Skeleton className="h-24 w-full" />
        </CardContent>
      </Card>
      <div className="grid gap-4 md:grid-cols-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i}>
            <CardContent className="p-4">
              <Skeleton className="h-20 w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
