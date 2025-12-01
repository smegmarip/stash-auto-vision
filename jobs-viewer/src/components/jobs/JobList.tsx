import { useJobsInfinite, useJobsCount } from '@/hooks/useJobs'
import { useJobsStore } from '@/store/jobsStore'
import { JobCard } from './JobCard'
import { JobFilters } from './JobFilters'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { RefreshCw } from 'lucide-react'
import { LazyLoader } from '@/components/shared/LazyLoader'

export function JobList() {
  const { filters, viewMode, setViewMode } = useJobsStore()

  // Service orchestrator - one infinite query per service
  const serviceQueries = {
    rollup: useJobsInfinite({ service: 'vision', ...filters }),
    faces: useJobsInfinite({ service: 'faces', ...filters }),
    scenes: useJobsInfinite({ service: 'scenes', ...filters }),
    semantics: useJobsInfinite({ service: 'semantics', ...filters }),
    objects: useJobsInfinite({ service: 'objects', ...filters })
  }

  // Active query driven by viewMode
  const activeQuery = serviceQueries[viewMode]
  const { data, isLoading, isFetching, fetchNextPage, hasNextPage, refetch } = activeQuery

  // Fetch job counts for accurate tab counts
  const { data: counts } = useJobsCount({
    status: filters.status,
    service: filters.service,
    source_id: filters.source_id,
    source: filters.source,
    start_date: filters.start_date,
    end_date: filters.end_date,
  })

  // Flatten all pages into single array
  const jobs = data?.pages.flatMap(page => page.jobs) ?? []
  const total = data?.pages[0]?.total ?? 0
  const hasMore = hasNextPage ?? false

  return (
    <div className="space-y-4">
      {/* Filters and refresh */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <JobFilters />
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetch()}
          disabled={isFetching}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Tabs */}
      <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as typeof viewMode)}>
        <TabsList>
          <TabsTrigger value="rollup">
            Vision ({counts?.by_service.vision ?? 0})
          </TabsTrigger>
          <TabsTrigger value="faces">
            Faces ({counts?.by_service.faces ?? 0})
          </TabsTrigger>
          <TabsTrigger value="scenes">
            Scenes ({counts?.by_service.scenes ?? 0})
          </TabsTrigger>
          <TabsTrigger value="semantics">
            Semantics ({counts?.by_service.semantics ?? 0})
          </TabsTrigger>
          <TabsTrigger value="objects">
            Objects ({counts?.by_service.objects ?? 0})
          </TabsTrigger>
        </TabsList>

        {/* Rollup view - vision composite jobs only */}
        <TabsContent value="rollup" className="mt-4">
          {isLoading ? (
            <JobListSkeleton />
          ) : jobs.length === 0 ? (
            <EmptyState message="No composite jobs found" />
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {jobs.map((job) => (
                <JobCard key={job.job_id} job={job} />
              ))}
            </div>
          )}
        </TabsContent>

        {/* Faces view */}
        <TabsContent value="faces" className="mt-4">
          {isLoading ? (
            <JobListSkeleton />
          ) : jobs.length === 0 ? (
            <EmptyState message="No faces jobs found" />
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {jobs.map((job) => (
                <JobCard key={job.job_id} job={job} />
              ))}
            </div>
          )}
        </TabsContent>

        {/* Scenes view */}
        <TabsContent value="scenes" className="mt-4">
          {isLoading ? (
            <JobListSkeleton />
          ) : jobs.length === 0 ? (
            <EmptyState message="No scenes jobs found" />
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {jobs.map((job) => (
                <JobCard key={job.job_id} job={job} />
              ))}
            </div>
          )}
        </TabsContent>

        {/* Semantics view */}
        <TabsContent value="semantics" className="mt-4">
          {isLoading ? (
            <JobListSkeleton />
          ) : jobs.length === 0 ? (
            <EmptyState message="No semantics jobs found" />
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {jobs.map((job) => (
                <JobCard key={job.job_id} job={job} />
              ))}
            </div>
          )}
        </TabsContent>

        {/* Objects view */}
        <TabsContent value="objects" className="mt-4">
          {isLoading ? (
            <JobListSkeleton />
          ) : jobs.length === 0 ? (
            <EmptyState message="No objects jobs found" />
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {jobs.map((job) => (
                <JobCard key={job.job_id} job={job} />
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* Lazy loading */}
      {hasMore && (
        <LazyLoader
          hasMore={hasMore}
          isLoading={isLoading || isFetching}
          onLoadMore={fetchNextPage}
        />
      )}

      {!hasMore && jobs.length > 0 && (
        <p className="text-center text-sm text-muted-foreground pt-4">
          All {total} jobs loaded
        </p>
      )}
    </div>
  )
}

function JobListSkeleton() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="border rounded-lg p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Skeleton className="h-8 w-8 rounded" />
            <div className="space-y-1">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-3 w-16" />
            </div>
          </div>
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-3 w-20" />
        </div>
      ))}
    </div>
  )
}

function EmptyState({ message = 'No jobs found' }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
      <p>{message}</p>
      <p className="text-sm mt-1">Jobs will appear here when you run video analysis</p>
    </div>
  )
}
